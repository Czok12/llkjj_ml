"""
ChromaDB Connection Pool für LLKJJ ML Pipeline
==============================================

Optimiertes Connection Pooling für ChromaDB mit:
- Thread-safe Connection Management
- Automatische Connection-Wiederverwendung
- Lazy Loading und Resource-Cleanup
- Performance-Monitoring
"""

import logging
import threading
import time
from collections import deque
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api import ClientAPI
from chromadb.config import Settings

logger = logging.getLogger(__name__)


@dataclass
class ConnectionStats:
    """Statistiken für Connection Pool."""

    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    total_requests: int = 0
    average_wait_time: float = 0.0
    peak_connections: int = 0
    connection_errors: int = 0


class ChromaDBConnection:
    """Wrapper für ChromaDB-Client mit Metadaten."""

    def __init__(self, client: ClientAPI, connection_id: str):
        self.client = client
        self.connection_id = connection_id
        self.created_at = time.time()
        self.last_used = time.time()
        self.usage_count = 0
        self.is_busy = False

    def mark_used(self) -> None:
        """Markiert Connection als verwendet."""
        self.last_used = time.time()
        self.usage_count += 1
        self.is_busy = True

    def mark_idle(self) -> None:
        """Markiert Connection als idle."""
        self.is_busy = False

    @property
    def age_seconds(self) -> float:
        """Alter der Connection in Sekunden."""
        return time.time() - self.created_at

    @property
    def idle_time(self) -> float:
        """Zeit seit letzter Verwendung."""
        return time.time() - self.last_used


class ChromaDBConnectionPool:
    """
    Thread-safe Connection Pool für ChromaDB.

    Features:
    - Connection-Wiederverwendung
    - Automatische Cleanup alter Connections
    - Performance-Monitoring
    - Thread-safe Operations
    """

    def __init__(
        self,
        db_path: Path,
        min_connections: int = 2,
        max_connections: int = 10,
        max_idle_time: int = 300,  # 5 Minuten
        max_connection_age: int = 3600,  # 1 Stunde
    ):
        self.db_path = Path(db_path)
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.max_idle_time = max_idle_time
        self.max_connection_age = max_connection_age

        # Thread-safe collections
        self._lock = threading.RLock()
        self._idle_connections: deque[ChromaDBConnection] = deque()
        self._active_connections: dict[str, ChromaDBConnection] = {}

        # Statistiken
        self.stats = ConnectionStats()

        # Cleanup-Thread
        self._cleanup_thread: threading.Thread | None = None
        self._shutdown_event = threading.Event()

        # ChromaDB Settings - Modern Configuration
        self._chroma_settings = Settings(anonymized_telemetry=False, allow_reset=False)

        # Initial connections erstellen
        self._initialize_pool()

        logger.info(f"ChromaDB Connection Pool initialisiert: {self.db_path}")

    def _initialize_pool(self) -> None:
        """Initialisiert den Connection Pool."""
        with self._lock:
            # Minimum connections erstellen
            for i in range(self.min_connections):
                try:
                    connection = self._create_connection(f"init_{i}")
                    self._idle_connections.append(connection)
                    self.stats.total_connections += 1
                except Exception as e:
                    logger.error(f"Fehler beim Erstellen der Initial-Connection: {e}")
                    self.stats.connection_errors += 1

            # Cleanup-Thread starten
            self._start_cleanup_thread()

    def _create_connection(self, connection_id: str) -> ChromaDBConnection:
        """Erstellt eine neue ChromaDB-Connection."""
        try:
            # Moderne ChromaDB Client API
            client = chromadb.PersistentClient(
                path=str(self.db_path)
                # Keine deprecated Settings mehr nötig
            )

            connection = ChromaDBConnection(client, connection_id)
            logger.debug(f"Neue ChromaDB-Connection erstellt: {connection_id}")
            return connection

        except Exception as e:
            logger.error(f"Fehler beim Erstellen der ChromaDB-Connection: {e}")
            raise

    def _start_cleanup_thread(self) -> None:
        """Startet den Cleanup-Thread."""
        if self._cleanup_thread is None or not self._cleanup_thread.is_alive():
            self._cleanup_thread = threading.Thread(
                target=self._cleanup_worker, name="ChromaDBCleanup", daemon=True
            )
            self._cleanup_thread.start()

    def _cleanup_worker(self) -> None:
        """Worker-Thread für Connection-Cleanup."""
        while not self._shutdown_event.wait(timeout=60):  # Alle 60 Sekunden
            try:
                self._cleanup_old_connections()
            except Exception as e:
                logger.error(f"Fehler im Cleanup-Worker: {e}")

    def _cleanup_old_connections(self) -> None:
        """Bereinigt alte und unbenutzte Connections."""
        time.time()
        cleaned_count = 0

        with self._lock:
            # Idle connections prüfen
            while self._idle_connections:
                connection = self._idle_connections[0]

                should_remove = (
                    connection.idle_time > self.max_idle_time
                    or connection.age_seconds > self.max_connection_age
                )

                if should_remove and len(self._idle_connections) > self.min_connections:
                    removed = self._idle_connections.popleft()
                    self.stats.total_connections -= 1
                    cleaned_count += 1
                    logger.debug(f"Connection bereinigt: {removed.connection_id}")
                else:
                    break

            # Active connections auf Alter prüfen (nur loggen)
            for conn_id, connection in list(self._active_connections.items()):
                if connection.age_seconds > self.max_connection_age:
                    logger.warning(
                        f"Alte aktive Connection: {conn_id} (Age: {connection.age_seconds:.1f}s)"
                    )

        if cleaned_count > 0:
            logger.info(f"Connection-Cleanup: {cleaned_count} Connections bereinigt")

    @contextmanager
    def get_connection(self) -> Generator[ClientAPI, None, None]:
        """
        Context Manager für Connection-Abruf.

        Yields:
            ChromaDB ClientAPI
        """
        start_time = time.time()
        connection = None

        try:
            connection = self._acquire_connection()
            self.stats.total_requests += 1

            wait_time = time.time() - start_time
            self.stats.average_wait_time = (
                self.stats.average_wait_time * (self.stats.total_requests - 1)
                + wait_time
            ) / self.stats.total_requests

            yield connection.client

        finally:
            if connection:
                self._release_connection(connection)

    def _acquire_connection(self) -> ChromaDBConnection:
        """Akquiriert eine Connection aus dem Pool."""
        with self._lock:
            # Idle connection verwenden wenn verfügbar
            if self._idle_connections:
                connection = self._idle_connections.popleft()
                connection.mark_used()
                self._active_connections[connection.connection_id] = connection
                self.stats.active_connections = len(self._active_connections)
                self.stats.idle_connections = len(self._idle_connections)
                logger.debug(f"Connection wiederverwendet: {connection.connection_id}")
                return connection

            # Neue Connection erstellen wenn unter Maximum
            if len(self._active_connections) < self.max_connections:
                connection_id = (
                    f"pool_{int(time.time())}_{len(self._active_connections)}"
                )
                try:
                    connection = self._create_connection(connection_id)
                    connection.mark_used()
                    self._active_connections[connection_id] = connection
                    self.stats.total_connections += 1
                    self.stats.active_connections = len(self._active_connections)

                    # Peak tracking
                    self.stats.peak_connections = max(
                        self.stats.active_connections, self.stats.peak_connections
                    )

                    logger.debug(f"Neue Connection erstellt: {connection_id}")
                    return connection

                except Exception as e:
                    self.stats.connection_errors += 1
                    logger.error(f"Fehler beim Erstellen neuer Connection: {e}")
                    raise

            # Pool ist voll - das sollte nicht passieren bei normalem Gebrauch
            raise RuntimeError(
                f"Connection Pool ist voll ({self.max_connections} Connections)"
            )

    def _release_connection(self, connection: ChromaDBConnection) -> None:
        """Gibt eine Connection zurück an den Pool."""
        with self._lock:
            # Connection aus active entfernen
            self._active_connections.pop(connection.connection_id, None)

            # Connection als idle markieren und zu Pool hinzufügen
            connection.mark_idle()
            self._idle_connections.append(connection)

            # Statistiken updaten
            self.stats.active_connections = len(self._active_connections)
            self.stats.idle_connections = len(self._idle_connections)

            logger.debug(f"Connection freigegeben: {connection.connection_id}")

    def get_stats(self) -> dict[str, Any]:
        """Holt aktuelle Pool-Statistiken."""
        with self._lock:
            return {
                "total_connections": self.stats.total_connections,
                "active_connections": self.stats.active_connections,
                "idle_connections": self.stats.idle_connections,
                "total_requests": self.stats.total_requests,
                "average_wait_time_ms": round(self.stats.average_wait_time * 1000, 2),
                "peak_connections": self.stats.peak_connections,
                "connection_errors": self.stats.connection_errors,
                "pool_utilization": round(
                    self.stats.active_connections / self.max_connections * 100, 1
                ),
                "pool_efficiency": round(
                    (self.stats.total_requests - self.stats.connection_errors)
                    / max(self.stats.total_requests, 1)
                    * 100,
                    1,
                ),
            }

    def shutdown(self) -> None:
        """Beendet den Connection Pool sauber."""
        logger.info("ChromaDB Connection Pool wird heruntergefahren...")

        # Cleanup-Thread stoppen
        self._shutdown_event.set()
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5)

        with self._lock:
            # Alle Connections schließen
            all_connections = list(self._idle_connections) + list(
                self._active_connections.values()
            )

            for connection in all_connections:
                try:
                    # ChromaDB-Connections haben keinen expliziten close()
                    # Cleanup wird durch Garbage Collection gemacht
                    pass
                except Exception as e:
                    logger.error(
                        f"Fehler beim Schließen der Connection {connection.connection_id}: {e}"
                    )

            # Collections leeren
            self._idle_connections.clear()
            self._active_connections.clear()
            self.stats = ConnectionStats()

        logger.info("ChromaDB Connection Pool heruntergefahren")

    def __enter__(self) -> "ChromaDBConnectionPool":
        """Context Manager Entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context Manager Exit."""
        self.shutdown()


# Global Pool Instance für einfache Verwendung
_global_pool: ChromaDBConnectionPool | None = None
_pool_lock = threading.Lock()


def get_chroma_pool(
    db_path: Path, min_connections: int = 2, max_connections: int = 10, **kwargs: Any
) -> ChromaDBConnectionPool:
    """
    Holt oder erstellt den globalen ChromaDB Connection Pool.

    Args:
        db_path: Pfad zur ChromaDB
        min_connections: Minimum Connections
        max_connections: Maximum Connections
        **kwargs: Weitere Pool-Parameter

    Returns:
        ChromaDBConnectionPool-Instanz
    """
    global _global_pool

    with _pool_lock:
        if _global_pool is None:
            _global_pool = ChromaDBConnectionPool(
                db_path=db_path,
                min_connections=min_connections,
                max_connections=max_connections,
                **kwargs,
            )
        return _global_pool


def shutdown_global_pool() -> None:
    """Beendet den globalen Pool."""
    global _global_pool

    with _pool_lock:
        if _global_pool is not None:
            _global_pool.shutdown()
            _global_pool = None
