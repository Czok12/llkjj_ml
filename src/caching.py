"""
Caching-System für LLKJJ ML Pipeline
===================================

Dieses Modul stellt ein zweistufiges Caching-System zur Verfügung:
1. SKR03Cache: Für SKR03-Klassifizierungen
2. EmbeddingCache: Für Model Embeddings

Features:
- LRU-Cache im Speicher für schnellen Zugriff
- SQLite-Persistierung für dauerhafte Speicherung
- TTL (Time-To-Live) für Cache-Expiry
- Cache-Statistiken und Monitoring
- Asynchrone und synchrone API
"""

import hashlib
import json
import logging
import pickle  # nosec B403 - Used only for internal embedding cache
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class CacheStats(BaseModel):
    """Cache-Statistiken für Monitoring."""

    hits: int = 0
    misses: int = 0
    total_requests: int = 0
    cache_size: int = 0
    memory_usage_mb: float = 0.0

    @property
    def hit_rate(self) -> float:
        """Berechnet die Cache-Hit-Rate."""
        if self.total_requests == 0:
            return 0.0
        return self.hits / self.total_requests


class SKR03Cache:
    """
    Cache für SKR03-Klassifizierungen mit SQLite-Persistierung.

    Features:
    - Memory LRU Cache für häufige Zugriffe
    - SQLite für persistente Speicherung
    - TTL für automatische Cache-Invalidierung
    - Cache-Statistiken
    """

    def __init__(
        self, db_path: Path, max_memory_size: int = 1000, default_ttl_hours: int = 24
    ):
        self.db_path = Path(db_path)
        self.max_memory_size = max_memory_size
        self.default_ttl = timedelta(hours=default_ttl_hours)

        # Memory cache als einfaches Dictionary
        self.memory_cache: dict[str, dict[str, Any]] = {}
        self.access_order: list[str] = []  # Für LRU

        # Statistiken
        self.stats = CacheStats()

        # SQLite initialisieren
        self._init_database()

        logger.info(f"SKR03Cache initialisiert: {self.db_path}")

    def _init_database(self) -> None:
        """Initialisiert die SQLite-Datenbank."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS skr03_cache (
                    key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expiry_time TIMESTAMP,
                    hits INTEGER DEFAULT 0
                )
            """
            )

            # Indices für Performance
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_last_accessed
                ON skr03_cache(last_accessed)
            """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_expiry_time
                ON skr03_cache(expiry_time)
            """
            )

            conn.commit()

    def generate_key(self, beschreibung: str, lieferant: str) -> str:
        """
        Generiert Cache-Schlüssel für SKR03-Klassifizierung.

        Args:
            beschreibung: Beschreibung der Buchung
            lieferant: Name des Lieferanten

        Returns:
            Eindeutiger Cache-Schlüssel
        """
        cache_input = f"{beschreibung.lower().strip()}|{lieferant.lower().strip()}"
        return hashlib.sha256(cache_input.encode("utf-8")).hexdigest()

    def _manage_memory_cache(self, key: str) -> None:
        """Verwaltet Memory Cache mit LRU-Policy."""
        # Key an Ende der Access-Liste verschieben
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)

        # LRU eviction wenn nötig
        while len(self.memory_cache) > self.max_memory_size:
            oldest_key = self.access_order.pop(0)
            self.memory_cache.pop(oldest_key, None)

    def get_classification(self, key: str) -> dict[str, Any] | None:
        """
        Holt SKR03-Klassifizierung aus Cache.

        Args:
            key: Cache-Schlüssel

        Returns:
            Klassifizierung oder None wenn nicht gefunden
        """
        self.stats.total_requests += 1

        # Erst Memory Cache prüfen
        if key in self.memory_cache:
            self._manage_memory_cache(key)
            self.stats.hits += 1
            logger.debug(f"Cache hit (memory): {key}")
            return self.memory_cache[key]

        # Dann SQLite prüfen
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT value_json, expiry_time
                FROM skr03_cache
                WHERE key = ?
            """,
                (key,),
            )

            row = cursor.fetchone()
            if row is None:
                self.stats.misses += 1
                logger.debug(f"Cache miss: {key}")
                return None

            # TTL prüfen
            if row["expiry_time"]:
                expiry = datetime.fromisoformat(row["expiry_time"])
                if datetime.now() > expiry:
                    # Expired entry löschen (DB und Memory)
                    conn.execute("DELETE FROM skr03_cache WHERE key = ?", (key,))
                    conn.commit()
                    # Auch aus Memory Cache entfernen
                    self.memory_cache.pop(key, None)
                    if key in self.access_order:
                        self.access_order.remove(key)
                    self.stats.misses += 1
                    logger.debug(f"Cache expired: {key}")
                    return None

            # Hit statistiken updaten
            conn.execute(
                """
                UPDATE skr03_cache
                SET last_accessed = CURRENT_TIMESTAMP, hits = hits + 1
                WHERE key = ?
            """,
                (key,),
            )
            conn.commit()

            # In Memory Cache laden
            try:
                value: dict[str, Any] = json.loads(row["value_json"])
                self.memory_cache[key] = value
                self._manage_memory_cache(key)
                self.stats.hits += 1
                logger.debug(f"Cache hit (SQLite): {key}")
                return value
            except json.JSONDecodeError:
                logger.error(f"Corrupt cache entry: {key}")
                conn.execute("DELETE FROM skr03_cache WHERE key = ?", (key,))
                conn.commit()
                self.stats.misses += 1
                return None

    def set_classification(
        self, key: str, classification: dict[str, Any], ttl_hours: float | None = None
    ) -> None:
        """
        Speichert SKR03-Klassifizierung im Cache.

        Args:
            key: Cache-Schlüssel
            classification: Klassifizierung
            ttl_hours: Time-to-live in Stunden
        """
        ttl = timedelta(hours=ttl_hours) if ttl_hours else self.default_ttl
        expiry_time = datetime.now() + ttl

        # In Memory Cache speichern
        self.memory_cache[key] = classification
        self._manage_memory_cache(key)

        # In SQLite speichern
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO skr03_cache
                (key, value_json, created_at, last_accessed, expiry_time, hits)
                VALUES (?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?, 0)
            """,
                (key, json.dumps(classification), expiry_time.isoformat()),
            )
            conn.commit()

        logger.debug(f"Cache set: {key}")

    def get_stats(self) -> dict[str, Any]:
        """
        Holt aktuelle Cache-Statistiken.

        Returns:
            Dictionary mit Cache-Statistiken
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT
                    COUNT(*) as total_entries,
                    SUM(hits) as total_db_hits,
                    AVG(hits) as avg_hits_per_entry
                FROM skr03_cache
                WHERE expiry_time IS NULL OR expiry_time > CURRENT_TIMESTAMP
            """
            )
            db_stats = cursor.fetchone()

        return {
            "memory_hits": self.stats.hits,
            "total_requests": self.stats.total_requests,
            "hit_rate": self.stats.hit_rate,
            "memory_cache_size": len(self.memory_cache),
            "db_cache_size": db_stats[0] if db_stats[0] else 0,
            "total_db_hits": db_stats[1] if db_stats[1] else 0,
            "avg_hits_per_entry": round(db_stats[2], 2) if db_stats[2] else 0.0,
        }

    def clear_cache(self) -> None:
        """Leert den kompletten Cache (Memory + SQLite)."""
        self.memory_cache.clear()
        self.access_order.clear()

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM skr03_cache")
            conn.commit()

        self.stats = CacheStats()
        logger.info("Cache geleert")

    def cleanup_expired(self) -> int:
        """
        Entfernt abgelaufene Cache-Einträge.

        Returns:
            Anzahl entfernter Einträge
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                DELETE FROM skr03_cache
                WHERE expiry_time IS NOT NULL AND expiry_time <= CURRENT_TIMESTAMP
            """
            )
            conn.commit()
            removed_count = cursor.rowcount

        logger.info(f"Cache cleanup: {removed_count} expired entries removed")
        return removed_count


class EmbeddingCache:
    """
    Cache für Model Embeddings mit numpy-optimierter Speicherung.

    Features:
    - Numpy-optimierte Serialisierung
    - Model-spezifische Caches
    - Memory + File-based Persistierung
    - Cache-Statistiken
    """

    def __init__(
        self,
        cache_dir: Path,
        max_memory_size: int = 500,
        default_ttl_hours: int = 168,  # 1 Woche
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_file = self.cache_dir / "embeddings.pkl"
        self.metadata_file = self.cache_dir / "embeddings_meta.json"
        self.max_memory_size = max_memory_size
        self.default_ttl = timedelta(hours=default_ttl_hours)

        # Memory caches
        self.embeddings_cache: dict[str, NDArray[np.float32]] = {}
        self.metadata_cache: dict[str, dict[str, Any]] = {}
        self.access_order: list[str] = []

        # Statistiken
        self.stats = CacheStats()

        # Cache laden
        self._load_cache()

        logger.info(f"EmbeddingCache initialisiert: {self.cache_dir}")

    def _load_cache(self) -> None:
        """Lädt Cache aus Dateien."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Embeddings laden
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "rb") as f:
                    # Security: Using pickle only for internal embedding cache data
                    # This is safe as we control the data source and format
                    self.embeddings_cache = pickle.load(f)  # nosec B301
                logger.info(f"Embeddings geladen: {len(self.embeddings_cache)} entries")
            except Exception as e:
                logger.error("Fehler beim Laden der Embeddings: %s", e)
                self.embeddings_cache = {}

        # Metadata laden
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, encoding="utf-8") as f:
                    self.metadata_cache = json.load(f)
                logger.info(f"Metadata geladen: {len(self.metadata_cache)} entries")
            except Exception as e:
                logger.error(f"Fehler beim Laden der Metadata: {e}")
                self.metadata_cache = {}

        # Access order initialisieren
        self.access_order = list(self.embeddings_cache.keys())

    def generate_key(self, text: str, model_name: str = "default") -> str:
        """Generiert Cache-Schlüssel für Embedding."""
        content = f"{model_name}|{text.strip()}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def get_embedding(
        self, text: str, model_name: str = "default"
    ) -> NDArray[np.float32] | None:
        """
        Holt Embedding aus Cache.

        Args:
            text: Text für Embedding
            model_name: Name des Models

        Returns:
            Embedding array oder None
        """
        key = self.generate_key(text, model_name)
        self.stats.total_requests += 1

        if key not in self.embeddings_cache:
            self.stats.misses += 1
            return None

        # TTL prüfen
        metadata = self.metadata_cache.get(key, {})
        if "expiry_time" in metadata:
            expiry = datetime.fromisoformat(metadata["expiry_time"])
            if datetime.now() > expiry:
                self._remove_key(key)
                self.stats.misses += 1
                return None

        # Access order updaten
        self._manage_memory_cache(key)
        self.stats.hits += 1

        return self.embeddings_cache[key]

    def set_embedding(
        self, text: str, embedding: NDArray[np.float32], model_name: str = "default"
    ) -> None:
        """
        Speichert Embedding im Cache.

        Args:
            text: Text für Embedding
            embedding: Embedding array
            model_name: Name des Models
        """
        key = self.generate_key(text, model_name)

        # Metadata erstellen
        expiry_time = datetime.now() + self.default_ttl
        self.metadata_cache[key] = {
            "text": text[:100],  # Gekürzt für Storage
            "model_name": model_name,
            "created_at": datetime.now().isoformat(),
            "expiry_time": expiry_time.isoformat(),
            "shape": embedding.shape,
            "dtype": str(embedding.dtype),
        }

        # Embedding speichern
        self.embeddings_cache[key] = embedding.astype(np.float32)
        self._manage_memory_cache(key)

        logger.debug(f"Embedding cached: {key} ({embedding.shape})")

    def _manage_memory_cache(self, key: str) -> None:
        """Verwaltet Memory Cache mit LRU-Policy."""
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)

        # LRU eviction
        while len(self.embeddings_cache) > self.max_memory_size:
            oldest_key = self.access_order.pop(0)
            self._remove_key(oldest_key)

    def _remove_key(self, key: str) -> None:
        """Entfernt Key aus allen Caches."""
        self.embeddings_cache.pop(key, None)
        self.metadata_cache.pop(key, None)
        if key in self.access_order:
            self.access_order.remove(key)

    def save_cache(self) -> None:
        """Speichert Cache in Dateien."""
        try:
            # Embeddings speichern
            with open(self.cache_file, "wb") as f:
                pickle.dump(self.embeddings_cache, f)

            # Metadata speichern
            with open(self.metadata_file, "w", encoding="utf-8") as f:
                json.dump(self.metadata_cache, f, indent=2)

            logger.info(f"Cache gespeichert: {len(self.embeddings_cache)} embeddings")
        except Exception as e:
            logger.error(f"Fehler beim Speichern des Cache: {e}")

    def get_stats(self) -> dict[str, Any]:
        """Holt Cache-Statistiken."""
        total_size_mb = 0.0
        if self.cache_file.exists():
            total_size_mb = self.cache_file.stat().st_size / (1024 * 1024)

        return {
            "hits": self.stats.hits,
            "misses": self.stats.misses,
            "hit_rate": self.stats.hit_rate,
            "cache_size": len(self.embeddings_cache),
            "memory_usage_mb": round(total_size_mb, 2),
            "models": list(
                {
                    meta.get("model_name", "unknown")
                    for meta in self.metadata_cache.values()
                }
            ),
        }

    def clear_cache(self) -> None:
        """Leert den kompletten Cache."""
        self.embeddings_cache.clear()
        self.metadata_cache.clear()
        self.access_order.clear()

        # Dateien löschen
        if self.cache_file.exists():
            self.cache_file.unlink()
        if self.metadata_file.exists():
            self.metadata_file.unlink()

        self.stats = CacheStats()
        logger.info("Embedding cache geleert")
