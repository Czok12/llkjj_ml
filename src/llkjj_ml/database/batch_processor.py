"""
Batch-Insert Optimierungen für LLKJJ ML ChromaDB
===============================================

Optimierte Batch-Operationen für:
- Effiziente Vektor-Einträge
- Bulk-Inserts für große Datenmengen
- Transaction-Management
- Performance-Monitoring
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..database.connection_pool import ChromaDBConnectionPool, get_chroma_pool

logger = logging.getLogger(__name__)


@dataclass
class BatchInsertStats:
    """Statistiken für Batch-Insert-Operationen."""

    total_documents: int = 0
    successful_inserts: int = 0
    failed_inserts: int = 0
    total_time_seconds: float = 0.0
    average_batch_time: float = 0.0
    documents_per_second: float = 0.0
    peak_batch_size: int = 0


class ChromaDBBatchProcessor:
    """
    Optimierter Batch-Processor für ChromaDB-Operationen.

    Features:
    - Effiziente Batch-Inserts
    - Automatische Batch-Größen-Optimierung
    - Retry-Mechanismus für fehlgeschlagene Inserts
    - Performance-Monitoring
    """

    def __init__(
        self,
        connection_pool: ChromaDBConnectionPool,
        collection_name: str = "llkjj_embeddings",
        default_batch_size: int = 100,
        max_batch_size: int = 500,
        retry_attempts: int = 3,
    ):
        self.connection_pool = connection_pool
        self.collection_name = collection_name
        self.default_batch_size = default_batch_size
        self.max_batch_size = max_batch_size
        self.retry_attempts = retry_attempts

        # Statistiken
        self.stats = BatchInsertStats()

        # Performance-Tracking
        self._batch_times: list[float] = []
        self._optimal_batch_size = default_batch_size

        logger.info(f"ChromaDB Batch Processor initialisiert: {collection_name}")

    def _get_or_create_collection(self, client: Any) -> Any:
        """Holt oder erstellt die ChromaDB-Collection."""
        try:
            # Versuche Collection zu holen
            collection = client.get_collection(name=self.collection_name)
            logger.debug(f"Collection '{self.collection_name}' gefunden")
            return collection

        except ValueError:
            # Collection existiert nicht - erstellen
            logger.info(f"Erstelle neue Collection: {self.collection_name}")
            collection = client.create_collection(
                name=self.collection_name,
                metadata={
                    "description": "LLKJJ ML Embeddings für SKR03-Klassifizierung",
                    "version": "1.0",
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
            )
            return collection

    def batch_insert_embeddings(
        self,
        documents: list[str],
        embeddings: list[NDArray[np.float32]],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
        batch_size: int | None = None,
    ) -> dict[str, Any]:
        """
        Führt optimierte Batch-Inserts für Embeddings durch.

        Args:
            documents: Liste der Dokument-Texte
            embeddings: Liste der Embedding-Vektoren
            metadatas: Optionale Metadaten für jedes Dokument
            ids: Optionale IDs für jedes Dokument
            batch_size: Batch-Größe (None = automatisch optimieren)

        Returns:
            Dictionary mit Insert-Statistiken
        """
        if not documents or not embeddings:
            raise ValueError("Dokumente und Embeddings dürfen nicht leer sein")

        if len(documents) != len(embeddings):
            raise ValueError("Anzahl Dokumente muss Anzahl Embeddings entsprechen")

        # Default-Werte setzen
        if metadatas is None:
            metadatas = [{} for _ in documents]
        if ids is None:
            ids = [f"doc_{i}_{int(time.time())}" for i in range(len(documents))]
        if batch_size is None:
            batch_size = self._get_optimal_batch_size()

        logger.info(
            f"Starte Batch-Insert: {len(documents)} Dokumente, Batch-Größe: {batch_size}"
        )

        total_start_time = time.time()
        successful_inserts = 0
        failed_inserts = 0

        with self.connection_pool.get_connection() as client:
            collection = self._get_or_create_collection(client)

            # In Batches aufteilen und verarbeiten
            for i in range(0, len(documents), batch_size):
                batch_start_time = time.time()

                # Batch-Daten extrahieren
                batch_end = min(i + batch_size, len(documents))
                batch_docs = documents[i:batch_end]
                batch_embeddings = [emb.tolist() for emb in embeddings[i:batch_end]]
                batch_metadatas = metadatas[i:batch_end]
                batch_ids = ids[i:batch_end]

                # Batch-Insert mit Retry-Mechanismus
                success = self._insert_batch_with_retry(
                    collection=collection,
                    documents=batch_docs,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas,
                    ids=batch_ids,
                    batch_number=i // batch_size + 1,
                )

                batch_time = time.time() - batch_start_time
                self._batch_times.append(batch_time)

                if success:
                    successful_inserts += len(batch_docs)
                    logger.debug(
                        f"Batch {i//batch_size + 1} erfolgreich: {len(batch_docs)} Dokumente in {batch_time:.2f}s"
                    )
                else:
                    failed_inserts += len(batch_docs)
                    logger.error(
                        f"Batch {i//batch_size + 1} fehlgeschlagen: {len(batch_docs)} Dokumente"
                    )

                # Adaptive Batch-Größen-Optimierung
                self._update_optimal_batch_size(batch_time, len(batch_docs))

        total_time = time.time() - total_start_time

        # Statistiken aktualisieren
        self._update_stats(
            total_documents=len(documents),
            successful_inserts=successful_inserts,
            failed_inserts=failed_inserts,
            total_time=total_time,
            batch_size=batch_size,
        )

        return self._get_insert_summary(total_time, successful_inserts, failed_inserts)

    def _insert_batch_with_retry(
        self,
        collection: Any,
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
        ids: list[str],
        batch_number: int,
    ) -> bool:
        """Führt Batch-Insert mit Retry-Mechanismus durch."""
        for attempt in range(self.retry_attempts):
            try:
                collection.add(
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    ids=ids,
                )
                return True

            except Exception as e:
                logger.warning(
                    f"Batch {batch_number}, Versuch {attempt + 1} fehlgeschlagen: {e}"
                )

                if attempt == self.retry_attempts - 1:
                    logger.error(
                        f"Batch {batch_number} nach {self.retry_attempts} Versuchen fehlgeschlagen"
                    )
                    return False

                # Kurz warten vor nächstem Versuch
                time.sleep(0.1 * (attempt + 1))

        return False

    def _get_optimal_batch_size(self) -> int:
        """Berechnet optimale Batch-Größe basierend auf Performance-History."""
        if not self._batch_times:
            return self.default_batch_size

        # Adaptive Batch-Größe basierend auf durchschnittlicher Zeit
        avg_time = np.mean(self._batch_times[-10:])  # Letzte 10 Batches

        if avg_time < 0.5:  # Sehr schnell - Batch-Größe erhöhen
            self._optimal_batch_size = min(
                self._optimal_batch_size + 10, self.max_batch_size
            )
        elif avg_time > 2.0:  # Zu langsam - Batch-Größe reduzieren
            self._optimal_batch_size = max(self._optimal_batch_size - 10, 10)

        return self._optimal_batch_size

    def _update_optimal_batch_size(self, batch_time: float, batch_size: int) -> None:
        """Aktualisiert optimale Batch-Größe basierend auf aktueller Performance."""
        # Documents per second für diesen Batch
        batch_size / batch_time if batch_time > 0 else 0

        # Ziel: ~1-2 Sekunden pro Batch für optimale Performance
        target_time = 1.5

        if batch_time < target_time * 0.5:  # Zu schnell
            suggested_size = min(batch_size + 20, self.max_batch_size)
        elif batch_time > target_time * 2:  # Zu langsam
            suggested_size = max(batch_size - 20, 10)
        else:
            suggested_size = batch_size  # Aktuell optimal

        # Graduelle Anpassung
        self._optimal_batch_size = int(
            self._optimal_batch_size * 0.8 + suggested_size * 0.2
        )

    def _update_stats(
        self,
        total_documents: int,
        successful_inserts: int,
        failed_inserts: int,
        total_time: float,
        batch_size: int,
    ) -> None:
        """Aktualisiert Performance-Statistiken."""
        self.stats.total_documents += total_documents
        self.stats.successful_inserts += successful_inserts
        self.stats.failed_inserts += failed_inserts
        self.stats.total_time_seconds += total_time

        # Durchschnittswerte berechnen
        if self._batch_times:
            self.stats.average_batch_time = float(np.mean(self._batch_times))

        if total_time > 0:
            self.stats.documents_per_second = successful_inserts / total_time

        self.stats.peak_batch_size = max(batch_size, self.stats.peak_batch_size)

    def _get_insert_summary(
        self, total_time: float, successful_inserts: int, failed_inserts: int
    ) -> dict[str, Any]:
        """Erstellt Zusammenfassung der Insert-Operation."""
        return {
            "total_time_seconds": round(total_time, 2),
            "successful_inserts": successful_inserts,
            "failed_inserts": failed_inserts,
            "success_rate": (
                round(
                    successful_inserts / (successful_inserts + failed_inserts) * 100, 1
                )
                if (successful_inserts + failed_inserts) > 0
                else 0
            ),
            "documents_per_second": (
                round(successful_inserts / total_time, 1) if total_time > 0 else 0
            ),
            "optimal_batch_size": self._optimal_batch_size,
            "average_batch_time": (
                round(np.mean(self._batch_times), 2) if self._batch_times else 0
            ),
        }

    def batch_query_embeddings(
        self,
        query_embeddings: list[NDArray[np.float32]],
        n_results: int = 10,
        where: dict[str, Any] | None = None,
        batch_size: int = 50,
    ) -> list[dict[str, Any]]:
        """
        Führt Batch-Queries für Embeddings durch.

        Args:
            query_embeddings: Liste der Query-Embedding-Vektoren
            n_results: Anzahl Ergebnisse pro Query
            where: Optionale WHERE-Bedingungen
            batch_size: Batch-Größe für Queries

        Returns:
            Liste der Query-Ergebnisse
        """
        logger.info(f"Starte Batch-Query: {len(query_embeddings)} Queries")

        all_results = []

        with self.connection_pool.get_connection() as client:
            collection = self._get_or_create_collection(client)

            # In Batches aufteilen
            for i in range(0, len(query_embeddings), batch_size):
                batch_end = min(i + batch_size, len(query_embeddings))
                batch_queries = [emb.tolist() for emb in query_embeddings[i:batch_end]]

                try:
                    # Batch-Query ausführen
                    batch_results = collection.query(
                        query_embeddings=batch_queries, n_results=n_results, where=where
                    )

                    # Ergebnisse für jede Query einzeln extrahieren
                    for j in range(len(batch_queries)):
                        result = {
                            "query_index": i + j,
                            "documents": (
                                batch_results["documents"][j]
                                if j < len(batch_results["documents"])
                                else []
                            ),
                            "distances": (
                                batch_results["distances"][j]
                                if j < len(batch_results["distances"])
                                else []
                            ),
                            "metadatas": (
                                batch_results["metadatas"][j]
                                if j < len(batch_results["metadatas"])
                                else []
                            ),
                            "ids": (
                                batch_results["ids"][j]
                                if j < len(batch_results["ids"])
                                else []
                            ),
                        }
                        all_results.append(result)

                    logger.debug(
                        f"Batch-Query {i//batch_size + 1} erfolgreich: {len(batch_queries)} Queries"
                    )

                except Exception as e:
                    logger.error(f"Batch-Query {i//batch_size + 1} fehlgeschlagen: {e}")
                    # Leere Ergebnisse für fehlgeschlagene Queries hinzufügen
                    for j in range(len(batch_queries)):
                        all_results.append(
                            {
                                "query_index": i + j,
                                "documents": [],
                                "distances": [],
                                "metadatas": [],
                                "ids": [],
                                "error": str(e),
                            }
                        )

        logger.info(f"Batch-Query abgeschlossen: {len(all_results)} Ergebnisse")
        return all_results

    def get_performance_stats(self) -> dict[str, Any]:
        """Holt umfassende Performance-Statistiken."""
        return {
            "insert_stats": {
                "total_documents": self.stats.total_documents,
                "successful_inserts": self.stats.successful_inserts,
                "failed_inserts": self.stats.failed_inserts,
                "success_rate": round(
                    self.stats.successful_inserts
                    / max(self.stats.total_documents, 1)
                    * 100,
                    1,
                ),
                "total_time_seconds": round(self.stats.total_time_seconds, 2),
                "documents_per_second": round(self.stats.documents_per_second, 1),
                "average_batch_time": round(self.stats.average_batch_time, 2),
                "peak_batch_size": self.stats.peak_batch_size,
            },
            "optimization_stats": {
                "current_optimal_batch_size": self._optimal_batch_size,
                "default_batch_size": self.default_batch_size,
                "max_batch_size": self.max_batch_size,
                "batch_history_length": len(self._batch_times),
                "recent_avg_batch_time": (
                    round(np.mean(self._batch_times[-10:]), 2)
                    if self._batch_times
                    else 0
                ),
            },
            "connection_pool_stats": self.connection_pool.get_stats(),
        }

    def clear_collection(self) -> bool:
        """
        Leert die gesamte Collection (Vorsicht!).

        Returns:
            True wenn erfolgreich
        """
        logger.warning(f"Lösche alle Daten aus Collection: {self.collection_name}")

        try:
            with self.connection_pool.get_connection() as client:
                try:
                    client.delete_collection(name=self.collection_name)
                    logger.info(f"Collection '{self.collection_name}' gelöscht")
                    return True
                except ValueError:
                    logger.info(f"Collection '{self.collection_name}' existierte nicht")
                    return True

        except Exception as e:
            logger.error(f"Fehler beim Löschen der Collection: {e}")
            return False


def create_batch_processor(
    db_path: Path, collection_name: str = "llkjj_embeddings", **kwargs: Any
) -> ChromaDBBatchProcessor:
    """
    Factory-Funktion für ChromaDBBatchProcessor.

    Args:
        db_path: Pfad zur ChromaDB
        collection_name: Name der Collection
        **kwargs: Weitere Parameter für Pool und Processor

    Returns:
        ChromaDBBatchProcessor-Instanz
    """
    # Connection Pool holen/erstellen
    pool = get_chroma_pool(db_path)

    # Batch Processor erstellen
    processor = ChromaDBBatchProcessor(
        connection_pool=pool, collection_name=collection_name, **kwargs
    )

    return processor
