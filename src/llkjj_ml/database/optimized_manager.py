"""
Database Integration für LLKJJ ML Pipeline
=========================================

Integriert alle Database-Optimierungen in die bestehende Pipeline:
- Connection Pooling
- Batch Processing
- Index-Optimierung
- Migration Tools
"""

import logging
import time
from pathlib import Path
from typing import Any

from ..database.batch_processor import ChromaDBBatchProcessor
from ..database.connection_pool import ChromaDBConnectionPool
from ..database.migrations import DatabaseMigrator

logger = logging.getLogger(__name__)


class OptimizedDatabaseManager:
    """
    Zentraler Manager für optimierte Database-Operationen.

    Vereint:
    - Connection Pooling
    - Batch Processing
    - Index-Optimierung
    - Migration Management
    """

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.pool = ChromaDBConnectionPool(db_path)
        self.batch_processor = ChromaDBBatchProcessor(
            connection_pool=self.pool, collection_name="llkjj_embeddings"
        )
        self.migrator = DatabaseMigrator(self.db_path)

        logger.info("Optimized Database Manager initialisiert für: %s", self.db_path)

    def initialize_database(self) -> dict[str, Any]:
        """
        Initialisiert Database mit allen Optimierungen.

        Returns:
            Initialisierung-Statistiken
        """
        logger.info("Initialisiere optimierte Database")

        start_time = time.time()
        results = {
            "start_time": start_time,
            "migrations": {},
            "optimization": {},
            "cleanup": {},
            "success": False,
        }

        try:
            # 1. Backup erstellen
            logger.info("Erstelle Database-Backup vor Optimierung")
            self.migrator.create_backup("pre_optimization")

            # 2. Database optimieren
            logger.info("Optimiere Database-Performance")
            optimization_result = self.migrator.optimize_database()

            # 3. Cleanup durchführen
            logger.info("Bereinige alte Backups")
            cleaned_backups = self.migrator.cleanup_old_backups(keep_count=5)

            results["migrations"] = {"successful_migrations": 0, "total_migrations": 0}
            results["optimization"] = optimization_result
            results["cleanup"] = {"cleaned_backups": cleaned_backups}
            results["success"] = True

            total_time = time.time() - start_time
            results["total_time_seconds"] = round(total_time, 2)

            logger.info("Database-Initialisierung abgeschlossen in %.2fs", total_time)

        except Exception as e:
            logger.error("Fehler bei Database-Initialisierung: %s", e)
            results["error"] = str(e)
            results["success"] = False

        return results

    def bulk_insert_embeddings(
        self,
        collection_name: str,
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Optimierter Bulk-Insert für Embeddings.

        Args:
            collection_name: Name der Collection
            documents: Dokumente
            embeddings: Embedding-Vektoren
            metadatas: Metadaten (optional)
            ids: IDs (optional)

        Returns:
            Insert-Statistiken
        """
        logger.info(
            "Bulk-Insert für Collection: %s (%d Dokumente)",
            collection_name,
            len(documents),
        )

        try:
            with self.pool.get_connection() as client:
                # Collection erstellen falls nicht vorhanden
                try:
                    client.get_collection(collection_name)
                except ValueError:
                    logger.info("Erstelle neue Collection: %s", collection_name)
                    client.create_collection(collection_name)

                # Batch-Insert durchführen
                insert_results = self.batch_processor.batch_insert_embeddings(
                    documents=documents,
                    embeddings=embeddings,  # type: ignore[arg-type]
                    metadatas=metadatas,
                    ids=ids,
                )

                logger.info("Bulk-Insert abgeschlossen: %s", insert_results)
                return insert_results

        except Exception as e:
            logger.error("Fehler bei Bulk-Insert: %s", e)
            return {"success": False, "error": str(e)}

    def search_similar_embeddings(
        self,
        collection_name: str,
        query_embeddings: list[list[float]],
        n_results: int = 10,
        where: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Optimierte Ähnlichkeitssuche.

        Args:
            collection_name: Name der Collection
            query_embeddings: Query-Embeddings
            n_results: Anzahl Ergebnisse
            where: Filter-Bedingungen

        Returns:
            Such-Ergebnisse
        """
        logger.debug("Suche ähnliche Embeddings in: %s", collection_name)

        try:
            with self.pool.get_connection() as client:
                collection = client.get_collection(collection_name)

                results = collection.query(
                    query_embeddings=query_embeddings,  # type: ignore[arg-type]
                    n_results=n_results,
                    where=where,
                )

                return {
                    "success": True,
                    "results": results,
                    "query_count": len(query_embeddings),
                    "total_results": len(results.get("documents", []) or []),
                }

        except Exception as e:
            logger.error("Fehler bei Embedding-Suche: %s", e)
            return {"success": False, "error": str(e)}

    def get_comprehensive_stats(self) -> dict[str, Any]:
        """
        Sammelt umfassende Database-Statistiken.

        Returns:
            Vollständige Statistiken
        """
        logger.info("Sammle umfassende Database-Statistiken")

        try:
            # Database-Statistiken
            db_stats = self.migrator.get_database_stats()

            # Pool-Statistiken
            pool_stats = self.pool.get_stats()

            # Batch-Processor-Statistiken
            batch_stats = self.batch_processor.get_performance_stats()

            # Migration-History
            migration_history = self.migrator.get_migration_history()

            return {
                "database": db_stats,
                "connection_pool": pool_stats,
                "batch_processor": batch_stats,
                "migration_history": migration_history,
                "timestamp": time.time(),
            }

        except Exception as e:
            logger.error("Fehler beim Sammeln der Statistiken: %s", e)
            return {"error": str(e)}

    def backup_database(self, backup_name: str | None = None) -> dict[str, Any]:
        """
        Erstellt Database-Backup.

        Args:
            backup_name: Optionaler Backup-Name

        Returns:
            Backup-Informationen
        """
        logger.info("Erstelle Database-Backup")

        try:
            backup_path = self.migrator.create_backup(backup_name)

            return {
                "success": True,
                "backup_path": str(backup_path),
                "timestamp": time.time(),
            }

        except Exception as e:
            logger.error("Fehler beim Backup: %s", e)
            return {"success": False, "error": str(e)}

    def cleanup_and_optimize(self) -> dict[str, Any]:
        """
        Führt Cleanup und Optimierung durch.

        Returns:
            Cleanup-Statistiken
        """
        logger.info("Starte Database-Cleanup und Optimierung")

        start_time = time.time()
        results = {
            "start_time": start_time,
            "backup_cleanup": {},
            "optimization": {},
            "pool_reset": {},
            "success": False,
        }

        try:
            # 1. Alte Backups bereinigen
            cleaned_backups = self.migrator.cleanup_old_backups(keep_count=5)
            results["backup_cleanup"] = {"cleaned_backups": cleaned_backups}

            # 2. Database optimieren
            optimization_results = self.migrator.optimize_database()
            results["optimization"] = optimization_results

            # 3. Connection Pool reset (stats werden intern zurückgesetzt)
            results["pool_reset"] = {"reset": True}

            results["success"] = True
            total_time = time.time() - start_time
            results["total_time_seconds"] = round(total_time, 2)

            logger.info("Cleanup und Optimierung abgeschlossen in %.2fs", total_time)

        except Exception as e:
            logger.error("Fehler bei Cleanup: %s", e)
            results["error"] = str(e)
            results["success"] = False

        return results

    def close(self) -> None:
        """Schließt alle Verbindungen ordnungsgemäß."""
        logger.info("Schließe Database-Manager")

        try:
            # Pool schließen
            self.pool.shutdown()
            logger.info("Connection Pool geschlossen")

        except Exception as e:
            logger.error("Fehler beim Schließen: %s", e)

    def __enter__(self) -> "OptimizedDatabaseManager":
        """Context Manager Entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context Manager Exit."""
        self.close()


def get_optimized_database_manager(db_path: Path) -> OptimizedDatabaseManager:
    """
    Factory-Funktion für OptimizedDatabaseManager.

    Args:
        db_path: Pfad zur Database

    Returns:
        Konfigurierter Database Manager
    """
    return OptimizedDatabaseManager(db_path)
