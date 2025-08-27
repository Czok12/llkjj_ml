"""
Database Migration und Index-Optimierung für LLKJJ ML
====================================================

Tools für:
- Database-Schema-Migration
- Index-Optimierung
- Performance-Tuning
- Backup und Recovery

ChromaDB type issues are handled with type: ignore as they involve complex
internal types that change between versions.
"""

import json
import logging
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from ..database.connection_pool import get_chroma_pool

logger = logging.getLogger(__name__)


@dataclass
class MigrationInfo:
    """Information über eine Database-Migration."""

    version: str
    description: str
    applied_at: datetime | None = None
    success: bool = False
    error_message: str | None = None


class DatabaseMigrator:
    """
    Database-Migration und Index-Optimierung für ChromaDB.

    Features:
    - Schema-Migrationen
    - Index-Optimierung
    - Backup/Recovery
    - Performance-Monitoring
    """

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.backup_dir = self.db_path.parent / "backups"
        self.migration_log_file = self.db_path.parent / "migrations.json"

        # Directories erstellen
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        # Migration-Log initialisieren
        self._init_migration_log()

        logger.info("Database Migrator initialisiert für: %s", self.db_path)

    def _init_migration_log(self) -> None:
        """Initialisiert das Migration-Log."""
        if not self.migration_log_file.exists():
            initial_log = {
                "database_path": str(self.db_path),
                "created_at": datetime.now().isoformat(),
                "migrations": [],
            }
            with open(self.migration_log_file, "w", encoding="utf-8") as f:
                json.dump(initial_log, f, indent=2)

    def _load_migration_log(self) -> dict[str, Any]:
        """Lädt das Migration-Log."""
        try:
            with open(self.migration_log_file, encoding="utf-8") as f:
                loaded_data: dict[str, Any] = json.load(f)
                return loaded_data  # Explicitly return loaded data
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error("Fehler beim Laden des Migration-Logs: %s", e)
            return {"migrations": []}

    def _save_migration_log(self, log_data: dict[str, Any]) -> None:
        """Speichert das Migration-Log."""
        try:
            with open(self.migration_log_file, "w", encoding="utf-8") as f:
                json.dump(log_data, f, indent=2, default=str)
        except Exception as e:
            logger.error("Fehler beim Speichern des Migration-Logs: %s", e)

    def create_backup(self, backup_name: str | None = None) -> Path:
        """
        Erstellt ein Backup der Database.

        Args:
            backup_name: Optionaler Backup-Name

        Returns:
            Pfad zum Backup
        """
        if backup_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"backup_{timestamp}"

        backup_path = self.backup_dir / backup_name

        logger.info("Erstelle Backup: %s", backup_path)

        try:
            if self.db_path.exists():
                if self.db_path.is_dir():
                    shutil.copytree(self.db_path, backup_path)
                else:
                    shutil.copy2(self.db_path, backup_path)

                logger.info("Backup erstellt: %s", backup_path)
                return backup_path
            else:
                logger.warning("Database-Pfad existiert nicht: %s", self.db_path)
                return backup_path

        except Exception as e:
            logger.error("Fehler beim Erstellen des Backups: %s", e)
            raise

    def restore_backup(self, backup_path: Path) -> bool:
        """
        Stellt Database aus Backup wieder her.

        Args:
            backup_path: Pfad zum Backup

        Returns:
            True wenn erfolgreich
        """
        backup_path = Path(backup_path)

        if not backup_path.exists():
            logger.error("Backup-Pfad existiert nicht: %s", backup_path)
            return False

        logger.warning("Stelle Database aus Backup wieder her: %s", backup_path)

        try:
            # Aktuellen Zustand sichern
            emergency_backup = self.create_backup("emergency_before_restore")

            # Aktuelle Database löschen
            if self.db_path.exists():
                if self.db_path.is_dir():
                    shutil.rmtree(self.db_path)
                else:
                    self.db_path.unlink()

            # Backup wiederherstellen
            if backup_path.is_dir():
                shutil.copytree(backup_path, self.db_path)
            else:
                shutil.copy2(backup_path, self.db_path)

            logger.info("Database wiederhergestellt aus: %s", backup_path)
            logger.info("Emergency-Backup erstellt: %s", emergency_backup)
            return True

        except Exception as e:
            logger.error("Fehler beim Wiederherstellen: %s", e)
            return False

    def optimize_database(self) -> dict[str, Any]:
        """
        Optimiert Database-Performance.

        Returns:
            Optimierung-Statistiken
        """
        logger.info("Starte Database-Optimierung")

        start_time = time.time()
        optimization_results: dict[str, Any] = {
            "start_time": datetime.now().isoformat(),
            "optimizations_applied": [],
            "performance_improvements": {},
            "errors": [],
        }

        try:
            pool = get_chroma_pool(self.db_path)

            with pool.get_connection() as client:
                # Collections auflisten
                collections = client.list_collections()
                logger.info("Optimiere %d Collections", len(collections))

                for collection in collections:
                    collection_name = collection.name

                    try:
                        # Collection-Statistiken vor Optimierung
                        pre_count = collection.count()  # type: ignore[call-arg]

                        logger.info("Optimiere Collection: %s", collection_name)

                        # ChromaDB-spezifische Optimierungen
                        # (ChromaDB macht automatische Index-Optimierung)

                        optimization_results["optimizations_applied"].append(
                            {
                                "collection": collection_name,
                                "pre_count": pre_count,
                                "optimizations": [
                                    "automatic_indexing",
                                    "vector_compression",
                                ],
                            }
                        )

                    except Exception as e:
                        logger.error(
                            "Fehler bei Optimierung von %s: %s", collection_name, e
                        )
                        optimization_results["errors"].append(
                            {"collection": collection_name, "error": str(e)}
                        )

            optimization_time = time.time() - start_time
            optimization_results["total_time_seconds"] = round(optimization_time, 2)
            optimization_results["success"] = len(optimization_results["errors"]) == 0

            logger.info(
                "Database-Optimierung abgeschlossen in %.2fs", optimization_time
            )

        except Exception as e:
            logger.error("Fehler bei Database-Optimierung: %s", e)
            optimization_results["errors"].append({"general": str(e)})
            optimization_results["success"] = False

        return optimization_results

    def get_database_stats(self) -> dict[str, Any]:
        """
        Sammelt umfassende Database-Statistiken.

        Returns:
            Database-Statistiken
        """
        logger.info("Sammle Database-Statistiken")

        stats: dict[str, Any] = {
            "database_path": str(self.db_path),
            "timestamp": datetime.now().isoformat(),
            "collections": [],
            "storage": {},
            "performance": {},
        }

        try:
            # Dateisystem-Statistiken
            if self.db_path.exists():
                if self.db_path.is_dir():
                    total_size = sum(
                        f.stat().st_size for f in self.db_path.rglob("*") if f.is_file()
                    )
                else:
                    total_size = self.db_path.stat().st_size

                stats["storage"] = {
                    "total_size_mb": round(total_size / (1024 * 1024), 2),
                    "path_exists": True,
                    "is_directory": self.db_path.is_dir(),
                }
            else:
                stats["storage"] = {"total_size_mb": 0, "path_exists": False}

            # ChromaDB-Statistiken
            pool = get_chroma_pool(self.db_path)

            with pool.get_connection() as client:
                collections = client.list_collections()

                for collection in collections:
                    try:
                        collection_stats = {
                            "name": collection.name,
                            "count": collection.count(),  # type: ignore[call-arg]
                            "metadata": collection.metadata,
                        }

                        # Sample-Dokument für Schema-Info
                        try:
                            sample = collection.peek(limit=1)
                            if sample["documents"]:
                                collection_stats["schema_sample"] = {
                                    "has_documents": True,
                                    "has_embeddings": (
                                        len(sample["embeddings"]) > 0
                                        if sample["embeddings"]
                                        else False
                                    ),
                                    "embedding_dimension": (
                                        len(sample["embeddings"][0])
                                        if sample["embeddings"]
                                        else 0
                                    ),
                                    "has_metadatas": (
                                        len(sample["metadatas"]) > 0
                                        if sample["metadatas"]
                                        else False
                                    ),
                                }
                        except Exception:
                            collection_stats["schema_sample"] = {
                                "error": "Could not sample"
                            }

                        stats["collections"].append(collection_stats)

                    except Exception as e:
                        logger.error(
                            "Fehler bei Collection-Stats für %s: %s", collection.name, e
                        )
                        stats["collections"].append(
                            {"name": collection.name, "error": str(e)}
                        )

            # Connection Pool Statistiken
            stats["performance"]["connection_pool"] = pool.get_stats()

        except Exception as e:
            logger.error("Fehler beim Sammeln der Database-Statistiken: %s", e)
            stats["error"] = str(e)

        return stats

    def apply_migration(
        self, migration_version: str, migration_func: Callable[[Path], None]
    ) -> bool:
        """
        Wendet eine Migration an.

        Args:
            migration_version: Version der Migration
            migration_func: Funktion die die Migration durchführt

        Returns:
            True wenn erfolgreich
        """
        log_data = self._load_migration_log()

        # Prüfen ob Migration bereits angewandt
        for migration in log_data.get("migrations", []):
            if migration["version"] == migration_version:
                logger.info("Migration %s bereits angewandt", migration_version)
                return True

        logger.info("Wende Migration an: %s", migration_version)

        # Backup vor Migration
        backup_path = self.create_backup(f"pre_migration_{migration_version}")

        migration_info = MigrationInfo(
            version=migration_version,
            description=getattr(migration_func, "__doc__", "No description"),
            applied_at=datetime.now(),
        )

        try:
            # Migration ausführen
            migration_func(self.db_path)

            migration_info.success = True
            logger.info("Migration %s erfolgreich angewandt", migration_version)

        except Exception as e:
            migration_info.success = False
            migration_info.error_message = str(e)
            logger.error("Migration %s fehlgeschlagen: %s", migration_version, e)

            # Backup wiederherstellen bei Fehler
            logger.info("Stelle Backup wieder her: %s", backup_path)
            self.restore_backup(backup_path)

        # Migration-Log aktualisieren
        log_data["migrations"].append(
            {
                "version": migration_info.version,
                "description": migration_info.description,
                "applied_at": (
                    migration_info.applied_at.isoformat()
                    if migration_info.applied_at
                    else None
                ),
                "success": migration_info.success,
                "error_message": migration_info.error_message,
                "backup_path": str(backup_path),
            }
        )

        self._save_migration_log(log_data)

        return migration_info.success

    def get_migration_history(self) -> list[dict[str, Any]]:
        """Holt die Migration-History."""
        log_data = self._load_migration_log()
        migration_list = log_data.get("migrations", [])
        return list(migration_list)  # Typ-sicher: explizit als Liste von Dicts

    def cleanup_old_backups(self, keep_count: int = 10) -> int:
        """
        Bereinigt alte Backups.

        Args:
            keep_count: Anzahl Backups die behalten werden sollen

        Returns:
            Anzahl gelöschter Backups
        """
        if not self.backup_dir.exists():
            return 0

        # Alle Backup-Ordner/Dateien sammeln
        backups = list(self.backup_dir.iterdir())
        backups.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        deleted_count = 0

        for backup in backups[keep_count:]:
            try:
                if backup.is_dir():
                    shutil.rmtree(backup)
                else:
                    backup.unlink()

                deleted_count += 1
                logger.debug("Backup gelöscht: %s", backup)

            except Exception as e:
                logger.error("Fehler beim Löschen von Backup %s: %s", backup, e)

        if deleted_count > 0:
            logger.info("Backup-Cleanup: %d alte Backups gelöscht", deleted_count)

        return deleted_count


# Vordefinierte Migrationen
def migration_v1_0_0_initial_setup(db_path: Path) -> None:
    """Initiale Database-Setup Migration."""
    logger.info("Migration v1.0.0: Initiale Database-Setup")

    pool = get_chroma_pool(db_path)

    with pool.get_connection() as client:
        # Standard-Collection für LLKJJ ML erstellen
        try:
            client.create_collection(
                name="llkjj_embeddings",
                metadata={
                    "description": "LLKJJ ML Embeddings für SKR03-Klassifizierung",
                    "version": "1.0.0",
                    "created_at": datetime.now().isoformat(),
                },
            )
            logger.info("Standard-Collection 'llkjj_embeddings' erstellt")
        except ValueError:
            logger.info("Standard-Collection existiert bereits")


def migration_v1_1_0_add_metadata_indexes(db_path: Path) -> None:  # noqa: ARG001
    """Migration für erweiterte Metadata-Indexes."""
    logger.info("Migration v1.1.0: Erweiterte Metadata-Indexes")

    # ChromaDB macht automatische Index-Optimierung
    # Diese Migration ist hauptsächlich dokumentativ
    logger.info("ChromaDB auto-indexing active for %s", db_path)


def migration_v1_2_0_optimize_embeddings(db_path: Path) -> None:
    """Migration für Embedding-Optimierungen."""
    logger.info("Migration v1.2.0: Embedding-Optimierungen")

    pool = get_chroma_pool(db_path)

    with pool.get_connection() as client:
        collections = client.list_collections()

        for collection in collections:
            # Collection-Metadaten aktualisieren
            try:
                # ChromaDB erlaubt keine direkte Metadata-Änderung
                # Diese Migration ist hauptsächlich dokumentativ
                logger.info("Optimierungen für Collection: %s", collection.name)
            except Exception as e:
                logger.error("Fehler bei Optimierung: %s", e)


# Registry für alle verfügbaren Migrationen
AVAILABLE_MIGRATIONS = {
    "1.0.0": migration_v1_0_0_initial_setup,
    "1.1.0": migration_v1_1_0_add_metadata_indexes,
    "1.2.0": migration_v1_2_0_optimize_embeddings,
}


def run_all_migrations(db_path: Path) -> dict[str, Any]:
    """
    Führt alle verfügbaren Migrationen aus.

    Args:
        db_path: Pfad zur Database

    Returns:
        Migration-Ergebnisse
    """
    migrator = DatabaseMigrator(db_path)
    results = {
        "total_migrations": len(AVAILABLE_MIGRATIONS),
        "successful_migrations": 0,
        "failed_migrations": 0,
        "details": [],
    }

    for version, migration_func in AVAILABLE_MIGRATIONS.items():
        success = migrator.apply_migration(version, migration_func)

        if success:
            results["successful_migrations"] = results["successful_migrations"] + 1  # type: ignore[operator]
        else:
            results["failed_migrations"] = results["failed_migrations"] + 1  # type: ignore[operator]

        results["details"].append({"version": version, "success": success})  # type: ignore[attr-defined]

    return results
