"""
Vereinfachter Database Manager für LLKJJ ML ChromaDB
===================================================

Basisfunktionalität für ChromaDB-Operationen ohne komplexe Dependencies.
"""

import logging
import time
from pathlib import Path
from typing import Any

import chromadb

logger = logging.getLogger(__name__)


class SimpleDatabaseManager:
    """
    Vereinfachter Database Manager für grundlegende ChromaDB-Operationen.

    Funktionen:
    - Initialisierung der Database
    - Grundlegende CRUD-Operationen
    - Einfache Statistiken
    """

    def __init__(self, db_path: str = "data/vectors"):
        """
        Initialisiert den SimpleDatabaseManager.

        Args:
            db_path: Pfad zur ChromaDB-Database
        """
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self._client: Any = None

    @property
    def client(self) -> Any:
        """Lazy-loaded ChromaDB client."""
        if self._client is None:
            self._client = chromadb.PersistentClient(path=str(self.db_path))
        return self._client

    def initialize_database(self) -> dict[str, Any]:
        """
        Initialisiert die Database und erstellt grundlegende Strukturen.

        Returns:
            Initialisierungs-Statistiken
        """
        start_time = time.time()
        results = {
            "success": False,
            "db_path": str(self.db_path),
            "collections_created": 0,
            "total_time_seconds": 0.0,
        }

        try:
            logger.info("Initialisiere Database in: %s", self.db_path)

            # Test der Database-Verbindung
            heartbeat = self.client.heartbeat()
            logger.info("Database-Heartbeat: %s", heartbeat)

            # Vorhandene Collections auflisten
            collections = self.client.list_collections()
            logger.info("Vorhandene Collections: %d", len(collections))

            results["success"] = True
            results["existing_collections"] = len(collections)
            results["heartbeat"] = heartbeat

            total_time = time.time() - start_time
            results["total_time_seconds"] = round(total_time, 2)

            logger.info("Database-Initialisierung abgeschlossen in %.2fs", total_time)
            return results

        except Exception as e:
            logger.error("Fehler bei Database-Initialisierung: %s", e)
            results["error"] = str(e)
            return results

    def get_database_stats(self) -> dict[str, Any]:
        """
        Sammelt Database-Statistiken.

        Returns:
            Database-Statistiken
        """
        try:
            collections = self.client.list_collections()
            stats = {
                "total_collections": len(collections),
                "collections": [],
                "db_path": str(self.db_path),
                "heartbeat": self.client.heartbeat(),
            }

            # Statistiken für jede Collection
            for collection in collections:
                col_info = {"name": collection.name, "count": collection.count()}
                stats["collections"].append(col_info)

            return stats

        except Exception as e:
            logger.error("Fehler beim Sammeln der Database-Statistiken: %s", e)
            return {"error": str(e)}

    def create_collection(self, name: str) -> dict[str, Any]:
        """
        Erstellt eine neue Collection.

        Args:
            name: Name der Collection

        Returns:
            Ergebnis der Collection-Erstellung
        """
        try:
            collection = self.client.create_collection(name=name)
            logger.info("Collection '%s' erstellt", name)
            return {
                "success": True,
                "collection_name": name,
                "count": collection.count(),
            }

        except Exception as e:
            logger.error("Fehler beim Erstellen der Collection '%s': %s", name, e)
            return {"success": False, "error": str(e)}

    def delete_collection(self, name: str) -> dict[str, Any]:
        """
        Löscht eine Collection.

        Args:
            name: Name der Collection

        Returns:
            Ergebnis der Collection-Löschung
        """
        try:
            self.client.delete_collection(name=name)
            logger.info("Collection '%s' gelöscht", name)
            return {"success": True, "collection_name": name}

        except Exception as e:
            logger.error("Fehler beim Löschen der Collection '%s': %s", name, e)
            return {"success": False, "error": str(e)}

    def optimize_database(self) -> dict[str, Any]:
        """
        Führt grundlegende Database-Optimierungen durch.

        Returns:
            Optimierungs-Ergebnisse
        """
        start_time = time.time()

        try:
            # Database-Statistiken vor Optimierung
            before_stats = self.get_database_stats()

            # Grundlegende Wartung (aktuell nur Statistiken sammeln)
            logger.info("Führe Database-Optimierung durch...")

            # Database-Statistiken nach Optimierung
            after_stats = self.get_database_stats()

            total_time = time.time() - start_time

            return {
                "success": True,
                "before_stats": before_stats,
                "after_stats": after_stats,
                "optimization_time_seconds": round(total_time, 2),
            }

        except Exception as e:
            logger.error("Fehler bei Database-Optimierung: %s", e)
            return {"success": False, "error": str(e)}

    def backup_database(self, backup_name: str | None = None) -> dict[str, Any]:
        """
        Erstellt ein einfaches Database-Backup.

        Args:
            backup_name: Name des Backups (optional)

        Returns:
            Backup-Ergebnisse
        """
        import shutil

        if backup_name is None:
            backup_name = f"backup_{int(time.time())}"

        backup_dir = self.db_path.parent / "backups"
        backup_dir.mkdir(exist_ok=True)
        backup_path = backup_dir / backup_name

        try:
            # Kopiere Database-Verzeichnis
            shutil.copytree(self.db_path, backup_path, dirs_exist_ok=True)

            logger.info("Database-Backup erstellt: %s", backup_path)
            return {
                "success": True,
                "backup_name": backup_name,
                "backup_path": str(backup_path),
                "backup_size_mb": sum(
                    f.stat().st_size for f in backup_path.rglob("*") if f.is_file()
                )
                / (1024 * 1024),
            }

        except Exception as e:
            logger.error("Fehler beim Erstellen des Database-Backups: %s", e)
            return {"success": False, "error": str(e)}


def create_simple_manager(db_path: str = "data/vectors") -> SimpleDatabaseManager:
    """
    Factory-Funktion für SimpleDatabaseManager.

    Args:
        db_path: Pfad zur ChromaDB-Database

    Returns:
        SimpleDatabaseManager-Instanz
    """
    return SimpleDatabaseManager(db_path=db_path)
