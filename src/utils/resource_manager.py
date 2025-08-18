#!/usr/bin/env python3
"""
LLKJJ ML Pipeline - Resource Manager
===================================

Singleton Resource Manager für schwere ML-Modelle und DB-Verbindungen.
Verhindert Memory-Leaks durch mehrfache Initialisierung und bietet
explizite Cleanup-Mechanismen.

Autor: LLKJJ ML Pipeline Team
Version: 4.0.0 (Hybrid Implementation)
Datum: 18. August 2025
"""

import gc
import logging
import os
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer

from src.skr03_manager import lade_skr03_manager

logger = logging.getLogger(__name__)


class ResourceManager:
    """
    Singleton Resource Manager für schwere ML-Modelle und DB-Verbindungen.

    Verhindert Memory-Leaks durch:
    - Einmalige Initialisierung schwerer Modelle
    - Explizite Cleanup-Mechanismen
    - Kontrolliertes Garbage Collection
    - Production-ready Resource-Management

    **USAGE:**
    ```python
    # Get singleton instance
    rm = ResourceManager()

    # Access resources (lazy-loaded)
    model = rm.get_embedding_model()
    client = rm.get_chromadb_client("/path/to/db")

    # Cleanup when done (production)
    rm.cleanup()
    ```
    """

    _instance: "ResourceManager | None" = None
    _embedding_model: SentenceTransformer | None = None
    _chromadb_client: Any = None
    _skr03_manager: Any = None
    _docling_processor: Any = None

    def __new__(cls) -> "ResourceManager":
        """Singleton pattern - nur eine Instanz"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            logger.info("🔧 ResourceManager Singleton erstellt")
        return cls._instance

    def get_embedding_model(
        self, model_name: str = "all-MiniLM-L12-v2"
    ) -> SentenceTransformer:
        """
        Lazy-loaded Singleton SentenceTransformer

        Args:
            model_name: Name des Embedding-Modells (L12 für höhere Qualität)

        Returns:
            SentenceTransformer Instanz
        """
        if self._embedding_model is None:
            logger.info(f"🔄 Lade SentenceTransformer '{model_name}' (Singleton)...")
            self._embedding_model = SentenceTransformer(model_name)
            logger.info("✅ SentenceTransformer geladen")
        return self._embedding_model

    def get_chromadb_client(self, db_path: str) -> Any:
        """
        Lazy-loaded Singleton ChromaDB Client

        Args:
            db_path: Pfad zur ChromaDB-Datenbank

        Returns:
            ChromaDB PersistentClient
        """
        if self._chromadb_client is None:
            logger.info("🔄 Lade ChromaDB Client (Singleton)...")
            # Deaktiviere ChromaDB Telemetry für saubere Logs
            os.environ["ANONYMIZED_TELEMETRY"] = "False"
            self._chromadb_client = chromadb.PersistentClient(path=db_path)
            logger.info("✅ ChromaDB Client geladen")
        return self._chromadb_client

    def get_skr03_manager(self) -> Any:
        """
        Lazy-loaded Singleton SKR03 Manager

        Returns:
            SKR03Manager Instanz
        """
        if self._skr03_manager is None:
            logger.info("🔄 Lade SKR03Manager (Singleton)...")
            self._skr03_manager = lade_skr03_manager()
            logger.info("✅ SKR03Manager geladen")
        return self._skr03_manager

    def get_docling_processor(self) -> Any:
        """
        Lazy-loaded Singleton DoclingProcessor für Alternative Pipeline

        Returns:
            AdvancedDoclingProcessor Instanz
        """
        if self._docling_processor is None:
            logger.info("🔄 Lade DoclingProcessor (Singleton)...")
            from src.extraction.docling_processor import AdvancedDoclingProcessor

            self._docling_processor = AdvancedDoclingProcessor(
                use_gpu=True,
                ocr_engine="rapid",
                table_mode="accurate",
                german_optimized=True,
            )
            logger.info("✅ DoclingProcessor geladen")
        return self._docling_processor

    def get_chromadb_collection(
        self,
        db_path: str,
        collection_name: str = "invoices",
        create_if_missing: bool = True,
    ) -> Any:
        """
        Get or create ChromaDB collection with error handling

        Args:
            db_path: Pfad zur ChromaDB-Datenbank
            collection_name: Name der Collection
            create_if_missing: Erstelle Collection falls nicht vorhanden

        Returns:
            ChromaDB Collection
        """
        client = self.get_chromadb_client(db_path)

        try:
            collection = client.get_collection(collection_name)
            logger.info(f"✅ ChromaDB Collection '{collection_name}' geladen")
            return collection
        except ValueError:
            if create_if_missing:
                collection = client.create_collection(
                    collection_name,
                    metadata={
                        "description": "German electrical contractor invoice items for SKR03 classification"
                    },
                )
                logger.info(f"✅ ChromaDB Collection '{collection_name}' erstellt")
                return collection
            else:
                logger.error(
                    f"❌ ChromaDB Collection '{collection_name}' nicht gefunden"
                )
                raise

    def cleanup(self) -> None:
        """
        Explizite Bereinigung aller Ressourcen für Production-Deployment.

        Führt folgende Bereinigungen durch:
        - ChromaDB Client schließen
        - Embedding Model mit CUDA Cache-Clear
        - DoclingProcessor bereinigen
        - SKR03 Manager freigeben
        - Multiple Garbage Collection Durchläufe
        """
        logger.info("🧹 Starte ResourceManager Cleanup...")

        # Cleanup ChromaDB client
        if self._chromadb_client:
            try:
                # ChromaDB hat keine explizite close(), aber reset das Objekt
                self._chromadb_client = None
                logger.info("  ✅ ChromaDB Client bereinigt")
            except Exception as e:
                logger.warning(f"  ⚠️ ChromaDB cleanup warning: {e}")

        # Cleanup embedding model with explicit torch cache clear
        if self._embedding_model:
            try:
                # Clear CUDA cache if available
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        logger.info("  ✅ CUDA Cache geleert")
                except ImportError:
                    pass  # Torch nicht verfügbar

                # Delete embedding model
                del self._embedding_model
                self._embedding_model = None
                logger.info("  ✅ Embedding Model bereinigt")
            except Exception as e:
                logger.warning(f"  ⚠️ Embedding model cleanup warning: {e}")

        # Cleanup Docling processor
        if self._docling_processor:
            try:
                # Delete docling processor
                del self._docling_processor
                self._docling_processor = None
                logger.info("  ✅ DoclingProcessor bereinigt")
            except Exception as e:
                logger.warning(f"  ⚠️ DoclingProcessor cleanup warning: {e}")

        # Cleanup SKR03 manager
        if self._skr03_manager:
            try:
                del self._skr03_manager
                self._skr03_manager = None
                logger.info("  ✅ SKR03Manager bereinigt")
            except Exception as e:
                logger.warning(f"  ⚠️ SKR03 manager cleanup warning: {e}")

        # Force garbage collection multiple times
        total_collected = 0
        for i in range(5):
            collected = gc.collect()
            total_collected += collected
            if collected > 0:
                logger.info(f"  🗑️ GC Pass {i+1}: {collected} Objekte freigegeben")

        logger.info(
            f"✅ ResourceManager cleanup abgeschlossen - {total_collected} Objekte freigegeben"
        )

    def get_memory_usage(self) -> dict[str, Any]:
        """
        Get current memory usage statistics

        Returns:
            Dict mit Memory-Usage-Informationen
        """
        import psutil

        process = psutil.Process()
        memory_info = process.memory_info()

        return {
            "memory_mb": memory_info.rss / 1024 / 1024,
            "memory_percent": process.memory_percent(),
            "cpu_percent": process.cpu_percent(),
            "loaded_resources": {
                "embedding_model": self._embedding_model is not None,
                "chromadb_client": self._chromadb_client is not None,
                "skr03_manager": self._skr03_manager is not None,
                "docling_processor": self._docling_processor is not None,
            },
            "python_objects": len(gc.get_objects()),
        }

    def force_memory_cleanup(self) -> dict[str, Any]:
        """
        Force aggressive memory cleanup and return before/after stats

        Returns:
            Dict mit Before/After Memory-Statistiken
        """
        before_stats = self.get_memory_usage()

        # Aggressive cleanup
        self.cleanup()

        # Multiple GC passes
        total_collected = 0
        for _ in range(10):
            total_collected += gc.collect()

        after_stats = self.get_memory_usage()

        return {
            "before": before_stats,
            "after": after_stats,
            "memory_freed_mb": before_stats["memory_mb"] - after_stats["memory_mb"],
            "objects_collected": total_collected,
        }


# Global Singleton Instance für einfachen Zugriff
_resource_manager = ResourceManager()


# Convenience Functions
def get_resource_manager() -> ResourceManager:
    """Get the global ResourceManager singleton"""
    return _resource_manager


def cleanup_resources() -> None:
    """Convenience function to cleanup all resources"""
    _resource_manager.cleanup()


def get_memory_stats() -> dict[str, Any]:
    """Convenience function to get memory statistics"""
    return _resource_manager.get_memory_usage()
