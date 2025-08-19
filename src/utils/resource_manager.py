#!/usr/bin/env python3
"""
LLKJJ ML Pipeline - Enhanced Resource Manager
============================================

🏭 PRODUCTION-READY RESOURCE MANAGEMENT

Centralized resource management with:
- ChromaDB Singleton (verhindert "instance already exists" Fehler)
- Apple Silicon Optimization
- Memory-Management Integration
- Production-Grade Error-Handling

Autor: LLKJJ Resource Team
Version: 2.0.0 (Production mit Apple Silicon)
"""

import gc
import logging
from typing import Any

import chromadb
import spacy
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Import optimization modules
from src.optimization.apple_silicon_optimizer import AppleSiliconOptimizer
from src.optimization.production_memory_manager import get_memory_manager

logger = logging.getLogger(__name__)


class ResourceManager:
    """
    🏭 PRODUCTION-READY RESOURCE MANAGER

    Singleton-basierte Verwaltung ressourcenintensiver Modelle mit:
    - ChromaDB Singleton-Pattern (verhindert "instance already exists" Fehler)
    - Apple Silicon Optimization
    - Memory-Management Integration
    - Production-Grade Error-Handling
    """

    _instance: "ResourceManager | None" = None
    _initialized: bool = False

    def __new__(cls) -> "ResourceManager":
        """Singleton-Pattern für einmalige Initialisierung."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            logger.info("🔧 ResourceManager Singleton erstellt")
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return

        logger.info("🏭 Initializing ResourceManager...")

        # Initialize optimization systems
        self.apple_optimizer = AppleSiliconOptimizer()
        self.memory_manager = get_memory_manager()

        # Resource instances
        self._sentence_transformer: SentenceTransformer | None = None
        self._spacy_model: Any = None
        self._chroma_client: Any = None

        self._initialized = True
        logger.info("🏭 ResourceManager initialized successfully")

    @property
    def sentence_transformer(self) -> SentenceTransformer:
        """
        🤖 Lazy-loaded SentenceTransformer für Embedding-Generierung

        Returns:
            Optimierte SentenceTransformer-Instanz
        """
        if self._sentence_transformer is None:
            logger.info("🤖 Loading SentenceTransformer...")

            with self.memory_manager.memory_tracked_context(
                "SentenceTransformer Loading"
            ):
                # Apply Apple Silicon optimization
                device = str(self.apple_optimizer.get_device())

                self._sentence_transformer = SentenceTransformer(
                    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                    device=device,
                )

                # Register with memory manager
                if hasattr(self._sentence_transformer._modules, "values"):
                    for module in self._sentence_transformer._modules.values():
                        if hasattr(module, "parameters"):
                            for param in module.parameters():
                                self.memory_manager.register_tensor(param)

                logger.info(f"🤖 SentenceTransformer loaded on {device}")

        return self._sentence_transformer

    @property
    def spacy_model(self) -> Any:
        """
        🇩🇪 Lazy-loaded spaCy Deutsches Modell

        Returns:
            spaCy-Modell für deutsche Textverarbeitung
        """
        if self._spacy_model is None:
            logger.info("🇩🇪 Loading spaCy German model...")

            with self.memory_manager.memory_tracked_context("spaCy Model Loading"):
                try:
                    self._spacy_model = spacy.load("de_core_news_sm")
                    logger.info("🇩🇪 spaCy German model loaded")
                except OSError as e:
                    logger.error(f"❌ spaCy German model not found: {e}")
                    logger.info(
                        "💡 Please install: python -m spacy download de_core_news_sm"
                    )
                    raise

        return self._spacy_model

    @property
    def chroma_client(self) -> Any:
        """
        🗃️ Singleton ChromaDB Client mit verbessertem Error-Handling

        Returns:
            ChromaDB Client-Instanz (verhindert "instance already exists")
        """
        if self._chroma_client is None:
            logger.info("🗃️ Initializing ChromaDB client...")

            try:
                with self.memory_manager.memory_tracked_context(
                    "ChromaDB Initialization"
                ):
                    # Fixed ChromaDB settings to prevent telemetry warnings
                    settings = Settings(
                        anonymized_telemetry=False, allow_reset=True, is_persistent=True
                    )

                    # Single client creation with error handling
                    self._chroma_client = chromadb.PersistentClient(
                        path="./data/vectors", settings=settings
                    )

                    # Register with memory manager
                    self.memory_manager.register_chroma_client(self._chroma_client)

                    logger.info("🗃️ ChromaDB client initialized successfully")

            except Exception as e:
                logger.error(f"❌ ChromaDB initialization failed: {e}")
                # Fallback to in-memory client
                try:
                    logger.info("🔄 Falling back to in-memory ChromaDB...")
                    settings = Settings(anonymized_telemetry=False, allow_reset=True)
                    self._chroma_client = chromadb.Client(settings=settings)
                    self.memory_manager.register_chroma_client(self._chroma_client)
                    logger.info("🗃️ In-memory ChromaDB client initialized")
                except Exception as fallback_error:
                    logger.critical(f"💥 ChromaDB completely failed: {fallback_error}")
                    raise

        return self._chroma_client

    def get_collection(self, collection_name: str) -> Any:
        """
        📚 Holt oder erstellt ChromaDB Collection

        Args:
            collection_name: Name der Collection

        Returns:
            ChromaDB Collection-Instanz
        """
        try:
            client = self.chroma_client

            # Versuche Collection zu holen, erstelle bei Bedarf
            try:
                collection = client.get_collection(name=collection_name)
                logger.debug(f"📚 Collection '{collection_name}' retrieved")
            except Exception:
                collection = client.create_collection(name=collection_name)
                logger.info(f"📚 Collection '{collection_name}' created")

            return collection

        except Exception as e:
            logger.error(f"❌ Collection error for '{collection_name}': {e}")
            raise

    def cleanup(self) -> dict[str, Any]:
        """
        🧹 Starte ResourceManager Cleanup
        """
        logger.info("🧹 Starte ResourceManager Cleanup...")

        collected_objects = {
            "sentence_transformer": 0,
            "spacy_model": 0,
            "chroma_client": 0,
            "general_gc": 0,
        }

        try:
            # Cleanup SentenceTransformer
            if self._sentence_transformer is not None:
                # Unregister tensors
                if hasattr(self._sentence_transformer._modules, "values"):
                    for module in self._sentence_transformer._modules.values():
                        if hasattr(module, "parameters"):
                            for param in module.parameters():
                                self.memory_manager.unregister_tensor(param)

                del self._sentence_transformer
                self._sentence_transformer = None
                collected_objects["sentence_transformer"] = 1
                logger.info("🔧 SentenceTransformer freigegeben")

            # Cleanup spaCy model
            if self._spacy_model is not None:
                del self._spacy_model
                self._spacy_model = None
                collected_objects["spacy_model"] = 1
                logger.info("🔧 spaCy model freigegeben")

            # Cleanup ChromaDB
            if self._chroma_client is not None:
                try:
                    self.memory_manager.unregister_chroma_client(self._chroma_client)
                    if hasattr(self._chroma_client, "reset"):
                        self._chroma_client.reset()
                except Exception as e:
                    logger.warning(f"⚠️ ChromaDB cleanup warning: {e}")

                del self._chroma_client
                self._chroma_client = None
                collected_objects["chroma_client"] = 1
                logger.info("🔧 ChromaDB client freigegeben")

            # Force memory cleanup
            self.memory_manager.force_cleanup(aggressive=True)

            # Force garbage collection
            collected_objects["general_gc"] = gc.collect()

            total_collected = sum(collected_objects.values())
            logger.info(
                f"✅ ResourceManager cleanup abgeschlossen - {total_collected} Objekte freigegeben"
            )

            return {
                "success": True,
                "collected_objects": collected_objects,
                "total_collected": total_collected,
            }

        except Exception as e:
            logger.error(f"❌ Fehler beim ResourceManager cleanup: {e}")
            return {
                "success": False,
                "error": str(e),
                "collected_objects": collected_objects,
            }

    def get_memory_report(self) -> dict[str, Any]:
        """
        📋 Memory-Report für alle verwalteten Ressourcen

        Returns:
            Detaillierter Memory-Status aller Ressourcen
        """
        report: dict[str, Any] = {
            "sentence_transformer_loaded": self._sentence_transformer is not None,
            "spacy_model_loaded": self._spacy_model is not None,
            "chroma_client_active": self._chroma_client is not None,
            "apple_silicon_optimized": self.apple_optimizer.mps_available,
            "memory_manager": self.memory_manager.get_memory_report(),
        }

        # Add model-specific memory info
        if self._sentence_transformer:
            device = next(self._sentence_transformer.parameters()).device
            report["sentence_transformer_device"] = str(device)

        return report

    def reset_chroma_client(self) -> None:
        """
        🔄 Forcibly reset ChromaDB client (für Testing/Debugging)
        """
        logger.warning("🔄 Resetting ChromaDB client...")

        if self._chroma_client:
            try:
                self.memory_manager.unregister_chroma_client(self._chroma_client)
                if hasattr(self._chroma_client, "reset"):
                    self._chroma_client.reset()
                logger.info("🔄 ChromaDB client reset completed")
            except Exception as e:
                logger.error(f"❌ ChromaDB reset error: {e}")
            finally:
                self._chroma_client = None

    @classmethod
    def reset_singleton(cls) -> None:
        """
        🔄 CRITICAL: Reset Singleton für Memory Leak Prevention

        Diese Methode ist essentiell für Long-Running Processes um
        Memory Leaks durch das Singleton Pattern zu verhindern.

        Usage:
            ResourceManager.reset_singleton()
            new_manager = ResourceManager()  # Fresh instance
        """
        logger.warning(
            "🔄 RESETTING ResourceManager Singleton - potential memory leak fix"
        )

        if cls._instance is not None:
            # Clean up existing instance
            try:
                cls._instance.cleanup()
                logger.info("✅ Existing ResourceManager cleaned up")
            except Exception as e:
                logger.error(f"❌ Error during singleton cleanup: {e}")

            # Reset class variables
            cls._instance = None
            cls._initialized = False

            # Force garbage collection
            import gc

            collected = gc.collect()
            logger.info(f"🧹 Singleton reset complete - {collected} objects collected")
        else:
            logger.info("ℹ️ No existing ResourceManager singleton to reset")

    @classmethod
    def cleanup_all(cls) -> dict[str, Any]:
        """
        🧹 PRODUCTION: Complete cleanup including singleton reset

        This method should be called in production environments
        during graceful shutdowns or periodic memory cleanup cycles.

        Returns:
            Dict with cleanup statistics
        """
        logger.info("🧹 PRODUCTION CLEANUP: Starting complete ResourceManager cleanup")

        cleanup_stats = {
            "singleton_existed": cls._instance is not None,
            "cleanup_results": {},
            "reset_successful": False,
        }

        if cls._instance is not None:
            try:
                cleanup_stats["cleanup_results"] = cls._instance.cleanup()
            except Exception as e:
                logger.error(
                    f"❌ PRODUCTION CLEANUP: Error during instance cleanup: {e}"
                )
                cleanup_stats["cleanup_error"] = str(e)

        try:
            cls.reset_singleton()
            cleanup_stats["reset_successful"] = True
            logger.info(
                "✅ PRODUCTION CLEANUP: Complete ResourceManager cleanup successful"
            )
        except Exception as e:
            logger.error(f"❌ PRODUCTION CLEANUP: Error during singleton reset: {e}")
            cleanup_stats["reset_error"] = str(e)

        return cleanup_stats


# 🏭 GLOBAL RESOURCE MANAGER - Now with proper cleanup support
_resource_manager: ResourceManager | None = None


def get_resource_manager() -> ResourceManager:
    """
    Get the global ResourceManager singleton

    Note: In production, call ResourceManager.cleanup_all() periodically
    to prevent memory leaks in long-running processes.
    """
    global _resource_manager
    if _resource_manager is None:
        _resource_manager = ResourceManager()
    return _resource_manager


def reset_global_resource_manager() -> None:
    """
    Reset the global resource manager for memory leak prevention

    This function should be called during:
    - Graceful shutdowns
    - Memory pressure situations
    - Testing scenarios
    - Long-running process maintenance
    """
    global _resource_manager

    if _resource_manager is not None:
        try:
            _resource_manager.cleanup()
            logger.info("🧹 Global ResourceManager cleaned up")
        except Exception as e:
            logger.error(f"❌ Error cleaning up global ResourceManager: {e}")

    _resource_manager = None
    ResourceManager.reset_singleton()

    logger.info("✅ Global ResourceManager reset complete")
