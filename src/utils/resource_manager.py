#!/usr/bin/env python3
"""
DEPRECATED: LLKJJ ML Pipeline - Enhanced Resource Manager
========================================================

⚠️  DEPRECATION WARNING: This ResourceManager is deprecated in v2.0.0 ⚠️

This file contains the legacy ResourceManager singleton that has been replaced
by the new stateless Repository-Pattern in v2.0.0. 

Use instead:
- llkjj_ml.MLPlugin (v2.0) with dependency injection
- llkjj_ml.src.embeddings.sentence_transformer_provider
- llkjj_ml.src.gemini.direct_processor

This file is kept only for backward compatibility and will be removed in v3.0.0.

Author: LLKJJ ML Team (Legacy Code)
Version: 1.0.0 (DEPRECATED)
Date: 2025-01-25
"""

import gc
import logging
import os
from typing import Any

# ChromaDB Telemetry vollständig deaktivieren um "capture() takes 1 positional argument" Fehler zu verhindern
# KRITISCH: Diese müssen VOR allen ChromaDB imports gesetzt werden
os.environ.setdefault("CHROMA_TELEMETRY", "false")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")
os.environ.setdefault("CHROMA_TELEMETRY_ENABLED", "false")
os.environ.setdefault("POSTHOG_ENABLED", "false")
os.environ.setdefault("CHROMA_SERVER_TELEMETRY", "false")
os.environ.setdefault("CHROMA_CLIENT_TELEMETRY", "false")

# Zusätzliche posthog-spezifische Deaktivierung
os.environ.setdefault("POSTHOG_CAPTURE", "false")
os.environ.setdefault("POSTHOG_DEBUG", "false")
os.environ.setdefault("POSTHOG_DISABLED", "true")

# ChromaDB optional importieren für graceful degradation
try:
    import chromadb
    from chromadb.config import Settings

    CHROMADB_AVAILABLE = True
except ImportError:
    chromadb = None
    Settings = None
    CHROMADB_AVAILABLE = False

import spacy
from sentence_transformers import SentenceTransformer

# Import optimization modules
from src.optimization.apple_silicon_optimizer import AppleSiliconOptimizer
from src.optimization.production_memory_manager import get_memory_manager

logger = logging.getLogger(__name__)


class ResourceManager:
    """
    🏭 DEPRECATED RESOURCE MANAGER (v1.x)

    ⚠️  DEPRECATED: Use llkjj_ml.MLPlugin (v2.0) with Repository-Pattern instead.

    This singleton-based resource manager is replaced by:
    - Dependency injection pattern
    - Explicit resource management
    - Repository-based data access
    
    Will be removed in v3.0.0.
    """

    _instance: "ResourceManager | None" = None
    _initialized: bool = False

    def __new__(cls) -> "ResourceManager":
        """Singleton-Pattern für einmalige Initialisierung."""
        import warnings
        warnings.warn(
            "ResourceManager singleton is deprecated in v2.0.0. "
            "Use llkjj_ml.MLPlugin with Repository-Pattern instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            logger.info("🔧 ResourceManager Singleton erstellt (DEPRECATED)")
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
        if self._chroma_client is None and CHROMADB_AVAILABLE:
            logger.info("🗃️ Attempting ChromaDB client initialization...")

            try:
                with self.memory_manager.memory_tracked_context(
                    "ChromaDB Initialization"
                ):
                    # Enhanced ChromaDB settings to prevent telemetry errors
                    settings = Settings(
                        anonymized_telemetry=False,
                        allow_reset=True,
                        is_persistent=True,
                        # Zusätzliche telemetry-Deaktivierung auf Settings-Ebene
                        telemetry_enabled=False,
                    )

                    # Create data directory if not exists
                    import os

                    os.makedirs("./data/vectors", exist_ok=True)

                    # Single client creation with enhanced error handling
                    self._chroma_client = chromadb.PersistentClient(
                        path="./data/vectors", settings=settings
                    )

                    # Register with memory manager
                    self.memory_manager.register_chroma_client(self._chroma_client)

                    logger.info("🗃️ ChromaDB client initialized successfully")

            except Exception as e:
                logger.error(
                    f"❌ ChromaDB persistent client initialization failed: {e}"
                )
                # Enhanced fallback to in-memory client with better error handling
                try:
                    logger.info(
                        "🔄 Falling back to in-memory ChromaDB with improved settings..."
                    )
                    settings = Settings(
                        anonymized_telemetry=False,
                        allow_reset=True,
                        # Zusätzliche telemetry-Deaktivierung für In-Memory Client
                        telemetry_enabled=False,
                    )
                    self._chroma_client = chromadb.Client(settings=settings)
                    self.memory_manager.register_chroma_client(self._chroma_client)
                    logger.info("🗃️ In-memory ChromaDB client initialized successfully")
                except Exception as fallback_error:
                    logger.warning(
                        f"⚠️ ChromaDB in-memory client also failed: {fallback_error}"
                    )
                    logger.info(
                        "✨ Operating without ChromaDB - using simplified storage backend"
                    )
                    self._chroma_client = None
                    # Don't raise - allow graceful degradation

        elif not CHROMADB_AVAILABLE and self._chroma_client is None:
            logger.info("✨ ChromaDB not installed - using simplified storage backend")

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

            # Graceful degradation if ChromaDB is completely unavailable
            if client is None:
                logger.warning(
                    f"⚠️ ChromaDB nicht verfügbar - Collection '{collection_name}' kann nicht erstellt werden"
                )
                return None

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
            logger.warning("⚠️ Graceful degradation - returning None")
            return None

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
    DEPRECATED: Get the global ResourceManager singleton
    
    ⚠️  WARNING: This function is deprecated in v2.0.0.
    Use llkjj_ml.MLPlugin with Repository-Pattern instead.
    
    Will be removed in v3.0.0.
    """
    import warnings
    warnings.warn(
        "get_resource_manager() is deprecated in v2.0.0. "
        "Use llkjj_ml.MLPlugin with dependency injection instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
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
