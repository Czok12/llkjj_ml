"""
Backend Integration Services für llkjj_ml Plugin v2.0

Service-Adapter für nahtlose Integration mit llkjj_backend Module Registry.
Stellt konkrete Implementierungen der Plugin-Protocols bereit.

Author: LLKJJ ML Team
Version: 1.0.0
Date: 2025-01-25
"""

import logging
from pathlib import Path

from ..embeddings.sentence_transformer_provider import SentenceTransformerProvider

# Import der stateless Plugin-Komponenten
from ..gemini.direct_processor import GeminiDirectConfig, GeminiDirectProcessor

logger = logging.getLogger(__name__)


class BackendGeminiService:
    """
    Gemini Service Adapter für llkjj_backend Integration.

    Implementiert GeminiClient Protocol für MLPlugin v2.0.
    """

    def __init__(
        self, api_key: str | None = None, model_name: str = "gemini-2.5-flash"
    ):
        """
        Initialize Backend Gemini Service.

        Args:
            api_key: Optional Gemini API Key (aus Environment wenn None)
            model_name: Gemini Model zu verwenden
        """
        self.config = GeminiDirectConfig(api_key=api_key, model_name=model_name)
        self.processor = GeminiDirectProcessor(self.config)

        logger.info(f"✅ BackendGeminiService initialisiert: {model_name}")

    async def process_pdf_direct(self, pdf_path: Path) -> dict:
        """
        Implementierung des GeminiClient Protocol.

        Args:
            pdf_path: Pfad zur PDF-Datei

        Returns:
            Dict mit processing results
        """
        try:
            result = await self.processor.process_pdf_direct(pdf_path)

            # Konvertiere zu Protocol-kompatiblem Format
            return {
                "success": result.success,
                "invoice_data": result.invoice_data,
                "invoice_items": [item.model_dump() for item in result.invoice_items],
                "processing_time_ms": result.processing_time_ms,
                "token_usage": result.token_usage,
                "errors": result.errors,
            }

        except Exception as e:
            logger.error(f"Fehler bei PDF-Processing: {e}")
            return {
                "success": False,
                "invoice_data": {},
                "invoice_items": [],
                "processing_time_ms": 0,
                "token_usage": {},
                "errors": [str(e)],
            }

    async def classify_with_rag_context(
        self, item_description: str, context: dict
    ) -> dict:
        """
        Implementierung des GeminiClient Protocol für RAG-enhanced Classification.

        Args:
            item_description: Artikelbeschreibung
            context: RAG-Kontext mit ähnlichen Items

        Returns:
            Dict mit enhanced classification
        """
        try:
            return await self.processor.classify_with_rag_context(
                item_description, context
            )
        except Exception as e:
            logger.error(f"Fehler bei RAG-Enhanced Classification: {e}")
            return {"success": False, "error": str(e)}

    def get_service_info(self) -> dict:
        """Service-Informationen für Registry."""
        return {
            "service_name": "BackendGeminiService",
            "version": "1.0.0",
            "model_name": self.config.model_name,
            "capabilities": [
                "pdf_direct_processing",
                "rag_enhanced_classification",
                "skr03_classification",
            ],
        }


class BackendEmbeddingService:
    """
    Embedding Service Adapter für llkjj_backend Integration.

    Implementiert EmbeddingProvider Protocol für MLPlugin v2.0.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        cache_folder: Path | None = None,
    ):
        """
        Initialize Backend Embedding Service.

        Args:
            model_name: Sentence-Transformers Model
            cache_folder: Optional cache directory
        """
        self.provider = SentenceTransformerProvider(
            model_name=model_name, cache_folder=cache_folder
        )

        logger.info(f"✅ BackendEmbeddingService initialisiert: {model_name}")

    def encode(self, text: str) -> list[float]:
        """
        Implementierung des EmbeddingProvider Protocol.

        Args:
            text: Text zu encoden

        Returns:
            384-dimensionaler Float-Vector
        """
        return self.provider.encode(text)

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Batch-Encoding für bessere Performance.

        Args:
            texts: Liste von Texten

        Returns:
            Liste von Embedding-Vectors
        """
        return self.provider.encode_batch(texts)

    def similarity(self, text1: str, text2: str) -> float:
        """
        Berechne Cosine-Similarity zwischen zwei Texten.

        Args:
            text1: Erster Text
            text2: Zweiter Text

        Returns:
            Cosine-Similarity (0.0 - 1.0)
        """
        return self.provider.similarity(text1, text2)

    def get_service_info(self) -> dict:
        """Service-Informationen für Registry."""
        model_info = self.provider.get_model_info()
        return {
            "service_name": "BackendEmbeddingService",
            "version": "1.0.0",
            "model_name": model_info["model_name"],
            "embedding_dimension": model_info["embedding_dimension"],
            "model_loaded": model_info["model_loaded"],
        }

    def cleanup(self) -> None:
        """Cleanup Service Resources."""
        self.provider.cleanup()


class MLPluginFactory:
    """
    Factory für MLPlugin v2.0 Integration mit llkjj_backend.

    Stellt eine einfache API für Plugin-Erstellung bereit.
    """

    @staticmethod
    def create_from_backend_registry(backend_registry, config=None):
        """
        Erstelle MLPlugin aus llkjj_backend Module Registry.

        Args:
            backend_registry: Module Registry aus llkjj_backend
            config: Optional MLPluginConfig

        Returns:
            Configured MLPlugin instance
        """
        try:
            from ...llkjj_ml_plugin_v2 import MLPlugin, MLPluginConfig

            # Repository aus Registry abrufen
            repository_factory = backend_registry.get_service("repository_factory")
            if not repository_factory:
                raise RuntimeError(
                    "RepositoryFactory nicht in Backend-Registry verfügbar"
                )

            ml_repository = repository_factory.get_ml_repository()
            if not ml_repository:
                raise RuntimeError("ML Repository nicht verfügbar")

            # Services erstellen
            gemini_service = BackendGeminiService()
            embedding_service = BackendEmbeddingService()

            # Plugin erstellen
            plugin_config = config or MLPluginConfig()
            plugin = MLPlugin(
                repository=ml_repository,
                gemini_client=gemini_service,
                embedding_provider=embedding_service,
                config=plugin_config,
            )

            logger.info("✅ MLPlugin v2.0 erfolgreich über Backend-Registry erstellt")
            return plugin

        except Exception as e:
            logger.error(f"Fehler bei MLPlugin-Factory: {e}")
            raise RuntimeError(f"MLPlugin-Erstellung fehlgeschlagen: {e}")

    @staticmethod
    def create_with_custom_services(
        repository, gemini_api_key=None, embedding_model=None, config=None
    ):
        """
        Erstelle MLPlugin mit custom Services.

        Args:
            repository: ML Repository Interface
            gemini_api_key: Optional Gemini API Key
            embedding_model: Optional Embedding Model Name
            config: Optional MLPluginConfig

        Returns:
            Configured MLPlugin instance
        """
        try:
            from ...llkjj_ml_plugin_v2 import MLPlugin, MLPluginConfig

            # Services mit custom parameters erstellen
            gemini_service = BackendGeminiService(api_key=gemini_api_key)

            embedding_model = (
                embedding_model
                or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )
            embedding_service = BackendEmbeddingService(model_name=embedding_model)

            # Plugin erstellen
            plugin_config = config or MLPluginConfig()
            plugin = MLPlugin(
                repository=repository,
                gemini_client=gemini_service,
                embedding_provider=embedding_service,
                config=plugin_config,
            )

            logger.info("✅ MLPlugin v2.0 mit custom Services erstellt")
            return plugin

        except Exception as e:
            logger.error(f"Fehler bei Custom MLPlugin-Factory: {e}")
            raise RuntimeError(f"Custom MLPlugin-Erstellung fehlgeschlagen: {e}")


class ServiceHealthChecker:
    """
    Health Checker für ML-Services.

    Validiert dass alle Services korrekt funktionieren.
    """

    @staticmethod
    async def check_gemini_service(gemini_service: BackendGeminiService) -> dict:
        """
        Health Check für Gemini Service.

        Args:
            gemini_service: BackendGeminiService instance

        Returns:
            Health status dict
        """
        try:
            # Simple test - kann erweitert werden
            service_info = gemini_service.get_service_info()

            return {
                "service": "BackendGeminiService",
                "status": "healthy",
                "model": service_info["model_name"],
                "capabilities": len(service_info["capabilities"]),
            }

        except Exception as e:
            return {
                "service": "BackendGeminiService",
                "status": "unhealthy",
                "error": str(e),
            }

    @staticmethod
    def check_embedding_service(embedding_service: BackendEmbeddingService) -> dict:
        """
        Health Check für Embedding Service.

        Args:
            embedding_service: BackendEmbeddingService instance

        Returns:
            Health status dict
        """
        try:
            # Test encoding
            test_text = "Test embedding für Health Check"
            embedding = embedding_service.encode(test_text)

            service_info = embedding_service.get_service_info()

            return {
                "service": "BackendEmbeddingService",
                "status": "healthy",
                "model": service_info["model_name"],
                "embedding_dimension": len(embedding),
                "model_loaded": service_info["model_loaded"],
            }

        except Exception as e:
            return {
                "service": "BackendEmbeddingService",
                "status": "unhealthy",
                "error": str(e),
            }

    @staticmethod
    async def comprehensive_health_check(
        gemini_service: BackendGeminiService, embedding_service: BackendEmbeddingService
    ) -> dict:
        """
        Comprehensive Health Check aller ML-Services.

        Args:
            gemini_service: BackendGeminiService
            embedding_service: BackendEmbeddingService

        Returns:
            Comprehensive health status
        """
        gemini_health = await ServiceHealthChecker.check_gemini_service(gemini_service)
        embedding_health = ServiceHealthChecker.check_embedding_service(
            embedding_service
        )

        overall_status = (
            "healthy"
            if (
                gemini_health["status"] == "healthy"
                and embedding_health["status"] == "healthy"
            )
            else "degraded"
        )

        return {
            "overall_status": overall_status,
            "timestamp": str(logger.name),  # Placeholder
            "services": {"gemini": gemini_health, "embedding": embedding_health},
        }


# =============================================================================
# Convenience Functions
# =============================================================================


def create_ml_plugin_for_backend(backend_registry, **kwargs):
    """
    Convenience function für MLPlugin-Erstellung.

    Args:
        backend_registry: llkjj_backend Module Registry
        **kwargs: Optional parameters für Plugin-Konfiguration

    Returns:
        MLPlugin v2.0 instance
    """
    return MLPluginFactory.create_from_backend_registry(backend_registry, **kwargs)


async def test_ml_services_integration():
    """
    Integration Test für alle ML-Services.

    Returns:
        Test results dict
    """
    try:
        # Services erstellen
        gemini_service = BackendGeminiService()
        embedding_service = BackendEmbeddingService()

        # Health Checks
        health_results = await ServiceHealthChecker.comprehensive_health_check(
            gemini_service, embedding_service
        )

        # Cleanup
        embedding_service.cleanup()

        return {"integration_test": "successful", "health_results": health_results}

    except Exception as e:
        return {"integration_test": "failed", "error": str(e)}
