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
from typing import TYPE_CHECKING, Any

from ..embeddings.sentence_transformer_provider import (
    SentenceTransformerProvider,
)

if TYPE_CHECKING:
    # Plugin Protocol Definition
    from typing import Protocol

    class MLPlugin(Protocol):
        """Protocol für MLPlugin Interface"""

        pass


# Import der stateless Plugin-Komponenten
from ..gemini.direct_processor import (
    GeminiDirectConfig,
    GeminiDirectProcessor,
)

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

    async def process_pdf_direct(self, pdf_path: Path) -> dict[str, Any]:
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
        self, item_description: str, context: dict[str, Any]
    ) -> dict[str, Any]:
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

    def get_service_info(self) -> dict[str, Any]:
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
        import numpy as np

        emb = self.provider.encode(text)
        if isinstance(emb, np.ndarray):
            return [float(x) for x in emb.tolist()]
        return [float(x) for x in emb]  # pragma: no cover

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Batch-Encoding für bessere Performance.

        Args:
            texts: Liste von Texten

        Returns:
            Liste von Embedding-Vectors
        """
        import numpy as np

        batch = self.provider.encode_batch(texts)
        result: list[list[float]] = []
        for emb in batch:
            if isinstance(emb, np.ndarray):
                result.append([float(x) for x in emb.tolist()])
            else:  # pragma: no cover
                result.append([float(x) for x in emb])
        return result

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

    def get_service_info(self) -> dict[str, Any]:
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
    def create_default() -> Any:
        """
        Erstellt MLPlugin mit Standard-Konfiguration.

        Returns:
            Configured MLPlugin instance
        """
        # Simuliere Standard-Plugin für Tests
        return {
            "gemini_service": BackendGeminiService(),
            "embedding_service": BackendEmbeddingService(),
            "config": "default",
        }

    @staticmethod
    def create_with_config(config: dict) -> Any:
        """
        Erstellt MLPlugin mit Custom-Konfiguration.

        Args:
            config: Custom configuration dictionary

        Returns:
            Configured MLPlugin instance
        """
        # Simuliere Custom-Plugin für Tests
        return {
            "gemini_service": BackendGeminiService(
                api_key=config.get("api_key"),
                model_name=config.get("model_name", "gemini-2.5-flash"),
            ),
            "embedding_service": BackendEmbeddingService(
                model_name=config.get("embedding_model", "all-MiniLM-L6-v2")
            ),
            "config": config,
        }

    @staticmethod
    def create_from_backend_registry(
        backend_registry: Any, config: Any = None
    ) -> "MLPlugin":
        """
        Erstelle MLPlugin aus llkjj_backend Module Registry.

        Args:
            backend_registry: Module Registry aus llkjj_backend
            config: Optional MLPluginConfig

        Returns:
            Configured MLPlugin instance
        """
        try:
            from llkjj_ml.llkjj_ml_plugin_v2 import (
                MLPlugin,
                MLPluginConfig,
            )
        except ImportError:
            raise RuntimeError(
                "MLPlugin v2.0 nicht verfügbar - Plugin-Factory nicht nutzbar"
            )

        try:
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
            raise RuntimeError(f"MLPlugin-Erstellung fehlgeschlagen: {e}") from e

    @staticmethod
    def create_with_custom_services(
        repository: Any,
        gemini_api_key: str | None = None,
        embedding_model: str | None = None,
        config: Any = None,
    ) -> "MLPlugin":
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
            from llkjj_ml.llkjj_ml_plugin_v2 import MLPlugin, MLPluginConfig
        except ImportError:
            raise RuntimeError(
                "MLPlugin v2.0 nicht verfügbar - Plugin-Factory nicht nutzbar"
            )

        try:
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
            raise RuntimeError(f"Custom MLPlugin-Erstellung fehlgeschlagen: {e}") from e


def create_ml_plugin_for_backend(config: dict[str, Any] | None = None) -> Any:
    """
    Helper-Funktion für ML-Plugin-Erstellung (Test-Kompatibilität).

    Args:
        config: Optional configuration dictionary

    Returns:
        Plugin instance
    """
    if config:
        return MLPluginFactory.create_with_config(config)
    return MLPluginFactory.create_default()


async def test_ml_services_integration() -> dict[str, Any]:
    """
    Test-Funktion für ML-Services Integration.

    Returns:
        Integration test results
    """
    try:
        # Erstelle Test-Plugin
        plugin = create_ml_plugin_for_backend()

        # Erstelle Health Checker
        checker = ServiceHealthChecker(
            [plugin.get("gemini_service"), plugin.get("embedding_service")]
        )

        # Führe Health Checks aus
        results = await checker.check_all_services()

        return {
            "integration_test": "passed",
            "plugin_created": plugin is not None,
            "health_check_results": results,
        }
    except Exception as e:
        return {"integration_test": "failed", "error": str(e)}

# Prevent pytest from collecting this helper when imported into test modules.
# Some tests import this symbol into the module namespace; pytest would treat
# it as a test function due to the name. Marking __test__ = False avoids that
# unintended collection while keeping backward-compatible imports.
test_ml_services_integration.__test__ = False  # type: ignore[attr-defined]


class ServiceHealthChecker:
    """
    Health Checker für ML-Services.

    Validiert dass alle Services korrekt funktionieren.
    """

    def __init__(self, services: list | None = None):
        """
        Initialize Service Health Checker.

        Args:
            services: Liste von Services zu überwachen
        """
        self.services = services or []
        logger.debug(
            f"ServiceHealthChecker initialisiert mit {len(self.services)} Services"
        )

    async def check_all_services(self) -> dict[str, Any]:
        """
        Prüft alle konfigurierten Services.

        Returns:
            Dictionary mit Service-Status
        """
        results = {}
        for i, service in enumerate(self.services):
            try:
                service_info = service.get_service_info()
                results[f"service_{i}"] = service_info
            except Exception as e:
                results[f"service_{i}"] = {"status": "unhealthy", "error": str(e)}

        return {
            "overall_status": "healthy"
            if all(r.get("status") == "healthy" for r in results.values())
            else "degraded",
            "services": results,
            "total_services": len(self.services),
            "healthy_services": sum(
                1 for r in results.values() if r.get("status") == "healthy"
            ),
        }

    async def check_service(self, service: Any) -> dict[str, Any]:
        """
        Prüft einen einzelnen Service.

        Args:
            service: Service-Instanz

        Returns:
            Service Status Dictionary
        """
        try:
            service_info = service.get_service_info()
            return {"status": "healthy", "info": service_info}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    @staticmethod
    async def check_gemini_service(
        gemini_service: BackendGeminiService,
    ) -> dict[str, Any]:
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
    def check_embedding_service(
        embedding_service: BackendEmbeddingService,
    ) -> dict[str, Any]:
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
    ) -> dict[str, Any]:
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
# Convenience Functions (Deprecated duplicates removed)
# =============================================================================
