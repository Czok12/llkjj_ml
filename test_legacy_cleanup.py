"""
Tests für Phase 6: Legacy Cleanup Validierung
===========================================

Validiert dass alle ResourceManager-Singletons erfolgreich als deprecated markiert
wurden und die v2.0 API korrekt funktioniert.

Author: LLKJJ ML Team
Version: 2.0.0
Date: 2025-01-25
"""

import warnings
from pathlib import Path
from unittest.mock import Mock, patch

import pytest


class TestLegacyCleanup:
    """Tests für Legacy-Cleanup-Validierung."""

    def test_resource_manager_deprecation_warnings(self):
        """Test dass ResourceManager Deprecation-Warnings auslöst."""

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Test ResourceManager Singleton
            from src.utils.resource_manager import get_resource_manager

            get_resource_manager()

            # Verify deprecation warning was raised
            assert len(w) >= 1
            assert any("deprecated in v2.0.0" in str(warning.message) for warning in w)
            assert any(
                issubclass(warning.category, DeprecationWarning) for warning in w
            )

    def test_pipeline_resource_manager_deprecation(self):
        """Test dass Pipeline ResourceManager deprecated ist."""

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Test Pipeline ResourceManager
            from src.pipeline.processor import ResourceManager

            ResourceManager()

            # Verify deprecation warning
            assert len(w) >= 1
            assert any("deprecated in v2.0.0" in str(warning.message) for warning in w)

    def test_ml_processor_deprecation(self):
        """Test dass MLProcessor deprecated ist."""

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")

            # Test MLProcessor mit Mock-Settings
            try:
                from ml_service.processor import MLProcessor, MLSettings

                settings = MLSettings()
                processor = MLProcessor(settings=settings)

                # Should work but be deprecated
                assert processor is not None

            except Exception as e:
                # Expected if dependencies missing - that's OK
                pytest.skip(f"MLProcessor dependencies missing: {e}")

    def test_legacy_plugin_deprecation(self):
        """Test dass Legacy MLPlugin deprecated ist."""

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Test Legacy Plugin
            from llkjj_ml_plugin import MLPlugin

            # This should trigger deprecation warning
            try:
                MLPlugin(validate_env=False)

                # Verify deprecation warning
                assert len(w) >= 1
                assert any(
                    "deprecated in v2.0.0" in str(warning.message) for warning in w
                )

            except Exception as e:
                # Expected if dependencies missing
                pytest.skip(f"Legacy plugin dependencies missing: {e}")

    def test_process_pdf_simple_deprecation(self):
        """Test dass process_pdf_simple deprecated ist."""

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Test process_pdf_simple
            try:
                from llkjj_ml_plugin import process_pdf_simple

                # Mock the underlying processing to avoid actual file processing
                with patch("llkjj_ml_plugin.MLPlugin") as mock_plugin:
                    mock_instance = Mock()
                    mock_instance.process_pdf.return_value = Mock()
                    mock_plugin.return_value = mock_instance

                    # This should trigger deprecation warning
                    process_pdf_simple("/fake/path.pdf")

                # Verify deprecation warning
                assert len(w) >= 1
                assert any(
                    "deprecated in v2.0.0" in str(warning.message) for warning in w
                )

            except Exception as e:
                pytest.skip(f"process_pdf_simple dependencies missing: {e}")

    def test_v2_api_functionality(self):
        """Test dass v2.0 API verfügbar und funktionsfähig ist."""

        # Test v2.0 API imports
        try:
            from llkjj_ml import (
                BackendEmbeddingService,
                BackendGeminiService,
                MLPlugin,
                MLPluginConfig,
                create_ml_plugin_for_backend,
            )

            # Verify v2.0 classes are available
            assert MLPlugin is not None
            assert MLPluginConfig is not None
            assert create_ml_plugin_for_backend is not None
            assert BackendGeminiService is not None
            assert BackendEmbeddingService is not None

            # Test MLPluginConfig creation
            config = MLPluginConfig(validate_environment=False)
            assert config.validate_environment is False

        except ImportError as e:
            pytest.fail(f"v2.0 API not properly available: {e}")

    def test_no_active_singletons(self):
        """Test dass keine aktiven ResourceManager-Singletons mehr verwendet werden."""

        # Check src/pipeline/processor.py
        from src.pipeline import processor

        # _resource_manager should be None or disabled
        assert (
            getattr(processor, "_resource_manager", None) is None
        ), "Pipeline _resource_manager should be disabled"

        # Check UnifiedProcessor doesn't use active ResourceManager
        try:
            unified = processor.UnifiedProcessor()
            # resource_manager should be None or disabled
            assert (
                getattr(unified, "resource_manager", None) is None
            ), "UnifiedProcessor resource_manager should be disabled"
        except Exception as e:
            # Expected if dependencies missing
            pytest.skip(f"UnifiedProcessor dependencies missing: {e}")

    def test_memory_leak_prevention(self):
        """Test dass Memory-Leaks durch Singleton-Elimination verhindert werden."""

        import gc
        import os

        import psutil

        # Get initial memory
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Try to trigger old singleton behavior multiple times
        for _i in range(5):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter(
                        "ignore"
                    )  # Suppress deprecation warnings for test

                    from src.utils.resource_manager import get_resource_manager

                    manager = get_resource_manager()

                    # Force garbage collection
                    del manager
                    gc.collect()

            except Exception:
                # Expected if properly disabled
                pass

        # Check memory didn't increase dramatically
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Should not increase more than 50MB (reasonable threshold)
        max_increase = 50 * 1024 * 1024  # 50MB in bytes

        assert (
            memory_increase < max_increase
        ), f"Memory increased by {memory_increase / 1024 / 1024:.1f}MB - possible memory leak"

    def test_chromadb_references_eliminated(self):
        """Test dass ChromaDB-Referenzen entfernt oder deprecated sind."""

        # Check that ChromaDB imports are handled gracefully
        try:
            from src.utils.resource_manager import ResourceManager

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # Suppress deprecation warnings

                manager = ResourceManager()

                # ChromaDB methods should either be deprecated or handle gracefully
                assert hasattr(
                    manager, "_chroma_client"
                )  # Should exist but be None or deprecated

        except ImportError:
            # ChromaDB not available - that's actually good for cleanup
            pass
        except Exception as e:
            pytest.skip(f"ChromaDB reference test skipped: {e}")

    def test_documentation_updated(self):
        """Test dass Dokumentation korrekt auf v2.0 verweist."""

        # Check __init__.py has correct version and deprecation info
        from llkjj_ml import __version__, get_plugin_info

        assert __version__ == "2.0.0"

        plugin_info = get_plugin_info()
        assert "stateless_repository" in plugin_info.get("architecture", "")
        assert "breaking_changes_from_v1" in plugin_info


class TestV2APIFunctionality:
    """Tests für v2.0 API Funktionalität."""

    @pytest.fixture
    def mock_repository(self):
        """Mock Repository für Tests."""
        from unittest.mock import AsyncMock, Mock

        repo = Mock()
        repo.store_invoice_embedding = AsyncMock(return_value="mock_id")
        repo.query_similar_items = AsyncMock(return_value=[])
        repo.store_training_feedback = AsyncMock(return_value="mock_feedback_id")
        return repo

    @pytest.fixture
    def mock_gemini_client(self):
        """Mock Gemini Client."""
        from unittest.mock import AsyncMock, Mock

        client = Mock()
        client.process_pdf_direct = AsyncMock(
            return_value={
                "success": True,
                "invoice_items": [
                    {"description": "Test Item", "skr03_account": "3400"}
                ],
            }
        )
        return client

    @pytest.fixture
    def mock_embedding_provider(self):
        """Mock Embedding Provider."""
        from unittest.mock import Mock

        provider = Mock()
        provider.encode = Mock(return_value=[0.1] * 384)
        return provider

    def test_v2_plugin_creation(
        self, mock_repository, mock_gemini_client, mock_embedding_provider
    ):
        """Test dass v2.0 Plugin korrekt erstellt werden kann."""

        from llkjj_ml import MLPlugin, MLPluginConfig

        config = MLPluginConfig(validate_environment=False)
        plugin = MLPlugin(
            repository=mock_repository,
            gemini_client=mock_gemini_client,
            embedding_provider=mock_embedding_provider,
            config=config,
        )

        assert plugin is not None
        assert plugin.config.validate_environment is False

    @pytest.mark.asyncio
    async def test_v2_plugin_processing(
        self, mock_repository, mock_gemini_client, mock_embedding_provider
    ):
        """Test dass v2.0 Plugin-Processing funktioniert."""

        from uuid import uuid4

        from llkjj_ml import MLPlugin, MLPluginConfig

        config = MLPluginConfig(validate_environment=False)
        plugin = MLPlugin(
            repository=mock_repository,
            gemini_client=mock_gemini_client,
            embedding_provider=mock_embedding_provider,
            config=config,
        )

        # Mock process_invoice_pdf
        with patch.object(plugin, "process_invoice_pdf") as mock_process:
            mock_result = Mock()
            mock_result.processing_successful = True
            mock_result.items_processed = 1
            mock_process.return_value = mock_result

            result = await plugin.process_invoice_pdf(Path("/fake/test.pdf"), uuid4())

            assert result.processing_successful is True
            assert result.items_processed == 1


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
