#!/usr/bin/env python3
"""
Test Script für stateless llkjj_ml v2.0 Integration

Testet die neue Repository-Pattern-basierte Implementierung mit llkjj_backend.

Author: LLKJJ ML Team
Usage: GOOGLE_API_KEY=development_mode python test_stateless_integration.py
"""

import asyncio
import logging
import sys
from pathlib import Path
from uuid import uuid4

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "llkjj_backend"))


async def test_backend_services_standalone():
    """Test Backend Services ohne vollständige Registry."""
    print("=" * 60)
    print("🧪 Test 1: Backend Services Standalone")
    print("=" * 60)

    try:
        # Import services
        from src.integration.backend_services import (
            BackendEmbeddingService,
            BackendGeminiService,
            ServiceHealthChecker,
        )

        # Test Embedding Service
        print("🔤 Teste Embedding Service...")
        embedding_service = BackendEmbeddingService()

        test_text = "WAGO-Klemme 221-412 für Elektroinstallation"
        embedding = embedding_service.encode(test_text)

        print(f"✅ Embedding erstellt: {len(embedding)} Dimensionen")
        print(f"   Erste 5 Werte: {embedding[:5]}")

        # Test Batch Encoding
        batch_texts = ["Siemens Schalter", "ABB Sicherung", "Phoenix Contact Klemme"]
        batch_embeddings = embedding_service.encode_batch(batch_texts)
        print(f"✅ Batch Encoding: {len(batch_embeddings)} Embeddings")

        # Test Similarity
        similarity = embedding_service.similarity(
            "WAGO-Klemme", "Phoenix Contact Klemme"
        )
        print(f"✅ Similarity: {similarity:.3f}")

        # Health Check Embedding Service
        embedding_health = ServiceHealthChecker.check_embedding_service(
            embedding_service
        )
        print(f"✅ Embedding Service Health: {embedding_health['status']}")

        # Test Gemini Service (Mock/Light)
        print("\n🤖 Teste Gemini Service...")
        gemini_service = BackendGeminiService()

        # Service Info
        gemini_info = gemini_service.get_service_info()
        print(f"✅ Gemini Service Info: {gemini_info['model_name']}")

        # Health Check Gemini Service
        gemini_health = await ServiceHealthChecker.check_gemini_service(gemini_service)
        print(f"✅ Gemini Service Health: {gemini_health['status']}")

        # Comprehensive Health Check
        print("\n📊 Comprehensive Health Check...")
        comprehensive_health = await ServiceHealthChecker.comprehensive_health_check(
            gemini_service, embedding_service
        )
        print(f"✅ Overall Status: {comprehensive_health['overall_status']}")

        # Cleanup
        embedding_service.cleanup()

        return True

    except Exception as e:
        logger.error(f"❌ Backend Services Test fehlgeschlagen: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_plugin_protocols():
    """Test dass Plugin-Protocols korrekt funktionieren."""
    print("\n" + "=" * 60)
    print("🔌 Test 2: Plugin Protocols & Interfaces")
    print("=" * 60)

    try:
        # Import Plugin und Protocols
        from llkjj_ml_plugin_v2 import (
            EmbeddingProvider,
            GeminiClient,
            MLPlugin,
            MLPluginConfig,
        )
        from src.integration.backend_services import (
            BackendEmbeddingService,
            BackendGeminiService,
        )

        print("📋 Teste Protocol Compliance...")

        # Test Services gegen Protocols
        gemini_service = BackendGeminiService()
        embedding_service = BackendEmbeddingService()

        # Protocol Compliance Tests
        print(f"✅ GeminiClient Protocol: {isinstance(gemini_service, GeminiClient)}")
        print(
            f"✅ EmbeddingProvider Protocol: {isinstance(embedding_service, EmbeddingProvider)}"
        )

        # Mock Repository für Test
        class MockMLRepository:
            """Mock Repository für Test."""

            async def store_invoice_embedding(
                self, invoice_id, invoice_item_id, embedding_data
            ):
                return uuid4()

            async def query_similar_items(
                self, query_text, query_embedding, limit=5, min_similarity=0.7
            ):
                return []  # Empty result für Test

            async def get_embedding_stats(self):
                return {"total_embeddings": 0}

            async def store_training_feedback(self, feedback):
                return uuid4()

            async def get_pending_training_data(
                self, limit=None, min_confidence_rating=3
            ):
                return []

            async def mark_training_data_used(self, feedback_ids):
                pass

            async def get_training_stats(self):
                return {"total_feedback": 0}

        mock_repository = MockMLRepository()

        # Test Plugin Creation mit Dependency Injection
        print("🏗️ Teste Plugin Creation...")

        config = MLPluginConfig(
            validate_environment=False,  # Für Test
            enable_rag_enhancement=False,  # Vereinfacht für Test
        )

        plugin = MLPlugin(
            repository=mock_repository,
            gemini_client=gemini_service,
            embedding_provider=embedding_service,
            config=config,
        )

        print("✅ MLPlugin v2.0 erfolgreich mit Dependency Injection erstellt")

        # Test Plugin Info
        plugin_info = plugin.get_plugin_info()
        print(f"✅ Plugin Info: {plugin_info['name']} v{plugin_info['version']}")
        print(f"   Capabilities: {len(plugin_info['capabilities'])} Features")

        # Test Statistics Methods
        embedding_stats = await plugin.get_embedding_statistics()
        training_stats = await plugin.get_training_statistics()

        print(f"✅ Embedding Stats: {embedding_stats}")
        print(f"✅ Training Stats: {training_stats}")

        # Cleanup
        embedding_service.cleanup()

        return True

    except Exception as e:
        logger.error(f"❌ Protocol Test fehlgeschlagen: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_integration_functions():
    """Test Integration Functions."""
    print("\n" + "=" * 60)
    print("🔗 Test 3: Integration Functions")
    print("=" * 60)

    try:
        # Import from the correct __init__.py (not from llkjj_ml package)
        sys.path.insert(0, str(Path(__file__).parent))
        from __init__ import check_environment, get_plugin_info

        env_status = check_environment()
        print(f"🌍 Environment Status: {env_status['status']}")

        if env_status["missing_requirements"]:
            print(f"⚠️ Missing: {env_status['missing_requirements']}")

        # Test Plugin Info
        plugin_info = get_plugin_info()
        print(f"📋 Plugin Info: {plugin_info['name']} v{plugin_info['version']}")
        print(f"   Architecture: {plugin_info['architecture']}")

        # Test ML Services Integration Function
        from src.integration.backend_services import test_ml_services_integration

        integration_result = await test_ml_services_integration()
        print(f"🧪 ML Services Integration: {integration_result['integration_test']}")

        if integration_result.get("health_results"):
            health = integration_result["health_results"]
            print(f"   Overall Health: {health['overall_status']}")

        return True

    except Exception as e:
        logger.error(f"❌ Integration Functions Test fehlgeschlagen: {e}")
        import traceback

        traceback.print_exc()
        return False


async def main():
    """Haupt-Test-Funktion."""
    print("🚀 LLKJJ ML v2.0 Stateless Integration Tests")
    print(f"📁 Working Directory: {Path.cwd()}")
    print(f"🐍 Python: {sys.version}")

    # Run all tests
    test_results = []

    test_results.append(await test_backend_services_standalone())
    test_results.append(await test_plugin_protocols())
    test_results.append(await test_integration_functions())

    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)

    passed = sum(test_results)
    total = len(test_results)

    if passed == total:
        print(f"🎉 Alle Tests erfolgreich! ({passed}/{total})")
        print("✅ llkjj_ml v2.0 Stateless Integration ist ready!")
        return 0
    else:
        print(f"❌ {total - passed} von {total} Tests fehlgeschlagen")
        print("🔧 Bitte Fehler beheben vor Production-Deploy")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
