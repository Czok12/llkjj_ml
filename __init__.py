"""
LLKJJ ML Package - Stateless Repository-Pattern Version

**VERSION 2.0.0 - BREAKING CHANGES:**
- Stateless Design mit Repository-Pattern
- Integration mit llkjj_backend
- Elimination von ResourceManager-Singleton
- Gemini-Direct als Standard (ohne Docling)

**Migration Guide:**
```python
# OLD (v1.x):
from llkjj_ml import MLPlugin
plugin = MLPlugin()
result = plugin.process_pdf("invoice.pdf")

# NEW (v2.x):
from llkjj_ml import create_ml_plugin_for_backend
from llkjj_backend.core.modules import registry

plugin = create_ml_plugin_for_backend(registry)
result = await plugin.process_invoice_pdf(Path("invoice.pdf"), invoice_id)
```

Author: LLKJJ ML Team
Version: 2.0.0
Date: 2025-01-25
"""

import logging
from pathlib import Path
from typing import Optional

# Set up package logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Version info
__version__ = "2.0.0"
__author__ = "LLKJJ ML Team"

# ============================================================================= 
# Primary Exports (v2.0 API)
# =============================================================================

# Stateless Plugin (v2.0)
from .llkjj_ml_plugin_v2 import (
    MLPlugin,
    MLPluginConfig,
    InvoiceProcessingResult,
    EmbeddingData,
    SimilarItem,
    TrainingFeedback,
    TrainingItem
)

# Training System
from .src.training.spacy_trainer import (
    SpacyTrainer,
    TrainingConfig,
    TrainingMetrics
)

# Integration Services
from .src.integration.backend_services import (
    MLPluginFactory,
    BackendGeminiService,
    BackendEmbeddingService,
    create_ml_plugin_for_backend,
    test_ml_services_integration
)

# Training Integration Services
from .src.integration.training_services import (
    BackendTrainingOrchestrator,
    create_training_orchestrator_for_backend,
    schedule_training_pipeline,
    get_training_recommendations
)

# Core Services  
from .src.gemini.direct_processor import (
    GeminiDirectProcessor,
    GeminiDirectConfig,
    create_gemini_direct_processor,
    process_pdf_with_gemini
)

from .src.embeddings.sentence_transformer_provider import (
    SentenceTransformerProvider,
    create_german_embedding_provider,
    create_multilingual_embedding_provider
)

# ============================================================================= 
# Legacy Exports removed in v2.0.0 final cleanup

# ============================================================================= 
# Convenience Functions
# =============================================================================

def get_version() -> str:
    """Get package version."""
    return __version__


def get_plugin_info() -> dict:
    """Get comprehensive plugin information."""
    return {
        "name": "LLKJJ ML Package",
        "version": __version__,
        "author": __author__,
        "architecture": "stateless_repository",
        "primary_api": "MLPlugin v2.0",
        "capabilities": [
            "pdf_processing",
            "gemini_direct", 
            "rag_enhancement",
            "pgvector_embeddings",
            "training_feedback",
            "skr03_classification"
        ],
        "dependencies": [
            "llkjj_backend",
            "llkjj_database", 
            "google-genai",
            "sentence-transformers"
        ],
        "breaking_changes_from_v1": [
            "Repository-Pattern required",
            "Async-only processing",
            "No ResourceManager singleton",
            "Backend integration mandatory"
        ]
    }


def check_environment() -> dict:
    """Check if environment is ready for ML processing."""
    import os
    
    checks = {
        "gemini_api_key": bool(os.getenv("GEMINI_API_KEY")),
        "sentence_transformers_available": False,
        "google_genai_available": False,
        "backend_integration_ready": False
    }
    
    # Check sentence-transformers
    try:
        import sentence_transformers
        checks["sentence_transformers_available"] = True
    except ImportError:
        pass
    
    # Check google-genai
    try:
        import google.genai
        checks["google_genai_available"] = True  
    except ImportError:
        pass
    
    # Check backend integration
    try:
        from llkjj_backend.core.modules import registry
        checks["backend_integration_ready"] = True
    except ImportError:
        pass
    
    # Overall status
    all_ready = all(checks.values())
    
    return {
        "status": "ready" if all_ready else "incomplete",
        "checks": checks,
        "missing_requirements": [
            name for name, status in checks.items() if not status
        ]
    }


# ============================================================================= 
# Public API Definition
# =============================================================================

__all__ = [
    # Version Info
    "__version__",
    "get_version", 
    "get_plugin_info",
    "check_environment",
    
    # Primary API (v2.0)
    "MLPlugin",
    "MLPluginConfig", 
    "InvoiceProcessingResult",
    
    # Data Models
    "EmbeddingData",
    "SimilarItem",
    "TrainingFeedback",
    "TrainingItem",
    
    # Integration
    "MLPluginFactory",
    "create_ml_plugin_for_backend",
    "test_ml_services_integration",
    
    # Training Integration
    "BackendTrainingOrchestrator",
    "create_training_orchestrator_for_backend",
    "schedule_training_pipeline",
    "get_training_recommendations",
    
    # Services
    "BackendGeminiService",
    "BackendEmbeddingService",
    "GeminiDirectProcessor",
    "SentenceTransformerProvider",
    
    # Factories
    "create_gemini_direct_processor",
    "create_german_embedding_provider",
    "create_multilingual_embedding_provider",
    "process_pdf_with_gemini",
    
    # Training System
    "SpacyTrainer",
    "TrainingConfig", 
    "TrainingMetrics",
    

]