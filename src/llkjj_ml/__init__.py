# Package marker

# Main modules
# Package info from package_api
from . import (
    config,
    database,
    embeddings,
    error_handling,
    extraction,
    features,
    gemini,
    integration,
    intelligence,
    models,
    monitoring,
    optimization,
    package_api,
    pipeline,
    processing,
    quality,
    security,
    utils,
)
from .settings_bridge import ConfigBridge

# Core functionality
from .skr03_manager import SKR03Manager
from .training_data_persistence import TrainingDataPersistence

__all__ = [
    # Submodules
    "database",
    "features",
    "models",
    "pipeline",
    "processing",
    "integration",
    "monitoring",
    "utils",
    "extraction",
    "intelligence",
    "optimization",
    "error_handling",
    "config",
    "security",
    "embeddings",
    "gemini",
    "quality",
    # Core classes
    "SKR03Manager",
    "ConfigBridge",
    "TrainingDataPersistence",
    "package_api",
]
