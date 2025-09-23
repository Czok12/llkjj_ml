"""Data extraction modules for LLKJJ ML Pipeline"""

# Suppress Pydantic warnings from third-party libraries (e.g., Docling)
import warnings

warnings.filterwarnings(
    "ignore",
    message=r'Field "model_.*" in .* has conflict with protected namespace "model_"',
    category=UserWarning,
)

# All extraction modules require docling - make everything conditional
__all__ = []

try:
    from .docling_processor import (
        AdvancedDoclingProcessor,
        OcrEngineConfig,
        create_advanced_docling_processor,
        create_optimized_docling_processor,
    )
    from .extractor import DataExtractor

    __all__.extend(
        [
            "DataExtractor",
            "AdvancedDoclingProcessor",
            "OcrEngineConfig",
            "create_advanced_docling_processor",
            "create_optimized_docling_processor",
        ]
    )
except ImportError:
    # docling not available - extraction modules unavailable
    pass
