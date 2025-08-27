"""
LLKJJ ML Package - Clean Architecture Version

**VERSION 3.0.0 - POST-MIGRATION:**
- Clean src/ structure implemented
- Import paths standardized
- Plugin interface preserved
- ML pipelines optimized

**Usage:**
Import from the src package:
    from llkjj_ml.src.llkjj_ml import *
"""

# Re-export main functionality from migrated src structure
try:
    from .src.llkjj_ml import *
except ImportError:
    # If direct import fails, try to import individual components
    try:
        from .src.llkjj_ml.models.processing_result import ProcessingResult
        from .src.llkjj_ml.pipeline.unified_processor import UnifiedProcessor
    except ImportError:
        pass  # Graceful degradation during development

__version__ = "3.0.0"
__author__ = "LLKJJ ML Team"
