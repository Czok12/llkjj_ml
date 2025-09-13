"""
Compatibility facade for tests expecting `llkjj_ml.services`.

Re-exports `UnifiedMLProcessor` from the modern pipeline package.
"""

from llkjj_ml.pipeline.unified_ml_processor import UnifiedMLProcessor

__all__ = ["UnifiedMLProcessor"]
