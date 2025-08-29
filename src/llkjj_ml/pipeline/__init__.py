"""
LLKJJ ML Pipeline - Unified Processing Module

This module provides the main entry point for PDF processing with ML.
The UnifiedMLProcessor consolidates all previous processor implementations
into a single, consistent API.

Key exports:
- UnifiedMLProcessor: Main processor class with strategy pattern
- ProcessorConfig: Configuration for all processing strategies  
- ProcessingOptions: Options for single PDF processing
- BatchOptions: Options for batch processing operations
- BatchResult: Results from batch processing

Legacy compatibility:
- GeminiDirectProcessor -> UnifiedMLProcessor (deprecated)
- AsyncGeminiDirectProcessor -> UnifiedMLProcessor (deprecated)
- GeminiFirstProcessor -> UnifiedMLProcessor (deprecated)

Migration guide:
Replace old imports:
    from llkjj_ml.pipeline.gemini_first_processor import GeminiFirstProcessor
    processor = GeminiFirstProcessor()
    
With new import:
    from llkjj_ml.pipeline import UnifiedMLProcessor
    processor = UnifiedMLProcessor(strategy="gemini_first")
"""

import warnings
from .unified_ml_processor import (
    UnifiedMLProcessor,
    ProcessorConfig,
    ProcessingOptions, 
    BatchOptions,
    BatchResult,
    CacheManager,
    MemoryManager,
    create_unified_ml_processor
)

# Backward compatibility aliases with deprecation warnings
class GeminiDirectProcessor(UnifiedMLProcessor):
    """DEPRECATED: Use UnifiedMLProcessor instead."""
    
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "GeminiDirectProcessor is deprecated. Use UnifiedMLProcessor(strategy='gemini_first') instead.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(strategy="gemini_first", *args, **kwargs)


class AsyncGeminiDirectProcessor(UnifiedMLProcessor):
    """DEPRECATED: Use UnifiedMLProcessor.process_async() instead."""
    
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "AsyncGeminiDirectProcessor is deprecated. Use UnifiedMLProcessor.process_async() instead.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(strategy="gemini_first", *args, **kwargs)


class GeminiFirstProcessor(UnifiedMLProcessor):
    """DEPRECATED: Use UnifiedMLProcessor instead."""
    
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "GeminiFirstProcessor is deprecated. Use UnifiedMLProcessor(strategy='gemini_first') instead.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(strategy="gemini_first", *args, **kwargs)


__all__ = [
    # Primary API
    "UnifiedMLProcessor",
    "ProcessorConfig", 
    "ProcessingOptions",
    "BatchOptions",
    "BatchResult",
    "CacheManager",
    "MemoryManager",
    "create_unified_ml_processor",
    
    # Deprecated compatibility aliases
    "GeminiDirectProcessor",  # DEPRECATED
    "AsyncGeminiDirectProcessor",  # DEPRECATED  
    "GeminiFirstProcessor",  # DEPRECATED
]
