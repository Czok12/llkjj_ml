"""Error handling module for LLKJJ ML."""

from .comprehensive_error_handler import (
    ErrorCategory,
    ErrorRecoveryManager,
    ErrorRecoveryResult,
    ErrorSeverity,
    MLPipelineError,
    create_fallback_processing_result,
    error_handler,
    get_error_manager,
    handle_api_error,
    handle_model_error,
    handle_pdf_error,
)

__all__ = [
    "ErrorSeverity",
    "ErrorCategory",
    "MLPipelineError",
    "ErrorRecoveryResult",
    "ErrorRecoveryManager",
    "error_handler",
    "create_fallback_processing_result",
    "get_error_manager",
    "handle_pdf_error",
    "handle_api_error",
    "handle_model_error",
]
