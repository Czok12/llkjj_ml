"""Utilities module for LLKJJ ML."""

# ResourceManager removed - was deprecated in v2.0.0
# Use MLPlugin v2 with dependency injection instead

from .german_errors import (
    GermanErrorMessages,
    wrap_pdf_errors,
    german_pdf_error,
    german_gemini_error,
    german_memory_error,
)

__all__: list[str] = [
    # German Error Messages - Pragmatic solution for 2 users
    "GermanErrorMessages",
    "wrap_pdf_errors", 
    "german_pdf_error",
    "german_gemini_error",
    "german_memory_error",
]
