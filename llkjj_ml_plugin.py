#!/usr/bin/env python3
"""
LLKJJ ML Plugin - Public API Interface
=====================================

This module provides the public "blackbox" interface for the LLKJJ ML Plugin.
It defines clear input/output contracts and hides all internal complexity.

**PUBLIC API:**
- Input: File path (str | Path)
- Output: ProcessingResult (Pydantic model)
- No external dependencies or system knowledge required

**USAGE:**
```python
from llkjj_ml_plugin import MLPlugin

# Initialize plugin (handles all internal setup)
plugin = MLPlugin()

# Process PDF (blackbox operation)
result = plugin.process_pdf("path/to/invoice.pdf")

# Access structured results
print(f"SKR03 classifications: {result.skr03_classifications}")
print(f"Confidence: {result.confidence_score}")
```

Author: LLKJJ ML Pipeline Team
Version: 3.0.0 (Blackbox Interface)
"""

import logging
from pathlib import Path

from src.config import Config
from src.pipeline.processor import ProcessingResult, UnifiedProcessor
from src.security.manager import validate_production_environment

logger = logging.getLogger(__name__)

__version__ = "3.0.0"
__all__ = ["MLPlugin", "ProcessingResult", "process_pdf_simple"]


class MLPlugin:
    """
    LLKJJ ML Plugin - Blackbox Interface

    Self-contained ML processing plugin that transforms PDF invoices
    into structured SKR03 accounting data. No external dependencies.

    **Interface Contract:**
    - Input: PDF file path (string or Path object)
    - Output: ProcessingResult with guaranteed schema
    - Internal: Manages all ML models, databases, and resources

    **Design Principles:**
    - Encapsulation: No knowledge of external systems
    - Singleton pattern: Efficient resource management
    - Type safety: Pydantic models for all data contracts
    - German optimization: Specialized for German electrical trade
    """

    def __init__(self, config: Config | None = None, validate_env: bool = True) -> None:
        """
        Initialize ML Plugin with internal configuration.

        Args:
            config: Optional configuration override. If None, uses default
                   internal configuration optimized for German invoices.
            validate_env: If True (default), validates critical environment
                         variables like GOOGLE_API_KEY at startup.

        Raises:
            RuntimeError: If critical environment variables are missing
                         and validate_env=True

        Note:
            All heavy ML models (Docling, Gemini, ChromaDB, spaCy) are
            lazy-loaded on first use via internal ResourceManager singleton.
        """
        # Critical environment validation for production readiness
        if validate_env:
            self._validate_critical_environment()

        self._config = config or Config()
        self._processor = UnifiedProcessor(self._config)

    def _validate_critical_environment(self) -> None:
        """
        Validate critical environment variables for production readiness.

        Raises:
            RuntimeError: If critical environment variables are missing
        """
        logger.info("🔍 Validating critical environment variables...")

        if not validate_production_environment():
            error_msg = (
                "❌ Critical environment validation failed! "
                "Missing required environment variables like GOOGLE_API_KEY. "
                "Set validate_env=False to skip validation in development."
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        logger.info("✅ Environment validation passed - production ready")

    def process_pdf(self, pdf_path: str | Path) -> ProcessingResult:
        """
        **MAIN PUBLIC API METHOD**

        Process a PDF invoice through the complete ML pipeline.
        This is the primary blackbox interface method.

        Args:
            pdf_path: Path to PDF file to process. Must be readable PDF.

        Returns:
            ProcessingResult: Structured result with guaranteed schema:
                - pdf_path: Original file path
                - processing_timestamp: ISO timestamp
                - raw_text: Extracted text content
                - structured_data: Docling extraction results
                - invoice_data: Enhanced invoice information
                - skr03_classifications: German accounting classifications
                - processing_time_ms: Performance metrics
                - confidence_score: Quality indicator (0.0-1.0)
                - extraction_quality: "high" | "medium" | "low"

        Raises:
            FileNotFoundError: If PDF file doesn't exist
            ValueError: If file is not a valid PDF
            ProcessingError: If ML pipeline fails (rare)

        Example:
            >>> plugin = MLPlugin()
            >>> result = plugin.process_pdf("sonepar_invoice.pdf")
            >>> print(f"Found {len(result.skr03_classifications)} items")
            >>> print(f"Quality: {result.extraction_quality}")
        """
        return self._processor.process_pdf(pdf_path)

    def get_version(self) -> str:
        """Return plugin version for compatibility checking."""
        return __version__

    def get_capabilities(self) -> dict[str, bool]:
        """
        Return plugin capabilities for feature detection.

        Returns:
            Dict with supported features:
            - pdf_processing: PDF text extraction
            - table_extraction: Table recognition
            - skr03_classification: German accounting integration
            - german_optimization: German language support
            - batch_processing: Multiple file support
        """
        return {
            "pdf_processing": True,
            "table_extraction": True,
            "skr03_classification": True,
            "german_optimization": True,
            "batch_processing": True,
            "ocr_support": True,
            "ai_enhancement": True,
        }

    def validate_input(self, pdf_path: str | Path) -> bool:
        """
        Validate if a file can be processed by this plugin.

        Args:
            pdf_path: File path to validate

        Returns:
            bool: True if file is processable, False otherwise
        """
        try:
            path = Path(pdf_path)
            return (
                path.exists()
                and path.is_file()
                and path.suffix.lower() == ".pdf"
                and path.stat().st_size > 0
            )
        except (OSError, TypeError):
            return False

    def cleanup(self) -> None:
        """
        Cleanup internal resources when plugin is no longer needed.

        Properly releases ML models, database connections, and memory.
        Call this when shutting down or switching to different processing.
        """
        if hasattr(self, "_processor"):
            self._processor.cleanup()


# Convenience function for simple use cases
def process_pdf_simple(pdf_path: str | Path) -> ProcessingResult:
    """
    Convenience function for one-shot PDF processing.

    Creates plugin instance, processes file, and cleans up automatically.
    Use this for simple scripts or single-file processing.

    Args:
        pdf_path: Path to PDF file

    Returns:
        ProcessingResult: Complete processing results

    Example:
        >>> from llkjj_ml_plugin import process_pdf_simple
        >>> result = process_pdf_simple("invoice.pdf")
        >>> print(result.get_summary())
    """
    plugin = MLPlugin()
    try:
        return plugin.process_pdf(pdf_path)
    finally:
        plugin.cleanup()
