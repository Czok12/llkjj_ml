#!/usr/bin/env python3
"""
LLKJJ ML Plugin - Blackbox API Documentation
============================================

**COMPLETE PUBLIC INTERFACE SPECIFICATION**

This document defines the complete "blackbox" interface for the LLKJJ ML Plugin.
External systems can use this as the definitive API contract.

Version: 3.0.0
Date: August 17, 2025
"""

# Required imports for type annotations
import json
import os
from pathlib import Path
from typing import Any

# =============================================================================
# TYPE DEFINITIONS FOR API DOCUMENTATION
# =============================================================================


class Config:
    """Configuration class placeholder for API documentation."""

    pass


class ProcessingResult:
    """ProcessingResult class placeholder for API documentation."""

    pass


# =============================================================================
# PUBLIC API INTERFACE
# =============================================================================


class MLPluginAPI:
    """
    **BLACKBOX INTERFACE SPECIFICATION**

    The LLKJJ ML Plugin provides a completely self-contained processing
    pipeline for German electrical trade invoices with SKR03 integration.

    **NO EXTERNAL DEPENDENCIES:**
    - No knowledge of DMS systems
    - No dependency on other plugins
    - No system-specific configurations required

    **GUARANTEED INTERFACE:**
    All methods and data structures in this specification are guaranteed
    to remain stable across minor version updates (3.x.x).
    """

    def __init__(self, config: Config | None = None) -> None:
        """
        Initialize ML Plugin instance.

        Args:
            config: Optional configuration. Uses optimized defaults if None.
                   Defaults are tuned for German electrical trade invoices.

        Internal Behavior:
            - All ML models are lazy-loaded on first use
            - ResourceManager singleton prevents memory waste
            - German language optimizations enabled by default
            - ChromaDB vector store initialized for SKR03 classification
        """

    def process_pdf(self, pdf_path: str | Path) -> ProcessingResult:
        """
        **PRIMARY BLACKBOX METHOD**

        Process PDF invoice through complete ML pipeline.

        Input Contract:
            - pdf_path: Path to readable PDF file
            - File must exist and be valid PDF format
            - No size restrictions (optimized for typical invoices)

        Output Contract:
            - Returns ProcessingResult with guaranteed schema
            - All fields documented in ProcessingResult specification
            - Processing never partially fails - always returns complete result

        Internal Pipeline (Hidden from External Systems):
            1. PDF → Docling (OCR + table extraction)
            2. Docling → Gemini (AI enhancement)
            3. Enhanced text → spaCy (German NLP)
            4. Items → ChromaDB RAG (SKR03 classification)
            5. Results → Quality assessment
            6. All data → ProcessingResult packaging

        Performance:
            - Typical processing: 30-60 seconds
            - Memory usage: ~500MB peak (cleaned up automatically)
            - Concurrent processing: Safe (ResourceManager handles locking)

        Error Handling:
            - FileNotFoundError: Invalid/missing PDF
            - ValueError: Corrupted PDF file
            - ProcessingError: Internal ML pipeline failure (rare)
            - All errors include helpful German error messages
        """
        # Documentation stub - implementation varies
        raise NotImplementedError("API Documentation stub")

    def validate_input(self, pdf_path: str | Path) -> bool:
        """
        Pre-validate file before processing.

        Returns:
            True if file is processable, False otherwise

        Use Case:
            Check files before batch processing to filter invalid inputs.
        """
        # Documentation stub - implementation varies
        return True

    def get_capabilities(self) -> dict[str, bool]:
        """
        Return plugin capability flags for feature detection.

        Returns:
            Dict with boolean flags:
            - pdf_processing: Basic PDF text extraction
            - table_extraction: Advanced table recognition
            - skr03_classification: German accounting integration
            - german_optimization: German language support
            - batch_processing: Multiple file handling
            - ocr_support: Image-based PDF processing
            - ai_enhancement: Gemini AI text improvement
        """
        # Documentation stub - implementation varies
        return {
            "pdf_processing": True,
            "table_extraction": True,
            "skr03_classification": True,
            "german_optimization": True,
            "batch_processing": True,
            "ocr_support": True,
            "ai_enhancement": True,
        }

    def cleanup(self) -> None:
        """
        Cleanup resources when plugin no longer needed.

        Releases:
            - ML model memory (Docling, Gemini, spaCy)
            - ChromaDB connections
            - Cache files and temporary data
            - GPU memory if applicable
        """


# =============================================================================
# OUTPUT DATA CONTRACT
# =============================================================================


class ProcessingResultSchema:
    """
    **GUARANTEED OUTPUT SCHEMA**

    This is the complete data contract returned by process_pdf().
    External systems can depend on this structure remaining stable.

    **VERSION**: 3.0.0
    **STABILITY**: All fields guaranteed in minor versions (3.x.x)
    """

    # SOURCE INFORMATION
    pdf_path: str  # Absolute path to processed file
    processing_timestamp: str  # ISO UTC timestamp (e.g., "2025-08-17T14:30:25Z")

    # EXTRACTION RESULTS
    raw_text: str  # Complete extracted text (10,000+ chars typical)
    structured_data: dict[str, Any]  # Docling results: tables, layout, metadata

    # INVOICE DATA
    invoice_data: dict[str, Any]  # Header info: supplier, dates, totals
    skr03_classifications: list[dict[str, Any]]  # Main results - see below

    # PERFORMANCE METRICS
    processing_time_ms: int  # Total time (30,000-60,000ms typical)
    ocr_time_ms: int  # OCR extraction time
    classification_time_ms: int  # SKR03 classification time

    # QUALITY INDICATORS
    confidence_score: float  # Overall confidence 0.0-1.0 (0.6+ is good)
    extraction_quality: str  # "high" | "medium" | "low"


class SKR03ClassificationSchema:
    """
    **SKR03 CLASSIFICATION ITEM SCHEMA**

    Each item in skr03_classifications list has this guaranteed structure:
    """

    position: int  # Line item number (1, 2, 3, ...)
    article_number: str  # Product code (e.g., "028203")
    description: str  # Item description (German)
    quantity: int  # Quantity ordered
    price_per_unit: float  # Unit price in EUR (optional)
    total_price: float  # Line total in EUR (optional)
    skr03_account: str  # German accounting code (e.g., "3400")
    skr03_category: str  # Account category name
    confidence: float  # Classification confidence 0.0-1.0
    classification_method: str  # "rag_dominant" | "rule_based" | "ai_enhanced"


# =============================================================================
# USAGE EXAMPLES
# =============================================================================


def example_simple_usage() -> None:
    """Example: Simple one-shot processing"""
    from llkjj_ml_plugin import process_pdf_simple

    result = process_pdf_simple("sonepar_invoice.pdf")

    print(f"Quality: {result.extraction_quality}")
    print(f"Found {len(result.skr03_classifications)} items")

    for item in result.skr03_classifications:
        print(f"  {item['description']} -> SKR03: {item['skr03_account']}")


def example_batch_processing() -> None:
    """Example: Batch processing with resource management"""
    from pathlib import Path

    from llkjj_ml_plugin import MLPlugin

    plugin = MLPlugin()

    try:
        for pdf_file in Path("invoices/").glob("*.pdf"):
            if plugin.validate_input(pdf_file):
                result = plugin.process_pdf(pdf_file)

                # Save results
                output_file = f"results/{pdf_file.stem}.json"
                with open(output_file, "w") as f:
                    json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)

                print(f"✅ {pdf_file.name}: {result.get_summary()}")
            else:
                print(f"❌ {pdf_file.name}: Invalid PDF")

    finally:
        plugin.cleanup()  # Important: cleanup resources


def example_integration_check() -> None:
    """Example: Check plugin capabilities before use"""
    from llkjj_ml_plugin import MLPlugin

    plugin = MLPlugin()
    caps = plugin.get_capabilities()

    if caps["skr03_classification"] and caps["german_optimization"]:
        print("✅ Plugin supports German SKR03 processing")

        # Process with confidence
        result = plugin.process_pdf("german_invoice.pdf")

        if result.extraction_quality == "high":
            print("✅ High quality extraction - safe for automated processing")
        else:
            print("⚠️ Medium/low quality - manual review recommended")

    plugin.cleanup()


# =============================================================================
# INTEGRATION PATTERNS
# =============================================================================


class IntegrationPatterns:
    """
    **RECOMMENDED INTEGRATION PATTERNS**

    These patterns show how external systems should integrate with
    the ML plugin while maintaining the blackbox principle.
    """

    @staticmethod
    def web_api_integration() -> None:
        """
        Pattern: Web API endpoint

        Integrate plugin into web service with proper resource management.
        """
        import tempfile

        from fastapi import FastAPI, HTTPException, UploadFile

        from llkjj_ml_plugin import MLPlugin

        app = FastAPI()

        # Global plugin instance for efficiency
        ml_plugin = MLPlugin()

        @app.post("/process-invoice")
        async def process_invoice(file: UploadFile) -> dict[str, Any]:
            if not file.filename or not file.filename.endswith(".pdf"):
                raise HTTPException(400, "Only PDF files allowed")

            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                content = await file.read()
                tmp.write(content)
                tmp_path = tmp.name

            try:
                # Process with ML plugin (blackbox operation)
                result = ml_plugin.process_pdf(tmp_path)

                # Return structured results
                return {
                    "status": "success",
                    "quality": result.extraction_quality,
                    "confidence": result.confidence_score,
                    "items": result.skr03_classifications,
                    "processing_time": result.processing_time_ms,
                }

            finally:
                os.unlink(tmp_path)  # Cleanup temp file

        @app.on_event("shutdown")
        async def shutdown() -> None:
            ml_plugin.cleanup()  # Cleanup on server shutdown

    @staticmethod
    def batch_processing_pattern() -> None:
        """
        Pattern: Batch processing with progress tracking

        Process multiple files with proper error handling and progress.
        """
        import logging
        from typing import Any

        from llkjj_ml_plugin import MLPlugin

        def process_invoice_batch(pdf_paths: list[str]) -> list[dict[str, Any]]:
            plugin = MLPlugin()
            results = []

            try:
                for i, pdf_path in enumerate(pdf_paths):
                    logging.info(f"Processing {i+1}/{len(pdf_paths)}: {pdf_path}")

                    try:
                        if plugin.validate_input(pdf_path):
                            result = plugin.process_pdf(pdf_path)

                            results.append(
                                {
                                    "file": pdf_path,
                                    "status": "success",
                                    "data": result.to_dict(),
                                }
                            )

                        else:
                            results.append(
                                {
                                    "file": pdf_path,
                                    "status": "invalid_file",
                                    "error": "File is not a valid PDF",
                                }
                            )

                    except Exception as e:
                        results.append(
                            {
                                "file": pdf_path,
                                "status": "processing_error",
                                "error": str(e),
                            }
                        )

            finally:
                plugin.cleanup()  # Always cleanup

            return results

    @staticmethod
    def quality_filtering_pattern() -> None:
        """
        Pattern: Quality-based processing decisions

        Use quality indicators to make automated vs manual decisions.
        """
        from llkjj_ml_plugin import MLPlugin

        def smart_processing(pdf_path: str) -> dict[str, Any]:
            plugin = MLPlugin()

            try:
                result = plugin.process_pdf(pdf_path)

                # Quality-based decision making
                if (
                    result.extraction_quality == "high"
                    and result.confidence_score > 0.9
                ):
                    return {
                        "recommendation": "auto_process",
                        "confidence": "very_high",
                        "data": result.skr03_classifications,
                    }

                elif (
                    result.extraction_quality in ["high", "medium"]
                    and result.confidence_score > 0.7
                ):
                    return {
                        "recommendation": "review_and_approve",
                        "confidence": "good",
                        "data": result.skr03_classifications,
                    }

                else:
                    return {
                        "recommendation": "manual_processing",
                        "confidence": "low",
                        "issues": "Quality too low for automated processing",
                    }

            finally:
                plugin.cleanup()


# =============================================================================
# ERROR HANDLING GUIDE
# =============================================================================


class ErrorHandling:
    """
    **COMPREHENSIVE ERROR HANDLING**

    Guide for handling all possible error conditions from the ML plugin.
    """

    @staticmethod
    def robust_processing_example() -> None:
        """
        Example: Robust error handling with proper logging
        """
        import logging

        from llkjj_ml_plugin import MLPlugin

        def process_with_error_handling(pdf_path: str) -> dict[str, Any]:
            plugin = MLPlugin()

            try:
                # Pre-validation
                if not plugin.validate_input(pdf_path):
                    return {
                        "status": "error",
                        "error_type": "invalid_input",
                        "message": "File is not a valid or readable PDF",
                    }

                # Main processing
                result = plugin.process_pdf(pdf_path)

                return {"status": "success", "result": result.to_dict()}

            except FileNotFoundError:
                return {
                    "status": "error",
                    "error_type": "file_not_found",
                    "message": f"PDF file not found: {pdf_path}",
                }

            except ValueError as e:
                return {
                    "status": "error",
                    "error_type": "invalid_pdf",
                    "message": f"Invalid PDF format: {str(e)}",
                }

            except Exception as e:
                logging.exception(f"Unexpected error processing {pdf_path}")
                return {
                    "status": "error",
                    "error_type": "processing_error",
                    "message": f"Internal processing error: {str(e)}",
                }

            finally:
                plugin.cleanup()


# =============================================================================
# PERFORMANCE OPTIMIZATION
# =============================================================================


class PerformanceOptimization:
    """
    **PERFORMANCE OPTIMIZATION GUIDE**

    Best practices for optimal performance with the ML plugin.
    """

    performance_characteristics = {
        "initialization_time": "2-5 seconds (one-time setup)",
        "processing_time": "30-60 seconds per PDF",
        "memory_usage": "~500MB peak (auto-cleanup)",
        "concurrent_processing": "Safe with ResourceManager singleton",
        "batch_optimization": "Reuse plugin instance for multiple files",
    }

    optimization_tips = [
        "Reuse MLPlugin instance for batch processing",
        "Call validate_input() before processing to filter bad files",
        "Monitor processing_time_ms to identify problematic files",
        "Use cleanup() only when completely done with plugin",
        "Consider quality thresholds to skip low-quality results",
    ]


# =============================================================================
# VERSION COMPATIBILITY
# =============================================================================

VERSION_COMPATIBILITY = {
    "3.0.0": {
        "breaking_changes": [
            "New blackbox interface design",
            "Enhanced ProcessingResult schema with examples",
            "Removed all external system dependencies",
        ],
        "backwards_compatibility": [
            "ProcessingResult.to_dict() still available",
            "All core processing methods unchanged",
            "SKR03 classification format stable",
        ],
        "migration_guide": [
            "Update imports: from llkjj_ml_plugin import MLPlugin",
            "Replace direct processor usage with MLPlugin",
            "Add plugin.cleanup() calls for resource management",
        ],
    }
}
