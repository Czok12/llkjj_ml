#!/usr/bin/env python3
"""
LLKJJ ML Pipeline - Slim Orchestrator (KISS Architecture)
========================================================

Schlanker Orchestrator der die spezialisierten Module koordiniert:
- DataExtractor: PDF-Extraktion und Datengewinnung
- DataClassifier: SKR03-Klassifizierung und RAG-System
- QualityAssessor: Qualitätsbewertung und Konfidenz-Scores

Folgt dem Single Responsibility Principle für bessere Wartbarkeit.

Autor: LLKJJ ML Pipeline Team
Version: 2.1.0 (Post-Konsolidierung Refactoring)
"""

import logging
from pathlib import Path

try:
    from google import genai
except ImportError:
    genai = None  # type: ignore[assignment]

from ..models.processing_result import ProcessingResult
from ..settings_bridge import ConfigBridge, config_instance
from .unified_processor import UnifiedProcessor

# dual_pipeline.py removed in cleanup - functionality integrated into UnifiedProcessor

logger = logging.getLogger(__name__)


class ResourceManager:
    """
    REMOVED: Legacy ResourceManager has been removed in v2.0.0 final cleanup

    ⚠️ This Singleton class has been completely replaced by dependency injection
       in llkjj_ml.MLPlugin (v2.0).

    Use MLPlugin with Repository-Pattern and service injection instead.
    """

    def __init__(self) -> None:
        raise RuntimeError(
            "ResourceManager has been removed in v2.0.0. "
            "Use llkjj_ml.MLPlugin (v2.0) with dependency injection instead."
        )


# DEPRECATED: Global Resource Manager Instance
# ⚠️  This is deprecated - use llkjj_ml.MLPlugin (v2.0) instead
_resource_manager = None  # ResourceManager() disabled in v2.0.0


class _DummyResourceManager:
    """Dummy ResourceManager für Tests."""

    def cleanup(self):
        pass


# Fallback für Tests
if _resource_manager is None:
    _resource_manager = _DummyResourceManager()


# ProcessingResult removed - using import from models.processing_result


# UnifiedProcessor removed - using import from unified_processor.py


# Convenience functions for backward compatibility
def create_unified_processor(cfg: ConfigBridge | None = None) -> UnifiedProcessor:
    """Factory function to create unified processor"""
    return UnifiedProcessor(cfg)


def process_single_pdf(
    pdf_path: str | Path, cfg: ConfigBridge | None = None
) -> ProcessingResult:
    """Process a single PDF file - simplified interface"""
    processor = UnifiedProcessor(cfg)
    return processor.process_pdf(pdf_path)


if __name__ == "__main__":
    # Quick test/demo
    import argparse

    parser = argparse.ArgumentParser(description="Test unified processor")
    parser.add_argument("pdf_path", help="Path to PDF file")
    parser.add_argument("--output", "-o", help="Output JSON file path")

    args = parser.parse_args()

    # Process PDF
    config = config_instance
    proc = UnifiedProcessor(config)
    final_result = proc.process_pdf(args.pdf_path)

    # Save result
    output_file_path: Path | None = None
    if args.output:
        output_file_path = Path(args.output)

    # Save result to file if output path provided
    import json

    saved_path = None
    if output_file_path:
        output_file_path.write_text(
            json.dumps(final_result.to_dict(), indent=2, ensure_ascii=False)
        )
        saved_path = str(output_file_path)

    print("✅ Processing complete!")
    print(f"📄 PDF: {final_result.pdf_path}")
    print(f"⏱️  Time: {final_result.processing_time_ms}ms")
    print(f"🎯 Confidence: {final_result.confidence_score:.1%}")
    print(f"📊 Positions: {len(final_result.skr03_classifications)}")
    print(f"💾 Saved: {saved_path}")
