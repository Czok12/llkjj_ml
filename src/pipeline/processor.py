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
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

try:
    import google.genai as genai
except ImportError:
    genai = None  # type: ignore[assignment]

from src.settings_bridge import Config

# dual_pipeline.py removed in cleanup - functionality integrated into UnifiedProcessor

logger = logging.getLogger(__name__)


class ResourceManager:
    """
    REMOVED: Legacy ResourceManager has been removed in v2.0.0 final cleanup

    ⚠️ This Singleton class has been completely replaced by dependency injection
       in llkjj_ml.MLPlugin (v2.0).

    Use MLPlugin with Repository-Pattern and service injection instead.
    """

    def __init__(self):
        raise RuntimeError(
            "ResourceManager has been removed in v2.0.0. "
            "Use llkjj_ml.MLPlugin (v2.0) with dependency injection instead."
        )


# DEPRECATED: Global Resource Manager Instance
# ⚠️  This is deprecated - use llkjj_ml.MLPlugin (v2.0) instead
_resource_manager = None  # ResourceManager() disabled in v2.0.0


class ProcessingResult(BaseModel):
    """
    **PUBLIC DATA CONTRACT** - LLKJJ ML Plugin Output Schema
    ========================================================

    This is the guaranteed public interface returned by the ML Plugin.
    External systems can rely on this schema remaining stable across versions.

    **BLACKBOX GUARANTEE:**
    All fields in this model are guaranteed to be populated by the plugin.
    No external system knowledge is required to interpret these results.

    **USAGE EXAMPLE:**
    ```python
    plugin = MLPlugin()
    result = plugin.process_pdf("invoice.pdf")

    # Access structured results
    for item in result.skr03_classifications:
        print(f"Item: {item['description']} -> SKR03: {item['account']}")

    # Check quality
    if result.extraction_quality == "high":
        print(f"High confidence: {result.confidence_score:.1%}")
    ```

    **DATA CONTRACT VERSION:** 3.0.0
    """

    # === SOURCE INFORMATION ===
    pdf_path: str = Field(
        ...,
        description="Absoluter Pfad zur verarbeiteten PDF-Datei",
        examples=["/path/to/sonepar_invoice.pdf"],
    )
    processing_timestamp: str = Field(
        ...,
        description="ISO-Zeitstempel der Verarbeitung (UTC)",
        examples=["2025-08-17T14:30:25.123456Z"],
    )

    # === EXTRACTION RESULTS ===
    raw_text: str = Field(
        ...,
        description="Kompletter extrahierter Text aus PDF (OCR + native)",
        examples=["Sonepar Deutschland...\nRechnung Nr. 123..."],
    )
    structured_data: dict[str, Any] = Field(
        default_factory=dict,
        description="Strukturierte Daten von Docling (Tabellen, Layout, Metadaten)",
        examples=[{"tables": [{"rows": 10, "cols": 7}], "document_type": "invoice"}],
    )

    # === CLASSIFICATION RESULTS ===
    invoice_data: dict[str, Any] = Field(
        default_factory=dict,
        description="Extrahierte Rechnungsheader (Lieferant, Datum, Beträge)",
        examples=[{"supplier": "Sonepar", "total": 328.82, "invoice_number": "123"}],
    )
    skr03_classifications: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Liste aller Positionen mit SKR03-Kontierungen",
        examples=[
            [
                {
                    "position": 1,
                    "article_number": "028203",
                    "description": "GIRA Adapterrahmen",
                    "quantity": 10,
                    "skr03_account": "3400",
                    "skr03_category": "wareneingang_elektro_allgemein",
                    "confidence": 0.95,
                }
            ]
        ],
    )

    # === PERFORMANCE METRICS ===
    processing_time_ms: int = Field(
        ...,
        ge=0,
        description="Gesamtverarbeitungszeit in Millisekunden",
        examples=[34567],
    )
    ocr_time_ms: int = Field(
        ...,
        ge=0,
        description="OCR-Verarbeitungszeit in Millisekunden (Docling)",
        examples=[10150],
    )
    classification_time_ms: int = Field(
        ...,
        ge=0,
        description="Klassifizierungszeit in Millisekunden (SKR03 + RAG)",
        examples=[1443],
    )

    # === QUALITY INDICATORS ===
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Gesamtkonfidenz-Score (0.0=niedrig, 1.0=perfekt)",
        examples=[0.633],
    )
    extraction_quality: Literal["high", "medium", "low"] = Field(
        ...,
        description="Qualitätskategorisierung für einfache Bewertung",
        examples=["medium"],
    )

    @field_validator("processing_timestamp")
    @classmethod
    def validate_timestamp_format(cls, v: str) -> str:
        """Validiere ISO-Zeitstempel-Format"""
        try:
            datetime.fromisoformat(v.replace("Z", "+00:00"))
            return v
        except ValueError as e:
            raise ValueError(f"Ungültiges Zeitstempel-Format: {v}") from e

    @field_validator("pdf_path")
    @classmethod
    def validate_pdf_path(cls, v: str) -> str:
        """Validiere PDF-Dateipfad"""
        if not v.lower().endswith(".pdf"):
            raise ValueError(f"Pfad muss eine PDF-Datei sein: {v}")
        return v

    def to_dict(self) -> dict[str, Any]:
        """
        Konvertiere zu Dictionary für Rückwärtskompatibilität.

        Returns:
            Dict representation of all results for JSON export
        """
        # Pydantic returns dict[str, Any]
        return self.model_dump()

    def get_summary(self) -> str:
        """
        Erstelle eine kompakte Zusammenfassung der Verarbeitungsergebnisse.

        Returns:
            String mit wichtigsten Kennzahlen für Logging/Display
        """
        return (
            f"PDF: {Path(self.pdf_path).name} | "
            f"Qualität: {self.extraction_quality} | "
            f"Konfidenz: {self.confidence_score:.2f} | "
            f"Zeit: {self.processing_time_ms}ms | "
            f"Positionen: {len(self.skr03_classifications)}"
        )

    def get_skr03_summary(self) -> dict[str, int]:
        """
        Erstelle Zusammenfassung der SKR03-Klassifizierungen.

        Returns:
            Dict mit SKR03-Konten und Anzahl zugeordneter Positionen
        """
        account_counts: dict[str, int] = {}
        for item in self.skr03_classifications:
            account = item.get("skr03_account", "unknown")
            account_counts[account] = account_counts.get(account, 0) + 1
        return account_counts

    @classmethod
    def get_schema_documentation(cls) -> str:
        """
        **PUBLIC API DOCUMENTATION**

        Returns complete schema documentation for external systems.
        This documents the guaranteed data contract for ProcessingResult.

        Returns:
            Formatted string with complete field documentation
        """
        return """
LLKJJ ML Plugin - ProcessingResult Schema v3.0.0
===============================================

**GUARANTEED OUTPUT FIELDS:**

📂 SOURCE INFORMATION:
  • pdf_path (str): Absolute path to processed PDF file
  • processing_timestamp (str): ISO timestamp in UTC format

📊 EXTRACTION RESULTS:
  • raw_text (str): Complete extracted text (OCR + native)
  • structured_data (dict): Docling results (tables, layout, metadata)

🏷️  CLASSIFICATION RESULTS:
  • invoice_data (dict): Invoice header (supplier, totals, dates)
  • skr03_classifications (list): List of items with German accounting codes
    Each item contains:
    - position (int): Line item number
    - article_number (str): Product/article code
    - description (str): Item description
    - quantity (int): Quantity ordered
    - skr03_account (str): German accounting code (e.g., "3400")
    - skr03_category (str): Account category
    - confidence (float): Classification confidence (0.0-1.0)

⏱️  PERFORMANCE METRICS:
  • processing_time_ms (int): Total processing time
  • ocr_time_ms (int): OCR extraction time
  • classification_time_ms (int): SKR03 classification time

✅ QUALITY INDICATORS:
  • confidence_score (float): Overall confidence (0.0-1.0)
  • extraction_quality (str): "high" | "medium" | "low"

**USAGE:**
```python
from llkjj_ml_plugin import MLPlugin

plugin = MLPlugin()
result = plugin.process_pdf("invoice.pdf")

# Access results
print(f"Found {len(result.skr03_classifications)} items")
for item in result.skr03_classifications:
    print(f"{item['description']} -> {item['skr03_account']}")
```
        """


class UnifiedProcessor:
    """
    REMOVED: Legacy UnifiedProcessor has been removed in v2.0.0 final cleanup

    ⚠️ This class has been completely replaced by llkjj_ml.MLPlugin (v2.0).

    Use MLPlugin with:
    - Repository-Pattern for data persistence
    - Dependency injection for services
    - Stateless design instead of singleton patterns
    """

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "UnifiedProcessor has been removed in v2.0.0. "
            "Use llkjj_ml.MLPlugin (v2.0) with Repository-Pattern instead."
        )


# Convenience functions for backward compatibility
def create_unified_processor(cfg: Config | None = None) -> UnifiedProcessor:
    """Factory function to create unified processor"""
    return UnifiedProcessor(cfg)


def process_single_pdf(
    pdf_path: str | Path, cfg: Config | None = None
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
    config = Config()
    proc = UnifiedProcessor(config)
    final_result = proc.process_pdf(args.pdf_path)

    # Save result
    output_file_path: Path | None = None
    if args.output:
        output_file_path = Path(args.output)

    saved_path = proc.save_result(final_result, output_file_path)

    print("✅ Processing complete!")
    print(f"📄 PDF: {final_result.pdf_path}")
    print(f"⏱️  Time: {final_result.processing_time_ms}ms")
    print(f"🎯 Confidence: {final_result.confidence_score:.1%}")
    print(f"📊 Positions: {len(final_result.skr03_classifications)}")
    print(f"💾 Saved: {saved_path}")
