#!/usr/bin/env python3
"""
LLKJJ ML Pipeline - ProcessingResult Data Model
==============================================

Einheitliche Datenstruktur für alle Pipeline-Ergebnisse.
Garantiert konsistente API zwischen Gemini-First und Docling-Alternative.

Autor: LLKJJ ML Pipeline Team
Version: 4.0.0 (Hybrid Implementation)
Datum: 18. August 2025
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class ProcessingResult(BaseModel):
    """
    **EINHEITLICHE DATENSTRUKTUR** - LLKJJ ML Plugin Output Schema
    ============================================================

    Garantierte öffentliche Schnittstelle für alle Pipeline-Ergebnisse.
    Sowohl Gemini-First als auch Docling-Alternative verwenden diese Struktur.

    **USAGE EXAMPLE:**
    ```python
    processor = GeminiDirectProcessor()
    result = processor.process_pdf_gemini_first("invoice.pdf")

    # Einheitlicher Zugriff auf Ergebnisse
    for item in result.skr03_classifications:
        print(f"Item: {item['description']} -> SKR03: {item['skr03_account']}")

    # Qualitätsprüfung
    if result.extraction_quality == "high":
        print(f"High confidence: {result.confidence_score:.1%}")
    ```

    **DATA CONTRACT VERSION:** 4.0.0 (Hybrid)
    """

    # Allow awaiting a ProcessingResult in async tests (returns itself)
    def __await__(self):
        async def _identity() -> "ProcessingResult":
            return self

        return _identity().__await__()

    # === SOURCE INFORMATION ===
    pdf_path: str = Field(
        ...,
        description="Absoluter Pfad zur verarbeiteten PDF-Datei",
        examples=["/path/to/sonepar_invoice.pdf"],
    )
    processing_timestamp: str = Field(
        ...,
        description="ISO-Zeitstempel der Verarbeitung (UTC)",
        examples=["2025-08-18T14:30:25.123456Z"],
    )
    processing_method: Literal[
        "gemini_first",
        "docling_alternative",
        "gemini",
        "spacy_rag",
        "hybrid",
        "auto",
        "gemini_fallback",
    ] = Field(
        ...,
        description="Verwendete Verarbeitungsmethode",
        examples=["gemini_first"],
    )

    # === EXTRACTION RESULTS ===
    raw_text: str = Field(
        default="",
        description="Kompletter extrahierter Text aus PDF (OCR + native)",
        examples=["Sonepar Deutschland...\nRechnung Nr. 123..."],
    )
    structured_data: dict[str, Any] = Field(
        default_factory=dict,
        description="Strukturierte Daten (Gemini-JSON oder Docling-Tabellen)",
        examples=[{"tables": [{"rows": 10, "cols": 7}], "document_type": "invoice"}],
    )

    # === CLASSIFICATION RESULTS ===
    invoice_header: dict[str, Any] = Field(
        default_factory=dict,
        description="Extrahierte Rechnungsheader (Lieferant, Datum, Beträge)",
        examples=[{"supplier": "Sonepar", "total": 328.82, "invoice_number": "123"}],
    )
    line_items: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Extrahierte Rechnungspositionen (raw)",
        examples=[
            [
                {
                    "position": 1,
                    "description": "GIRA Rahmen",
                    "quantity": 10,
                    "amount": 34.50,
                }
            ]
        ],
    )
    skr03_classifications: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Liste aller Positionen mit SKR03-Kontierungen",
        examples=[
            [
                {
                    "position": 1,
                    "confidence": 0.95,
                    "skr03_account": "3400",
                    "skr03_category": "wareneingang_elektro_allgemein",
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
    gemini_time_ms: int = Field(
        default=0,
        ge=0,
        description="Gemini-AI-Verarbeitungszeit in Millisekunden",
        examples=[5200],
    )
    ocr_time_ms: int = Field(
        default=0,
        ge=0,
        description="OCR-Verarbeitungszeit in Millisekunden (Docling)",
        examples=[10150],
    )
    classification_time_ms: int = Field(
        default=0,
        ge=0,
        description="SKR03-Klassifizierungszeit in Millisekunden",
        examples=[1443],
    )

    # === MODEL INFORMATION ===
    gemini_model: str = Field(
        default="gemini-2.5-flash",
        description="Verwendetes Gemini-Modell für die Verarbeitung",
        examples=["gemini-2.5-flash", "gemini-1.5-flash"],
    )

    # === QUALITY INDICATORS ===
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Gesamtkonfidenz-Score (0.0=niedrig, 1.0=perfekt)",
        examples=[0.633],
    )
    extraction_quality: Literal["high", "medium", "low", "poor"] = Field(
        ...,
        description="Qualitätskategorisierung für einfache Bewertung",
        examples=["medium"],
    )

    # === TRAINING DATA ===
    training_annotations: list[dict[str, Any]] = Field(
        default_factory=list,
        description="spaCy-Training-Annotationen für kontinuierliches Lernen",
        examples=[
            [
                {
                    "text": "Sonepar Deutschland",
                    "label": "HÄNDLER",
                    "start_char": 0,
                    "end_char": 19,
                }
            ]
        ],
    )

    # === STATISTICS ===
    extracted_positions: int = Field(
        default=0,
        ge=0,
        description="Anzahl extrahierter Rechnungspositionen",
        examples=[12],
    )
    classified_positions: int = Field(
        default=0,
        ge=0,
        description="Anzahl SKR03-klassifizierter Positionen",
        examples=[10],
    )

    # === ERROR HANDLING ===
    errors: list[str] = Field(
        default_factory=list,
        description="List of error messages encountered during processing",
        examples=[["Memory limit exceeded", "Timeout during classification"]],
    )

    @property
    def success(self) -> bool:
        """
        Determine if processing was successful based on quality indicators.

        Returns:
            True if processing was successful (no errors and good quality)
        """
        return (
            len(self.errors) == 0
            and self.confidence_score >= 0.5
            and self.extraction_quality in ["high", "medium"]
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
        # Pydantic returns dict[str, Any] (kein cast nötig)
        return self.model_dump()

    def get_summary(self) -> str:
        """
        Erstelle eine kompakte Zusammenfassung der Verarbeitungsergebnisse.

        Returns:
            String mit wichtigsten Kennzahlen für Logging/Display
        """
        return (
            f"PDF: {Path(self.pdf_path).name} | "
            f"Methode: {self.processing_method} | "
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

    def get_performance_summary(self) -> dict[str, Any]:
        """
        Erstelle Performance-Zusammenfassung mit Timing-Breakdown.

        Returns:
            Dict mit detaillierten Performance-Metriken
        """
        total_time = self.processing_time_ms
        return {
            "total_time_ms": total_time,
            "gemini_time_ms": self.gemini_time_ms,
            "ocr_time_ms": self.ocr_time_ms,
            "classification_time_ms": self.classification_time_ms,
            "gemini_percentage": (
                (self.gemini_time_ms / total_time * 100) if total_time > 0 else 0
            ),
            "ocr_percentage": (
                (self.ocr_time_ms / total_time * 100) if total_time > 0 else 0
            ),
            "classification_percentage": (
                (self.classification_time_ms / total_time * 100)
                if total_time > 0
                else 0
            ),
            "positions_per_second": (
                (len(self.skr03_classifications) / (total_time / 1000))
                if total_time > 0
                else 0
            ),
        }

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
LLKJJ ML Plugin - ProcessingResult Schema v4.0.0 (Hybrid)
=========================================================

**GUARANTEED OUTPUT FIELDS:**

📂 SOURCE INFORMATION:
  • pdf_path (str): Absolute path to processed PDF file
  • processing_timestamp (str): ISO timestamp in UTC format
  • processing_method (str): "gemini_first" | "docling_alternative"

📊 EXTRACTION RESULTS:
  • raw_text (str): Complete extracted text (OCR + native)
  • structured_data (dict): Method-specific results (Gemini JSON or Docling tables)

🏷️  CLASSIFICATION RESULTS:
  • invoice_header (dict): Invoice header (supplier, totals, dates)
  • line_items (list): Raw extracted line items
  • skr03_classifications (list): List of items with German accounting codes
    Each item contains:
    - position (int): Line item number
    - description (str): Item description
    - quantity (int): Quantity ordered
    - skr03_account (str): German accounting code (e.g., "3400")
    - skr03_category (str): Account category
    - confidence (float): Classification confidence (0.0-1.0)

⏱️  PERFORMANCE METRICS:
  • processing_time_ms (int): Total processing time
  • gemini_time_ms (int): Gemini AI processing time
  • ocr_time_ms (int): OCR extraction time (Docling)
  • classification_time_ms (int): SKR03 classification time

✅ QUALITY INDICATORS:
  • confidence_score (float): Overall confidence (0.0-1.0)
  • extraction_quality (str): "high" | "medium" | "low" | "poor"

🧠 TRAINING DATA:
  • training_annotations (list): spaCy NER annotations for continuous learning

📈 STATISTICS:
  • extracted_positions (int): Number of extracted line items
  • classified_positions (int): Number of SKR03-classified items

**USAGE:**
```python
from ...llkjj_ml_plugin_v2 import GeminiDirectProcessor

processor = GeminiDirectProcessor()
result = processor.process_pdf_gemini_first("invoice.pdf")

# Access results
print(f"Found {len(result.skr03_classifications)} items")
for item in result.skr03_classifications:
    print(f"{item['description']} -> {item['skr03_account']}")
```

    """

    @classmethod
    def from_gemini_analysis(
        cls,
        pdf_path: str,
        gemini_result: dict[str, Any],
        skr03_classifications: list[dict[str, Any]],
        confidence_score: float,
        quality_level: str,
        training_annotations: list[dict[str, Any]],
        processing_time_ms: int,
        gemini_time_ms: int,
        classification_time_ms: int,
        gemini_model: str = "gemini-2.5-flash",
    ) -> "ProcessingResult":
        """
        Factory method für Gemini-First Pipeline Results.

        Args:
            pdf_path: Pfad zur PDF-Datei
            gemini_result: Structured result from Gemini analysis
            skr03_classifications: Enhanced SKR03 classifications
            confidence_score: Overall confidence score (0.0-1.0)
            quality_level: Quality assessment ("high", "medium", "low", "poor")
            training_annotations: spaCy training annotations as list
            processing_time_ms: Total processing time
            gemini_time_ms: Gemini API processing time
            classification_time_ms: Classification processing time
            gemini_model: Used Gemini model version

        Returns:
            ProcessingResult instance
        """
        # Extract structured data from Gemini result
        invoice_header = gemini_result.get("invoice_header", {})
        line_items = gemini_result.get("line_items", [])

        # Ensure quality_level is valid
        quality_options: dict[str, Literal["high", "medium", "low", "poor"]] = {
            "high": "high",
            "medium": "medium",
            "low": "low",
            "poor": "poor",
        }
        validated_quality = quality_options.get(quality_level, "medium")

        return cls(
            # Source Information
            pdf_path=pdf_path,
            processing_method="gemini_first",
            processing_timestamp=datetime.now().isoformat(),
            # Extraction Results
            raw_text=gemini_result.get("raw_text", ""),
            structured_data=gemini_result,
            invoice_header=invoice_header,
            line_items=line_items,
            skr03_classifications=skr03_classifications,
            # Performance Metrics
            processing_time_ms=processing_time_ms,
            gemini_time_ms=gemini_time_ms,
            ocr_time_ms=0,  # No OCR in Gemini-First
            classification_time_ms=classification_time_ms,
            # Model Information
            gemini_model=gemini_model,
            # Quality Indicators
            confidence_score=confidence_score,
            extraction_quality=validated_quality,
            # Training Data
            training_annotations=training_annotations,
            # Statistics
            extracted_positions=len(line_items),
            classified_positions=len(skr03_classifications),
        )

    @classmethod
    def create_error(
        cls,
        pdf_path: str,
        error_message: str,
        processing_method: str = "unknown",
        processing_time_ms: int = 0,
    ) -> "ProcessingResult":
        """
        Factory method to create error ProcessingResult.

        Args:
            pdf_path: Path to the PDF file that failed
            error_message: Error message describing the failure
            processing_method: Method that was attempted
            processing_time_ms: Time spent before failure

        Returns:
            ProcessingResult instance with error state
        """
        return cls(
            # Source Information
            pdf_path=pdf_path,
            processing_method=processing_method,  # type: ignore
            processing_timestamp=datetime.now().isoformat(),
            # Extraction Results (empty)
            raw_text="",
            structured_data={},
            invoice_header={},
            line_items=[],
            skr03_classifications=[],
            # Performance Metrics
            processing_time_ms=processing_time_ms,
            gemini_time_ms=0,
            ocr_time_ms=0,
            classification_time_ms=0,
            # Quality Indicators (low quality for errors)
            confidence_score=0.0,
            extraction_quality="poor",
            # Training Data (empty)
            training_annotations=[],
            # Statistics (empty)
            extracted_positions=0,
            classified_positions=0,
            # Errors
            errors=[error_message],
        )


# Convenience functions für Backward Compatibility
def create_processing_result(
    pdf_path: str,
    processing_method: Literal["gemini_first", "docling_alternative"],
    **kwargs: Any,
) -> ProcessingResult:
    """
    Factory function für ProcessingResult mit Default-Werten.

    Args:
        pdf_path: Pfad zur PDF-Datei
        processing_method: Verarbeitungsmethode
        **kwargs: Zusätzliche Felder

    Returns:
        ProcessingResult mit Default-Werten
    """
    defaults: dict[str, Any] = {
        "processing_timestamp": datetime.now().isoformat(),
        "processing_time_ms": 0,
        "confidence_score": 0.0,
        "extraction_quality": "low",
    }
    defaults.update(kwargs)

    return ProcessingResult(
        pdf_path=pdf_path, processing_method=processing_method, **defaults
    )
