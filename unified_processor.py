#!/usr/bin/env python3
"""
LLKJJ ML Pipeline - Unified Processor (KISS Architecture)
========================================================

This module consolidates all PDF processing functionality into a single,
simple pipeline following KISS principles:

PDF → Docling OCR → Gemini Enhancement → SKR03 Classification → Results

Replaces the complex multi-service architecture with a streamlined approach
optimized for German electrical contractor invoices.

Author: LLKJJ ML Pipeline
Version: 2.0.0 (Post-Consolidation)
"""

import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import google.generativeai as genai  # type: ignore
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    EasyOcrOptions,
    PdfPipelineOptions,
    TableFormerMode,
    TableStructureOptions,
)
from docling.document_converter import DocumentConverter

from src.config import Config

logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """Unified result from PDF processing pipeline"""

    # Source information
    pdf_path: str
    processing_timestamp: str

    # Extraction results
    raw_text: str
    structured_data: dict[str, Any]

    # Classification results
    invoice_data: dict[str, Any]
    skr03_classifications: list[dict[str, Any]]

    # Performance metrics
    processing_time_ms: int
    ocr_time_ms: int
    classification_time_ms: int

    # Quality indicators
    confidence_score: float
    extraction_quality: str  # "high", "medium", "low"


class UnifiedProcessor:
    """
    Unified PDF-to-SKR03 processor combining all functionality.

    This replaces the complex service architecture with a single,
    optimized pipeline for electrical contractor invoices.
    """

    def __init__(self, config: Config | None = None):
        """Initialize unified processor with configuration"""
        self.config = config or Config()
        self._setup_logging()
        self._setup_docling()
        self._setup_gemini()
        self._setup_skr03_rules()

    def _setup_logging(self) -> None:
        """Configure logging for the processor"""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    def _setup_docling(self) -> None:
        """Setup Docling document converter with optimized settings"""

        # Optimized pipeline options for German invoices
        pipeline_options = PdfPipelineOptions(
            # OCR Configuration
            do_ocr=True,
            ocr_options=EasyOcrOptions(lang=["de"]),  # German language
            # Table Structure (critical for invoice positions)
            do_table_structure=True,
            table_structure_options=TableStructureOptions(
                do_cell_matching=True,  # Better cell recognition
                mode=TableFormerMode.ACCURATE,  # Highest accuracy
            ),
            # Performance optimizations
            generate_page_images=True,  # For debugging
            images_scale=2.0,  # Higher quality OCR
            # Hardware acceleration
            accelerator_options=AcceleratorOptions(
                device=AcceleratorDevice.AUTO,  # Auto-detect best device
                num_threads=4,  # Optimize for typical systems
            ),
        )

        self.document_converter = DocumentConverter(
            format_options={InputFormat.PDF: pipeline_options}
        )

        logger.info("Docling processor initialized with German optimization")

    def _setup_gemini(self) -> None:
        """Setup Gemini API for enhanced extraction"""
        if self.config.google_api_key:
            genai.configure(api_key=self.config.google_api_key)
            self.gemini_model = genai.GenerativeModel(self.config.gemini_model)
            logger.info(f"Gemini {self.config.gemini_model} initialized")
        else:
            self.gemini_model = None
            logger.warning("No Gemini API key - enhanced extraction disabled")

    def _setup_skr03_rules(self) -> None:
        """Setup SKR03 classification rules for electrical contractors"""

        # Core SKR03 mappings for Elektrotechnik Handwerk UG
        self.skr03_rules = {
            # Elektromaterial (most common)
            "elektromaterial": {
                "keywords": [
                    "kabel",
                    "leitung",
                    "schalter",
                    "steckdose",
                    "lampe",
                    "led",
                    "sicherung",
                    "schütz",
                    "relais",
                    "verteiler",
                    "klemme",
                    "installationsmaterial",
                    "elektroinstallation",
                ],
                "konto": "3400",  # Wareneingang 19% Vorsteuer
                "description": "Elektromaterial mit 19% Vorsteuer",
            },
            # Werkzeuge und Betriebsausstattung
            "werkzeuge": {
                "keywords": [
                    "bohrmaschine",
                    "schraubendreher",
                    "zange",
                    "multimeter",
                    "werkzeug",
                    "bohrer",
                    "bits",
                    "messgerät",
                ],
                "konto": "4985",  # Werkzeuge unter 800€
                "description": "Werkzeuge und Kleingeräte",
            },
            # Anlagegüter über GWG-Grenze
            "anlagen": {
                "keywords": [
                    "maschine",
                    "generator",
                    "prüfgerät",
                    "messplatz",
                    "anlage",
                ],
                "konto": "0200",  # Maschinen und Anlagen (bei >800€)
                "description": "Anlagegüter über GWG-Grenze",
            },
            # Büromaterial und Software
            "buero": {
                "keywords": [
                    "papier",
                    "drucker",
                    "computer",
                    "software",
                    "lizenz",
                    "büromaterial",
                    "office",
                    "druckerpapier",
                ],
                "konto": "4930",  # Bürobedarf
                "description": "Büromaterial und Software",
            },
        }

        logger.info(f"SKR03 rules loaded: {len(self.skr03_rules)} categories")

    def process_pdf(self, pdf_path: str | Path) -> ProcessingResult:
        """
        Main processing pipeline: PDF → OCR → Enhancement → Classification

        Args:
            pdf_path: Path to PDF file to process

        Returns:
            ProcessingResult with all extracted and classified data
        """

        start_time = time.time()
        pdf_path = Path(pdf_path)

        logger.info(f"Starting unified processing: {pdf_path.name}")

        # Step 1: Docling OCR Extraction
        ocr_start = time.time()
        raw_extraction = self._extract_with_docling(pdf_path)
        ocr_time = int((time.time() - ocr_start) * 1000)

        # Step 2: Gemini Enhancement (if available)
        if self.gemini_model:
            enhanced_data = self._enhance_with_gemini(raw_extraction)
        else:
            enhanced_data = self._extract_structured_data(raw_extraction)

        # Step 3: SKR03 Classification
        classification_start = time.time()
        classifications = self._classify_skr03(enhanced_data)
        classification_time = int((time.time() - classification_start) * 1000)

        # Step 4: Quality Assessment
        confidence_score = self._calculate_confidence(enhanced_data, classifications)
        quality = self._assess_quality(confidence_score)

        total_time = int((time.time() - start_time) * 1000)

        result = ProcessingResult(
            pdf_path=str(pdf_path),
            processing_timestamp=datetime.now().isoformat(),
            raw_text=raw_extraction["raw_text"],
            structured_data=enhanced_data,
            invoice_data=enhanced_data.get("invoice_header", {}),
            skr03_classifications=classifications,
            processing_time_ms=total_time,
            ocr_time_ms=ocr_time,
            classification_time_ms=classification_time,
            confidence_score=confidence_score,
            extraction_quality=quality,
        )

        logger.info(
            f"Processing complete: {pdf_path.name} "
            f"({total_time}ms, {len(classifications)} positions, "
            f"{confidence_score:.1%} confidence)"
        )

        return result

    def _extract_with_docling(self, pdf_path: Path) -> dict[str, Any]:
        """Extract data using Docling with optimized settings"""

        try:
            # Convert PDF with optimized pipeline
            conv_result = self.document_converter.convert(str(pdf_path))

            # Extract structured data
            raw_text = conv_result.document.export_to_markdown()

            # Extract tables (invoice positions)
            tables = []
            for table in conv_result.document.tables:
                # Convert table to structured format
                table_data = self._parse_table_data(table)
                if table_data:
                    tables.append(table_data)

            result = {
                "raw_text": raw_text,
                "tables": tables,
                "page_count": len(conv_result.document.pages),
                "extraction_method": "docling_optimized",
            }

            logger.debug(f"Docling extracted {len(tables)} tables from {pdf_path.name}")
            return result

        except Exception as e:
            logger.error(f"Docling extraction failed for {pdf_path}: {e}")
            # Fallback to basic text extraction
            return {
                "raw_text": f"Extraction failed: {str(e)}",
                "tables": [],
                "page_count": 0,
                "extraction_method": "fallback",
            }

    def _parse_table_data(self, table: Any) -> dict[str, Any] | None:
        """Parse Docling table into structured invoice positions"""

        try:
            # Extract table structure
            rows = []
            if hasattr(table, "data") and table.data:
                for row in table.data:
                    if hasattr(row, "cells"):
                        row_data = [
                            cell.text.strip() if hasattr(cell, "text") else str(cell)
                            for cell in row.cells
                        ]
                        if any(row_data):  # Skip empty rows
                            rows.append(row_data)

            if not rows:
                return None

            # Detect header and data rows
            header_row = rows[0] if rows else []
            data_rows = rows[1:] if len(rows) > 1 else []

            return {
                "headers": header_row,
                "rows": data_rows,
                "row_count": len(data_rows),
            }

        except Exception as e:
            logger.warning(f"Table parsing failed: {e}")
            return None

    def _enhance_with_gemini(self, raw_data: dict[str, Any]) -> dict[str, Any]:
        """Enhance extraction using Gemini AI"""

        prompt = f"""
Analysiere diese deutsche Elektrotechnik-Rechnung und extrahiere strukturierte Daten:

TEXT:
{raw_data['raw_text'][:3000]}

Extrahiere:
1. Rechnungsdaten (Nummer, Datum, Betrag)
2. Lieferant-Information
3. Positionen mit Beschreibung, Menge, Preis
4. Kategorisierung für SKR03 (Elektromaterial, Werkzeuge, etc.)

Format: JSON
"""

        try:
            response = self.gemini_model.generate_content(prompt)

            # Parse Gemini response
            enhanced_data = json.loads(response.text)
            enhanced_data.update(raw_data)  # Merge with raw data
            enhanced_data["enhancement_method"] = "gemini"

            logger.debug("Gemini enhancement successful")
            return enhanced_data

        except Exception as e:
            logger.warning(f"Gemini enhancement failed: {e}")
            return self._extract_structured_data(raw_data)

    def _extract_structured_data(self, raw_data: dict[str, Any]) -> dict[str, Any]:
        """Fallback structured extraction using patterns"""

        text = raw_data.get("raw_text", "")

        # Basic pattern extraction for German invoices
        structured = {
            "invoice_header": self._extract_invoice_header(text),
            "line_items": self._extract_line_items(raw_data.get("tables", [])),
            "totals": self._extract_totals(text),
            "enhancement_method": "pattern_based",
        }

        structured.update(raw_data)
        return structured

    def _extract_invoice_header(self, text: str) -> dict[str, Any]:
        """Extract invoice header information using patterns"""

        import re

        # German invoice patterns
        patterns = {
            "invoice_number": r"(?:Rechn(?:ungs)?[-.\s]*Nr\.?|Rechnungs-Nr\.)\s*:?\s*(\d+)",
            "date": r"(?:Datum|Rechnungsdatum|vom)\s*:?\s*(\d{1,2}[.\s/]\d{1,2}[.\s/]\d{2,4})",
            "customer_number": r"(?:Kunden[-.\s]*Nr\.?)\s*:?\s*(\d+)",
            "total_amount": r"(?:Gesamt|Gesamtbetrag|Endbetrag)\s*:?\s*(\d+[.,]\d{2})\s*€?",
        }

        header = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                header[key] = match.group(1).strip()

        return header

    def _extract_line_items(self, tables: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Extract line items from table data"""

        items = []

        for table in tables:
            headers = table.get("headers", [])
            rows = table.get("rows", [])

            # Map common German invoice table headers
            header_map = self._map_table_headers(headers)

            for row in rows:
                if len(row) >= len(headers):
                    item = {}
                    for i, header in enumerate(headers):
                        if i < len(row) and header_map.get(header):
                            item[header_map[header]] = row[i]

                    if item.get("description") or item.get("artikel"):
                        items.append(item)

        return items

    def _map_table_headers(self, headers: list[str]) -> dict[str, str]:
        """Map German table headers to standard fields"""

        mapping = {}
        for header in headers:
            header_lower = header.lower().strip()

            if any(
                term in header_lower
                for term in ["beschreibung", "artikel", "bezeichnung"]
            ):
                mapping[header] = "description"
            elif any(term in header_lower for term in ["menge", "anz", "stück"]):
                mapping[header] = "quantity"
            elif any(term in header_lower for term in ["preis", "ep", "einzelpreis"]):
                mapping[header] = "unit_price"
            elif any(term in header_lower for term in ["gesamt", "summe", "betrag"]):
                mapping[header] = "total_price"

        return mapping

    def _extract_totals(self, text: str) -> dict[str, Any]:
        """Extract invoice totals"""

        import re

        totals = {}

        # German total patterns
        patterns = {
            "net_total": r"(?:Netto|Zwischensumme)\s*:?\s*(\d+[.,]\d{2})",
            "vat_amount": r"(?:MwSt|USt|19%)\s*:?\s*(\d+[.,]\d{2})",
            "gross_total": r"(?:Brutto|Gesamt|Endbetrag)\s*:?\s*(\d+[.,]\d{2})",
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Convert German decimal comma to float
                amount_str = match.group(1).replace(",", ".")
                try:
                    totals[key] = float(amount_str)
                except ValueError:
                    totals[key] = amount_str

        return totals

    def _classify_skr03(self, structured_data: dict[str, Any]) -> list[dict[str, Any]]:
        """Classify line items according to SKR03 rules"""

        classifications = []
        line_items = structured_data.get("line_items", [])

        for i, item in enumerate(line_items):
            description = item.get("description", "").lower()

            # Find best matching SKR03 category
            best_match = self._find_best_skr03_match(description)

            classification = {
                "position": i + 1,
                "description": item.get("description", ""),
                "skr03_konto": best_match["konto"],
                "category": best_match["category"],
                "confidence": best_match["confidence"],
                "reasoning": best_match["reasoning"],
                "amount": item.get("total_price", "0"),
                "quantity": item.get("quantity", "1"),
            }

            classifications.append(classification)

        return classifications

    def _find_best_skr03_match(self, description: str) -> dict[str, Any]:
        """Find best SKR03 category match for description"""

        best_score = 0
        best_category = "elektromaterial"  # Default
        matched_keywords = []

        for category, rules in self.skr03_rules.items():
            score = 0
            keywords_found = []

            for keyword in rules["keywords"]:
                if keyword.lower() in description:
                    score += 1
                    keywords_found.append(keyword)

            if score > best_score:
                best_score = score
                best_category = category
                matched_keywords = keywords_found

        category_info = self.skr03_rules[best_category]

        # Calculate confidence based on keyword matches
        confidence = min(0.9, 0.3 + (best_score * 0.2))

        return {
            "category": best_category,
            "konto": category_info["konto"],
            "confidence": confidence,
            "reasoning": (
                f"Matched keywords: {', '.join(matched_keywords)}"
                if matched_keywords
                else "Default classification"
            ),
            "matched_keywords": matched_keywords,
        }

    def _calculate_confidence(
        self, structured_data: dict[str, Any], classifications: list[dict[str, Any]]
    ) -> float:
        """Calculate overall processing confidence score"""

        scores = []

        # Invoice header completeness
        header = structured_data.get("invoice_header", {})
        header_score = len([v for v in header.values() if v]) / 4  # 4 key fields
        scores.append(header_score)

        # Classification confidence
        if classifications:
            avg_classification_confidence = sum(
                c.get("confidence", 0) for c in classifications
            ) / len(classifications)
            scores.append(avg_classification_confidence)

        # Table extraction success
        line_items = structured_data.get("line_items", [])
        extraction_score = min(1.0, len(line_items) / 5)  # Assume 5 items is good
        scores.append(extraction_score)

        return sum(scores) / len(scores) if scores else 0.5

    def _assess_quality(self, confidence_score: float) -> str:
        """Assess extraction quality based on confidence"""

        if confidence_score >= 0.8:
            return "high"
        elif confidence_score >= 0.6:
            return "medium"
        else:
            return "low"

    def save_result(
        self, result: ProcessingResult, output_path: Path | None = None
    ) -> Path:
        """Save processing result to JSON file"""

        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pdf_name = Path(result.pdf_path).stem
            output_path = (
                self.config.data_processed_path / f"{pdf_name}_{timestamp}.json"
            )

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save as JSON
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(asdict(result), f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved to: {output_path}")
        return output_path


# Convenience functions for backward compatibility
def create_unified_processor(config: Config | None = None) -> UnifiedProcessor:
    """Factory function to create unified processor"""
    return UnifiedProcessor(config)


def process_single_pdf(
    pdf_path: str | Path, config: Config | None = None
) -> ProcessingResult:
    """Process a single PDF file - simplified interface"""
    processor = UnifiedProcessor(config)
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
    processor = UnifiedProcessor(config)
    result = processor.process_pdf(args.pdf_path)

    # Save result
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = None

    saved_path = processor.save_result(result, output_path)

    print("✅ Processing complete!")
    print(f"📄 PDF: {result.pdf_path}")
    print(f"⏱️  Time: {result.processing_time_ms}ms")
    print(f"🎯 Confidence: {result.confidence_score:.1%}")
    print(f"📊 Positions: {len(result.skr03_classifications)}")
    print(f"💾 Saved: {saved_path}")
