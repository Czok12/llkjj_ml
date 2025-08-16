#!/usr/bin/env python3
"""
LLKJJ ML Pipeline - Data Extraction Module
==========================================

This module handles all data extraction functionality:
- PDF text extraction with Docling
- Table parsing and structure recognition
- Gemini AI enhancement
- Pattern-based fallback extraction
- Invoice header and line item extraction

Extracted from unified processor following KISS modularization principles.

Author: LLKJJ ML Pipeline
Version: 1.0.0 (Post-Modularization)
"""

import json
import logging
import re
from pathlib import Path
from typing import Any

from docling.document_converter import DocumentConverter

logger = logging.getLogger(__name__)


class DataExtractor:
    """
    Handles all data extraction functionality for PDF invoice processing.

    Responsibilities:
    - PDF text extraction with Docling
    - Table parsing and structure recognition
    - Gemini AI enhancement (when available)
    - Pattern-based fallback extraction
    - Invoice header and line item extraction
    """

    def __init__(self, gemini_model: Any = None) -> None:
        """Initialize data extractor with optional Gemini model"""
        self.gemini_model = gemini_model
        self.document_converter = DocumentConverter()
        logger.info("DataExtractor initialized with German optimization")

    def process_pdf(self, pdf_path: Path) -> dict[str, Any]:
        """
        Hauptmethode für PDF-Extraktion - Öffentliche API.

        Args:
            pdf_path: Pfad zur PDF-Datei

        Returns:
            Dictionary mit strukturierten Extraktionsergebnissen
        """
        try:
            logger.info("🔍 Starte PDF-Extraktion: %s", pdf_path.name)

            # Phase 1: Docling-Extraktion
            raw_result = self.extract_with_docling(pdf_path)

            # Phase 2: Strukturierte Datenextraktion
            structured_data = self.extract_structured_data(raw_result)

            # Phase 3: Invoice-Header extrahieren
            invoice_data = self.extract_invoice_header(raw_result.get("text", ""))

            # Phase 4: Line Items extrahieren
            line_items = self.extract_line_items(
                raw_result.get("tables", []), raw_result.get("text", "")
            )

            # Gemini-Enhancement (falls verfügbar)
            if self.gemini_model:
                try:
                    enhanced_data = self.enhance_with_gemini(raw_result)
                    if enhanced_data:
                        # Merge enhanced data into line_items
                        logger.info("✅ Gemini-Enhancement erfolgreich")
                except Exception as e:
                    logger.warning("⚠️ Gemini-Enhancement fehlgeschlagen: %s", e)

            result = {
                "raw_text": raw_result.get("text", ""),
                "structured_data": structured_data,
                "invoice_data": invoice_data,
                "line_items": line_items,
                "tables": raw_result.get("tables", []),
                "metadata": raw_result.get("metadata", {}),
            }

            logger.info(
                "✅ PDF-Extraktion abgeschlossen: %d Positionen gefunden",
                len(line_items),
            )

            return result

        except Exception as e:
            logger.error("❌ Fehler bei PDF-Extraktion: %s", e)
            raise

    def extract_with_docling(self, pdf_path: Path) -> dict[str, Any]:
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
                table_data = self.parse_table_data(table)
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

    def parse_table_data(self, table: Any) -> dict[str, Any] | None:
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

    def enhance_with_gemini(self, raw_data: dict[str, Any]) -> dict[str, Any]:
        """Enhance extraction using Gemini AI with German language optimization"""
        prompt = f"""
Du bist ein deutschsprachiger Experte für Elektrotechnik-Rechnungen und SKR03-Buchhaltung.

AUFGABE: Analysiere diese deutsche Elektrotechnik-Rechnung und extrahiere strukturierte Daten.
SPRACHE: Alle Antworten bitte auf Deutsch.
KONTEXT: Deutsche Geschäftsdokumente, deutsches Rechnungswesen, deutsche Elektrobranche.

RECHNUNGSTEXT:
{raw_data['raw_text'][:3000]}

EXTRAHIERE FOLGENDE DATEN (als valides JSON):
{{
  "rechnung": {{
    "nummer": "string",
    "datum": "YYYY-MM-DD",
    "betrag_netto": "number",
    "betrag_brutto": "number",
    "waehrung": "EUR"
  }},
  "lieferant": {{
    "name": "string",
    "adresse": "string",
    "ust_id": "string"
  }},
  "positionen": [
    {{
      "beschreibung": "deutsche Artikelbeschreibung",
      "menge": "number",
      "einheit": "string",
      "einzelpreis": "number",
      "gesamtpreis": "number",
      "kategorie": "Elektromaterial|Werkzeuge|Anlagen|Buero|Sonstiges"
    }}
  ]
}}

WICHTIG:
- Alle Texte auf Deutsch belassen
- Deutsche Elektroterminologie verwenden
- Preise als Zahlen (ohne Währungszeichen)
- Nur valides JSON zurückgeben
"""

        try:
            if self.gemini_model is None:
                logger.warning("Gemini model not available, using fallback extraction")
                return self.extract_structured_data(raw_data)

            response = self.gemini_model.generate_content(prompt)

            # Parse Gemini response
            enhanced_data: dict[str, Any] = json.loads(response.text)
            enhanced_data.update(raw_data)  # Merge with raw data
            enhanced_data["enhancement_method"] = "gemini"
            enhanced_data["language"] = "de"  # Explicitly mark as German

            logger.debug("Gemini enhancement successful (German optimized)")
            return enhanced_data

        except Exception as e:
            logger.warning(f"Gemini enhancement failed: {e}")
            return self.extract_structured_data(raw_data)

    def extract_structured_data(self, raw_data: dict[str, Any]) -> dict[str, Any]:
        """Fallback structured extraction using patterns"""
        text = raw_data.get("raw_text", "")

        # Basic pattern extraction for German invoices
        structured = {
            "invoice_header": self.extract_invoice_header(text),
            "line_items": self.extract_line_items(raw_data.get("tables", []), text),
            "totals": self.extract_totals(text),
            "enhancement_method": "pattern_based",
        }

        structured.update(raw_data)
        return structured

    def extract_invoice_header(self, text: str) -> dict[str, Any]:
        """Extract invoice header information using patterns"""
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

    def extract_line_items(
        self, tables: list[dict[str, Any]], text: str = ""
    ) -> list[dict[str, Any]]:
        """Extract line items from table data or fallback to pattern extraction"""
        items = []

        # First try to extract from Docling tables
        for table in tables:
            headers = table.get("headers", [])
            rows = table.get("rows", [])

            # Map common German invoice table headers
            header_map = self.map_table_headers(headers)

            for row in rows:
                if len(row) >= len(headers):
                    item = {}
                    for i, header in enumerate(headers):
                        if i < len(row) and header_map.get(header):
                            item[header_map[header]] = row[i]

                    if item.get("description") or item.get("artikel"):
                        items.append(item)

        # If no table data found, try pattern-based extraction
        if not items and text:
            items = self.extract_line_items_from_text(text)

        return items

    def extract_line_items_from_text(self, text: str) -> list[dict[str, Any]]:
        """Extract line items using regex patterns for German invoices"""
        items = []
        lines = text.split("\n")

        logger.info(f"Trying to extract line items from {len(lines)} lines of text")

        for i, line in enumerate(lines):
            line = line.strip()

            # Skip header lines and empty lines
            if (
                not line
                or "Pos" in line
                or "Artikel-Nr" in line
                or "Bestellposition" in line
            ):
                continue

            # Skip unwanted lines (bank details, addresses, etc.)
            if any(
                skip in line.lower()
                for skip in [
                    "iban:",
                    "swift",
                    "commerzbank",
                    "erfüllungsort",
                    "amtsgericht",
                    "hannover",
                ]
            ):
                continue

            # Debug: log lines that contain potential product info
            if (
                any(brand in line.upper() for brand in ["GIRA", "SIEMENS"])
                or "|" in line
            ):
                logger.info(
                    f"Analyzing line {i}: '{line[:100]}{'...' if len(line) > 100 else ''}'"
                )

            # Handle Docling table format: fields separated by pipes
            if "|" in line and any(
                brand in line.upper()
                for brand in ["GIRA", "SIEMENS", "ABB", "SCHNEIDER", "LEGRAND", "HAGER"]
            ):
                parts = [p.strip() for p in line.split("|")]

                # Find the article number (usually first numeric field with 6+ digits)
                article_no = None
                for part in parts:
                    if re.match(r"^\d{6,}$", part.strip()):
                        article_no = part.strip()
                        break

                # Find description (contains brand name)
                description = None
                for part in parts:
                    if any(
                        brand in part.upper()
                        for brand in [
                            "GIRA",
                            "SIEMENS",
                            "ABB",
                            "SCHNEIDER",
                            "LEGRAND",
                            "HAGER",
                        ]
                    ):
                        # Extract just the product part
                        desc_match = re.search(
                            r"(GIRA|SIEMENS|ABB|SCHNEIDER|LEGRAND|HAGER)[^|]*",
                            part,
                            re.IGNORECASE,
                        )
                        if desc_match:
                            description = desc_match.group(0).strip()
                            # Clean up description - take only the product name part
                            if "GIRA" in description.upper():
                                gira_match = re.search(
                                    r"GIRA\s+[^|]+?(?=\s+\w+\.|$)",
                                    description,
                                    re.IGNORECASE,
                                )
                                if gira_match:
                                    description = gira_match.group(0).strip()
                        break

                # Find numeric values (quantity, unit price, total price)
                numeric_parts = []
                for part in parts:
                    if re.match(r"^\d+$", part.strip()) and len(part.strip()) <= 6:
                        numeric_parts.append(part.strip())

                # Extract values if we have enough data
                if article_no and description and len(numeric_parts) >= 3:
                    quantity = numeric_parts[0]
                    unit_price = numeric_parts[-2]  # Second to last number
                    total_price = numeric_parts[-1]  # Last number

                    logger.info(
                        f"Extracted: {article_no} | {description} | {quantity} | {unit_price} | {total_price}"
                    )

                    item = {
                        "article_number": article_no,
                        "description": description,
                        "quantity": quantity,
                        "unit_price": unit_price,
                        "total_price": total_price,
                    }
                    items.append(item)

        logger.info(f"Extracted {len(items)} line items using pattern matching")
        return items

    def map_table_headers(self, headers: list[str]) -> dict[str, str]:
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

    def extract_totals(self, text: str) -> dict[str, Any]:
        """Extract invoice totals"""
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
                totals[key] = match.group(1).replace(",", ".")

        return totals
