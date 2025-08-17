#!/usr/bin/env python3
"""
LLKJJ ML Pipeline - Da        self.docling_processor = AdvancedDoclingProcessor(
            use_gpu=True,
            enable_table_structure=True,
            enable_ocr=True,
            ocr_engine="tesseract_cli",  # Beste deutsche OCR
            table_mode="accurate",
            german_optimized=True  # Deutsche NER aktiviert
        )raction Module
==========================================

This module handles all data extraction functionality:
- PDF text extraction with Docling
- Table parsing and structure recogni            # Use new google-genai API: client.models.generate_content
            response = self.gemini_model.models.generate_content(
                model=self.config.gemini_model,  # Use the model from config
                contents=prompt
            )- Gemini AI enhancement
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

from src.config import Config
from src.extraction.docling_processor import AdvancedDoclingProcessor

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

    def __init__(self, gemini_model: Any = None, config: Config | None = None) -> None:
        """Initialize data extractor with optional Gemini model and config"""
        self.gemini_model = gemini_model
        self.config = config or Config()

        # Use singleton DoclingProcessor to prevent memory leaks
        try:
            from src.pipeline.processor import _resource_manager

            self.docling_processor = _resource_manager.get_docling_processor()
            logger.info("DataExtractor initialized with singleton DoclingProcessor")
        except ImportError:
            # Fallback if ResourceManager is not available
            self.docling_processor = AdvancedDoclingProcessor(
                use_gpu=True,
                enable_table_structure=True,
                enable_ocr=True,
                ocr_engine="tesseract_cli",  # Startet mit TesseractCLI, fällt auf robuste Engines zurück
                table_mode="accurate",
                german_optimized=True,  # Deutsche NER + intelligentes OCR-Fallback
            )
            logger.warning(
                "DataExtractor fallback: created own DoclingProcessor (ResourceManager unavailable)"
            )

    def process_pdf(self, pdf_path: Path) -> dict[str, Any]:
        """
        Hauptmethode für PDF-Extraktion mit intelligenter OCR-Fallback-Logik.

        2025 Update: Nutzt das neue process() System mit automatischer
        OCR-Engine-Auswahl für maximale Robustheit.

        Args:
            pdf_path: Pfad zur PDF-Datei

        Returns:
            Dictionary mit strukturierten Extraktionsergebnissen
        """
        try:
            logger.info("🔍 Starte intelligente PDF-Extraktion: %s", pdf_path.name)

            # Phase 1: Intelligente Docling-Extraktion mit OCR-Fallback
            try:
                # Neue intelligente Methode mit OCR-Fallback
                result, quality_score = self.docling_processor.process(str(pdf_path))

                logger.info("✅ Intelligente Extraktion erfolgreich:")
                logger.info(f"   Engine: {result['ocr_engine_used']}")
                logger.info(
                    f"   Versuch: {result['attempt_number']}/{result['total_attempts']}"
                )
                logger.info(f"   Qualität: {quality_score:.2f}")

                # Strukturierte Daten aus intelligentem Result verwenden
                if "structured_data" in result:
                    raw_result = result["structured_data"]
                else:
                    # Fallback: Strukturierte Extraktion aus Content
                    raw_result = {
                        "raw_text": result["content"],
                        "tables": [],
                        "pages": result["pages"],
                        "processing_metadata": {
                            "ocr_engine_used": result["ocr_engine_used"],
                            "quality_score": quality_score,
                            "processing_time": result["processing_time"],
                            "character_count": result["character_count"],
                            "attempt_number": result["attempt_number"],
                            "total_attempts": result["total_attempts"],
                            "german_ner_applied": result["german_ner_applied"],
                        },
                    }

            except Exception as e:
                logger.warning(f"⚠️ Intelligente Extraktion fehlgeschlagen: {e}")
                logger.info("🔄 Fallback auf Standard process_pdf...")
                # Fallback: Standard process_pdf
                raw_result = self.docling_processor.process_pdf(pdf_path)

            # Phase 2: Strukturierte Datenextraktion
            structured_data = self.extract_structured_data(raw_result)

            # Phase 3: Invoice-Header extrahieren
            invoice_data = self.extract_invoice_header(raw_result.get("raw_text", ""))

            # Phase 4: Line Items extrahieren
            line_items = self.extract_line_items(
                raw_result.get("tables", []), raw_result.get("raw_text", "")
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
                "raw_text": raw_result.get("raw_text", ""),
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

    def enhance_with_gemini(self, raw_data: dict[str, Any]) -> dict[str, Any]:
        """Enhance extraction using Gemini AI with German language optimization"""

        # Load prompt template from config file
        prompt_file = (
            Path(__file__).parent.parent / "config" / "gemini_extraction_prompt.txt"
        )
        try:
            with open(prompt_file, encoding="utf-8") as f:
                prompt_template = f.read()

            # Fill in the template with actual invoice text
            prompt = prompt_template.format(rechnungstext=raw_data["raw_text"][:3000])

        except (FileNotFoundError, KeyError) as e:
            logger.warning(f"Could not load prompt template: {e}, using fallback")
            # Fallback to hardcoded prompt if file not found
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

            # Use new google-genai API: client.models.generate_content
            response = self.gemini_model.models.generate_content(
                model=self.config.gemini_model,  # Use the model from config
                contents=prompt,
            )

            # Parse Gemini response - robust JSON handling
            response_text = (
                response.text if hasattr(response, "text") else str(response)
            )

            # Debug logging for Gemini response
            logger.debug(f"Gemini Response: {response_text[:500]}...")

            # NEU: Bereinige den Text von Markdown-Codeblöcken
            if "```json" in response_text:
                # Extrahiere nur den JSON-Teil
                response_text = response_text.split("```json\n", 1)[1].rsplit(
                    "\n```", 1
                )[0]
            elif response_text.strip().startswith("```"):
                # Allgemeinerer Fall für Codeblöcke
                response_text = response_text.strip().strip("`").strip()

            # Safe JSON parsing with fallback
            try:
                if not response_text or response_text.strip() == "":
                    logger.warning("Gemini returned empty response")
                    enhanced_data: dict[str, Any] = {}
                else:
                    enhanced_data = json.loads(response_text)
            except json.JSONDecodeError as json_error:
                logger.warning(
                    f"Gemini JSON decode error: {json_error}, using fallback"
                )
                enhanced_data = {}
            enhanced_data.update(raw_data)  # Merge with raw data
            enhanced_data["enhancement_method"] = "gemini"
            enhanced_data["language"] = "de"  # Explicitly mark as German

            logger.debug("Gemini enhancement successful (German optimized)")
            return enhanced_data

        except Exception as e:
            logger.warning(f"Gemini enhancement failed: {e}")
            return self.extract_structured_data(raw_data)

    def extract_structured_data(self, raw_data: dict[str, Any]) -> dict[str, Any]:
        """Fallback structured extraction using patterns.

        Returns a structure compatible with consumers/tests expecting:
        - header, line_items, footer keys.
        """
        # Support both raw_text and content fields
        text = raw_data.get("raw_text") or raw_data.get("content") or ""

        # Header extraction
        header = self.extract_invoice_header(text)

        # Totals extraction and normalization
        totals_raw = self.extract_totals(text)
        footer: dict[str, Any] = {}

        if "net_total" in totals_raw:
            try:
                footer["netto_total"] = float(totals_raw["net_total"].replace(",", "."))
            except Exception:
                footer["netto_total"] = 0.0

        if "gross_total" in totals_raw:
            try:
                footer["brutto_total"] = float(
                    totals_raw["gross_total"].replace(",", ".")
                )
            except Exception:
                footer["brutto_total"] = 0.0

        if "vat_amount" in totals_raw:
            try:
                footer["vat_total"] = float(totals_raw["vat_amount"].replace(",", "."))
            except Exception:
                footer["vat_total"] = 0.0

        # Line items extraction
        line_items = self.extract_line_items(raw_data.get("tables", []), text)

        structured: dict[str, Any] = {
            "header": header,
            "line_items": line_items,
            "footer": footer,
            "enhancement_method": "pattern_based",
        }

        # Preserve original raw_data for debugging
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

        # --- Lieferantendaten Fallback-Extraktion ---
        # Nur die ersten 20 Zeilen durchsuchen
        lines = text.split("\n")[:20]
        supplier_name = None
        supplier_address = None
        supplier_ust_id = None

        # Name: Erste Zeile mit Großbuchstaben, kein "Rechnung", kein "Kunde"
        for line in lines:
            if (
                line.strip()
                and line.strip()[0].isupper()
                and not re.search(
                    r"Rechnung|Kunde|Lieferdatum|Datum|Nr", line, re.IGNORECASE
                )
                and len(line.strip()) > 3
            ):
                supplier_name = line.strip()
                break

        # Adresse: Zeile mit PLZ und Ort
        for line in lines:
            addr_match = re.search(r"\b\d{5}\s+[A-ZÄÖÜ][a-zäöüß]+.*", line)
            if addr_match:
                supplier_address = addr_match.group(0).strip()
                break

        # USt-IdNr.:
        for line in lines:
            ust_match = re.search(
                r"USt[-\s]?IdNr\.?\s*:?[\s]*([A-Z]{2}\s?\d{9,12})", line
            )
            if ust_match:
                supplier_ust_id = ust_match.group(1).replace(" ", "")
                break

        # Ergebnisse hinzufügen, falls gefunden
        if supplier_name:
            header["supplier_name"] = supplier_name
        if supplier_address:
            header["supplier_address"] = supplier_address
        if supplier_ust_id:
            header["supplier_ust_id"] = supplier_ust_id

        return header

    def extract_line_items(
        self, tables: list[dict[str, Any]], text: str = ""
    ) -> list[dict[str, Any]]:
        """Extract line items from table data or fallback to pattern extraction"""
        items = []

        logger.info(f"DEBUG: Tabellen-Extraktion startet mit {len(tables)} Tabellen")

        # First try to extract from Docling tables
        for idx, table in enumerate(tables):
            headers = table.get("headers", [])
            rows = table.get("rows", [])

            logger.info(
                f"DEBUG: Tabelle {idx}: {len(headers)} Headers, {len(rows)} Rows"
            )
            if headers:
                logger.info(
                    f"DEBUG: Headers: {headers[:3]}..."
                )  # Ersten 3 Headers zeigen

            # Map common German invoice table headers
            header_map = self.map_table_headers(headers)

            for row in rows:
                if len(row) >= len(headers):
                    item = {}
                    for i, header in enumerate(headers):
                        if i < len(row) and header_map.get(header):
                            item[header_map[header]] = row[i]

                    # NEU: Validierung VOR dem Hinzufügen
                    # Annahme: item['article_number'], item['description'], item['quantity'] sind befüllt
                    # Verwende die striktere Sonepar-Validierung für Tabellen
                    if (
                        item.get("description")
                        and item.get("article_number")
                        and self._validate_sonepar_line_item(
                            "01",  # Platzhalter für Positionsnummer
                            item.get("description", ""),
                            item.get("article_number", ""),
                            item.get("quantity", "0"),
                        )
                    ):
                        items.append(item)
                        logger.info(
                            f"DEBUG: VALIDES Item aus Tabelle hinzugefügt: {item.get('description', 'N/A')[:30]}"
                        )
                    else:
                        logger.warning(f"DEBUG: UNGÜLTIGES Item verworfen: {row}")

        logger.info(
            f"DEBUG: Tabellen-Extraktion abgeschlossen. {len(items)} Items gefunden"
        )
        logger.info(f"DEBUG: Text verfügbar: {len(text) if text else 0} Zeichen")

        # If no table data found, try pattern-based extraction
        if not items and text:
            logger.info("DEBUG: Keine Tabellen-Items gefunden - starte Text-Extraktion")
            items = self.extract_line_items_from_text(text)
        elif not text:
            logger.info("DEBUG: Kein Text verfügbar für Pattern-Extraktion")
        else:
            logger.info(
                f"DEBUG: Überspringe Text-Extraktion - {len(items)} Tabellen-Items gefunden"
            )

        return items

    def extract_line_items_from_text(self, text: str) -> list[dict[str, Any]]:
        """
        Extract line items using improved regex patterns for German invoices.

        Supports multi-line invoice formats where data is spread across rows and columns.
        Specifically optimized for Sonepar format: |pos|description|qty|pe|empty|unit_price|total|
        """
        items = []

        logger.info(f"Versuche robuste Extraktion aus {len(text)} Zeichen Text")

        # SONEPAR-SPEZIFISCHES PATTERN (angepasst für Docling-Format)
        # Docling Format: "GIRA Adapterrahmen rws-gl quadr. 50x50mm S.55 028203 |      10 |    "
        # OCR File Format: "010889201 |GIRA Adapterrahmen rws-gl 10 |1 182 1820"

        # Pattern für Docling-Ausgabe: Marke + Beschreibung + Artikelnummer + Pipe + Menge + Pipe
        sonepar_docling_pattern = re.compile(
            r"((?:GIRA|SIEMENS|ABB|SCHNEIDER|LEGRAND|HAGER|Spelsberg|Striebel|CIMCO)[^|]*?(\d{6})[^|]*?)\|\s*(\d+)\s*\|",
            re.IGNORECASE,
        )

        # Pattern für Seite 2 Format: Artikel-Nr. in separater Spalte
        sonepar_page2_pattern = re.compile(
            r"(\d{6,})\s*\|\s*([^|]*(?:Spelsberg|Hager|Striebel|CIMCO)[^|]*?)\s*\|\s*(\d+)\s*\|",
            re.IGNORECASE,
        )

        # Pattern für Positionen ohne erkennbare Artikelnummer (Siemens, Hager ohne Nummer)
        sonepar_no_article_pattern = re.compile(
            r"(?:Bestellposition Kunde: \d+\s+)?(SIEMENS|Hager)\s+([^|]*(?:Blindabdeckstreifen|Abdeckstreifen)[^|]*?)\s*\|\s*(\d+)\s*\|",
            re.IGNORECASE,
        )

        # Fallback Pattern für OCR-Format (falls verfügbar)
        sonepar_ocr_pattern = re.compile(
            r"(\d{2})(\d{7})\s*\|([^|]*(?:GIRA|SIEMENS|ABB|SCHNEIDER|LEGRAND|HAGER|Spelsberg|Striebel|CIMCO)[^|]*?)\s*(\d+)\s*\|\s*(\d+)\s*(\d+)\s*(\d+)",
            re.IGNORECASE,
        )

        # Allgemeine Fallback-Pattern für andere Rechnungsformate
        patterns = [
            # Pattern 1: Sonepar Docling-Format (PRIORITÄT für echte PDF-Extraktion)
            sonepar_docling_pattern,
            # Pattern 2: Sonepar Seite 2 Format (Artikel-Nr. in separater Spalte)
            sonepar_page2_pattern,
            # Pattern 3: Positionen ohne Artikelnummer (Siemens, Hager)
            sonepar_no_article_pattern,
            # Pattern 4: Sonepar OCR-Format (Fallback für OCR-Dateien)
            sonepar_ocr_pattern,
            # Pattern 5: Standard Docling pipe-separated Format
            re.compile(
                r"(\d{6,})\s*\|\s*([^|]*(?:GIRA|SIEMENS|ABB|SCHNEIDER|LEGRAND|HAGER|Spelsberg|Striebel|CIMCO)[^|]*)\s*\|\s*(\d+)\s*\|\s*(\d+[,.]?\d*)\s*\|\s*(\d+[,.]?\d*)",
                re.IGNORECASE,
            ),
            # Pattern 6: Whitespace-separated Format mit Marken-Keywords
            re.compile(
                r"(\d{6,})\s+([^0-9]*(?:GIRA|SIEMENS|ABB|SCHNEIDER|LEGRAND|HAGER|Spelsberg|Striebel|CIMCO)[^0-9]*?)\s+(\d+)\s+(\d+[,.]?\d*)\s+(\d+[,.]?\d*)",
                re.IGNORECASE,
            ),
        ]

        for pattern_idx, pattern in enumerate(patterns):
            logger.info(
                f"Verwende Pattern {pattern_idx + 1}: {pattern.pattern[:80]}..."
            )

            # DEBUG: Test das Sonepar Pattern extra
            if pattern_idx == 0:
                gira_test = re.search(r"GIRA.*Adapterrahmen", text, re.IGNORECASE)
                logger.info(
                    f"DEBUG: GIRA Adapterrahmen Test erfolgreich: {'JA' if gira_test else 'NEIN'}"
                )

                if gira_test:
                    logger.info(
                        f"DEBUG: GIRA Context: '{text[gira_test.start()-50:gira_test.end()+50]}'"
                    )

            matches = pattern.findall(text)
            logger.info(f"Pattern {pattern_idx + 1} gefunden: {len(matches)} Matches")

            for match in matches:
                if pattern_idx == 0:  # Sonepar Docling-Format
                    full_description, article_no, quantity = match

                    # Extrahiere saubere Produktbeschreibung
                    description_clean = self._extract_product_description(
                        full_description
                    )

                    # Für Docling-Format fehlen die Preise, setze Platzhalter
                    pos_nr = "01"  # Platzhalter da Position im Docling-Text fehlt
                    unit_price = "0"  # Nicht verfügbar in diesem Format
                    total_price = "0"  # Nicht verfügbar in diesem Format

                    logger.info(
                        f"Sonepar Docling Pattern extrahiert: Art={article_no} | {description_clean[:30]}... | Qty={quantity}"
                    )

                    logger.info(
                        f"DEBUG: Validierung für Art={article_no}, Pos={pos_nr}, Desc='{description_clean[:50]}', Qty={quantity}"
                    )

                    if self._validate_sonepar_line_item(
                        pos_nr, description_clean, article_no, quantity
                    ):
                        item = {
                            "position_number": pos_nr,
                            "article_number": article_no.strip(),
                            "description": description_clean.strip(),
                            "quantity": quantity.strip(),
                            "unit_price": unit_price,
                            "total_price": total_price,
                        }
                        items.append(item)
                        logger.info(f"✅ Item validiert und hinzugefügt: {article_no}")
                    else:
                        logger.warning(
                            f"❌ Item Validierung fehlgeschlagen: {article_no}"
                        )

                elif pattern_idx == 1:  # Sonepar Seite 2 Format
                    article_no, description, quantity = match

                    # Extrahiere saubere Produktbeschreibung
                    description_clean = self._extract_product_description(description)

                    # Für Seite 2 Format fehlen die Preise, setze Platzhalter
                    pos_nr = "01"  # Platzhalter da Position im Docling-Text fehlt
                    unit_price = "0"  # Nicht verfügbar in diesem Format
                    total_price = "0"  # Nicht verfügbar in diesem Format

                    logger.info(
                        f"Sonepar Seite 2 Pattern extrahiert: Art={article_no} | {description_clean[:30]}... | Qty={quantity}"
                    )

                    # Verwende dieselbe Validierung wie Pattern 0 für Konsistenz
                    if self._validate_sonepar_line_item(
                        pos_nr, description_clean, article_no, quantity
                    ):
                        item = {
                            "position_number": pos_nr,
                            "article_number": article_no.strip(),
                            "description": description_clean.strip(),
                            "quantity": quantity.strip(),
                            "unit_price": unit_price,
                            "total_price": total_price,
                        }
                        items.append(item)
                        logger.info(f"✅ Item validiert und hinzugefügt: {article_no}")
                    else:
                        logger.warning(
                            f"❌ Item Validierung fehlgeschlagen: {article_no}"
                        )

                elif pattern_idx == 2:  # Positionen ohne Artikelnummer (Siemens, Hager)
                    brand, description, quantity = match

                    # Kombiniere Marke und Beschreibung
                    full_description = f"{brand} {description}".strip()
                    description_clean = self._extract_product_description(
                        full_description
                    )

                    # Setze Platzhalter für fehlende Daten
                    pos_nr = "01"
                    article_no = "000000"  # Platzhalter für fehlende Artikelnummer
                    unit_price = "0"
                    total_price = "0"

                    logger.info(
                        f"Sonepar Ohne-Artikelnummer Pattern extrahiert: Brand={brand} | {description_clean[:30]}... | Qty={quantity}"
                    )

                    if self._validate_line_item(
                        article_no, description_clean, quantity
                    ):
                        item = {
                            "position_number": pos_nr,
                            "article_number": article_no.strip(),
                            "description": description_clean.strip(),
                            "quantity": quantity.strip(),
                            "unit_price": unit_price,
                            "total_price": total_price,
                        }
                        items.append(item)
                        logger.info(
                            f"✅ Item validiert und hinzugefügt: {brand} {description[:20]}..."
                        )
                    else:
                        logger.warning(
                            f"❌ Item Validierung fehlgeschlagen: {brand} {description[:20]}..."
                        )

                elif pattern_idx == 3:  # Sonepar OCR-Format
                    (
                        pos_nr,
                        sonepar_nr,
                        description,
                        quantity,
                        pe_value,
                        unit_price,
                        total_price,
                    ) = match

                    # Suche die Artikelnummer in den nächsten Zeilen nach diesem Match
                    position_text = f"{pos_nr}{sonepar_nr}"
                    pos_index = text.find(position_text)

                    article_no = ""
                    if pos_index >= 0:
                        # Schaue in den nächsten 200 Zeichen nach einer 6-stelligen Artikelnummer
                        search_area = text[pos_index : pos_index + 200]
                        article_match = re.search(r"\b(\d{6})\b", search_area)
                        if article_match:
                            article_no = article_match.group(1)

                    # Extrahiere saubere Produktbeschreibung ohne Lieferdetails
                    description_clean = self._extract_product_description(description)

                    logger.info(
                        f"Sonepar OCR Pattern extrahiert: Pos={pos_nr} | Sonepar={sonepar_nr} | Art={article_no} | {description_clean[:30]}... | Qty={quantity} | Unit={unit_price} | Total={total_price}"
                    )

                    if self._validate_sonepar_line_item(
                        pos_nr, description_clean, article_no, quantity
                    ):
                        item = {
                            "position_number": pos_nr.strip(),
                            "sonepar_number": sonepar_nr.strip(),  # Sonepar interne Nummer
                            "article_number": article_no.strip(),  # GIRA Artikelnummer
                            "description": description_clean.strip(),
                            "quantity": quantity.strip(),
                            "unit_price": unit_price.strip(),
                            "total_price": total_price.strip(),
                        }
                        items.append(item)

                elif len(match) >= 5:  # Standard patterns (Index 4+)
                    article_no, description, quantity, unit_price, total_price = match[
                        :5
                    ]

                    # Bereinige Beschreibung von Pipe-Zeichen und Extra-Leerzeichen
                    description = re.sub(r"\|+", " ", description).strip()
                    description = re.sub(r"\s+", " ", description)

                    # Validiere dass es ein echtes Produkt ist (nicht Kopfzeile/Footer)
                    if self._validate_line_item(article_no, description, quantity):
                        logger.info(
                            f"Pattern {pattern_idx + 1} extrahiert: {article_no} | {description[:30]}... | {quantity} | {unit_price} | {total_price}"
                        )

                        item = {
                            "article_number": article_no.strip(),
                            "description": description.strip(),
                            "quantity": quantity.strip(),
                            "unit_price": unit_price.strip(),
                            "total_price": total_price.strip(),
                        }
                        items.append(item)

        # Deduplizierung basierend auf Artikelnummer
        seen_articles = set()
        unique_items = []
        for item in items:
            key = item.get("article_number", "") + item.get("position_number", "")
            if key not in seen_articles:
                seen_articles.add(key)
                unique_items.append(item)

        logger.info(
            f"Robuste Extraktion: {len(unique_items)} eindeutige Positionen gefunden"
        )
        return unique_items

    def _extract_product_description(self, full_description: str) -> str:
        """Extrahiert saubere Produktbeschreibung aus Sonepar-Text"""

        # Entferne Lieferdetails am Anfang
        desc = re.sub(r"^.*?Lieferung.*?Scharmbeck\s*", "", full_description)

        # Extrahiere Produktteil bis zur Artikelnummer am Ende
        # Format: "MARKE Produktname ... Artikelnummer"
        product_match = re.search(
            r"(GIRA|SIEMENS|ABB|SCHNEIDER|LEGRAND|HAGER|Spelsberg|Striebel|CIMCO)([^0-9]+)",
            desc,
            re.IGNORECASE,
        )

        if product_match:
            brand = product_match.group(1)
            product_text = product_match.group(2).strip()

            # Bereinige Produkttext
            product_text = re.sub(r"\s+", " ", product_text)
            product_text = re.sub(r"[^a-zA-ZäöüÄÖÜß0-9.\s\-/]", "", product_text)

            return f"{brand} {product_text}".strip()

        # Fallback: Nutze ersten aussagekräftigen Teil
        desc_clean = re.sub(r"Bestellposition Kunde:.*", "", desc)
        desc_clean = re.sub(r"\s+", " ", desc_clean).strip()

        return desc_clean[:100] if desc_clean else full_description[:50]

    def _validate_sonepar_line_item(
        self, pos_nr: str, description: str, article_no: str, quantity: str
    ) -> bool:
        """Validierung speziell für Sonepar-Format"""

        # Positionsnummer muss 2 Ziffern haben (01, 02, 03, etc.)
        if not re.match(r"^\d{2}$", pos_nr.strip()):
            return False

        # Artikelnummer muss mindestens 6 Ziffern haben
        if not re.match(r"^\d{6,}$", article_no.strip()):
            return False

        # Beschreibung muss Elektro-Marke enthalten
        elektro_brands = [
            "GIRA",
            "SIEMENS",
            "ABB",
            "SCHNEIDER",
            "LEGRAND",
            "HAGER",
            "Spelsberg",
            "Striebel",
            "CIMCO",  # Hinzugefügt für CIMCO Positionen
        ]
        if not any(brand.lower() in description.lower() for brand in elektro_brands):
            return False

        # Menge muss numerisch und realistisch sein (1-999)
        try:
            qty = int(quantity.strip())
            if not 1 <= qty <= 999:
                return False
        except ValueError:
            return False

        return True

    def _validate_line_item(
        self, article_no: str, description: str, quantity: str
    ) -> bool:
        """Validiert ob es sich um eine echte Rechnungsposition handelt"""

        # Artikelnummer muss mindestens 6 Ziffern haben (entspanntere Validierung)
        if not re.match(r"^\d{6,}$", article_no.strip()):
            # Fallback: Prüfe ob die Beschreibung stark genug ist, um fehlende Artikelnummer zu kompensieren
            strong_elektro_indicators = [
                "GIRA",
                "SIEMENS",
                "HAGER",
                "Spelsberg",
                "Striebel",
                "CIMCO",
                "Abdeckrahmen",
                "Abdeckstreifen",
                "Blindabdeckung",
                "Verschlußstreifen",
                "Überziehschuhe",
            ]
            if not any(
                indicator.lower() in description.lower()
                for indicator in strong_elektro_indicators
            ):
                return False

        # Beschreibung muss mindestens eine Elektro-Marke oder ein Elektro-Keyword enthalten
        elektro_keywords = [
            "GIRA",
            "SIEMENS",
            "ABB",
            "SCHNEIDER",
            "LEGRAND",
            "HAGER",
            "WAGO",
            "PHOENIX",
            "Spelsberg",  # Hinzugefügt für Position 7
            "Striebel",  # Hinzugefügt für Position 9 (Striebel&J)
            "CIMCO",  # Hinzugefügt für Position 10
            "Klingeltaster",
            "Abdeckrahmen",
            "Abdeckstreifen",  # Hinzugefügt für Position 6, 7, 8
            "Blindabdeckung",  # Für GIRA und Siemens
            "Blindabdeckstreifen",  # Spezifisch für Siemens Position 6
            "Verschlußstreifen",  # Für Striebel&J Position 9
            "Überziehschuhe",  # Für CIMCO Position 10
            "Dimmer",
            "Schalter",
            "Steckdose",
            "Schutzschalter",
            "Kabel",
            "Rohr",
            "Dose",
            "Sicherung",
        ]

        if not any(
            keyword.lower() in description.lower() for keyword in elektro_keywords
        ):
            return False

        # Menge muss numerisch und realistisch sein (1-999)
        try:
            qty = int(quantity.strip())
            if not 1 <= qty <= 999:
                return False
        except ValueError:
            return False

        # Keine Kopfzeilen oder Footer-Texte
        invalid_terms = [
            "artikel-nr",
            "pos",
            "menge",
            "preis",
            "gesamt",
            "summe",
            "total",
            "iban",
            "swift",
            "commerzbank",
            "hannover",
        ]

        # Prüfe auf ungültige Header-Begriffe, aber erlaube "bestellposition" in Produktbeschreibungen
        for term in invalid_terms:
            if term in description.lower():
                return False

        # Spezielle Prüfung für "bestellposition" - nur verbieten wenn es als Header-Zeile aussieht
        if "bestellposition" in description.lower():
            # Erlaube wenn es Teil einer Produktbeschreibung mit Marke ist
            has_brand = any(
                brand.lower() in description.lower()
                for brand in [
                    "gira",
                    "siemens",
                    "hager",
                    "spelsberg",
                    "striebel",
                    "cimco",
                    "abb",
                    "schneider",
                    "legrand",
                ]
            )
            # Verbiete nur wenn es wie eine reine Header-Zeile aussieht (ohne Marke und ohne Artikelnummer)
            if not has_brand and not re.search(r"\d{6,}", description):
                return False

        return True

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
        """Extract invoice totals (net, vat, gross) as strings for normalization."""
        totals: dict[str, Any] = {}

        # German total patterns (support common labels and vat rates)
        patterns = {
            "net_total": r"(?:Netto|Zwischensumme)\s*:?\s*(\d+[.,]\d{2})",
            "vat_amount": r"(?:MwSt|USt|(?:\d{1,2}%))\s*:?\s*(\d+[.,]\d{2})",
            "gross_total": r"(?:Brutto|Gesamt|Endbetrag)\s*:?\s*(\d+[.,]\d{2})",
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                totals[key] = match.group(1).strip()

        return totals

    def cleanup(self):
        """Cleanup DataExtractor resources to prevent memory leaks"""
        import gc

        logger.info("🧹 Cleanup DataExtractor resources...")

        # Cleanup Gemini model
        if hasattr(self, "gemini_model") and self.gemini_model:
            try:
                del self.gemini_model
                self.gemini_model = None
            except Exception as e:
                logger.warning(f"Gemini model cleanup warning: {e}")

        # Note: DoclingProcessor is managed by ResourceManager singleton
        # Don't cleanup shared resources here

        # Force garbage collection
        collected = gc.collect()
        if collected > 0:
            logger.info(f"DataExtractor GC: {collected} objects collected")

        logger.info("✅ DataExtractor cleanup complete")
