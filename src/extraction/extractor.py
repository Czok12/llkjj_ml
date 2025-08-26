"""
DataExtractor für LLKJJ ML Plugin - Intelligente PDF-Rechnungsextraktion.

Nutzt Docling 2.44.0 für robuste PDF-Verarbeitung mit TableFormer AI.
Optimiert für deutsche Elektrotechnik-Rechnungen (Sonepar, etc.).
"""

# Logging Setup
import logging
import re
from pathlib import Path
from typing import Any

from docling.document_converter import DocumentConverter

from src.settings_bridge import Config

logger = logging.getLogger(__name__)


class DataExtractor:
    """
    Intelligenter PDF-Datenextraktor für deutsche Elektrotechnik-Rechnungen.

    Features:
    - Docling 2.44.0 Integration mit TableFormer AI
    - Robuste Tabellenerkennung für Line Items
    - Deutsche Rechnungsformat-Optimierung
    - Optionale Gemini AI-Verbesserung
    """

    def __init__(self, config: Any = None) -> None:
        """Initialisiert DataExtractor mit Docling DocumentConverter."""
        logger.info("🔄 Initialisiere DataExtractor...")

        # Config für Kompatibilität akzeptieren
        self.config = config

        # Docling DocumentConverter für PDF-Verarbeitung
        self.document_converter = DocumentConverter()

        # Optional: Gemini AI Client für Verbesserung
        self.gemini_client = None
        try:
            if hasattr(Config, "USE_GEMINI") and hasattr(Config, "GEMINI_API_KEY"):
                # Gemini würde hier initialisiert werden
                pass
        except Exception as e:
            logger.warning(f"⚠️ Gemini AI nicht verfügbar: {e}")

        logger.info("✅ DataExtractor initialisiert")

    def process_pdf(self, pdf_path: Path) -> dict[str, Any]:
        """
        Verarbeitet eine PDF-Datei und extrahiert strukturierte Daten.

        Args:
            pdf_path: Pfad zur PDF-Datei

        Returns:
            Dictionary mit extrahierten Daten
        """
        try:
            logger.info(f"🔄 Extrahiere Daten aus: {pdf_path.name}")

            # PDF mit Docling konvertieren
            docling_result = self.document_converter.convert(pdf_path)
            doc = docling_result.document
            raw_text = doc.export_to_text()

            # Strukturierte Daten extrahieren
            structured_data: dict[str, Any] = {
                "invoice_number": "",
                "invoice_date": "",
                "supplier": "",
                "customer": "",
                "line_items": [],
                "total_amount": "",
                "tables": [],
                "raw_text": raw_text[:2000],  # Erste 2000 Zeichen für Kontext
            }

            # Basic Info aus Text extrahieren
            self._extract_basic_info(raw_text, structured_data)

            # Line Items aus Tabellen extrahieren
            self._extract_line_items_from_tables(doc, structured_data)

            # Metadata sammeln
            metadata: dict[str, Any] = {
                "file_size": pdf_path.stat().st_size,
                "file_name": pdf_path.name,
                "extraction_method": "docling",
            }

            result: dict[str, Any] = {
                "raw_text": raw_text,
                "structured_data": structured_data,
                "line_items": structured_data.get(
                    "line_items", []
                ),  # Für compatibility mit processor.py
                "metadata": metadata,
                "success": True,
            }

            logger.info(f"✅ Extraktion erfolgreich: {len(raw_text)} Zeichen")
            return result

        except Exception as e:
            logger.error(f"❌ Fehler bei PDF-Extraktion: {e}")
            return {
                "raw_text": "",
                "structured_data": {},
                "metadata": {"error": str(e)},
                "success": False,
            }

    def _extract_basic_info(
        self, raw_text: str, structured_data: dict[str, Any]
    ) -> None:
        """Extrahiert Basis-Informationen aus dem Text."""
        lines = raw_text.split("\n")

        for line in lines:
            line = line.strip()

            # Rechnungsnummer extrahieren
            if "Rechnungs-Nr." in line or "Rechnung" in line:
                match = re.search(r"(\d{8,})", line)
                if match:
                    structured_data["invoice_number"] = match.group(1)

            # Datum extrahieren
            if "Datum" in line:
                match = re.search(r"(\d{1,2}\.\d{1,2}\.\d{2,4})", line)
                if match:
                    structured_data["invoice_date"] = match.group(1)

            # Lieferant (Sonepar)
            if "Sonepar" in line:
                structured_data["supplier"] = line.strip()

            # Kunde
            if "Elektro" in line and "UG" in line:
                structured_data["customer"] = line.strip()

    def _extract_line_items_from_tables(
        self, doc: Any, structured_data: dict[str, Any]
    ) -> None:
        """Extrahiert Line Items aus Docling-Tabellen mit TableCell-Unterstützung."""
        if not hasattr(doc, "tables") or not doc.tables:
            return

        for table_idx, table in enumerate(doc.tables):
            try:
                if not hasattr(table, "data") or not hasattr(table.data, "table_cells"):
                    continue

                table_cells = table.data.table_cells
                logger.info(
                    f"📊 Verarbeite Tabelle {table_idx} mit {len(table_cells)} Zellen"
                )

                # Konvertiere TableCells zu lesbarer Struktur
                rows_dict: dict[int, dict[int, str]] = {}
                for cell in table_cells:
                    if hasattr(cell, "start_row_offset_idx") and hasattr(cell, "text"):
                        row_idx = cell.start_row_offset_idx
                        col_idx = (
                            cell.start_col_offset_idx
                            if hasattr(cell, "start_col_offset_idx")
                            else 0
                        )

                        if row_idx not in rows_dict:
                            rows_dict[row_idx] = {}
                        rows_dict[row_idx][col_idx] = cell.text

                # Konvertiere zu sortierten Zeilen
                table_rows: list[list[str]] = []
                for row_idx in sorted(rows_dict.keys()):
                    row_data: list[str] = []
                    for col_idx in sorted(rows_dict[row_idx].keys()):
                        row_data.append(rows_dict[row_idx][col_idx])
                    table_rows.append(row_data)

                logger.info(f"📋 Konvertierte Tabelle: {len(table_rows)} Zeilen")

                # Suche nach Rechnungsposten (Header überspringen)
                for _row_idx, row in enumerate(table_rows[1:], 1):  # Skip header
                    if len(row) < 4:
                        continue

                    row_text = " ".join(str(cell) for cell in row)

                    # Deutsche Elektrotechnik-Marken erkennen
                    product_keywords = [
                        "gira",
                        "siemens",
                        "hager",
                        "artikel",
                        "pos",
                        "spelsberg",
                        "striebel",
                        "cimco",
                        "bestellposition",
                        "adapterrahmen",
                        "abdeckrahmen",
                        "blindabdeckung",
                    ]

                    if any(keyword in row_text.lower() for keyword in product_keywords):
                        # Extrahiere Line Item mit intelligenter Spaltenerkennung
                        line_item = self._parse_line_item_from_row(
                            row, table_rows[0] if table_rows else []
                        )
                        if line_item:
                            structured_data["line_items"].append(line_item)
                            logger.info(
                                f"✅ Line Item gefunden: {line_item['description'][:50]}..."
                            )

                # Speichere auch Tabellendaten für Debugging
                table_data = {
                    "rows": len(table_rows),
                    "cols": len(table_rows[0]) if table_rows else 0,
                    "data": table_rows[:10],  # Nur erste 10 Zeilen speichern
                }
                structured_data["tables"].append(table_data)

            except Exception as table_error:
                logger.warning(f"⚠️ Tabellenfehler ignoriert: {table_error}")
                continue

    def _parse_line_item_from_row(
        self, row: list[str], header: list[str]
    ) -> dict[str, Any] | None:
        """Parst eine Tabellenzeile zu einem Line Item."""
        try:
            # Standard-Spaltenerkennung für Sonepar-Format
            line_item = {
                "position": "",
                "description": "",
                "quantity": "",
                "price": "",
                "total": "",
            }

            # Intelligente Zuordnung basierend auf Spaltenanzahl
            if len(row) >= 7:  # Vollständige Sonepar-Tabelle
                line_item["position"] = str(row[0]).strip() if row[0] else ""
                line_item["description"] = str(row[1]).strip() if row[1] else ""
                line_item["quantity"] = str(row[2]).strip() if row[2] else ""
                line_item["price"] = (
                    str(row[5]).strip() if len(row) > 5 and row[5] else ""
                )
                line_item["total"] = (
                    str(row[6]).strip() if len(row) > 6 and row[6] else ""
                )
            elif len(row) >= 4:  # Minimale Struktur
                line_item["position"] = str(row[0]).strip() if row[0] else ""
                line_item["description"] = str(row[1]).strip() if row[1] else ""
                line_item["quantity"] = str(row[2]).strip() if row[2] else ""
                line_item["total"] = str(row[-1]).strip() if row[-1] else ""
            else:
                return None

            # Validierung: Mindestens Beschreibung muss vorhanden sein
            if not line_item["description"].strip():
                return None

            return line_item

        except Exception as e:
            logger.warning(f"⚠️ Fehler beim Parsen der Zeile: {e}")
            return None

    def extract_structured_data(self, raw_text: str) -> dict[str, Any]:
        """
        Extrahiert strukturierte Daten aus bereits extrahiertem Text.

        Args:
            raw_text: Der bereits extrahierte Text

        Returns:
            Dictionary mit strukturierten Daten
        """
        try:
            structured_data = {
                "invoice_number": "",
                "invoice_date": "",
                "supplier": "",
                "customer": "",
                "line_items": [],
                "total_amount": "",
            }

            # Basis-Informationen extrahieren
            self._extract_basic_info(raw_text, structured_data)

            return structured_data

        except Exception as e:
            logger.error(f"❌ Fehler bei Strukturierung: {e}")
            return {}

    def enhance_with_gemini(self, raw_text: str) -> dict[str, Any]:
        """
        Verbessert extrahierte Daten mit Gemini AI (optional).

        Args:
            raw_text: Roher extrahierter Text

        Returns:
            Dictionary mit verbesserten/strukturierten Daten
        """
        # Gemini Client ist aktuell deaktiviert - verwende Fallback
        logger.warning("⚠️ Kein Gemini-Client verfügbar für Verbesserung")
        return {"enhanced_text": raw_text, "enhancement_applied": False}
