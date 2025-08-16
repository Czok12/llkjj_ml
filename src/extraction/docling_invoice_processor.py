"""
Docling Invoice Processor für LLKJJ ML Pipeline

Spezialisierte Klasse für die Verarbeitung von PDF-Eingangsrechnungen
mit IBM Docling für strukturierte Dokumentenerkennung.

Features:
- Strukturierte PDF-Verarbeitung mit Docling
- Automatische Tabellenerkennung für Rechnungspositionen
- Intelligent Text-Extraktion mit Layout-Erkennung
- Integration mit Gemini für AI-basierte Datenextraktion
- Unterstützung für deutsche Elektrotechnik-Rechnungen
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from docling.chunking import HybridChunker
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption

logger = logging.getLogger(__name__)


@dataclass
class DoclingInvoiceItem:
    """Strukturierte Rechnungsposition aus Docling-Extraktion"""

    position_number: str | None = None
    description: str = ""
    article_number: str | None = None
    quantity: float = 0.0
    unit: str = "Stk"
    unit_price: float = 0.0
    total_price: float = 0.0
    vat_rate: float = 0.19
    category: str = "elektromaterial"
    skr03_account: str = "3400"

    # Docling-spezifische Metadaten
    table_source: bool = False
    confidence: float = 0.0
    bounding_box: dict[str, float] | None = None


@dataclass
class DoclingInvoiceResult:
    """Vollständiges Ergebnis der Docling-Verarbeitung"""

    # Rechnungsmetadaten
    invoice_number: str = ""
    invoice_date: str = ""
    supplier: str = ""
    supplier_number: str | None = None
    order_number: str | None = None

    # Finanzielle Daten
    net_amount: float = 0.0
    vat_amount: float = 0.0
    gross_amount: float = 0.0
    currency: str = "EUR"

    # Rechnungspositionen
    line_items: list[DoclingInvoiceItem] = field(default_factory=list)

    # Docling-spezifische Metadaten
    document_structure: dict[str, Any] = field(default_factory=dict)
    tables_found: int = 0
    processing_time_ms: int = 0
    docling_confidence: float = 0.0

    # Extrahierte Rohdaten
    raw_text: str = ""
    structured_content: dict[str, Any] = field(default_factory=dict)


class DoclingInvoiceProcessor:
    """
    Spezialisierter Processor für PDF-Eingangsrechnungen mit Docling

    Kombination aus struktureller PDF-Analyse (Docling) und
    AI-basierter Datenextraktion (Gemini) für optimale Ergebnisse.
    """

    def __init__(
        self,
        enable_table_structure: bool = True,
        table_mode: TableFormerMode = TableFormerMode.ACCURATE,
        max_pages: int = 50,
        max_file_size: int = 50 * 1024 * 1024,
    ):  # 50MB
        """
        Initialisiert den Docling Invoice Processor

        Args:
            enable_table_structure: Tabellenerkennung aktivieren
            table_mode: TableFormer-Modus (FAST oder ACCURATE)
            max_pages: Maximale Seitenzahl pro Dokument
            max_file_size: Maximale Dateigröße in Bytes
        """
        self.enable_table_structure = enable_table_structure
        self.table_mode = table_mode
        self.max_pages = max_pages
        self.max_file_size = max_file_size

        # Docling Pipeline konfigurieren
        self.pipeline_options = self._setup_pipeline()
        self.converter = self._setup_converter()
        self.chunker = HybridChunker()

        logger.info(
            f"DoclingInvoiceProcessor initialisiert mit TableFormer: {table_mode}"
        )

    def _setup_pipeline(self) -> PdfPipelineOptions:
        """Konfiguriert die Docling PDF-Pipeline für Rechnungsverarbeitung"""

        pipeline_options = PdfPipelineOptions(
            do_table_structure=self.enable_table_structure,
            do_ocr=True,  # OCR für gescannte PDFs
            do_cell_matching=True,  # Verbesserte Tabellenzellenextraktion
        )

        if self.enable_table_structure:
            # TableFormer für präzise Tabellenerkennung
            pipeline_options.table_structure_options.mode = self.table_mode
            pipeline_options.table_structure_options.do_cell_matching = True

        return pipeline_options

    def _setup_converter(self) -> DocumentConverter:
        """Erstellt den konfigurierten DocumentConverter"""

        format_options = {
            InputFormat.PDF: PdfFormatOption(pipeline_options=self.pipeline_options)
        }

        return DocumentConverter(format_options=format_options)

    def process_invoice(
        self, pdf_path: Path, gemini_extractor=None
    ) -> DoclingInvoiceResult:
        """
        Verarbeitet eine PDF-Rechnung mit Docling + optional Gemini

        Args:
            pdf_path: Pfad zur PDF-Datei
            gemini_extractor: Optional - GeminiExtractor für AI-basierte Extraktion

        Returns:
            DoclingInvoiceResult: Vollständige Extraktionsergebnisse
        """
        logger.info(f"Starte Docling-Verarbeitung: {pdf_path}")
        start_time = datetime.now()

        try:
            # Schritt 1: Docling Strukturanalyse
            docling_result = self._process_with_docling(pdf_path)

            # Schritt 2: Tabellendaten extrahieren
            table_data = self._extract_table_data(docling_result.document)

            # Schritt 3: Text-Chunks für AI-Verarbeitung
            text_chunks = self._create_text_chunks(docling_result.document)

            # Schritt 4: Optional - Gemini AI-Extraktion
            ai_data = {}
            if gemini_extractor:
                ai_data = self._process_with_gemini(text_chunks, gemini_extractor)

            # Schritt 5: Daten kombinieren und strukturieren
            result = self._combine_results(
                docling_result=docling_result,
                table_data=table_data,
                ai_data=ai_data,
                processing_time=(datetime.now() - start_time).total_seconds() * 1000,
            )

            logger.info(f"Docling-Verarbeitung erfolgreich: {result.invoice_number}")
            return result

        except Exception as e:
            logger.error(f"Fehler bei Docling-Verarbeitung: {e}")
            raise

    def _process_with_docling(self, pdf_path: Path) -> DoclingInvoiceResult:
        """Führt die Docling-Dokumentenkonvertierung durch"""

        logger.info("Starte Docling-Dokumentenkonvertierung...")

        # Dateigröße prüfen
        if pdf_path.stat().st_size > self.max_file_size:
            raise ValueError(
                f"PDF zu groß: {pdf_path.stat().st_size} > {self.max_file_size}"
            )

        # Konvertierung mit Limits
        conv_result = self.converter.convert(
            source=str(pdf_path),
            max_num_pages=self.max_pages,
            max_file_size=self.max_file_size,
        )

        if conv_result.status.name != "SUCCESS":
            logger.warning(f"Docling Conversion Status: {conv_result.status}")

        # Erstelle DoclingInvoiceResult aus ConversionResult
        return DoclingInvoiceResult(
            raw_text=conv_result.document.export_to_markdown(),
            processing_time_ms=int((time.time() - start_time) * 1000),
            structured_content={"document": conv_result.document},
            tables_found=(
                len(conv_result.document.tables)
                if hasattr(conv_result.document, "tables")
                else 0
            ),
        )

    def _extract_table_data(self, docling_document) -> list[dict[str, Any]]:
        """Extrahiert strukturierte Tabellendaten aus Docling-Dokument"""

        tables = []

        try:
            # Durchsuche Dokument nach Tabellen
            for item in docling_document.body:
                if hasattr(item, "label") and item.label == "table":
                    table_data = self._parse_docling_table(item)
                    if table_data:
                        tables.append(table_data)

            logger.info(f"Gefunden: {len(tables)} Tabellen in Dokument")

        except Exception as e:
            logger.error(f"Fehler bei Tabellenextraktion: {e}")

        return tables

    def _parse_docling_table(self, table_item) -> dict[str, Any] | None:
        """Parst eine einzelne Docling-Tabelle zu strukturierten Daten"""

        try:
            if not hasattr(table_item, "data") or not table_item.data:
                return None

            table_data = {
                "rows": [],
                "headers": [],
                "num_rows": 0,
                "num_cols": 0,
                "table_type": "invoice_positions",  # Annahme für Rechnungen
            }

            # Konvertiere Tabellendaten
            if hasattr(table_item.data, "table"):
                table = table_item.data.table

                # Headers extrahieren
                if table.grid and len(table.grid) > 0:
                    table_data["headers"] = [cell.text for cell in table.grid[0]]
                    table_data["num_cols"] = len(table.grid[0])

                    # Datenzeilen extrahieren
                    for row_idx, row in enumerate(table.grid[1:]):
                        row_data = [cell.text for cell in row]
                        table_data["rows"].append(
                            {
                                "row_index": row_idx,
                                "cells": row_data,
                                "is_header": False,
                            }
                        )

                    table_data["num_rows"] = len(table.grid) - 1

            return table_data

        except Exception as e:
            logger.error(f"Fehler beim Tabellen-Parsing: {e}")
            return None

    def _create_text_chunks(self, docling_document) -> list[str]:
        """Erstellt Text-Chunks für AI-Verarbeitung"""

        try:
            # Verwende HybridChunker für optimale Chunk-Größen
            chunks = []
            for chunk in self.chunker.chunk(docling_document):
                if chunk.text.strip():
                    chunks.append(chunk.text)

            logger.info(f"Erstellt: {len(chunks)} Text-Chunks")
            return chunks

        except Exception as e:
            logger.error(f"Fehler beim Chunking: {e}")
            # Fallback: Volltext
            return [docling_document.export_to_markdown()]

    def _process_with_gemini(
        self, text_chunks: list[str], gemini_extractor
    ) -> dict[str, Any]:
        """Verarbeitet Text-Chunks mit Gemini AI"""

        try:
            # Kombiniere Chunks für Gemini-Verarbeitung
            combined_text = "\n\n".join(text_chunks)

            logger.info("Starte Gemini AI-Extraktion...")
            gemini_result = gemini_extractor.extract_from_text(combined_text)

            return {
                "invoice_number": gemini_result.invoice_number,
                "invoice_date": gemini_result.invoice_date,
                "supplier": gemini_result.supplier,
                "supplier_number": gemini_result.supplier_number,
                "order_number": gemini_result.order_number,
                "net_amount": gemini_result.net_amount,
                "vat_amount": gemini_result.vat_amount,
                "gross_amount": gemini_result.gross_amount,
                "currency": gemini_result.currency,
                "line_items": gemini_result.line_items,
                "confidence_score": gemini_result.confidence_score,
            }

        except Exception as e:
            logger.error(f"Fehler bei Gemini-Verarbeitung: {e}")
            return {}

    def _combine_results(
        self,
        docling_result,
        table_data: list[dict[str, Any]],
        ai_data: dict[str, Any],
        processing_time: float,
    ) -> DoclingInvoiceResult:
        """Kombiniert Docling- und AI-Ergebnisse zu finalem Result"""

        # Basis-Ergebnis aus AI-Daten oder Fallback
        result = DoclingInvoiceResult(
            invoice_number=ai_data.get("invoice_number", "UNKNOWN"),
            invoice_date=ai_data.get("invoice_date", ""),
            supplier=ai_data.get("supplier", ""),
            supplier_number=ai_data.get("supplier_number"),
            order_number=ai_data.get("order_number"),
            net_amount=ai_data.get("net_amount", 0.0),
            vat_amount=ai_data.get("vat_amount", 0.0),
            gross_amount=ai_data.get("gross_amount", 0.0),
            currency=ai_data.get("currency", "EUR"),
            processing_time_ms=int(processing_time),
            tables_found=len(table_data),
            docling_confidence=ai_data.get("confidence_score", 0.0),
            raw_text=docling_result.document.export_to_markdown(),
        )

        # Line Items aus Tabellen und AI kombinieren
        result.line_items = self._merge_line_items(
            table_data, ai_data.get("line_items", [])
        )

        # Dokument-Struktur für Debugging
        result.document_structure = {
            "docling_status": docling_result.status.name,
            "num_pages": (
                len(docling_result.document.pages)
                if hasattr(docling_result.document, "pages")
                else 0
            ),
            "tables_detected": len(table_data),
            "processing_pipeline": "docling + gemini" if ai_data else "docling_only",
        }

        return result

    def _merge_line_items(
        self, table_data: list[dict[str, Any]], ai_items: list[dict[str, Any]]
    ) -> list[DoclingInvoiceItem]:
        """Kombiniert Tabellendaten mit AI-extrahierten Line Items"""

        items = []

        try:
            # Priorität: Strukturierte Tabellendaten
            for table in table_data:
                table_items = self._convert_table_to_items(table)
                items.extend(table_items)

            # Falls keine Tabellen gefunden: AI-Daten verwenden
            if not items and ai_items:
                for ai_item in ai_items:
                    item = DoclingInvoiceItem(
                        description=ai_item.get("description", ""),
                        article_number=ai_item.get("article_number"),
                        quantity=ai_item.get("quantity", 0.0),
                        unit=ai_item.get("unit", "Stk"),
                        unit_price=ai_item.get("unit_price", 0.0),
                        total_price=ai_item.get("total_price", 0.0),
                        vat_rate=ai_item.get("vat_rate", 0.19),
                        category=ai_item.get("category", "elektromaterial"),
                        table_source=False,
                        confidence=ai_item.get("confidence", 0.0),
                    )
                    # SKR03-Klassifikation
                    item.skr03_account = self._classify_skr03(item.category)
                    items.append(item)

            logger.info(f"Finalisiert: {len(items)} Rechnungspositionen")

        except Exception as e:
            logger.error(f"Fehler beim Line-Item Merge: {e}")

        return items

    def _convert_table_to_items(
        self, table: dict[str, Any]
    ) -> list[DoclingInvoiceItem]:
        """Konvertiert Tabellendaten zu DoclingInvoiceItem-Objekten"""

        items = []
        headers = table.get("headers", [])

        # Header-Mapping für deutsche Rechnungen
        header_mapping = self._create_header_mapping(headers)

        for row in table.get("rows", []):
            if row.get("is_header", False):
                continue

            cells = row.get("cells", [])
            if len(cells) < 2:  # Mindestens Beschreibung + Preis
                continue

            item = self._extract_item_from_row(cells, header_mapping)
            if item:
                item.table_source = True
                item.confidence = 0.9  # Hohe Confidence für Tabellendaten
                items.append(item)

        return items

    def _create_header_mapping(self, headers: list[str]) -> dict[str, int]:
        """Erstellt Mapping von deutschen Spaltennamen zu Indizes"""

        mapping = {}

        for idx, header in enumerate(headers):
            header_lower = header.lower().strip()

            # Deutsche Rechnungsfelder
            if any(
                term in header_lower for term in ["pos", "position", "nr", "nummer"]
            ):
                mapping["position"] = idx
            elif any(
                term in header_lower
                for term in ["artikel", "beschreibung", "bezeichnung"]
            ):
                mapping["description"] = idx
            elif any(
                term in header_lower for term in ["art", "artikel", "produktnummer"]
            ):
                mapping["article_number"] = idx
            elif any(term in header_lower for term in ["menge", "anzahl", "stück"]):
                mapping["quantity"] = idx
            elif any(term in header_lower for term in ["einheit", "me", "uom"]):
                mapping["unit"] = idx
            elif any(term in header_lower for term in ["einzelpreis", "ep", "preis"]):
                mapping["unit_price"] = idx
            elif any(term in header_lower for term in ["gesamt", "summe", "betrag"]):
                mapping["total_price"] = idx
            elif any(term in header_lower for term in ["mwst", "ust", "steuer"]):
                mapping["vat_rate"] = idx

        return mapping

    def _extract_item_from_row(
        self, cells: list[str], header_mapping: dict[str, int]
    ) -> DoclingInvoiceItem | None:
        """Extrahiert ein DoclingInvoiceItem aus einer Tabellenzeile"""

        try:
            item = DoclingInvoiceItem()

            # Position
            if "position" in header_mapping:
                item.position_number = cells[header_mapping["position"]].strip()

            # Beschreibung (Required)
            if "description" in header_mapping:
                item.description = cells[header_mapping["description"]].strip()
            elif len(cells) > 0:
                item.description = cells[0].strip()  # Fallback: erste Spalte

            if not item.description:
                return None  # Keine gültige Position ohne Beschreibung

            # Artikelnummer
            if "article_number" in header_mapping:
                item.article_number = cells[header_mapping["article_number"]].strip()

            # Menge
            if "quantity" in header_mapping:
                item.quantity = self._parse_number(cells[header_mapping["quantity"]])

            # Einheit
            if "unit" in header_mapping:
                item.unit = cells[header_mapping["unit"]].strip() or "Stk"

            # Einzelpreis
            if "unit_price" in header_mapping:
                item.unit_price = self._parse_number(
                    cells[header_mapping["unit_price"]]
                )

            # Gesamtpreis
            if "total_price" in header_mapping:
                item.total_price = self._parse_number(
                    cells[header_mapping["total_price"]]
                )

            # MwSt-Satz
            if "vat_rate" in header_mapping:
                vat_text = cells[header_mapping["vat_rate"]].strip()
                if "19" in vat_text:
                    item.vat_rate = 0.19
                elif "7" in vat_text:
                    item.vat_rate = 0.07

            # Kategorie klassifizieren
            item.category = self._classify_category(item.description)
            item.skr03_account = self._classify_skr03(item.category)

            return item

        except Exception as e:
            logger.error(f"Fehler beim Item-Parsing: {e}")
            return None

    def _parse_number(self, text: str) -> float:
        """Parst deutsche Zahlenformate zu float"""

        if not text or not isinstance(text, str):
            return 0.0

        try:
            # Deutsche Zahlenformate: 1.234,56 -> 1234.56
            text = text.replace("€", "").replace("EUR", "").strip()
            text = text.replace(".", "").replace(",", ".")

            # Extrahiere nur Zahlen und Dezimalpunkt
            import re

            number_match = re.search(r"\d+\.?\d*", text)
            if number_match:
                return float(number_match.group())

        except ValueError:
            pass

        return 0.0

    def _classify_category(self, description: str) -> str:
        """Klassifiziert Produktkategorie aus Beschreibung"""

        desc_lower = description.lower()

        if any(term in desc_lower for term in ["kabel", "leitung", "draht"]):
            return "elektromaterial"
        elif any(term in desc_lower for term in ["schalter", "steckdose", "dose"]):
            return "elektromaterial"
        elif any(term in desc_lower for term in ["lampe", "leuchte", "led"]):
            return "elektromaterial"
        elif any(term in desc_lower for term in ["sicherung", "automat", "fi"]):
            return "elektromaterial"
        elif any(term in desc_lower for term in ["werkzeug", "bohrer", "zange"]):
            return "werkzeuge"
        elif any(term in desc_lower for term in ["maschine", "gerät", "bohrmaschine"]):
            return "maschinen"
        elif any(term in desc_lower for term in ["montage", "installation", "service"]):
            return "fremdleistung"
        else:
            return "elektromaterial"  # Default

    def _classify_skr03(self, category: str) -> str:
        """SKR03-Kontenzuordnung basierend auf Kategorie"""

        skr03_mapping = {
            "elektromaterial": "3400",  # Wareneingang 19% Vorsteuer
            "werkzeuge": "4985",  # Werkzeuge und Kleingeräte
            "maschinen": "0210",  # Maschinen (über 800€)
            "fremdleistung": "4400",  # Fremdleistungen
            "bueroartikel": "0420",  # Büroeinrichtung
        }

        return skr03_mapping.get(category, "3400")

    def batch_process(
        self, pdf_directory: Path, output_directory: Path, gemini_extractor=None
    ) -> list[DoclingInvoiceResult]:
        """
        Verarbeitet mehrere PDF-Rechnungen in einem Batch

        Args:
            pdf_directory: Verzeichnis mit PDF-Dateien
            output_directory: Ausgabeverzeichnis für Ergebnisse
            gemini_extractor: Optional - GeminiExtractor

        Returns:
            List[DoclingInvoiceResult]: Verarbeitete Rechnungen
        """
        logger.info(f"Starte Batch-Verarbeitung: {pdf_directory}")

        pdf_files = list(pdf_directory.glob("*.pdf"))
        results = []

        output_directory.mkdir(parents=True, exist_ok=True)

        for pdf_file in pdf_files:
            try:
                logger.info(f"Verarbeite: {pdf_file.name}")

                result = self.process_invoice(pdf_file, gemini_extractor)
                results.append(result)

                # Einzelergebnis speichern
                output_file = output_directory / f"{pdf_file.stem}_docling.json"
                self._save_result(result, output_file)

                logger.info(f"✅ Erfolgreich: {result.invoice_number}")

            except Exception as e:
                logger.error(f"❌ Fehler bei {pdf_file.name}: {e}")
                continue

        # Batch-Zusammenfassung
        summary_file = output_directory / "batch_summary_docling.json"
        self._save_batch_summary(results, summary_file)

        logger.info(
            f"Batch-Verarbeitung abgeschlossen: {len(results)}/{len(pdf_files)} erfolgreich"
        )
        return results

    def _save_result(self, result: DoclingInvoiceResult, output_file: Path) -> None:
        """Speichert Einzelergebnis als JSON"""

        try:
            result_dict = {
                "invoice_number": result.invoice_number,
                "invoice_date": result.invoice_date,
                "supplier": result.supplier,
                "supplier_number": result.supplier_number,
                "order_number": result.order_number,
                "net_amount": result.net_amount,
                "vat_amount": result.vat_amount,
                "gross_amount": result.gross_amount,
                "currency": result.currency,
                "line_items": [
                    {
                        "position_number": item.position_number,
                        "description": item.description,
                        "article_number": item.article_number,
                        "quantity": item.quantity,
                        "unit": item.unit,
                        "unit_price": item.unit_price,
                        "total_price": item.total_price,
                        "vat_rate": item.vat_rate,
                        "category": item.category,
                        "skr03_account": item.skr03_account,
                        "table_source": item.table_source,
                        "confidence": item.confidence,
                    }
                    for item in result.line_items
                ],
                "document_structure": result.document_structure,
                "tables_found": result.tables_found,
                "processing_time_ms": result.processing_time_ms,
                "docling_confidence": result.docling_confidence,
                "processed_at": datetime.now().isoformat(),
            }

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result_dict, f, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error(f"Fehler beim Speichern: {e}")

    def _save_batch_summary(
        self, results: list[DoclingInvoiceResult], summary_file: Path
    ):
        """Speichert Batch-Zusammenfassung"""

        try:
            summary = {
                "batch_info": {
                    "processed_at": datetime.now().isoformat(),
                    "total_invoices": len(results),
                    "total_amount": sum(r.gross_amount for r in results),
                    "average_confidence": (
                        sum(r.docling_confidence for r in results) / len(results)
                        if results
                        else 0
                    ),
                    "total_line_items": sum(len(r.line_items) for r in results),
                    "total_tables_found": sum(r.tables_found for r in results),
                },
                "invoice_summary": [
                    {
                        "invoice_number": r.invoice_number,
                        "supplier": r.supplier,
                        "gross_amount": r.gross_amount,
                        "line_items_count": len(r.line_items),
                        "tables_found": r.tables_found,
                        "confidence": r.docling_confidence,
                        "processing_time_ms": r.processing_time_ms,
                    }
                    for r in results
                ],
            }

            with open(summary_file, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error(f"Fehler beim Speichern der Zusammenfassung: {e}")


# Factory Functions
def create_docling_processor(**kwargs) -> DoclingInvoiceProcessor:
    """Factory Function für DoclingInvoiceProcessor"""
    return DoclingInvoiceProcessor(**kwargs)


def create_accurate_processor() -> DoclingInvoiceProcessor:
    """Erstellt Processor mit höchster Genauigkeit"""
    return DoclingInvoiceProcessor(
        table_mode=TableFormerMode.ACCURATE, enable_table_structure=True
    )


def create_fast_processor() -> DoclingInvoiceProcessor:
    """Erstellt Processor für schnelle Verarbeitung"""
    return DoclingInvoiceProcessor(
        table_mode=TableFormerMode.FAST, enable_table_structure=True
    )


if __name__ == "__main__":
    # Test des Docling Invoice Processors
    from pathlib import Path

    # Setup Logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Test-Verarbeitung
    processor = create_accurate_processor()

    # Teste mit Beispiel-PDF (falls vorhanden)
    test_pdf = Path("data/input/pdfs/test_rechnung.pdf")

    if test_pdf.exists():
        try:
            result = processor.process_invoice(test_pdf)
            print("\n✅ Test erfolgreich!")
            print(f"Rechnung: {result.invoice_number}")
            print(f"Lieferant: {result.supplier}")
            print(f"Betrag: {result.gross_amount} {result.currency}")
            print(f"Positionen: {len(result.line_items)}")
            print(f"Tabellen: {result.tables_found}")
            print(f"Verarbeitungszeit: {result.processing_time_ms}ms")

        except Exception as e:
            print(f"❌ Test-Fehler: {e}")
    else:
        print("ℹ️  Kein Test-PDF gefunden. Processor bereit für Verwendung.")
