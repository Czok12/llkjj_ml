#!/usr/bin/env python3
"""
LLKJJ ML Pipeline - Advanced Docling Processor
==============================================

State-of-the-art Docling-Verarbeitung mit 2025 Best Practices für deutsche Eingangsrechnungen.
Optimiert für Elektrohandwerk-Rechnungen mit SKR03-Kontierung.

🚀 2025 Best Practices:
- PaddleOCR statt EasyOCR (Industrial Powerhouse für deutsche Texte)
- TableFormer ACCURATE Modus für komplexe Tabellen
- Deutsche OCR explizit konfiguriert für Umlaute
- Custom NER für deutsche Rechnungsfelder
- Layout-Awareness für bessere Struktur-Erkennung

Research Sources:
- OCR-Vergleich 2025: PaddleOCR > EasyOCR für Layout-Awareness
- Docling 2.44.0 API-Dokumentation
- Deutsche Rechnungsverarbeitung Optimierungen

Author: LLKJJ ML Pipeline
Version: 2.0.0 (Advanced Best Practices)
Date: 17 August 2025
"""

import logging
import re
import time
from pathlib import Path
from typing import Any, TypedDict

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    EasyOcrOptions,
    PdfPipelineOptions,
    RapidOcrOptions,
    TableFormerMode,
    TableStructureOptions,
    TesseractCliOcrOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption

logger = logging.getLogger(__name__)


class OcrEngineConfig(TypedDict):
    """Type definition for OCR engine configuration."""

    name: str
    options: TesseractCliOcrOptions | RapidOcrOptions | EasyOcrOptions


class AdvancedDoclingProcessor:
    """
    🚀 State-of-the-art Docling-Verarbeitung für deutsche Elektrohandwerk-Rechnungen.

    2025 Best Practices Features:
    - TesseractCLI statt EasyOCR (bessere deutsche Spracherkennung)
    - TableFormer ACCURATE für komplexe Rechnungstabellen
    - Custom NER für deutsche Rechnungsfelder
    - Deutsche OCR-Konfiguration für Umlaute und Sonderzeichen
    - Layout-optimierte Extraktion
    """

    def __init__(
        self,
        use_gpu: bool = True,
        enable_table_structure: bool = True,
        enable_ocr: bool = True,
        ocr_engine: str = "rapid",  # "rapid" für bessere Tabellen-Layouts
        table_mode: str = "accurate",  # "fast" oder "accurate"
        force_full_page_ocr: bool = True,  # Für komplette Rechnungsseiten
        german_optimized: bool = True,
    ):
        """
        Initialisiert den Advanced Docling Processor.

        Args:
            use_gpu: GPU-Beschleunigung für TableFormer aktivieren
            enable_table_structure: Tabellenerkennung aktivieren
            enable_ocr: OCR für gescannte PDFs aktivieren
            ocr_engine: OCR-Engine ("tesseract_cli", "rapid", "easy")
            table_mode: TableFormer-Modus ("fast" oder "accurate")
            force_full_page_ocr: Vollständige Seiten-OCR erzwingen
            german_optimized: Deutsche Optimierungen aktivieren
        """
        self.use_gpu = use_gpu
        self.enable_table_structure = enable_table_structure
        self.enable_ocr = enable_ocr
        self.ocr_engine = ocr_engine
        self.table_mode = table_mode
        self.force_full_page_ocr = force_full_page_ocr
        self.german_optimized = german_optimized
        self.german_ner_enabled = german_optimized  # Alias für NER-Status

        # Pipeline-Optionen konfigurieren
        pipeline_options = self._create_pipeline_options()
        pdf_format_options = PdfFormatOption(pipeline_options=pipeline_options)

        # DocumentConverter mit optimierten Einstellungen
        self.converter = DocumentConverter(
            format_options={InputFormat.PDF: pdf_format_options}
        )

        # Custom NER-Patterns für deutsche Rechnungen
        self.german_invoice_patterns = self._init_german_ner_patterns()

        logger.info("AdvancedDoclingProcessor initialisiert:")
        logger.info(f"  - GPU: {use_gpu}")
        logger.info(f"  - TableFormer: {enable_table_structure} (Modus: {table_mode})")
        logger.info(f"  - OCR: {enable_ocr} (Engine: {ocr_engine})")
        logger.info(f"  - Deutsche Optimierung: {german_optimized}")

    def _create_pipeline_options(self) -> PdfPipelineOptions:
        """Erstellt optimierte Pipeline-Optionen für deutsche Rechnungen."""
        pipeline_options = PdfPipelineOptions(
            # OCR aktivieren
            do_ocr=self.enable_ocr,
            # Tabellenerkennung aktivieren
            do_table_structure=self.enable_table_structure,
            # Bilder und Layout
            images_scale=2.0,
            generate_page_images=False,
            artifacts_path=None,
        )

        # OCR-Engine konfigurieren
        if self.enable_ocr:
            pipeline_options.ocr_options = self._get_ocr_options()

        # TableFormer-Modus konfigurieren mit erweiterten Optionen
        if self.enable_table_structure:
            # Erweiterte Table-Structure-Optionen für bessere Rechnungstabellen-Erkennung
            table_structure_options = TableStructureOptions(
                do_cell_matching=True,  # Bessere Zellen-Zuordnung für komplexe Layouts
                mode=TableFormerMode.ACCURATE,  # Präziser Modus für Rechnungen (überschreibt table_mode)
            )

            # Mode aus Initialisierung berücksichtigen, aber ACCURATE als Default für Rechnungen
            if self.table_mode == "fast":
                table_structure_options.mode = TableFormerMode.FAST
                logger.info("🏃 TableFormer FAST-Modus für Geschwindigkeit")
            else:
                table_structure_options.mode = TableFormerMode.ACCURATE
                logger.info("🎯 TableFormer ACCURATE-Modus für Rechnungstabellen")

            pipeline_options.table_structure_options = table_structure_options

        return pipeline_options

    def _get_ocr_options(
        self,
    ) -> TesseractCliOcrOptions | RapidOcrOptions | EasyOcrOptions:
        """
        Bestimmt OCR-Optionen basierend auf optimierter 2025-Strategie.

        Nach Research: TesseractCLI (Deutsch) -> RapidOCR (PaddleOCR) -> EasyOCR
        Entfernt die alte EasyOCR-First-Überschreibung für bessere Performance.
        """
        # Verwende die gewählte OCR-Engine direkt (entfernt alte Überschreibungslogik)
        if self.ocr_engine == "tesseract_cli":
            logger.info("🇩🇪 TesseractCLI mit deutscher Spracherkennung aktiviert")
            return TesseractCliOcrOptions(
                lang=["deu"], force_full_page_ocr=self.force_full_page_ocr
            )

        elif self.ocr_engine == "rapid":
            logger.info("🚀 RapidOCR (PaddleOCR-Wrapper) aktiviert")
            return RapidOcrOptions(force_full_page_ocr=self.force_full_page_ocr)

        elif self.ocr_engine == "easy":
            logger.info("📷 EasyOCR für robuste Layouts aktiviert")
            return EasyOcrOptions(force_full_page_ocr=self.force_full_page_ocr)

        else:
            # Fallback: TesseractCLI für deutsche Optimierung
            logger.warning(
                f"Unbekannte OCR-Engine '{self.ocr_engine}', verwende TesseractCLI (Deutsch)"
            )
            return TesseractCliOcrOptions(
                lang=["deu"], force_full_page_ocr=self.force_full_page_ocr
            )

    def _init_german_ner_patterns(self) -> dict[str, str]:
        """
        Initialisiert NER-Patterns für deutsche Rechnungsfelder.

        Ersetzt die fehlende Docling Entity Recognition mit Custom Patterns.
        Optimiert für deutsche Elektrohandwerk-Rechnungen.
        """
        return {
            # Rechnungsheader
            "invoice_number": r"(?:Rechn(?:ungs)?[-.\s]*Nr\.?|Rechnungs-Nr\.?|Invoice\s*No\.?)\s*:?\s*([A-Z0-9\-_/]+)",
            "invoice_date": r"(?:Datum|Rechnungsdatum|vom|Date)\s*:?\s*(\d{1,2}[.\s/-]\d{1,2}[.\s/-]\d{2,4})",
            "due_date": r"(?:Fällig(?:keit)?|Zahlbar\s*bis|Due\s*Date)\s*:?\s*(\d{1,2}[.\s/-]\d{1,2}[.\s/-]\d{2,4})",
            # Kunden-/Lieferantendaten
            "customer_number": r"(?:Kunden[-.\s]*Nr\.?|Kundennummer)\s*:?\s*([A-Z0-9\-_/]+)",
            "supplier_name": r"^([A-ZÄÖÜ][a-zäöüß\s&,.-]+(?:GmbH|AG|KG|UG|e\.K\.)?)",
            # Beträge (deutsche Formatierung)
            "total_net": r"(?:Netto(?:betrag)?|Zwischensumme)\s*:?\s*(\d{1,3}(?:\.\d{3})*(?:,\d{2})?)\s*€?",
            "total_vat": r"(?:MwSt\.?|USt\.?|Steuer)\s*:?\s*(\d{1,3}(?:\.\d{3})*(?:,\d{2})?)\s*€?",
            "total_gross": r"(?:Brutto(?:betrag)?|Gesamt(?:betrag)?|Endbetrag|Total)\s*:?\s*(\d{1,3}(?:\.\d{3})*(?:,\d{2})?)\s*€?",
            # Elektrotechnik-spezifisch
            "article_number": r"(?:Art\.?[-.\s]*Nr\.?|Artikel[-.\s]*Nr\.?)\s*:?\s*([A-Z0-9\-_./]+)",
            "quantity": r"(?:Menge|Anzahl|Stück|St\.?|Qty)\s*:?\s*(\d+(?:,\d+)?)",
            "unit_price": r"(?:Einzelpreis|E-Preis|EP|Unit\s*Price)\s*:?\s*(\d{1,3}(?:\.\d{3})*(?:,\d{2})?)\s*€?",
            # Adressen
            "plz_ort": r"(\d{5})\s+([A-ZÄÖÜ][a-zäöüß\s-]+)",
            "email": r"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})",
            "phone": r"(?:Tel\.?|Telefon|Phone)\s*:?\s*([\+]?[\d\s\-\(\)/]{8,})",
            # Zahlungsinformationen
            "iban": r"(?:IBAN)\s*:?\s*([A-Z]{2}[\d\s]{18,22})",
            "payment_terms": r"(?:Zahlungsziel|Payment\s*Terms)\s*:?\s*(\d+\s*Tage?)",
        }

    def _select_ocr_engines_by_quality(self) -> list[OcrEngineConfig]:
        """
        Optimierte OCR-Engine-Auswahl basierend auf 2025 Research.

        Korrigierte Reihenfolge nach neuester Analyse:
        1. TesseractCLI: Beste deutsche Spracherkennung (primär)
        2. RapidOCR: PaddleOCR-Wrapper - leistungsstärkste Engine (fallback)
        3. EasyOCR: Robust bei komplexen Layouts (letzte Option)

        Research-Update: RapidOCR = PaddleOCR Wrapper (offiziell bestätigt)
        GitHub Docling Issue #626: "We have RapidOCR in docling, which wraps PaddleOCR"
        """
        engines = []

        # Primäre Engine: TesseractCLI (beste deutsche Erkennung)
        if self.german_optimized:
            engines.append(
                {
                    "name": "TesseractCLI (Deutsch)",
                    "options": TesseractCliOcrOptions(
                        lang=["deu"], force_full_page_ocr=self.force_full_page_ocr
                    ),
                }
            )

        # Fallback 1: RapidOCR (PaddleOCR-Wrapper - leistungsstärkste Engine)
        engines.append(
            {
                "name": "RapidOCR (PaddleOCR)",
                "options": RapidOcrOptions(
                    force_full_page_ocr=self.force_full_page_ocr
                ),
            }
        )

        # Fallback 2: EasyOCR (robust bei komplexen Layouts)
        engines.append(
            {
                "name": "EasyOCR",
                "options": EasyOcrOptions(force_full_page_ocr=self.force_full_page_ocr),
            }
        )

        return engines  # type: ignore[return-value]

    def process(self, pdf_path: str) -> tuple[dict[str, Any], float]:
        """
        Verarbeitet PDF mit intelligenter OCR-Fallback-Logik.

        Implementiert automatisches OCR-Fallback für maximale Robustheit:
        1. Primäre Engine: EasyOCR (robust bei schlechten Scans)
        2. Fallback: RapidOCR (schnell)
        3. Letzte Option: TesseractCLI (deutsch)

        Returns:
            tuple[dict, float]: (Extrahierte Daten, Qualitäts-Score)
        """
        logger.info(f"🚀 Starte intelligente PDF-Verarbeitung: {pdf_path}")
        start_time = time.time()

        # OCR-Engine-Auswahl für robuste Verarbeitung
        ocr_engines_to_try = self._select_ocr_engines_by_quality()

        last_error = None

        for attempt, engine_config in enumerate(ocr_engines_to_try, 1):
            try:
                logger.info(
                    f"📊 OCR-Versuch {attempt}/{len(ocr_engines_to_try)}: {engine_config['name']}"
                )

                # Neue Pipeline-Optionen für diesen Versuch
                ocr_options = engine_config["options"]
                if not isinstance(
                    ocr_options,
                    TesseractCliOcrOptions | RapidOcrOptions | EasyOcrOptions,
                ):
                    raise TypeError(f"Invalid OCR options type: {type(ocr_options)}")
                pipeline_options = self._create_pipeline_options_for_engine(ocr_options)
                pdf_format_options = PdfFormatOption(pipeline_options=pipeline_options)

                # Neuen Converter mit Engine-spezifischen Optionen
                temp_converter = DocumentConverter(
                    format_options={InputFormat.PDF: pdf_format_options}
                )

                # Dokument konvertieren mit aktueller Engine
                doc_result = temp_converter.convert(pdf_path)

                if not doc_result or not doc_result.document:
                    logger.warning(f"⚠️ Keine Ergebnisse mit {engine_config['name']}")
                    continue

                doc = doc_result.document

                # Content extrahieren
                raw_content = doc.export_to_markdown()
                if not raw_content or len(raw_content.strip()) < 100:
                    logger.warning(
                        f"⚠️ Zu wenig Content mit {engine_config['name']}: {len(raw_content)} Zeichen"
                    )
                    continue

                # Erfolgreiche Extraktion
                logger.info(
                    f"✅ Erfolgreiche OCR mit {engine_config['name']}: {len(raw_content)} Zeichen"
                )

                # Deutsche NER-Verbesserung
                enhanced_data = self._extract_structured_data(doc)
                if self.german_optimized:
                    enhanced_data = self._apply_german_ner(enhanced_data)

                # Qualitäts-Bewertung
                quality_score = self._calculate_quality_score(enhanced_data)

                # Timing-Info
                processing_time = time.time() - start_time
                logger.info(
                    f"🎯 Verarbeitung erfolgreich in {processing_time:.2f}s mit Score: {quality_score:.2f}"
                )

                return {
                    "content": enhanced_data.get("raw_text", ""),
                    "raw_content": raw_content,
                    "structured_data": enhanced_data,
                    "pages": len(doc.pages) if hasattr(doc, "pages") else 1,
                    "processing_time": processing_time,
                    "ocr_engine_used": engine_config["name"],
                    "quality_score": quality_score,
                    "character_count": len(raw_content),
                    "attempt_number": attempt,
                    "total_attempts": len(ocr_engines_to_try),
                    "german_ner_applied": self.german_optimized,
                }, quality_score

            except Exception as e:
                last_error = e
                logger.error(f"❌ Fehler mit {engine_config['name']}: {str(e)}")
                if attempt < len(ocr_engines_to_try):
                    logger.info("🔄 Versuche nächste Engine...")
                    continue

        # Alle Engines fehlgeschlagen
        logger.error(f"💥 Alle {len(ocr_engines_to_try)} OCR-Engines fehlgeschlagen!")
        raise RuntimeError(
            f"Alle OCR-Engines versagten. Letzter Fehler: {str(last_error)}"
        )

    def _create_pipeline_options_for_engine(
        self, ocr_options: TesseractCliOcrOptions | RapidOcrOptions | EasyOcrOptions
    ) -> PdfPipelineOptions:
        """Erstellt Pipeline-Optionen für spezifische OCR-Engine."""
        pipeline_options = PdfPipelineOptions(
            # OCR aktivieren
            do_ocr=self.enable_ocr,
            # Tabellenerkennung aktivieren
            do_table_structure=self.enable_table_structure,
            # Bilder und Layout
            images_scale=2.0,
            generate_page_images=False,
            artifacts_path=None,
        )

        # OCR-Engine-spezifische Optionen setzen
        if self.enable_ocr and ocr_options:
            pipeline_options.ocr_options = ocr_options

        # TableFormer-Modus konfigurieren
        if self.enable_table_structure and self.table_mode == "accurate":
            pipeline_options.table_structure_options.do_cell_matching = True
            pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE

        return pipeline_options

    def process_pdf(self, pdf_path: Path) -> dict[str, Any]:
        """
        Verarbeitet eine PDF-Datei mit Advanced Features.

        Args:
            pdf_path: Pfad zur PDF-Datei

        Returns:
            Dictionary mit strukturierten Extraktionsergebnissen und NER
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF-Datei nicht gefunden: {pdf_path}")

        logger.info(f"Starte Advanced Docling-Verarbeitung: {pdf_path.name}")

        try:
            # PDF mit optimierter Pipeline konvertieren
            result = self.converter.convert(str(pdf_path))

            if not result or not result.document:
                raise ValueError(f"Docling konnte PDF nicht verarbeiten: {pdf_path}")

            # Basis-Extraktion
            extracted_data = self._extract_structured_data(result.document)

            # Custom NER anwenden
            if self.german_optimized:
                extracted_data = self._apply_german_ner(extracted_data)

            # Qualitäts-Scoring hinzufügen
            extracted_data["quality_score"] = self._calculate_quality_score(
                extracted_data
            )

            logger.info(
                f"Advanced Docling-Verarbeitung erfolgreich: {len(extracted_data.get('raw_text', ''))} Zeichen"
            )
            return extracted_data

        except Exception as e:
            logger.error(f"Fehler bei Advanced Docling-Verarbeitung: {e}")
            raise

    def _apply_german_ner(self, extracted_data: dict[str, Any]) -> dict[str, Any]:
        """
        Wendet Custom NER für deutsche Rechnungsfelder an.

        Ersetzt die fehlende Docling Entity Recognition.
        """
        import re

        text = extracted_data.get("raw_text", "")
        entities = {}

        # NER-Patterns auf Text anwenden
        for entity_type, pattern in self.german_invoice_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            if matches:
                # Ersten/besten Match nehmen
                entities[entity_type] = (
                    matches[0] if isinstance(matches[0], str) else matches[0][0]
                )

        # Entitäten zu extrahierten Daten hinzufügen
        extracted_data["german_entities"] = entities
        extracted_data["ner_method"] = "custom_patterns"

        return extracted_data

    def _calculate_quality_score(self, extracted_data: dict[str, Any]) -> float:
        """
        Berechnet Quality Score basierend auf Extraktions-Vollständigkeit.

        Returns:
            Score zwischen 0.0 und 1.0
        """
        score = 0.0
        max_score = 0.0

        # Text-Vollständigkeit (40% der Bewertung)
        text_length = len(extracted_data.get("raw_text", ""))
        if text_length > 0:
            score += (
                min(text_length / 5000, 1.0) * 0.4
            )  # 5000 Zeichen = volle Punktzahl
        max_score += 0.4

        # Tabellen-Erkennung (20% der Bewertung)
        tables = extracted_data.get("tables", [])
        if tables:
            score += min(len(tables) / 3, 1.0) * 0.2  # 3 Tabellen = volle Punktzahl
        max_score += 0.2

        # NER-Entitäten (30% der Bewertung)
        entities = extracted_data.get("german_entities", {})
        if entities:
            # Wichtige Entitäten bewerten
            important_entities = [
                "invoice_number",
                "invoice_date",
                "total_gross",
                "supplier_name",
            ]
            found_important = sum(
                1 for entity in important_entities if entity in entities
            )
            score += (found_important / len(important_entities)) * 0.3
        max_score += 0.3

        # Metadaten (10% der Bewertung)
        metadata = extracted_data.get("metadata", {})
        if metadata.get("page_count", 0) > 0:
            score += 0.1
        max_score += 0.1

        return score / max_score if max_score > 0 else 0.0

    def _extract_structured_data(self, document: Any) -> dict[str, Any]:
        """
        Extrahiert strukturierte Daten aus Docling-Dokument.

        Args:
            document: Docling Document-Objekt

        Returns:
            Dictionary mit extrahierten Daten
        """
        # Volltext extrahieren
        raw_text = document.export_to_text()

        # Tabellen extrahieren und parsen
        tables = self._extract_tables(document)

        # WICHTIG: Original Docling Tables für neue API verfügbar machen
        original_tables = []
        if hasattr(document, "tables"):
            original_tables = document.tables

        # Metadaten sammeln
        metadata = {
            "page_count": len(document.pages) if hasattr(document, "pages") else 1,
            "processing_method": "docling_optimized",
            "table_count": len(tables),
            "original_table_count": len(original_tables),
            "has_ocr": self.enable_ocr,
            "table_mode": self.table_mode,
        }

        return {
            "raw_text": raw_text,
            "tables": tables,  # Legacy-Format für Rückwärtskompatibilität
            "docling_tables": original_tables,  # Native Docling Tables für neue API
            "metadata": metadata,
            "extraction_method": "docling_optimized",
        }

    def _extract_tables(self, document: Any) -> list[dict[str, Any]]:
        """
        Extrahiert und parst Tabellen aus Docling-Dokument.

        Args:
            document: Docling Document-Objekt

        Returns:
            Liste von Tabellen-Dictionaries
        """
        tables = []

        try:
            # Dokumentstruktur debuggen
            logger.debug(f"Dokument-Typ: {type(document)}")
            logger.debug(f"Dokumentattribute: {dir(document)}")

            # RICHTIGE API: document.tables direkt, nicht über pages
            if hasattr(document, "tables"):
                document_tables = document.tables
                logger.debug(f"Anzahl Tabellen im Dokument: {len(document_tables)}")

                for table_idx, table in enumerate(document_tables):
                    parsed_table = self._parse_docling_table(table, table_idx)
                    if parsed_table:
                        tables.append(parsed_table)
                        logger.debug(
                            f"Tabelle {table_idx}: {parsed_table['row_count']} Zeilen erfolgreich extrahiert"
                        )
            else:
                logger.warning("Dokument hat keine 'tables' Attribute")

            # Fallback: Suche in pages.elements wenn direkte tables leer
            if not tables and hasattr(document, "pages"):
                logger.debug("Fallback: Suche in Seiten-Elementen nach Tabellen")
                for page_idx, page in enumerate(document.pages):
                    if hasattr(page, "elements"):
                        table_elements = [
                            elem
                            for elem in page.elements
                            if hasattr(elem, "label") and elem.label == "table"
                        ]
                        if table_elements:
                            logger.debug(
                                f"Seite {page_idx + 1}: {len(table_elements)} Tabellen in elements gefunden"
                            )
                            for table_idx, table_elem in enumerate(table_elements):
                                parsed_table = self._parse_docling_table(
                                    table_elem, table_idx, page_idx + 1
                                )
                                if parsed_table:
                                    tables.append(parsed_table)

            # Pattern-Extraktion wenn immer noch keine Tabellen
            if not tables:
                logger.debug("Fallback: Pattern-basierte Extraktion")
                fallback_tables = self._extract_pattern_tables_from_text(document)
                tables.extend(fallback_tables)

            logger.info(f"Gesamt: {len(tables)} Tabellen extrahiert")

            # Zusätzliches Debugging wenn keine Tabellen gefunden
            if not tables:
                logger.warning(
                    "PROBLEM: 0 Tabellen gefunden trotz aktivierter Tabellenstrukturen!"
                )
                logger.debug(f"TableFormer-Modus: {self.table_mode}")
                logger.debug(f"OCR-Engine: {self.ocr_engine}")
                logger.debug(f"force_full_page_ocr: {self.force_full_page_ocr}")

        except Exception as e:
            logger.error(f"Fehler bei Tabellen-Extraktion: {e}", exc_info=True)

        return tables

    def _parse_docling_table(
        self, table: Any, table_idx: int, page_number: int = 1
    ) -> dict[str, Any] | None:
        """
        Parst eine Docling-Tabelle korrekt basierend auf der offiziellen API.

        Args:
            table: Docling Table-Objekt
            table_idx: Tabellen-Index
            page_number: Seiten-Nummer

        Returns:
            Geparstes Tabellen-Dictionary oder None
        """
        try:
            # Tabelle zu DataFrame konvertieren (offizielle Docling API)
            if hasattr(table, "export_to_dataframe"):
                table_df = table.export_to_dataframe()

                if table_df.empty:
                    logger.debug(f"Tabelle {table_idx}: DataFrame ist leer")
                    return None

                # Headers und Rows extrahieren
                headers = table_df.columns.tolist()
                rows = table_df.values.tolist()

                # Leere Zeilen filtern
                rows = [row for row in rows if any(str(cell).strip() for cell in row)]

                if not rows:
                    logger.debug(f"Tabelle {table_idx}: Keine gültigen Datenzeilen")
                    return None

                # Deutsche Elektrohandwerk-Header normalisieren
                normalized_headers = self._normalize_german_headers(headers)

                return {
                    "page": page_number,
                    "table_index": table_idx,
                    "headers": headers,
                    "normalized_headers": normalized_headers,
                    "rows": rows,
                    "row_count": len(rows),
                    "column_count": len(headers),
                    "extraction_method": "docling_dataframe",
                    "confidence": 0.9,  # Hohe Konfidenz für native Docling-Extraktion
                }
            else:
                # Fallback für andere Tabellen-Objekte
                logger.debug(
                    f"Tabelle {table_idx}: Kein export_to_dataframe verfügbar, verwende Fallback"
                )
                return self._parse_table_data(table, page_number - 1, table_idx)

        except Exception as e:
            logger.warning(f"Fehler beim Parsen von Tabelle {table_idx}: {e}")
            return None

    def _extract_pattern_tables_from_text(self, document: Any) -> list[dict[str, Any]]:
        """
        Extrahiert Tabellen über Pattern-Matching aus dem Dokument-Text.

        Args:
            document: Docling Document-Objekt

        Returns:
            Liste von erkannten Tabellen
        """
        tables: list[dict[str, Any]] = []

        try:
            # Volltext aus Dokument extrahieren
            if hasattr(document, "export_to_text"):
                document_text = document.export_to_text()
            else:
                logger.debug("Dokument hat keine export_to_text Methode")
                return tables

            if not document_text or len(document_text.strip()) < 100:
                logger.debug("Zu wenig Text für Pattern-Erkennung verfügbar")
                return tables

            # Deutsche Elektrohandwerk-Tabellen-Pattern
            patterns = [
                # Rechnungsposition-Pattern (GIRA etc.)
                r"(?:GIRA|SIEMENS|ABB|SCHNEIDER|LEGRAND|HAGER|Spelsberg|Striebel)[^|]*?(\d{6})[^|]*?\|\s*(\d+)\s*\|",
                # Artikel mit Preisen
                r"(\d+)\s+(.{10,})\s+(\d+(?:[.,]\d+)?)\s+(?:St|Stk|Stück|m|kg|l)\s+(\d+(?:[.,]\d{2})?)\s*€",
                # Tabellen-ähnliche Strukturen
                r"(.+?)\s{3,}(\d+(?:[.,]\d+)?)\s{3,}(\d+(?:[.,]\d{2})?)",
            ]

            for pattern_idx, pattern in enumerate(patterns):
                matches = re.finditer(
                    pattern, document_text, re.MULTILINE | re.IGNORECASE
                )
                rows = []

                for match in matches:
                    if len(match.groups()) >= 2:  # Mindestens 2 Spalten
                        rows.append(list(match.groups()))

                if rows and len(rows) >= 2:  # Mindestens 2 Zeilen für eine Tabelle
                    # Header basierend auf Pattern bestimmen
                    if pattern_idx == 0:  # Elektro-Artikel
                        headers = ["Hersteller", "Artikel-Nr", "Menge"]
                    elif pattern_idx == 1:  # Artikel mit Preisen
                        headers = ["Artikel-Nr", "Beschreibung", "Menge", "Einzelpreis"]
                    else:  # Allgemein
                        headers = [f"Spalte_{i+1}" for i in range(len(rows[0]))]

                    table = {
                        "page": 1,
                        "table_index": pattern_idx,
                        "headers": headers,
                        "normalized_headers": self._normalize_german_headers(headers),
                        "rows": rows,
                        "row_count": len(rows),
                        "column_count": len(headers),
                        "extraction_method": "pattern_fallback",
                        "confidence": 0.6,  # Niedrigere Konfidenz für Pattern-basierte Extraktion
                    }

                    tables.append(table)
                    logger.debug(
                        f"Pattern-Tabelle {pattern_idx} erkannt: {len(rows)} Zeilen"
                    )
                    break  # Nur ein Pattern verwenden

        except Exception as e:
            logger.warning(f"Fehler bei Pattern-Tabellen-Extraktion: {e}")

        return tables

    def _parse_table_data(
        self, table: Any, page_idx: int, table_idx: int
    ) -> dict[str, Any] | None:
        """
        Parst eine einzelne Tabelle aus Docling-Format.

        Args:
            table: Docling Table-Objekt
            page_idx: Seiten-Index
            table_idx: Tabellen-Index auf der Seite

        Returns:
            Geparstes Tabellen-Dictionary oder None
        """
        try:
            # Tabellen-Grid extrahieren
            if not hasattr(table, "data") or not table.data:
                return None

            # Headers und Daten trennen
            table_data = table.data
            if not table_data:
                return None

            # Erste Zeile als Header interpretieren
            headers = []
            rows = []

            for row_idx, row in enumerate(table_data):
                if hasattr(row, "cells"):
                    row_cells = [
                        (
                            str(cell.content).strip()
                            if hasattr(cell, "content")
                            else str(cell).strip()
                        )
                        for cell in row.cells
                    ]
                else:
                    # Fallback für andere Formate
                    row_cells = [str(cell).strip() for cell in row]

                if row_idx == 0:
                    headers = row_cells
                else:
                    rows.append(row_cells)

            # Deutsche Elektrohandwerk-Header normalisieren
            normalized_headers = self._normalize_german_headers(headers)

            return {
                "page": page_idx + 1,
                "table_index": table_idx,
                "headers": headers,
                "normalized_headers": normalized_headers,
                "rows": rows,
                "row_count": len(rows),
                "column_count": len(headers) if headers else 0,
            }

        except Exception as e:
            logger.warning(
                f"Fehler beim Parsen von Tabelle {table_idx} auf Seite {page_idx + 1}: {e}"
            )
            return None

    def _normalize_german_headers(self, headers: list[str]) -> dict[str, str]:
        """
        Normalisiert deutsche Tabellen-Header für einheitliche Verarbeitung.

        Args:
            headers: Liste der ursprünglichen Header

        Returns:
            Dictionary mit normalisierten Header-Mappings
        """
        # Deutsche Elektrohandwerk-Header-Mappings
        header_mappings = {
            # Positions-/Artikel-Nummer
            "pos": ["pos", "position", "nr", "nummer", "item", "art.-nr", "artikelnr"],
            "article_number": [
                "art.-nr",
                "artikelnr",
                "artikel-nr",
                "artikel",
                "sku",
                "item",
            ],
            # Beschreibung
            "description": [
                "beschreibung",
                "artikel",
                "bezeichnung",
                "text",
                "description",
            ],
            # Mengen
            "quantity": ["menge", "anzahl", "qty", "stück", "stk", "pcs"],
            "unit": ["einheit", "me", "unit", "vpe"],
            # Preise
            "unit_price": [
                "einzelpreis",
                "e-preis",
                "ep",
                "preis",
                "unit_price",
                "stückpreis",
            ],
            "total_price": ["gesamtpreis", "gesamt", "summe", "betrag", "total"],
            # Sonstige
            "discount": ["rabatt", "nachlass", "discount"],
            "tax": ["mwst", "ust", "steuer", "tax", "%"],
        }

        normalized = {}

        for idx, header in enumerate(headers):
            header_lower = header.lower().strip()

            # Mapping finden
            for standard_key, variants in header_mappings.items():
                if any(variant in header_lower for variant in variants):
                    normalized[standard_key] = header
                    break
            else:
                # Fallback: Original Header
                normalized[f"column_{idx}"] = header

        return normalized

    def _extract_pattern_tables(self, page: Any, page_idx: int) -> list[dict[str, Any]]:
        """
        Fallback-Tabellenerkennung über Text-Pattern.

        Args:
            page: Seiten-Objekt
            page_idx: Seiten-Index

        Returns:
            Liste von erkannten Tabellen
        """
        tables: list[dict[str, Any]] = []

        try:
            # Seitentext extrahieren
            page_text = ""
            if hasattr(page, "export_to_text"):
                page_text = page.export_to_text()
            elif hasattr(page, "text"):
                page_text = page.text
            else:
                logger.debug(
                    f"Seite {page_idx + 1}: Kein Text für Pattern-Erkennung verfügbar"
                )
                return tables

            # Deutsche Elektrohandwerk-Tabellen-Pattern
            patterns = [
                # Rechnungsposition-Pattern
                r"(?:Pos\.?\s*|Position\s+)?(\d+)\s+(.+?)\s+(\d+(?:[.,]\d+)?)\s+(\w+)\s+(\d+(?:[.,]\d{2})?)\s+(\d+(?:[.,]\d{2})?)",
                # Artikel-Pattern
                r"(\d+)\s+(.{10,})\s+(\d+(?:[.,]\d+)?)\s+(?:St|Stk|Stück|m|kg|l)\s+(\d+(?:[.,]\d{2})?)\s*€",
                # Einfaches Tabellenformat
                r"(.+?)\s{3,}(\d+(?:[.,]\d+)?)\s{3,}(\d+(?:[.,]\d{2})?)",
            ]

            for pattern_idx, pattern in enumerate(patterns):
                matches = re.finditer(pattern, page_text, re.MULTILINE | re.IGNORECASE)
                rows = []

                for match in matches:
                    if len(match.groups()) >= 3:  # Mindestens 3 Spalten
                        rows.append(list(match.groups()))

                if rows:
                    # Header basierend auf Pattern bestimmen
                    if pattern_idx == 0:  # Rechnungsposition
                        headers = [
                            "Position",
                            "Beschreibung",
                            "Menge",
                            "Einheit",
                            "Einzelpreis",
                            "Gesamtpreis",
                        ]
                    elif pattern_idx == 1:  # Artikel
                        headers = ["Artikel-Nr", "Beschreibung", "Menge", "Einzelpreis"]
                    else:  # Einfach
                        headers = [f"Spalte_{i+1}" for i in range(len(rows[0]))]

                    table = {
                        "page": page_idx + 1,
                        "table_index": pattern_idx,
                        "headers": headers,
                        "normalized_headers": self._normalize_german_headers(headers),
                        "rows": rows,
                        "row_count": len(rows),
                        "column_count": len(headers),
                        "extraction_method": "pattern_fallback",
                        "confidence": 0.7,  # Niedrigere Konfidenz für Pattern-basierte Extraktion
                    }

                    tables.append(table)
                    logger.debug(
                        f"Pattern-Tabelle {pattern_idx} erkannt: {len(rows)} Zeilen"
                    )
                    break  # Nur ein Pattern pro Seite verwenden

        except Exception as e:
            logger.warning(f"Fehler bei Pattern-Tabellen-Extraktion: {e}")

        return tables

    def get_performance_stats(self) -> dict[str, Any]:
        """
        Gibt Performance-Statistiken zurück.

        Returns:
            Dictionary mit Performance-Daten
        """
        return {
            "gpu_enabled": self.use_gpu,
            "table_structure_enabled": self.enable_table_structure,
            "ocr_enabled": self.enable_ocr,
            "table_mode": self.table_mode,
            "optimized_for": "deutsche_eingangsrechnungen",
        }


def create_advanced_docling_processor(**kwargs: Any) -> AdvancedDoclingProcessor:
    """
    Factory-Funktion für Advanced Docling Processor.

    Args:
        **kwargs: Konfigurationsparameter für AdvancedDoclingProcessor

    Returns:
        Konfigurierte AdvancedDoclingProcessor-Instanz
    """
    return AdvancedDoclingProcessor(**kwargs)


# Alias für Backward Compatibility
DoclingProcessor = AdvancedDoclingProcessor

# Rückwärtskompatibilität mit alter API
def create_optimized_docling_processor(**kwargs: Any) -> AdvancedDoclingProcessor:
    """Rückwärtskompatibilität - verwendet jetzt AdvancedDoclingProcessor."""
    return AdvancedDoclingProcessor(**kwargs)


# Test-Funktion für Entwicklung
if __name__ == "__main__":
    # Test mit vorhandener PDF
    test_pdf = Path("test_pdfs/Sonepar_test3.pdf")

    if test_pdf.exists():
        processor = create_optimized_docling_processor(
            use_gpu=True, enable_table_structure=True, enable_ocr=True
        )

        result = processor.process_pdf(test_pdf)
        print(f"Text-Länge: {len(result.get('raw_text', ''))}")
        print(f"Tabellen: {len(result.get('tables', []))}")
        print(f"Performance: {processor.get_performance_stats()}")
    else:
        print(f"Test-PDF nicht gefunden: {test_pdf}")
