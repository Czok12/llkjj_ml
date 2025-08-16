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

import chromadb
from docling.document_converter import DocumentConverter
from sentence_transformers import SentenceTransformer

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
        self._setup_skr03_regeln()
        self._setup_vector_db()  # NEW: Add ChromaDB setup

    def _setup_logging(self) -> None:
        """Configure logging for the processor"""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    def _setup_docling(self) -> None:
        """Setup Docling document converter with simplified settings"""

        # Initialize document converter with default configuration
        self.document_converter = DocumentConverter()

        logger.info("Docling processor initialized with German optimization")

    def _setup_gemini(self) -> None:
        """Setup Gemini API for enhanced extraction - temporarily disabled"""
        self.gemini_model = None
        logger.warning("Gemini temporarily disabled - using basic extraction only")

    def _setup_skr03_regeln(self) -> None:
        """Lädt SKR03-Klassifizierungsregeln über den neuen SKR03Manager"""
        try:
            # Import des neuen SKR03Managers
            from src.skr03_manager import SKR03Manager

            # Pfade zu den Dateien
            regeln_pfad = (
                Path(__file__).parent.parent.parent / "config" / "skr03_regeln.yaml"
            )
            kontenplan_pfad = Path(__file__).parent.parent.parent / "Kontenplan.csv"

            # Initialisiere SKR03Manager
            self.skr03_manager = SKR03Manager(regeln_pfad, kontenplan_pfad)

            if self.skr03_manager.ist_bereit():
                stats = self.skr03_manager.get_statistiken()
                logger.info(
                    f"✅ SKR03-Manager bereit: {stats['regeln_geladen']} Kategorien, "
                    f"{stats['konten_geladen']} Konten aus CSV"
                )

                # Für Rückwärtskompatibilität: Setze alte Attribute
                self.skr03_regeln = self.skr03_manager.klassifizierungsregeln
                self.kontenplan_parser = self.skr03_manager.kontenplan_parser
            else:
                logger.error("❌ SKR03-Manager konnte nicht initialisiert werden")
                self._setup_fallback_regeln()

        except Exception as e:
            logger.error(f"❌ Fehler beim Setup des SKR03-Managers: {e}")
            self._setup_fallback_regeln()

    def _setup_fallback_regeln(self) -> None:
        """Fallback zu alten YAML-Regeln wenn Manager nicht funktioniert"""
        import yaml

        # Versuche zuerst erweiterte Regeln, dann Fallback auf Basic-Regeln
        regeln_pfade = [
            Path(__file__).parent.parent.parent
            / "config"
            / "skr03_regeln_erweitert.yaml",
            Path(__file__).parent.parent.parent / "config" / "skr03_regeln.yaml",
        ]

        for regeln_pfad in regeln_pfade:
            if regeln_pfad.exists():
                try:
                    with open(regeln_pfad, encoding="utf-8") as f:
                        regeln_daten = yaml.safe_load(f)
                    self.skr03_regeln = regeln_daten.get("kategorien", {})

                    if self.skr03_regeln:
                        regeln_typ = (
                            "erweitert" if "erweitert" in str(regeln_pfad) else "basic"
                        )
                        logger.info(
                            f"SKR03-Regeln geladen (fallback {regeln_typ}) aus {regeln_pfad}: {len(self.skr03_regeln)} Kategorien"
                        )
                        self._lade_kontenplan_fuer_validierung()
                        return

                except Exception as e:
                    logger.warning(
                        f"Fehler beim Laden der SKR03-Regeln aus {regeln_pfad}: {e}"
                    )
                    continue

        # Fallback: Leere Regeln
        logger.error("Keine gültigen SKR03-Regeln gefunden - verwende leere Regeln")
        self.skr03_regeln = {}

    def _lade_kontenplan_fuer_validierung(self) -> None:
        """Lädt optional den vollständigen Kontenplan für erweiterte Validierung"""
        try:
            # Der SKR03Manager enthält bereits einen integrierten KontenplanParser
            # Diese Funktion ist nicht mehr notwendig, da die Validierung
            # direkt über den SKR03Manager erfolgt
            logger.info("Kontenplan-Validierung erfolgt über SKR03Manager")
            self.kontenplan_parser = None

        except Exception as e:
            logger.warning(f"Kontenplan für Validierung nicht verfügbar: {e}")
            self.kontenplan_parser = None

    def _setup_vector_db(self) -> None:
        """
        Erweiterte ChromaDB-Initialisierung für validierte Buchungsbeispiele.

        Erstellt separate Collections für:
        - Rechnungsdaten (invoices)
        - Validierte SKR03-Buchungsbeispiele (validated_bookings)
        - Elektrotechnik-Artikelkatalog (elektro_articles)
        """
        try:
            # Initialize embedding model for semantic search
            self.embedding_model = SentenceTransformer(
                self.config.sentence_transformer_model
            )

            # Initialize ChromaDB client
            db_path = str(self.config.vector_db_path)
            self.vector_db_client = chromadb.PersistentClient(path=db_path)

            # Collection 1: Standard-Rechnungsdaten
            self.invoice_collection = self.vector_db_client.get_or_create_collection(
                name=self.config.chroma_collection_name,
                metadata={
                    "type": "invoice_data",
                    "description": "PDF-Rechnungsextraktion",
                },
            )

            # Collection 2: Validierte Buchungsbeispiele für SKR03
            self.booking_collection = self.vector_db_client.get_or_create_collection(
                name="validated_skr03_bookings",
                metadata={
                    "type": "booking_examples",
                    "description": "Validierte SKR03-Buchungssätze",
                },
            )

            # Collection 3: Elektrotechnik-Artikelkatalog mit historischen Daten
            self.article_collection = self.vector_db_client.get_or_create_collection(
                name="elektro_article_catalog",
                metadata={
                    "type": "article_catalog",
                    "description": "Elektrotechnik-Artikelkatalog mit Klassifizierungen",
                },
            )

            # Lade validierte Buchungsbeispiele beim Start
            self._load_validated_booking_examples()

            logger.info(
                f"✅ ChromaDB erweitert initialisiert: "
                f"Invoices: {self.invoice_collection.count()}, "
                f"Bookings: {self.booking_collection.count()}, "
                f"Articles: {self.article_collection.count()}"
            )

        except Exception as e:
            logger.error(f"Fehler bei der Initialisierung der Vektordatenbank: {e}")
            self.embedding_model = None
            self.invoice_collection = None
            self.booking_collection = None
            self.article_collection = None

    def _load_validated_booking_examples(self) -> None:
        """
        Lädt validierte Buchungsbeispiele für Elektrotechnik-Artikel.

        Diese Beispiele dienen als Wissensbasis für intelligente Klassifizierung
        neuer oder unbekannter Artikel basierend auf Ähnlichkeit.
        """

        # Prüfe ob bereits Buchungsbeispiele vorhanden sind
        if self.booking_collection.count() > 0:
            logger.info(
                f"📚 {self.booking_collection.count()} validierte Buchungsbeispiele bereits geladen"
            )
            return

        # Standard Elektrotechnik-Buchungsbeispiele
        validated_bookings = [
            # GIRA-Produkte
            {
                "description": "GIRA Adapterrahmen System 55 reinweiß",
                "supplier": "GIRA",
                "category": "Installationsmaterial",
                "skr03_account": "3400",
                "vat_rate": "19%",
                "material_group": "Elektroinstallation",
                "keywords": ["gira", "adapterrahmen", "system55", "reinweiß"],
                "confidence": 0.95,
                "validation_date": "2025-08-16",
                "validation_source": "manual_expert_classification",
            },
            {
                "description": "GIRA Abdeckrahmen 2fach System 55",
                "supplier": "GIRA",
                "category": "Installationsmaterial",
                "skr03_account": "3400",
                "vat_rate": "19%",
                "material_group": "Elektroinstallation",
                "keywords": ["gira", "abdeckrahmen", "2fach", "system55"],
                "confidence": 0.95,
                "validation_date": "2025-08-16",
                "validation_source": "manual_expert_classification",
            },
            # Hager-Produkte
            {
                "description": "Hager Abdeckstreifen brechbar 18 Module",
                "supplier": "Hager",
                "category": "Verteilermaterial",
                "skr03_account": "3400",
                "vat_rate": "19%",
                "material_group": "Verteilerausbau",
                "keywords": ["hager", "abdeckstreifen", "brechbar", "module"],
                "confidence": 0.95,
                "validation_date": "2025-08-16",
                "validation_source": "manual_expert_classification",
            },
            # Kabel und Leitungen
            {
                "description": "NYM-J 3x1,5mm² 100m Mantelleitung",
                "supplier": "Various",
                "category": "Kabel und Leitungen",
                "skr03_account": "3400",
                "vat_rate": "19%",
                "material_group": "Installationskabel",
                "keywords": ["nym", "mantelleitung", "kabel", "3x1,5"],
                "confidence": 0.98,
                "validation_date": "2025-08-16",
                "validation_source": "din_norm_classification",
            },
            # Werkzeuge (andere Klassifizierung)
            {
                "description": "WAGO Verbindungsklemme 2-polig",
                "supplier": "WAGO",
                "category": "Verbindungstechnik",
                "skr03_account": "3400",
                "vat_rate": "19%",
                "material_group": "Klemmen",
                "keywords": ["wago", "klemme", "verbindung", "2-polig"],
                "confidence": 0.96,
                "validation_date": "2025-08-16",
                "validation_source": "manual_expert_classification",
            },
            # Werkzeuge (abweichende Klassifizierung)
            {
                "description": "Bosch Professional Schlagbohrmaschine",
                "supplier": "Bosch",
                "category": "Werkzeuge",
                "skr03_account": "4985",  # Andere Klassifizierung!
                "vat_rate": "19%",
                "material_group": "Werkzeuge",
                "keywords": ["bosch", "bohrmaschine", "werkzeug", "professional"],
                "confidence": 0.92,
                "validation_date": "2025-08-16",
                "validation_source": "manual_expert_classification",
            },
            # LED-Beleuchtung
            {
                "description": "LED Panel 60x60cm 40W neutralweiß",
                "supplier": "Various",
                "category": "Beleuchtung",
                "skr03_account": "3400",
                "vat_rate": "19%",
                "material_group": "LED-Technik",
                "keywords": ["led", "panel", "beleuchtung", "40w", "neutralweiß"],
                "confidence": 0.94,
                "validation_date": "2025-08-16",
                "validation_source": "manual_expert_classification",
            },
        ]

        # Vektorisiere und speichere Buchungsbeispiele
        documents = []
        metadatas = []
        ids = []

        for i, booking in enumerate(validated_bookings):
            # Erstelle semantischen Text für Vektorsuche
            search_text = f"{booking['description']} {booking['supplier']} {booking['category']} {' '.join(booking['keywords'])}"
            documents.append(search_text)

            # Metadata für Retrieval
            metadatas.append(
                {
                    "id": f"booking_{i:03d}",
                    "description": booking["description"],
                    "supplier": booking["supplier"],
                    "category": booking["category"],
                    "skr03_account": booking["skr03_account"],
                    "vat_rate": booking["vat_rate"],
                    "material_group": booking["material_group"],
                    "confidence": booking["confidence"],
                    "validation_date": booking["validation_date"],
                    "validation_source": booking["validation_source"],
                    "keywords": ",".join(booking["keywords"]),
                }
            )

            ids.append(f"validated_booking_{i:03d}")

        # Speichere in ChromaDB
        self.booking_collection.add(documents=documents, metadatas=metadatas, ids=ids)

        logger.info(
            f"📚 {len(validated_bookings)} validierte Buchungsbeispiele geladen"
        )

    def add_validated_booking(
        self,
        description: str,
        supplier: str,
        skr03_account: str,
        category: str = "Elektromaterial",
        material_group: str = "Unbekannt",
        confidence: float = 1.0,
        validation_source: str = "manual_validation",
    ) -> None:
        """
        Fügt ein neues validiertes Buchungsbeispiel zur Wissensbasis hinzu.

        Args:
            description: Artikelbeschreibung
            supplier: Lieferant
            skr03_account: SKR03-Kontonummer
            category: Artikelkategorie
            material_group: Materialgruppe
            confidence: Konfidenzwert (0.0-1.0)
            validation_source: Quelle der Validierung
        """
        if not self.booking_collection:
            logger.warning(
                "❌ ChromaDB nicht verfügbar - Buchungsbeispiel nicht gespeichert"
            )
            return

        # Erstelle ID basierend auf aktueller Anzahl
        current_count = self.booking_collection.count()
        booking_id = f"validated_booking_{current_count:03d}"

        # Erstelle Suchtext
        search_text = f"{description} {supplier} {category}"

        # Metadata
        metadata = {
            "id": booking_id,
            "description": description,
            "supplier": supplier,
            "category": category,
            "skr03_account": skr03_account,
            "material_group": material_group,
            "confidence": confidence,
            "validation_date": datetime.now().strftime("%Y-%m-%d"),
            "validation_source": validation_source,
        }

        # Speichere in ChromaDB
        self.booking_collection.add(
            documents=[search_text], metadatas=[metadata], ids=[booking_id]
        )

        logger.info(
            f"✅ Neues Buchungsbeispiel hinzugefügt: {description} -> {skr03_account}"
        )

    def find_similar_bookings(
        self, description: str, n_results: int = 5, similarity_threshold: float = 0.7
    ) -> list[dict[str, Any]]:
        """
        Findet ähnliche validierte Buchungsbeispiele für intelligente Klassifizierung.

        Args:
            description: Zu klassifizierende Artikelbeschreibung
            n_results: Anzahl ähnlicher Beispiele
            similarity_threshold: Mindest-Ähnlichkeitsschwelle

        Returns:
            Liste ähnlicher Buchungsbeispiele mit Konfidenzwerten
        """
        if not self.booking_collection:
            logger.warning(
                "❌ ChromaDB nicht verfügbar - keine ähnlichen Buchungen gefunden"
            )
            return []

        try:
            # Suche ähnliche Buchungsbeispiele
            results = self.booking_collection.query(
                query_texts=[description], n_results=n_results
            )

            similar_bookings = []

            if results["metadatas"]:
                for i, metadata in enumerate(results["metadatas"][0]):
                    # Berechne Ähnlichkeitsscore (ChromaDB gibt Distanz zurück)
                    distance = results["distances"][0][i]
                    similarity = max(
                        0.0, 1.0 - distance
                    )  # Konvertiere Distanz zu Ähnlichkeit

                    if similarity >= similarity_threshold:
                        booking_info = {
                            "description": metadata["description"],
                            "supplier": metadata["supplier"],
                            "category": metadata["category"],
                            "skr03_account": metadata["skr03_account"],
                            "material_group": metadata["material_group"],
                            "confidence": metadata["confidence"],
                            "similarity": similarity,
                            "validation_source": metadata["validation_source"],
                        }
                        similar_bookings.append(booking_info)

            logger.info(
                f"🔍 {len(similar_bookings)} ähnliche Buchungsbeispiele für '{description}' gefunden"
            )
            return similar_bookings

        except Exception as e:
            logger.error(f"❌ Fehler bei der Suche ähnlicher Buchungen: {e}")
            return []

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

        # Debug: Log extraction results
        logger.info(
            f"Raw extraction found {len(raw_extraction.get('tables', []))} tables"
        )
        logger.info(
            f"Enhanced data has {len(enhanced_data.get('line_items', []))} line items"
        )

        # Debug: Log extracted line items
        for i, item in enumerate(enhanced_data.get("line_items", [])):
            logger.info(
                f"Line item {i+1}: {item.get('description', 'NO_DESC')} - {item.get('total_price', 'NO_PRICE')}"
            )

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

        # Schritt 5: Vektorisieren und Speichern
        self._vectorize_and_store(result)

        # Schritt 6: Ergebnis als JSON speichern
        json_path = self.save_result(result)
        logger.info(f"Results saved to: {json_path}")

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
                return self._extract_structured_data(raw_data)

            response = self.gemini_model.generate_content(prompt)

            # Parse Gemini response
            enhanced_data = json.loads(response.text)
            enhanced_data.update(raw_data)  # Merge with raw data
            enhanced_data["enhancement_method"] = "gemini"
            enhanced_data["language"] = "de"  # Explicitly mark as German

            logger.debug("Gemini enhancement successful (German optimized)")
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
            "line_items": self._extract_line_items(raw_data.get("tables", []), text),
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

    def _extract_line_items(
        self, tables: list[dict[str, Any]], text: str = ""
    ) -> list[dict[str, Any]]:
        """Extract line items from table data or fallback to pattern extraction"""

        items = []

        # First try to extract from Docling tables
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

        # If no table data found, try pattern-based extraction
        if not items and text:
            items = self._extract_line_items_from_text(text)

        return items

    def _extract_line_items_from_text(self, text: str) -> list[dict[str, Any]]:
        """Extract line items using regex patterns for German invoices"""

        import re

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
            # Format: | article_no | description with lots of info | qty | unit | | unit_price | total |
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
        """
        Erweiterte SKR03-Klassifizierung mit RAG-System für intelligente Artikel-Klassifizierung.

        Kombiniert:
        1. Regelbasierte Klassifizierung (bestehende Keywords)
        2. RAG-basierte Klassifizierung (ähnliche validierte Buchungsbeispiele)
        3. Intelligente Konfidenz-Bewertung
        """

        classifications = []
        line_items = structured_data.get("line_items", [])

        logger.info(f"🏷️ Klassifiziere {len(line_items)} Positionen mit RAG-System")

        for i, item in enumerate(line_items):
            description = item.get("description", "")

            # Schritt 1: Regelbasierte Klassifizierung (bestehend)
            rule_match = self._find_best_skr03_match(description)

            # Schritt 2: RAG-basierte Klassifizierung (NEU)
            rag_match = self._classify_with_rag_system(description)

            # Schritt 3: Intelligente Kombination der Ergebnisse
            final_classification = self._combine_classification_results(
                description, rule_match, rag_match
            )

            classification = {
                "position": i + 1,
                "description": description,
                "skr03_konto": final_classification["konto"],
                "category": final_classification["category"],
                "confidence": final_classification["confidence"],
                "reasoning": final_classification["reasoning"],
                "amount": item.get("total_price", "0"),
                "quantity": item.get("quantity", "1"),
                # Erweiterte RAG-Informationen
                "rule_based_confidence": rule_match["confidence"],
                "rag_based_confidence": rag_match["confidence"],
                "similar_articles": rag_match.get("similar_articles", []),
                "classification_method": final_classification["method"],
                "supplier_detected": rag_match.get("supplier", "Unknown"),
            }

            logger.info(
                f"📊 Position {i+1}: '{description[:50]}...' -> "
                f"SKR03 {final_classification['konto']} "
                f"({final_classification['category']}, "
                f"{final_classification['confidence']:.1%} via {final_classification['method']})"
            )

            classifications.append(classification)

        return classifications

    def _classify_with_rag_system(self, description: str) -> dict[str, Any]:
        """
        RAG-basierte Klassifizierung: Nutzt ähnliche validierte Buchungsbeispiele
        für intelligente Klassifizierung neuer oder unbekannter Artikel.
        """

        # Suche ähnliche validierte Buchungsbeispiele
        similar_bookings = self.find_similar_bookings(
            description=description, n_results=5, similarity_threshold=0.6
        )

        if not similar_bookings:
            return {
                "konto": "3400",  # Fallback
                "category": "Unbekannt",
                "confidence": 0.1,
                "reasoning": "Keine ähnlichen Buchungsbeispiele gefunden",
                "method": "fallback",
                "similar_articles": [],
            }

        # Gewichtete Bewertung basierend auf Ähnlichkeit und Konfidenz
        weighted_votes = {}
        total_weight = 0.0

        for booking in similar_bookings:
            skr03_account = booking["skr03_account"]
            weight = booking["similarity"] * booking["confidence"]

            if skr03_account not in weighted_votes:
                weighted_votes[skr03_account] = {
                    "weight": 0.0,
                    "category": booking["category"],
                    "examples": [],
                }

            weighted_votes[skr03_account]["weight"] += weight
            weighted_votes[skr03_account]["examples"].append(
                {
                    "description": booking["description"],
                    "similarity": booking["similarity"],
                    "supplier": booking["supplier"],
                }
            )
            total_weight += weight

        # Finde bestes Ergebnis
        if not weighted_votes:
            return {
                "konto": "3400",
                "category": "Unbekannt",
                "confidence": 0.1,
                "reasoning": "Keine gewichteten Stimmen gefunden",
                "method": "fallback",
                "similar_articles": [],
            }

        best_account = max(
            weighted_votes.keys(), key=lambda k: weighted_votes[k]["weight"]
        )
        best_info = weighted_votes[best_account]

        # Berechne RAG-Konfidenz
        rag_confidence = min(0.95, best_info["weight"] / total_weight)

        # Erkenne Lieferant aus ähnlichen Beispielen
        suppliers = [ex["supplier"] for ex in best_info["examples"]]
        most_common_supplier = (
            max(set(suppliers), key=suppliers.count) if suppliers else "Unknown"
        )

        return {
            "konto": best_account,
            "category": best_info["category"],
            "confidence": rag_confidence,
            "reasoning": f"RAG: {len(similar_bookings)} ähnliche Artikel, stärkste Übereinstimmung mit {best_info['category']}",
            "method": "rag_similarity",
            "similar_articles": similar_bookings[:3],  # Top 3 für Debugging
            "supplier": most_common_supplier,
        }

    def _combine_classification_results(
        self, description: str, rule_match: dict[str, Any], rag_match: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Intelligente Kombination von regelbasierter und RAG-basierter Klassifizierung.

        Strategie:
        - Bei hoher Regel-Konfidenz: Bevorzuge Regeln (bewährte Keywords)
        - Bei niedriger Regel-Konfidenz: Bevorzuge RAG (ähnliche Beispiele)
        - Bei Konflikt: Wähle höhere Konfidenz, aber protokolliere beide
        """

        rule_confidence = rule_match["confidence"]
        rag_confidence = rag_match["confidence"]

        # Schwellenwerte für Entscheidungslogik
        HIGH_CONFIDENCE_THRESHOLD = 0.7
        MEDIUM_CONFIDENCE_THRESHOLD = 0.4

        # Szenario 1: Beide niedrige Konfidenz -> Regelbasiert (konservativer Ansatz)
        if (
            rule_confidence < MEDIUM_CONFIDENCE_THRESHOLD
            and rag_confidence < MEDIUM_CONFIDENCE_THRESHOLD
        ):
            return {
                "konto": rule_match.get("account", rule_match.get("konto", "3400")),
                "category": rule_match["category"],
                "confidence": max(rule_confidence, rag_confidence),
                "reasoning": f"Niedrige Konfidenz beider Methoden. Regel: {rule_confidence:.2f}, RAG: {rag_confidence:.2f}. Standardklassifizierung gewählt.",
                "method": "rule_fallback",
            }

        # Szenario 2: Regelbasiert hat hohe Konfidenz -> Bevorzuge Regeln
        if rule_confidence >= HIGH_CONFIDENCE_THRESHOLD:
            return {
                "konto": rule_match.get("account", rule_match.get("konto", "3400")),
                "category": rule_match["category"],
                "confidence": rule_confidence,
                "reasoning": f"Starke Keywords gefunden: {rule_match['reasoning']}",
                "method": "rule_dominant",
            }

        # Szenario 3: RAG hat deutlich höhere Konfidenz -> Bevorzuge RAG
        if rag_confidence > rule_confidence + 0.2:
            return {
                "konto": rag_match.get("account", rag_match.get("konto", "3400")),
                "category": rag_match["category"],
                "confidence": rag_confidence,
                "reasoning": f"RAG-System überzeugt: {rag_match['reasoning']}",
                "method": "rag_dominant",
            }

        # Szenario 4: Beide haben ähnliche mittlere Konfidenz -> Hybrid
        if abs(rule_confidence - rag_confidence) <= 0.2:
            # Prüfe ob beide zur gleichen Klassifizierung kommen
            regel_konto = rule_match.get("account", rule_match.get("konto", "3400"))
            rag_konto = rag_match.get("account", rag_match.get("konto", "3400"))
            if regel_konto == rag_konto:
                hybrid_confidence = (rule_confidence + rag_confidence) / 2
                return {
                    "konto": regel_konto,
                    "category": rule_match["category"],
                    "confidence": hybrid_confidence,
                    "reasoning": f"Hybrid: Beide Methoden stimmen überein (Regel: {rule_confidence:.2f}, RAG: {rag_confidence:.2f})",
                    "method": "hybrid_consensus",
                }
            else:
                # Konflikt -> Wähle höhere Konfidenz, aber reduziere sie leicht
                if rule_confidence >= rag_confidence:
                    return {
                        "konto": rule_match["konto"],
                        "category": rule_match["category"],
                        "confidence": rule_confidence
                        * 0.9,  # Leichte Reduktion wegen Konflik
                        "reasoning": f"Konflikt gelöst: Regel ({rule_confidence:.2f}) vs RAG ({rag_confidence:.2f})",
                        "method": "rule_conflict_resolution",
                    }
                else:
                    return {
                        "konto": rag_match["konto"],
                        "category": rag_match["category"],
                        "confidence": rag_confidence * 0.9,
                        "reasoning": f"Konflikt gelöst: RAG ({rag_confidence:.2f}) vs Regel ({rule_confidence:.2f})",
                        "method": "rag_conflict_resolution",
                    }

        # Fallback: Standardregeln
        return {
            "konto": rule_match["konto"],
            "category": rule_match["category"],
            "confidence": rule_confidence,
            "reasoning": f"Fallback zu Regeln: {rule_match['reasoning']}",
            "method": "rule_fallback",
        }

    def _find_best_skr03_match(self, description: str) -> dict[str, Any]:
        """
        SKR03-Klassifizierung über den neuen SKR03Manager.

        Verwendet saubere Trennung von Regeln (YAML) und Konten (CSV).
        """
        # Verwende den neuen Manager falls verfügbar
        if hasattr(self, "skr03_manager") and self.skr03_manager:
            kategorie, konto, konfidenz, keywords = (
                self.skr03_manager.klassifiziere_artikel(description)
            )

            # Konvertiere Manager-Ergebnis zu erwartetem Format für Kompatibilität
            return {
                "category": kategorie,
                "konto": konto,  # Verwende "konto" statt "account" für Kompatibilität
                "confidence": konfidenz,
                "matched_keywords": keywords,
                "method": "skr03_manager_v2",
                "reasoning": f"SKR03Manager Klassifizierung basierend auf Keywords: {keywords}",
            }

        # Fallback zu alter Logik
        return self._find_best_skr03_match_fallback(description)

    def _find_best_skr03_match_fallback(self, description: str) -> dict[str, Any]:
        """
        Fallback-Klassifizierung für den Fall dass Manager nicht verfügbar ist.
        """
        best_score = 0
        best_category = "elektromaterial"  # Default
        matched_keywords = []

        # Normalisiere Beschreibung für besseres Matching
        description_lower = description.lower()

        for kategorie, regeln in self.skr03_regeln.items():
            score = 0
            gefundene_schluesselwoerter = []

            for schluesselwort in regeln["schlüsselwörter"]:
                if schluesselwort.lower() in description_lower:
                    # Gewichte längere Schlüsselwörter höher
                    gewicht = len(schluesselwort) / 10.0 + 1.0
                    score += gewicht
                    gefundene_schluesselwoerter.append(schluesselwort)

            if score > best_score:
                best_score = score
                best_category = kategorie
                matched_keywords = gefundene_schluesselwoerter

        category_info = self.skr03_regeln.get(best_category, {})

        # Erweiterte Konfidenz-Berechnung
        base_confidence = min(0.9, 0.3 + (best_score * 0.1))

        # Bonus für mehrere Keyword-Matches
        keyword_bonus = min(0.2, len(matched_keywords) * 0.05)

        # Bonus für bekannte Elektrotechnik-Marken
        brand_bonus = 0.0
        elektro_marken = [
            "gira",
            "hager",
            "siemens",
            "abb",
            "schneider",
            "wago",
            "phoenix",
        ]
        for marke in elektro_marken:
            if marke in description_lower:
                brand_bonus = 0.15
                break

        final_confidence = min(0.95, base_confidence + keyword_bonus + brand_bonus)

        # Hole Konto aus Kategorie-Info
        konto = category_info.get("konto", "3400") if category_info else "3400"

        # Optional: Validiere Konto gegen Kontenplan
        konto_validierung = self._validiere_konto_zuordnung(konto)

        result = {
            "category": best_category,
            "account": konto,
            "confidence": final_confidence,
            "reasoning": (
                f"Matched keywords: {', '.join(matched_keywords)}"
                if matched_keywords
                else "Default classification"
            ),
            "method": "rule_fallback",
            "validation_info": {
                "konto_gueltig": (
                    bool(konto_validierung) if konto_validierung else False
                ),
                "konto_info": konto_validierung,
            },
        }

        return result

    def _validiere_konto_zuordnung(self, kontonummer: str) -> dict[str, Any] | None:
        """
        Validiert eine Kontenzuordnung gegen den vollständigen Kontenplan.

        Returns:
            Validierungsergebnis oder None falls Kontenplan nicht verfügbar
        """
        if hasattr(self, "skr03_manager") and self.skr03_manager:
            ist_gueltig = self.skr03_manager.validiere_konto(kontonummer)
            if ist_gueltig and self.skr03_manager.kontenplan_parser:
                konto_info = self.skr03_manager.kontenplan_parser.get_konto_info(
                    kontonummer
                )
                return {"ist_gueltig": True, "konto_info": konto_info}
            else:
                return {"ist_gueltig": False, "konto_info": None}

        return None

    def _calculate_confidence(
        self, structured_data: dict[str, Any], classifications: list[dict[str, Any]]
    ) -> float:
        """
        Erweiterte Konfidenz-Bewertung mit umfassender Qualitätskontrolle.

        Berücksichtigt:
        1. Rechnungskopf-Vollständigkeit
        2. Klassifizierungs-Konfidenz (regel- und RAG-basiert)
        3. Tabellenextraktion-Erfolg
        4. Betragskonsistenz
        5. Lieferanten-Erkennung
        6. Elektrotechnik-spezifische Qualitätsindikatoren
        """

        scores = []
        quality_metrics = {}

        # 1. Rechnungskopf-Vollständigkeit (Gewichtung: 20%)
        header = structured_data.get("invoice_header", {})
        critical_fields = ["invoice_number", "date", "total_amount", "supplier_name"]

        header_completeness = 0
        for field in critical_fields:
            if header.get(field):
                header_completeness += 1

        header_score = header_completeness / len(critical_fields)
        scores.append(("header_completeness", header_score, 0.2))
        quality_metrics["header_completeness"] = header_completeness

        # 2. Klassifizierungs-Konfidenz (Gewichtung: 35%)
        classification_scores = []
        rag_usage_count = 0
        high_confidence_count = 0

        for classification in classifications:
            conf = classification.get("confidence", 0)
            classification_scores.append(conf)

            # Zähle RAG-Nutzung und hohe Konfidenz
            method = classification.get("classification_method", "unknown")
            if "rag" in method:
                rag_usage_count += 1
            if conf >= 0.8:
                high_confidence_count += 1

        if classification_scores:
            avg_classification_confidence = sum(classification_scores) / len(
                classification_scores
            )
            # Bonus für hohen Anteil hoher Konfidenz-Werte
            confidence_bonus = high_confidence_count / len(classification_scores) * 0.1
            final_classification_score = min(
                1.0, avg_classification_confidence + confidence_bonus
            )
        else:
            final_classification_score = 0.0

        scores.append(("classification_confidence", final_classification_score, 0.35))
        quality_metrics["avg_classification_confidence"] = (
            avg_classification_confidence if classification_scores else 0
        )
        quality_metrics["rag_usage_percentage"] = (
            rag_usage_count / len(classifications) if classifications else 0
        )

        # 3. Tabellenextraktion-Erfolg (Gewichtung: 15%)
        line_items = structured_data.get("line_items", [])
        expected_items = max(
            3, len(classifications)
        )  # Mindestens 3 Positionen erwartet

        extraction_score = min(1.0, len(line_items) / expected_items)

        # Bonus für vollständige Extraktion (Beschreibung, Menge, Preis)
        complete_items = 0
        for item in line_items:
            if (
                item.get("description")
                and item.get("quantity")
                and item.get("total_price")
            ):
                complete_items += 1

        completeness_bonus = complete_items / len(line_items) * 0.2 if line_items else 0
        final_extraction_score = min(1.0, extraction_score + completeness_bonus)

        scores.append(("extraction_success", final_extraction_score, 0.15))
        quality_metrics["extracted_items"] = len(line_items)
        quality_metrics["complete_items_percentage"] = (
            complete_items / len(line_items) if line_items else 0
        )

        # 4. Betragskonsistenz (Gewichtung: 15%)
        amount_consistency_score = self._check_amount_consistency(
            structured_data, classifications
        )
        scores.append(("amount_consistency", amount_consistency_score, 0.15))
        quality_metrics["amount_consistency"] = amount_consistency_score

        # 5. Elektrotechnik-spezifische Qualitätsindikatoren (Gewichtung: 10%)
        elektro_quality_score = self._assess_elektro_quality(classifications)
        scores.append(("elektro_quality", elektro_quality_score, 0.10))
        quality_metrics["elektro_quality"] = elektro_quality_score

        # 6. Lieferanten-Erkennung (Gewichtung: 5%)
        supplier_score = self._assess_supplier_recognition(
            structured_data, classifications
        )
        scores.append(("supplier_recognition", supplier_score, 0.05))
        quality_metrics["supplier_recognition"] = supplier_score

        # Gewichtete Gesamtbewertung
        weighted_score = sum(score * weight for _, score, weight in scores)

        # Speichere detaillierte Metriken für Debugging
        self._log_quality_metrics(quality_metrics, weighted_score)

        return weighted_score

    def _check_amount_consistency(
        self, structured_data: dict[str, Any], classifications: list[dict[str, Any]]
    ) -> float:
        """Prüft Konsistenz zwischen Einzelbeträgen und Gesamtbetrag"""

        try:
            # Hole Gesamtbetrag aus Header
            total_amount_str = structured_data.get("invoice_header", {}).get(
                "total_amount", "0"
            )
            total_amount = float(
                str(total_amount_str).replace(",", ".").replace("€", "").strip()
            )

            if total_amount <= 0:
                return 0.5  # Neutral wenn kein Gesamtbetrag gefunden

            # Summiere Einzelbeträge
            sum_line_items = 0
            valid_amounts = 0

            for classification in classifications:
                amount_str = classification.get("amount", "0")
                try:
                    amount = float(
                        str(amount_str).replace(",", ".").replace("€", "").strip()
                    )
                    if amount > 0:
                        sum_line_items += amount
                        valid_amounts += 1
                except (ValueError, TypeError):
                    continue

            if valid_amounts == 0:
                return 0.3  # Schlechte Bewertung wenn keine gültigen Beträge

            # Berechne Abweichung
            difference = abs(total_amount - sum_line_items)
            relative_difference = difference / total_amount if total_amount > 0 else 1.0

            # Bewertung basierend auf relativer Abweichung
            if relative_difference <= 0.05:  # <= 5% Abweichung
                return 1.0
            elif relative_difference <= 0.15:  # <= 15% Abweichung
                return 0.8
            elif relative_difference <= 0.30:  # <= 30% Abweichung
                return 0.5
            else:
                return 0.2

        except Exception as e:
            logger.warning(f"❌ Fehler bei Betragsvergleich: {e}")
            return 0.5

    def _assess_elektro_quality(self, classifications: list[dict[str, Any]]) -> float:
        """Bewertet Elektrotechnik-spezifische Qualitätsindikatoren"""

        if not classifications:
            return 0.0

        quality_indicators = 0
        total_checks = 0

        # Check 1: Verhältnis bekannter Elektrotechnik-Artikel
        elektro_keywords = [
            "gira",
            "hager",
            "siemens",
            "abb",
            "schneider",
            "wago",
            "kabel",
            "leitung",
            "schalter",
            "steckdose",
            "led",
        ]

        elektro_items = 0
        for classification in classifications:
            description = classification.get("description", "").lower()
            if any(keyword in description for keyword in elektro_keywords):
                elektro_items += 1

        elektro_ratio = elektro_items / len(classifications)
        quality_indicators += elektro_ratio
        total_checks += 1

        # Check 2: Verwendung von SKR03-Konten für Elektromaterial
        elektro_accounts = ["3400", "4985", "0200"]  # Typische Elektrotechnik-Konten
        correct_accounts = 0

        for classification in classifications:
            account = classification.get("skr03_konto", "")
            if account in elektro_accounts:
                correct_accounts += 1

        account_ratio = correct_accounts / len(classifications)
        quality_indicators += account_ratio
        total_checks += 1

        # Check 3: Konsistenz der Klassifizierungsmethoden
        method_consistency = self._check_method_consistency(classifications)
        quality_indicators += method_consistency
        total_checks += 1

        return quality_indicators / total_checks if total_checks > 0 else 0.0

    def _check_method_consistency(self, classifications: list[dict[str, Any]]) -> float:
        """Prüft Konsistenz der verwendeten Klassifizierungsmethoden"""

        method_counts = {}

        for classification in classifications:
            method = classification.get("classification_method", "unknown")
            method_counts[method] = method_counts.get(method, 0) + 1

        if not method_counts:
            return 0.0

        # Bewerte Vielfalt der Methoden (zeigt robuste Klassifizierung)
        len(classifications)

        # Ideal: Mix aus regel- und RAG-basiert
        rule_based = sum(
            count for method, count in method_counts.items() if "rule" in method
        )
        rag_based = sum(
            count for method, count in method_counts.items() if "rag" in method
        )

        if rule_based > 0 and rag_based > 0:
            return 0.9  # Gute Balance
        elif rule_based > 0 or rag_based > 0:
            return 0.7  # Nur eine Methode dominant
        else:
            return 0.3  # Unbekannte Methoden

    def _assess_supplier_recognition(
        self, structured_data: dict[str, Any], classifications: list[dict[str, Any]]
    ) -> float:
        """Bewertet Qualität der Lieferanten-Erkennung"""

        # Lieferant aus Header
        header_supplier = structured_data.get("invoice_header", {}).get(
            "supplier_name", ""
        )

        # Lieferanten aus RAG-Klassifizierungen
        rag_suppliers = []
        for classification in classifications:
            supplier = classification.get("supplier_detected", "")
            if supplier and supplier != "Unknown":
                rag_suppliers.append(supplier)

        # Bewertung
        if header_supplier and rag_suppliers:
            # Prüfe Konsistenz
            most_common_rag_supplier = (
                max(set(rag_suppliers), key=rag_suppliers.count)
                if rag_suppliers
                else ""
            )
            if (
                header_supplier.lower() in most_common_rag_supplier.lower()
                or most_common_rag_supplier.lower() in header_supplier.lower()
            ):
                return 1.0  # Perfekte Übereinstimmung
            else:
                return 0.7  # Beide erkannt, aber unterschiedlich
        elif header_supplier or rag_suppliers:
            return 0.6  # Nur eine Quelle hat Lieferant erkannt
        else:
            return 0.2  # Keine Lieferanten-Erkennung

    def _log_quality_metrics(
        self, metrics: dict[str, Any], overall_score: float
    ) -> None:
        """Protokolliert detaillierte Qualitätsmetriken für Debugging"""

        logger.info(f"📊 QUALITÄTSMETRIKEN (Gesamt: {overall_score:.1%}):")
        logger.info(
            f"   📋 Header-Vollständigkeit: {metrics.get('header_completeness', 0)}/4 Felder"
        )
        logger.info(
            f"   🎯 Ø Klassifizierungs-Konfidenz: {metrics.get('avg_classification_confidence', 0):.1%}"
        )
        logger.info(f"   🤖 RAG-Nutzung: {metrics.get('rag_usage_percentage', 0):.1%}")
        logger.info(
            f"   📊 Vollständige Positionen: {metrics.get('complete_items_percentage', 0):.1%}"
        )
        logger.info(
            f"   💰 Betragskonsistenz: {metrics.get('amount_consistency', 0):.1%}"
        )
        logger.info(
            f"   ⚡ Elektrotechnik-Qualität: {metrics.get('elektro_quality', 0):.1%}"
        )
        logger.info(
            f"   🏢 Lieferanten-Erkennung: {metrics.get('supplier_recognition', 0):.1%}"
        )

    def _assess_quality(self, confidence_score: float) -> str:
        """
        Erweiterte Qualitätsbewertung mit detaillierteren Kategorien.

        Berücksichtigt die umfassende Konfidenz-Bewertung für präzisere Kategorisierung.
        """

        if confidence_score >= 0.9:
            return "excellent"  # Exzellente Qualität - produktionsbereit
        elif confidence_score >= 0.8:
            return "high"  # Hohe Qualität - minimale Nachbearbeitung
        elif confidence_score >= 0.65:
            return "good"  # Gute Qualität - moderate Nachbearbeitung
        elif confidence_score >= 0.5:
            return "medium"  # Mittlere Qualität - erhebliche Nachbearbeitung
        elif confidence_score >= 0.3:
            return "low"  # Niedrige Qualität - umfassende Überprüfung nötig
        else:
            return "poor"  # Schlechte Qualität - manuelle Bearbeitung empfohlen

    def _vectorize_and_store(self, result: ProcessingResult) -> None:
        """Vektorisiert Rechnungspositionen und speichert sie in ChromaDB."""
        if not self.invoice_collection or not self.embedding_model:
            logger.warning(
                "Vektordatenbank nicht initialisiert. Überspringe Speicherung."
            )
            return

        items = result.skr03_classifications
        if not items:
            return

        documents_to_embed = []
        metadatas_to_store = []
        ids_to_add = []

        for i, item in enumerate(items):
            # Der Text, der in einen Vektor umgewandelt wird (Kontext ist König!)
            content = f"Lieferant: {result.invoice_data.get('supplier', '')} | Artikel: {item.get('description', '')}"
            documents_to_embed.append(content)

            # Die Metadaten, die wir abrufen wollen
            metadatas_to_store.append(
                {
                    "supplier": result.invoice_data.get("supplier", ""),
                    "description": item.get("description", ""),
                    "skr03_konto": item.get("skr03_konto", "N/A"),
                    "amount": float(str(item.get("amount", "0")).replace(",", ".")),
                    "pdf_path": result.pdf_path,
                }
            )

            # Eindeutige ID
            ids_to_add.append(f"{Path(result.pdf_path).stem}_{i}")

        try:
            # Vektoren erstellen
            embeddings = self.embedding_model.encode(documents_to_embed).tolist()

            # In ChromaDB speichern
            self.invoice_collection.add(
                embeddings=embeddings,
                documents=documents_to_embed,
                metadatas=metadatas_to_store,
                ids=ids_to_add,
            )
            logger.info(
                f"{len(items)} Positionen für '{Path(result.pdf_path).name}' in Vektordatenbank gespeichert."
            )
        except Exception as e:
            logger.error(f"Fehler beim Speichern in Vektordatenbank: {e}")

    # =================================================================
    # EXPORT-FUNKTIONEN FÜR IMPORTFERTIGE BUCHHALTUNGSDATEN
    # =================================================================

    def export_for_accounting_import(
        self,
        result: ProcessingResult,
        export_format: str = "datev",
        output_path: Path | str | None = None,
    ) -> dict[str, Path]:
        """
        Exportiert verarbeitete Rechnungsdaten für den direkten Import in Buchhaltungssoftware.

        Unterstützte Formate:
        - datev: DATEV CSV-Format für deutsche Buchhaltung
        - csv: Standard CSV für Excel/allgemeine Verwendung
        - json: Strukturierte JSON für API-Integration
        - bmv: Buchung mit Vorlage für DATEV

        Args:
            result: Verarbeitetes Rechnungsergebnis
            export_format: Gewünschtes Exportformat
            output_path: Ausgabepfad (optional)

        Returns:
            Dictionary mit Pfaden zu den exportierten Dateien
        """

        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pdf_name = Path(result.pdf_path).stem
            output_path = self.config.output_path / f"{pdf_name}_accounting_{timestamp}"
        else:
            output_path = Path(output_path)

        output_path.mkdir(parents=True, exist_ok=True)

        exported_files = {}

        if export_format in ["datev", "all"]:
            datev_path = self._export_datev_format(result, output_path)
            exported_files["datev"] = datev_path

        if export_format in ["csv", "all"]:
            csv_path = self._export_csv_format(result, output_path)
            exported_files["csv"] = csv_path

        if export_format in ["json", "all"]:
            json_path = self._export_structured_json(result, output_path)
            exported_files["json"] = json_path

        if export_format in ["bmv", "all"]:
            bmv_path = self._export_bmv_format(result, output_path)
            exported_files["bmv"] = bmv_path

        # Zusätzlich: Zusammenfassungsbericht
        summary_path = self._generate_accounting_summary(result, output_path)
        exported_files["summary"] = summary_path

        logger.info(
            f"📤 Buchhaltungsexport abgeschlossen: {len(exported_files)} Dateien erstellt"
        )
        return exported_files

    def _export_datev_format(self, result: ProcessingResult, output_path: Path) -> Path:
        """Exportiert im DATEV CSV-Format für deutsche Buchhaltung"""

        import csv

        datev_path = output_path / "datev_import.csv"

        # DATEV CSV-Header (vereinfacht)
        datev_headers = [
            "Konto",  # SKR03-Kontonummer
            "Gegenkonto",  # Kreditorenkonto (Lieferant)
            "Buchungstext",  # Beschreibung
            "Belegfeld1",  # Rechnungsnummer
            "Datum",  # Rechnungsdatum
            "Betrag",  # Nettobetrag
            "Steuerschlüssel",  # USt-Schlüssel
            "USt-Betrag",  # USt-Betrag
            "Kostenstelle",  # Kostenstelle (optional)
            "Zusatzinformation",  # Weitere Infos
        ]

        with open(datev_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile, delimiter=";", quoting=csv.QUOTE_MINIMAL)

            # Header schreiben
            writer.writerow(datev_headers)

            # Rechnungsdaten
            invoice_data = result.invoice_data
            supplier_name = invoice_data.get("supplier_name", "Unbekannter Lieferant")
            invoice_number = invoice_data.get("invoice_number", "")
            invoice_date = invoice_data.get("date", datetime.now().strftime("%d.%m.%Y"))

            # Jede Position als separate Buchungszeile
            for classification in result.skr03_classifications:
                description = classification.get("description", "")
                skr03_account = classification.get("skr03_konto", "3400")
                amount = self._parse_amount(classification.get("amount", "0"))

                # USt-Berechnung (19% für Elektromaterial)
                vat_rate = 0.19
                net_amount = amount / (1 + vat_rate)
                vat_amount = amount - net_amount

                # DATEV-Zeile
                datev_row = [
                    skr03_account,  # Konto
                    "7000",  # Gegenkonto (Kreditor)
                    f"{supplier_name}: {description[:40]}",  # Buchungstext (gekürzt)
                    invoice_number,  # Belegfeld1
                    invoice_date,  # Datum
                    f"{net_amount:.2f}".replace(".", ","),  # Betrag (deutsches Format)
                    "19",  # Steuerschlüssel (19% USt)
                    f"{vat_amount:.2f}".replace(".", ","),  # USt-Betrag
                    "",  # Kostenstelle
                    f"Konfidenz: {classification.get('confidence', 0):.1%}",  # Zusatzinfo
                ]

                writer.writerow(datev_row)

        logger.info(f"📋 DATEV-Export erstellt: {datev_path}")
        return datev_path

    def _export_csv_format(self, result: ProcessingResult, output_path: Path) -> Path:
        """Exportiert als Standard-CSV für Excel und allgemeine Verwendung"""

        import csv

        csv_path = output_path / "accounting_data.csv"

        with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile, delimiter=",")

            # Header
            headers = [
                "Position",
                "Beschreibung",
                "SKR03_Konto",
                "Kategorie",
                "Betrag_EUR",
                "Menge",
                "Einzelpreis",
                "Lieferant",
                "Rechnungsnummer",
                "Rechnungsdatum",
                "Konfidenz",
                "Klassifizierungsmethode",
                "Qualitätsbewertung",
            ]
            writer.writerow(headers)

            # Rechnungsdaten
            invoice_data = result.invoice_data
            supplier = invoice_data.get("supplier_name", "")
            invoice_number = invoice_data.get("invoice_number", "")
            invoice_date = invoice_data.get("date", "")

            # Positionen
            for classification in result.skr03_classifications:
                amount = self._parse_amount(classification.get("amount", "0"))
                quantity = classification.get("quantity", "1")

                try:
                    unit_price = (
                        amount / float(quantity) if float(quantity) > 0 else amount
                    )
                except (ValueError, ZeroDivisionError):
                    unit_price = amount

                row = [
                    classification.get("position", ""),
                    classification.get("description", ""),
                    classification.get("skr03_konto", ""),
                    classification.get("category", ""),
                    f"{amount:.2f}",
                    quantity,
                    f"{unit_price:.2f}",
                    supplier,
                    invoice_number,
                    invoice_date,
                    f"{classification.get('confidence', 0):.1%}",
                    classification.get("classification_method", ""),
                    result.extraction_quality,
                ]
                writer.writerow(row)

        logger.info(f"📊 CSV-Export erstellt: {csv_path}")
        return csv_path

    def _export_structured_json(
        self, result: ProcessingResult, output_path: Path
    ) -> Path:
        """Exportiert strukturierte JSON für API-Integration"""

        json_path = output_path / "structured_accounting.json"

        # Strukturierte Daten für API-Integration
        structured_export = {
            "metadata": {
                "export_timestamp": datetime.now().isoformat(),
                "pdf_source": result.pdf_path,
                "processing_time_ms": result.processing_time_ms,
                "confidence_score": result.confidence_score,
                "extraction_quality": result.extraction_quality,
                "total_positions": len(result.skr03_classifications),
            },
            "invoice_header": {
                "supplier_name": result.invoice_data.get("supplier_name", ""),
                "invoice_number": result.invoice_data.get("invoice_number", ""),
                "invoice_date": result.invoice_data.get("date", ""),
                "total_amount": result.invoice_data.get("total_amount", ""),
                "customer_number": result.invoice_data.get("customer_number", ""),
                "due_date": result.invoice_data.get("due_date", ""),
            },
            "line_items": [],
            "summary": {
                "total_net_amount": 0.0,
                "total_vat_amount": 0.0,
                "total_gross_amount": 0.0,
                "classification_methods_used": {},
                "quality_indicators": {
                    "high_confidence_items": 0,
                    "rag_classified_items": 0,
                    "rule_classified_items": 0,
                },
            },
        }

        # Verarbeite Positionen
        total_amount = 0.0
        method_counts = {}
        high_confidence_count = 0

        for classification in result.skr03_classifications:
            amount = self._parse_amount(classification.get("amount", "0"))
            total_amount += amount

            # Klassifizierungsmethode zählen
            method = classification.get("classification_method", "unknown")
            method_counts[method] = method_counts.get(method, 0) + 1

            # Konfidenz bewerten
            if classification.get("confidence", 0) >= 0.8:
                high_confidence_count += 1

            # Position hinzufügen
            item = {
                "position": classification.get("position", ""),
                "description": classification.get("description", ""),
                "skr03_account": classification.get("skr03_konto", ""),
                "category": classification.get("category", ""),
                "amount_eur": amount,
                "quantity": classification.get("quantity", ""),
                "confidence": classification.get("confidence", 0),
                "classification_method": method,
                "reasoning": classification.get("reasoning", ""),
                "similar_articles": classification.get("similar_articles", []),
            }

            structured_export["line_items"].append(item)

        # Summary aktualisieren
        vat_amount = total_amount * 0.19 / 1.19  # 19% USt zurückrechnen
        net_amount = total_amount - vat_amount

        structured_export["summary"].update(
            {
                "total_net_amount": round(net_amount, 2),
                "total_vat_amount": round(vat_amount, 2),
                "total_gross_amount": round(total_amount, 2),
                "classification_methods_used": method_counts,
                "quality_indicators": {
                    "high_confidence_items": high_confidence_count,
                    "rag_classified_items": method_counts.get("rag_dominant", 0)
                    + method_counts.get("rag_similarity", 0),
                    "rule_classified_items": method_counts.get("rule_dominant", 0)
                    + method_counts.get("rule_fallback", 0),
                },
            }
        )

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(structured_export, f, indent=2, ensure_ascii=False)

        logger.info(f"🔗 JSON-Export erstellt: {json_path}")
        return json_path

    def _export_bmv_format(self, result: ProcessingResult, output_path: Path) -> Path:
        """Exportiert Buchungsvorlage (BMV) für DATEV"""

        bmv_path = output_path / "buchungsvorlage.txt"

        with open(bmv_path, "w", encoding="utf-8") as f:
            f.write("# DATEV Buchungsvorlage - Automatisch generiert\n")
            f.write(f"# Quelle: {Path(result.pdf_path).name}\n")
            f.write(f"# Erstellt: {datetime.now().strftime('%d.%m.%Y %H:%M')}\n")
            f.write(
                f"# Qualität: {result.extraction_quality} ({result.confidence_score:.1%})\n\n"
            )

            invoice_data = result.invoice_data
            f.write(f"Lieferant: {invoice_data.get('supplier_name', 'Unbekannt')}\n")
            f.write(f"Rechnungsnr: {invoice_data.get('invoice_number', '')}\n")
            f.write(f"Datum: {invoice_data.get('date', '')}\n\n")

            f.write("BUCHUNGSSÄTZE:\n")
            f.write("=" * 50 + "\n")

            for i, classification in enumerate(result.skr03_classifications, 1):
                amount = self._parse_amount(classification.get("amount", "0"))

                f.write(f"\nPosition {i}:\n")
                f.write(f"  Beschreibung: {classification.get('description', '')}\n")
                f.write(
                    f"  Konto: {classification.get('skr03_konto', '')} ({classification.get('category', '')})\n"
                )
                f.write(f"  Betrag: {amount:.2f} EUR\n")
                f.write(f"  Konfidenz: {classification.get('confidence', 0):.1%}\n")
                f.write(
                    f"  Methode: {classification.get('classification_method', '')}\n"
                )

                if classification.get("reasoning"):
                    f.write(f"  Begründung: {classification.get('reasoning', '')}\n")

        logger.info(f"📝 BMV-Vorlage erstellt: {bmv_path}")
        return bmv_path

    def _generate_accounting_summary(
        self, result: ProcessingResult, output_path: Path
    ) -> Path:
        """Generiert Zusammenfassungsbericht für die Buchhaltung"""

        summary_path = output_path / "accounting_summary.txt"

        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("ELEKTROTECHNIK UG - RECHNUNGSVERARBEITUNGS-ZUSAMMENFASSUNG\n")
            f.write("=" * 60 + "\n\n")

            # Allgemeine Informationen
            f.write(f"PDF-Datei: {Path(result.pdf_path).name}\n")
            f.write(f"Verarbeitet am: {result.processing_timestamp}\n")
            f.write(f"Verarbeitungszeit: {result.processing_time_ms}ms\n")
            f.write(
                f"Gesamtqualität: {result.extraction_quality} ({result.confidence_score:.1%})\n\n"
            )

            # Rechnungsdaten
            invoice_data = result.invoice_data
            f.write("RECHNUNGSDATEN:\n")
            f.write("-" * 20 + "\n")
            f.write(
                f"Lieferant: {invoice_data.get('supplier_name', 'Nicht erkannt')}\n"
            )
            f.write(
                f"Rechnungsnummer: {invoice_data.get('invoice_number', 'Nicht erkannt')}\n"
            )
            f.write(f"Rechnungsdatum: {invoice_data.get('date', 'Nicht erkannt')}\n")
            f.write(
                f"Gesamtbetrag: {invoice_data.get('total_amount', 'Nicht erkannt')}\n\n"
            )

            # Positionen-Analyse
            f.write("POSITIONEN-ANALYSE:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Anzahl Positionen: {len(result.skr03_classifications)}\n")

            # Konfidenz-Statistiken
            confidences = [c.get("confidence", 0) for c in result.skr03_classifications]
            if confidences:
                avg_confidence = sum(confidences) / len(confidences)
                high_confidence = sum(1 for c in confidences if c >= 0.8)
                f.write(f"Durchschnittliche Konfidenz: {avg_confidence:.1%}\n")
                f.write(
                    f"Hohe Konfidenz (≥80%): {high_confidence}/{len(confidences)}\n"
                )

            # Methoden-Statistiken
            methods = [
                c.get("classification_method", "unknown")
                for c in result.skr03_classifications
            ]
            method_counts = {}
            for method in methods:
                method_counts[method] = method_counts.get(method, 0) + 1

            f.write("\nKLASSIFIZIERUNGSMETHODEN:\n")
            for method, count in method_counts.items():
                f.write(f"  {method}: {count} Positionen\n")

            # SKR03-Verteilung
            accounts = [c.get("skr03_konto", "") for c in result.skr03_classifications]
            account_counts = {}
            for account in accounts:
                account_counts[account] = account_counts.get(account, 0) + 1

            f.write("\nSKR03-KONTEN-VERTEILUNG:\n")
            for account, count in sorted(account_counts.items()):
                f.write(f"  Konto {account}: {count} Positionen\n")

            # Empfehlungen
            f.write("\nEMPFEHLUNGEN:\n")
            f.write("-" * 15 + "\n")

            if result.confidence_score >= 0.9:
                f.write("✅ Exzellente Qualität - Direkter Import empfohlen\n")
            elif result.confidence_score >= 0.8:
                f.write("✅ Hohe Qualität - Minimale Überprüfung empfohlen\n")
            elif result.confidence_score >= 0.6:
                f.write("⚠️  Mittlere Qualität - Moderate Überprüfung empfohlen\n")
            else:
                f.write("❌ Niedrige Qualität - Umfassende Überprüfung erforderlich\n")

            low_confidence_items = [
                c for c in result.skr03_classifications if c.get("confidence", 0) < 0.5
            ]
            if low_confidence_items:
                f.write(
                    f"\n⚠️  {len(low_confidence_items)} Positionen mit niedriger Konfidenz prüfen:\n"
                )
                for item in low_confidence_items[:5]:  # Zeige nur erste 5
                    f.write(f"   - {item.get('description', '')[:50]}...\n")

        logger.info(f"📋 Zusammenfassung erstellt: {summary_path}")
        return summary_path

    def _parse_amount(self, amount_str: str | float) -> float:
        """Hilfsfunktion zum Parsen von Geldbeträgen"""

        if isinstance(amount_str, int | float):
            return float(amount_str)

        try:
            # Entferne Währungssymbole und Leerzeichen
            clean_amount = str(amount_str).replace("€", "").replace("EUR", "").strip()
            # Ersetze Komma durch Punkt für deutschen Zahlenformat
            clean_amount = clean_amount.replace(",", ".")
            return float(clean_amount)
        except (ValueError, TypeError):
            return 0.0

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
