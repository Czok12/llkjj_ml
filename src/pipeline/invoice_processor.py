"""
LLKJJ ML Pipeline - PDF Verarbeitungs-Pipeline

Hauptklasse für die automatisierte Verarbeitung von PDF-Rechnungen
für Elektrotechnik-Handwerk UG.
"""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# Setup basic logging for now (dependencies will be installed later)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class InvoiceData:
    """Strukturierte Rechnungsdaten"""

    # Rechnungsmetadaten (required fields first)
    invoice_number: str
    invoice_date: str
    supplier: str
    net_amount: float
    vat_amount: float
    gross_amount: float

    # Optional fields with defaults
    supplier_number: str | None = None
    order_number: str | None = None
    currency: str = "EUR"
    line_items: list[dict[str, Any]] | None = None
    skr03_mappings: dict[str, str] | None = None
    processing_timestamp: str | None = None
    confidence_score: float = 0.0

    def __post_init__(self) -> None:
        if self.line_items is None:
            self.line_items = []
        if self.skr03_mappings is None:
            self.skr03_mappings = {}
        if self.processing_timestamp is None:
            self.processing_timestamp = datetime.now().isoformat()


@dataclass
class LineItem:
    """Einzelne Rechnungsposition"""

    description: str
    item_number: str | None = None
    quantity: float = 1.0
    unit_price: float = 0.0
    total_price: float = 0.0
    vat_rate: float = 0.19
    product_category: str = "elektromaterial"
    skr03_account: str = "4830"


class InvoiceProcessor:
    """Hauptklasse für PDF-Rechnungsverarbeitung"""

    def __init__(self, config_path: Path | None = None):
        """
        Initialisiert den Invoice Processor

        Args:
            config_path: Pfad zur Konfigurationsdatei
        """
        self.config_path = config_path
        self.setup_logging()

        # Placeholder für Services (werden später implementiert)
        self.pdf_extractor = None
        self.gemini_client = None
        self.vector_store = None
        self.spacy_trainer = None

        logger.info("InvoiceProcessor initialisiert")

    def setup_logging(self) -> None:
        """Logging-Konfiguration"""
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler("logs/pipeline.log", encoding="utf-8"),
                logging.StreamHandler(),
            ],
        )

    def process_pdf(self, pdf_path: Path) -> InvoiceData:
        """
        Verarbeitet eine PDF-Rechnung vollständig

        Args:
            pdf_path: Pfad zur PDF-Datei

        Returns:
            InvoiceData: Extrahierte und klassifizierte Rechnungsdaten
        """
        logger.info(f"Starte PDF-Verarbeitung: {pdf_path}")

        try:
            # Schritt 1: PDF Text extrahieren
            pdf_text = self._extract_pdf_text(pdf_path)

            # Schritt 2: Gemini 2.5 Pro Analyse
            gemini_result = self._analyze_with_gemini(pdf_text)

            # Schritt 3: SKR03 Klassifizierung
            classified_data = self._classify_skr03(gemini_result)

            # Schritt 4: Daten strukturieren
            invoice_data = self._structure_invoice_data(classified_data)

            # Schritt 5: Vektorisierung für Training
            self._vectorize_and_store(invoice_data, pdf_text)

            # Schritt 6: JSONL für spaCy Training
            self._export_training_data(invoice_data, pdf_text)

            logger.info(f"PDF-Verarbeitung erfolgreich: {invoice_data.invoice_number}")
            return invoice_data

        except Exception as e:
            logger.error(f"Fehler bei PDF-Verarbeitung: {e}")
            raise

    def _extract_pdf_text(self, pdf_path: Path) -> str:
        """PDF Text-Extraktion (Placeholder)"""
        logger.info("Extrahiere Text aus PDF...")

        # Placeholder - wird später mit pdfplumber implementiert
        return f"PDF Inhalt von {pdf_path.name} - Placeholder Text für Entwicklung"

    def _analyze_with_gemini(self, text: str) -> dict[str, Any]:
        """Gemini 2.5 Pro Analyse (Placeholder)"""
        logger.info("Analysiere mit Gemini 2.5 Pro...")

        # Placeholder - wird später mit Google AI implementiert
        return {
            "invoice_number": "RE-2024-001",
            "invoice_date": "2024-08-14",
            "supplier": "Elektro Conrad GmbH",
            "net_amount": 156.78,
            "vat_amount": 29.79,
            "gross_amount": 186.57,
            "line_items": [
                {
                    "description": "Installationsleitung NYM-J 3x1,5 mm²",
                    "quantity": 100.0,
                    "unit_price": 1.25,
                    "total_price": 125.00,
                    "product_category": "elektromaterial",
                },
                {
                    "description": "Steckdose SCHUKO weiß",
                    "quantity": 5.0,
                    "unit_price": 6.36,
                    "total_price": 31.78,
                    "product_category": "elektromaterial",
                },
            ],
        }

    def _classify_skr03(self, gemini_data: dict[str, Any]) -> dict[str, Any]:
        """SKR03 Klassifizierung"""
        logger.info("Klassifiziere nach SKR03...")

        # SKR03 Mapping für Elektrohandwerk
        category_mapping = {
            "elektromaterial": "4830",
            "werkzeug": "0490",
            "bueroartikel": "4935",
            "fremdleistung": "4400",
        }

        # Füge SKR03 Konten zu Positionen hinzu
        for item in gemini_data.get("line_items", []):
            category = item.get("product_category", "elektromaterial")
            item["skr03_account"] = category_mapping.get(category, "4830")

        gemini_data["skr03_mappings"] = category_mapping
        return gemini_data

    def _structure_invoice_data(self, classified_data: dict[str, Any]) -> InvoiceData:
        """Strukturiert Daten in InvoiceData Objekt"""
        logger.info("Strukturiere Rechnungsdaten...")

        return InvoiceData(
            invoice_number=classified_data.get("invoice_number", ""),
            invoice_date=classified_data.get("invoice_date", ""),
            supplier=classified_data.get("supplier", ""),
            net_amount=classified_data.get("net_amount", 0.0),
            vat_amount=classified_data.get("vat_amount", 0.0),
            gross_amount=classified_data.get("gross_amount", 0.0),
            line_items=classified_data.get("line_items", []),
            skr03_mappings=classified_data.get("skr03_mappings", {}),
            confidence_score=0.95,  # Placeholder
        )

    def _vectorize_and_store(
        self, invoice_data: InvoiceData, original_text: str
    ) -> None:
        """Vektorisierung für ML Training (Placeholder)"""
        logger.info("Vektorisiere Daten...")

        # Placeholder - wird später mit sentence-transformers implementiert
        vector_data = {
            "invoice_id": invoice_data.invoice_number,
            "text": original_text,
            "structured_data": asdict(invoice_data),
            "embedding_placeholder": [0.1] * 384,  # Placeholder Vektor
        }

        # Speichere in JSON (später ChromaDB)
        output_path = (
            Path("data/vectors") / f"{invoice_data.invoice_number}_vector.json"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(vector_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Vektor gespeichert: {output_path}")

    def _export_training_data(
        self, invoice_data: InvoiceData, original_text: str
    ) -> None:
        """Exportiert Daten für spaCy Training"""
        logger.info("Exportiere Training-Daten...")

        # spaCy JSONL Format
        training_entry: dict[str, object] = {
            "text": original_text,
            "entities": [],
            "cats": {
                "INVOICE": 1.0,
                "ELEKTRO": 1.0 if "elektro" in invoice_data.supplier.lower() else 0.0,
            },
            "meta": {
                "invoice_number": invoice_data.invoice_number,
                "supplier": invoice_data.supplier,
                "skr03_mappings": invoice_data.skr03_mappings,
            },
        }

        # Placeholder Entitäten (später NER training)
        training_entry["entities"] = [
            {
                "start": 0,
                "end": len(invoice_data.invoice_number),
                "label": "INVOICE_NUMBER",
            },
            {"start": 0, "end": len(invoice_data.supplier), "label": "SUPPLIER"},
        ]

        # Speichere JSONL
        output_path = Path("data/training") / "spacy_training.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(training_entry, ensure_ascii=False) + "\n")

        logger.info(f"Training-Daten exportiert: {output_path}")

    def batch_process(
        self, pdf_directory: Path, output_directory: Path | None = None
    ) -> list[InvoiceData]:
        """
        Verarbeitet mehrere PDFs in einem Batch

        Args:
            pdf_directory: Verzeichnis mit PDF-Dateien
            output_directory: Ausgabeverzeichnis für Ergebnisse

        Returns:
            List[InvoiceData]: Liste aller verarbeiteten Rechnungen
        """
        logger.info(f"Starte Batch-Verarbeitung: {pdf_directory}")

        pdf_files = list(pdf_directory.glob("*.pdf"))
        results = []

        for pdf_file in pdf_files:
            try:
                invoice_data = self.process_pdf(pdf_file)
                results.append(invoice_data)

                # Speichere Einzelergebnis
                if output_directory:
                    output_file = (
                        output_directory / f"{invoice_data.invoice_number}.json"
                    )
                    with open(output_file, "w", encoding="utf-8") as f:
                        json.dump(asdict(invoice_data), f, ensure_ascii=False, indent=2)

            except Exception as e:
                logger.error(f"Fehler bei {pdf_file}: {e}")
                continue

        logger.info(
            f"Batch-Verarbeitung abgeschlossen: {len(results)} von {len(pdf_files)} erfolgreich"
        )
        return results


if __name__ == "__main__":
    # Test der Pipeline
    processor = InvoiceProcessor()

    # Erstelle Test-Verzeichnisse
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("data/vectors").mkdir(parents=True, exist_ok=True)
    Path("data/training").mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(parents=True, exist_ok=True)

    logger.info("LLKJJ ML Pipeline bereit für PDF-Verarbeitung!")
    logger.info("Nächste Schritte:")
    logger.info("1. Dependencies installieren: pip install -r requirements.txt")
    logger.info("2. .env Datei konfigurieren")
    logger.info("3. PDF-Dateien in data/raw/ ablegen")
    logger.info("4. Pipeline mit echten PDFs testen")
