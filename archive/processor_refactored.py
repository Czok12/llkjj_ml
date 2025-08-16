#!/usr/bin/env python3
"""
LLKJJ ML Pipeline - Refactored Unified Processor (KISS Architecture)
===================================================================

Schlanker Orchestrator der die spezialisierten Module koordiniert:
- DataExtractor: PDF-Extraktion und Datengewinnung
- DataClassifier: SKR03-Klassifizierung und RAG-System
- QualityAssessor: Qualitätsbewertung und Konfidenz-Scores

Folgt dem Single Responsibility Principle für bessere Wartbarkeit.

Autor: LLKJJ ML Pipeline Team
Version: 2.1.0 (Post-Konsolidierung Refactoring)
"""

import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer

from src.config import Config
from src.processing import DataClassifier, DataExtractor, QualityAssessor

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
    Schlanker Orchestrator für die modulare PDF-zu-SKR03-Pipeline.

    Koordiniert die spezialisierten Module:
    - DataExtractor für PDF-Verarbeitung und Extraktion
    - DataClassifier für SKR03-Klassifizierung und RAG-System
    - QualityAssessor für Qualitätsbewertung

    Folgt KISS-Prinzipien durch klare Verantwortungstrennung.
    """

    def __init__(self, config: Config | None = None):
        """Initialize unified processor with modular components"""
        self.config = config or Config()
        self._setup_logging()
        self._setup_vector_db()

        # Initialize specialized modules
        self.extractor = DataExtractor()
        self.classifier = DataClassifier(vector_store=self.invoice_collection)
        self.quality_assessor = QualityAssessor()

    def _setup_logging(self) -> None:
        """Configure logging for the processor"""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    def _setup_vector_db(self) -> None:
        """Setup ChromaDB and embedding model for RAG system"""
        try:
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=str(self.config.vector_db_path)
            )

            # Get or create collection for invoices
            self.invoice_collection = self.client.get_or_create_collection(
                name="invoice_line_items",
                metadata={
                    "description": "German electrical contractor invoice items for SKR03 classification"
                },
            )

            # Initialize embedding model for semantic search
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

            logger.info("✅ ChromaDB und Embedding-Model initialisiert")

        except Exception as e:
            logger.error("❌ Fehler bei Vektordatenbank-Setup: %s", e)
            self.invoice_collection = None
            self.embedding_model = None

    def process_pdf(self, pdf_path: str | Path) -> ProcessingResult:
        """
        Hauptfunktion: Verarbeitet eine PDF-Datei durch die modulare Pipeline.

        Args:
            pdf_path: Pfad zur PDF-Datei

        Returns:
            ProcessingResult mit allen Extraktions- und Klassifizierungsergebnissen
        """
        start_time = time.time()
        pdf_path = Path(pdf_path)

        logger.info("🚀 Starte modulare PDF-Verarbeitung: %s", pdf_path.name)

        try:
            # Phase 1: Extraktion mit DataExtractor
            extraction_start = time.time()
            extraction_result = self.extractor.process_pdf(pdf_path)
            ocr_time_ms = int((time.time() - extraction_start) * 1000)

            logger.info("✅ Extraktion abgeschlossen (%d ms)", ocr_time_ms)

            # Phase 2: Klassifizierung mit DataClassifier
            classification_start = time.time()
            classifications = self.classifier.process_classifications(
                extraction_result["line_items"], extraction_result["structured_data"]
            )
            classification_time_ms = int((time.time() - classification_start) * 1000)

            logger.info(
                "✅ Klassifizierung abgeschlossen (%d ms)", classification_time_ms
            )

            # Phase 3: Qualitätsbewertung mit QualityAssessor
            confidence_score = self.quality_assessor.calculate_confidence(
                extraction_result["structured_data"], classifications
            )
            quality_assessment = self.quality_assessor.assess_quality(confidence_score)

            logger.info(
                "✅ Qualitätsbewertung: %s (%.1f%%)",
                quality_assessment,
                confidence_score * 100,
            )

            # Phase 4: Vektorisierung für RAG-System (optional)
            if self.invoice_collection and self.embedding_model:
                result = ProcessingResult(
                    pdf_path=str(pdf_path),
                    processing_timestamp=datetime.now().isoformat(),
                    raw_text=extraction_result["raw_text"],
                    structured_data=extraction_result["structured_data"],
                    invoice_data=extraction_result["invoice_data"],
                    skr03_classifications=classifications,
                    processing_time_ms=int((time.time() - start_time) * 1000),
                    ocr_time_ms=ocr_time_ms,
                    classification_time_ms=classification_time_ms,
                    confidence_score=confidence_score,
                    extraction_quality=quality_assessment,
                )
                self._vectorize_and_store(result)

            # Ergebnis zusammenstellen
            processing_time_ms = int((time.time() - start_time) * 1000)

            result = ProcessingResult(
                pdf_path=str(pdf_path),
                processing_timestamp=datetime.now().isoformat(),
                raw_text=extraction_result["raw_text"],
                structured_data=extraction_result["structured_data"],
                invoice_data=extraction_result["invoice_data"],
                skr03_classifications=classifications,
                processing_time_ms=processing_time_ms,
                ocr_time_ms=ocr_time_ms,
                classification_time_ms=classification_time_ms,
                confidence_score=confidence_score,
                extraction_quality=quality_assessment,
            )

            logger.info(
                "🎉 PDF-Verarbeitung erfolgreich abgeschlossen: %d Positionen, %.1f%% Konfidenz",
                len(classifications),
                confidence_score * 100,
            )

            return result

        except Exception as e:
            logger.error("❌ Fehler bei PDF-Verarbeitung: %s", e)
            raise

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
                "%d Positionen für '%s' in Vektordatenbank gespeichert.",
                len(items),
                Path(result.pdf_path).name,
            )
        except Exception as e:
            logger.error("Fehler beim Speichern in Vektordatenbank: %s", e)

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
            output_path = (
                self.config.data_processed_path / f"{pdf_name}_accounting_{timestamp}"
            )
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
            "📤 Buchhaltungsexport abgeschlossen: %d Dateien erstellt",
            len(exported_files),
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

        logger.info("📋 DATEV-Export erstellt: %s", datev_path)
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

        logger.info("📊 CSV-Export erstellt: %s", csv_path)
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
        method_counts: dict[str, int] = {}
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

        logger.info("🔗 JSON-Export erstellt: %s", json_path)
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

        logger.info("📝 BMV-Vorlage erstellt: %s", bmv_path)
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
            method_counts: dict[str, int] = {}
            for method in methods:
                method_counts[method] = method_counts.get(method, 0) + 1

            f.write("\nKLASSIFIZIERUNGSMETHODEN:\n")
            for method, count in method_counts.items():
                f.write(f"  {method}: {count} Positionen\n")

            # SKR03-Verteilung
            accounts = [c.get("skr03_konto", "") for c in result.skr03_classifications]
            account_counts: dict[str, int] = {}
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

        logger.info("📋 Zusammenfassung erstellt: %s", summary_path)
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

        logger.info("Results saved to: %s", output_path)
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
