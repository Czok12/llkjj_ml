#!/usr/bin/env python3
"""
LLKJJ ML Pipeline - Training Service (KISS Architecture)
=====================================================

Consolidated training functionality combining data export and spaCy training
into a single, streamlined service following KISS principles.

Replaces separate data_exporter.py and spacy_trainer.py modules.

Author: LLKJJ ML Pipeline
Version: 2.0.0 (Post-Consolidation)
"""

import json
import logging
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import spacy
from spacy.pipeline import TextCategorizer
from spacy.tokens import DocBin
from spacy.training import Example
from spacy.util import minibatch

from src.config import Config

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Training performance metrics"""

    # Training data
    total_examples: int
    training_examples: int
    validation_examples: int

    # Performance metrics
    precision: float
    recall: float
    f1_score: float

    # Training info
    epochs_trained: int
    training_time_seconds: float
    model_size_mb: float

    # File paths
    model_path: str
    training_data_path: str


@dataclass
class ExportResult:
    """Data export result"""

    jsonl_path: str
    total_records: int
    skr03_classifications: int
    export_timestamp: str


class TrainingService:
    """
    Unified training service for spaCy NER models.

    Combines data export and model training functionality into a single
    service optimized for German electrical contractor invoices.
    """

    def __init__(self, config: Config | None = None):
        """Initialize training service"""
        self.config = config or Config()
        self._setup_logging()
        self._setup_spacy()

    def _setup_logging(self) -> None:
        """Configure logging"""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    def _setup_spacy(self) -> None:
        """Setup spaCy with German language model"""

        try:
            # Try to load German model
            self.nlp = spacy.load("de_core_news_sm")
            logger.info("German spaCy model loaded successfully")
        except OSError:
            logger.warning("German spaCy model not found, creating blank model")
            self.nlp = spacy.blank("de")
        # Add NER pipeline if not present
        if "ner" not in self.nlp.pipe_names:
            self.nlp.add_pipe("ner", last=True)
        ner = self.nlp.get_pipe("ner")
        # Add comprehensive Elektrotechnik & FIBU entity labels
        elektro_labels = [
            # Grundlegende Rechnungsdaten
            "INVOICE_NUMBER",
            "INVOICE_DATE",
            "DUE_DATE",
            "SUPPLIER_NAME",
            "SUPPLIER_ADDRESS",
            "SUPPLIER_VAT_ID",
            "CUSTOMER_NUMBER",
            "CUSTOMER_NAME",
            "DELIVERY_DATE",
            "ORDER_NUMBER",
            "PROJECT_NUMBER",
            # Steuerrelevante Beträge
            "NET_AMOUNT",
            "VAT_AMOUNT",
            "VAT_RATE",
            "GROSS_AMOUNT",
            "TOTAL_NET",
            "TOTAL_VAT",
            "TOTAL_GROSS",
            "DISCOUNT_AMOUNT",
            "FREIGHT_COSTS",
            # Produktdaten
            "PRODUCT_NAME",
            "PRODUCT_CODE",
            "MANUFACTURER_CODE",
            "QUANTITY",
            "UNIT_PRICE",
            "LINE_TOTAL",
            "UNIT_OF_MEASUREMENT",
            # SKR03 & Buchungsrelevant
            "SKR03_ACCOUNT",
            "COST_CENTER",
            "ELEKTRO_CATEGORY",
            "MATERIAL_GROUP",
            "ACCOUNT_ASSIGNMENT",
            "BOOKING_TEXT",
            # Elektrotechnik-spezifisch
            "CABLE_TYPE",
            "VOLTAGE_RATING",
            "AMPERAGE",
            "INSTALLATION_TYPE",
            "SAFETY_CLASS",
            "CERTIFICATION",
            "ENERGY_EFFICIENCY",
            # Zusätzliche FIBU-Daten
            "PAYMENT_TERMS",
            "BANK_DETAILS",
            "REFERENCE_NUMBER",
            "DUNNING_LEVEL",
            "CURRENCY",
        ]

        # Fix: Ensure ner is a spaCy NER pipe before adding labels
        if hasattr(ner, "add_label"):
            for label in elektro_labels:
                ner.add_label(label)
        else:
            # Remove and re-add NER pipe if misconfigured
            self.nlp.remove_pipe("ner")
            ner = self.nlp.add_pipe("ner", last=True)
            for label in elektro_labels:
                ner.add_label(label)

        self.entity_labels = elektro_labels
        logger.info(f"Added {len(elektro_labels)} entity labels for Elektrotechnik")

    def export_training_data(
        self, processed_data_dir: Path | str, output_path: Path | str | None = None
    ) -> ExportResult:
        """
        Export processed PDF data to spaCy training format.

        Args:
            processed_data_dir: Directory containing processed JSON files
            output_path: Output JSONL file path (optional)

        Returns:
            ExportResult with export statistics
        """

        processed_dir = Path(processed_data_dir)

        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = (
                self.config.training_data_path / f"elektro_training_{timestamp}.jsonl"
            )
        else:
            output_path = Path(output_path)

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Exporting training data from {processed_dir} to {output_path}")

        # Process all JSON files in directory
        training_examples = []
        skr03_count = 0

        for json_file in processed_dir.glob("*.json"):
            try:
                with open(json_file, encoding="utf-8") as f:
                    data = json.load(f)

                # Convert to spaCy training format
                spacy_examples = self._convert_to_spacy_format(data)
                training_examples.extend(spacy_examples)

                # Count SKR03 classifications
                if "skr03_classifications" in data:
                    skr03_count += len(data["skr03_classifications"])

                logger.debug(
                    f"Processed {json_file.name}: {len(spacy_examples)} examples"
                )

            except Exception as e:
                logger.warning(f"Failed to process {json_file}: {e}")
                continue

        # Write to JSONL format
        with open(output_path, "w", encoding="utf-8") as f:
            for example in training_examples:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")

        result = ExportResult(
            str(output_path),
            len(training_examples),
            skr03_count,
            datetime.now().isoformat(),
        )

        logger.info(
            f"Export complete: {len(training_examples)} training examples, "
            f"{skr03_count} SKR03 classifications"
        )

        return result

    def _convert_to_spacy_format(
        self, processed_data: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Convert processed PDF data to spaCy training format"""

        examples = []

        # Extract text and create entities from structured data
        raw_text = processed_data.get("raw_text", "")

        if not raw_text:
            return examples

        # Create entities from invoice header
        entities = []
        header = processed_data.get("invoice_data", {})

        # Find entities in text using simple string matching
        for field, value in header.items():
            if value and isinstance(value, str):
                start_idx = raw_text.find(value)
                if start_idx != -1:
                    end_idx = start_idx + len(value)
                    entity_label = self._map_field_to_entity(field)
                    entities.append([start_idx, end_idx, entity_label])

        # Add entities from line items
        line_items = processed_data.get("structured_data", {}).get("line_items", [])
        for item in line_items:
            for field, value in item.items():
                if value and isinstance(value, str) and len(value) > 2:
                    start_idx = raw_text.find(value)
                    if start_idx != -1:
                        end_idx = start_idx + len(value)
                        entity_label = self._map_field_to_entity(field)
                        entities.append([start_idx, end_idx, entity_label])

        # Create spaCy training example
        if entities:
            # Remove overlapping entities
            entities = self._remove_overlapping_entities(entities)

            example = {"text": raw_text, "entities": entities}
            examples.append(example)

        return examples

    def _map_field_to_entity(self, field: str) -> str:
        """Map field names to entity labels with comprehensive German/Elektrotechnik coverage"""

        field_mapping = {
            # Grundlegende Rechnungsdaten
            "invoice_number": "INVOICE_NUMBER",
            "rechnungsnummer": "INVOICE_NUMBER",
            "re_nr": "INVOICE_NUMBER",
            "belegnummer": "INVOICE_NUMBER",
            "date": "INVOICE_DATE",
            "invoice_date": "INVOICE_DATE",
            "rechnungsdatum": "INVOICE_DATE",
            "datum": "INVOICE_DATE",
            "due_date": "DUE_DATE",
            "fälligkeitsdatum": "DUE_DATE",
            "zahlungsziel": "DUE_DATE",
            "supplier_name": "SUPPLIER_NAME",
            "lieferant": "SUPPLIER_NAME",
            "firma": "SUPPLIER_NAME",
            "unternehmen": "SUPPLIER_NAME",
            "customer_number": "CUSTOMER_NUMBER",
            "kundennummer": "CUSTOMER_NUMBER",
            "debitorennummer": "CUSTOMER_NUMBER",
            # Steuerrelevante Beträge
            "total_amount": "GROSS_AMOUNT",
            "net_total": "NET_AMOUNT",
            "netto": "NET_AMOUNT",
            "nettobetrag": "NET_AMOUNT",
            "vat_amount": "VAT_AMOUNT",
            "mwst": "VAT_AMOUNT",
            "umsatzsteuer": "VAT_AMOUNT",
            "mehrwertsteuer": "VAT_AMOUNT",
            "vat_rate": "VAT_RATE",
            "steuersatz": "VAT_RATE",
            "mwst_satz": "VAT_RATE",
            "gross_total": "GROSS_AMOUNT",
            "brutto": "GROSS_AMOUNT",
            "gesamtbetrag": "GROSS_AMOUNT",
            "endbetrag": "GROSS_AMOUNT",
            # Produktdaten
            "description": "PRODUCT_NAME",
            "artikel": "PRODUCT_NAME",
            "bezeichnung": "PRODUCT_NAME",
            "produkt": "PRODUCT_NAME",
            "material": "PRODUCT_NAME",
            "product_code": "PRODUCT_CODE",
            "artikelnummer": "PRODUCT_CODE",
            "materialnummer": "PRODUCT_CODE",
            "bestellnummer": "PRODUCT_CODE",
            "quantity": "QUANTITY",
            "menge": "QUANTITY",
            "anzahl": "QUANTITY",
            "stück": "QUANTITY",
            "unit_price": "UNIT_PRICE",
            "einzelpreis": "UNIT_PRICE",
            "stückpreis": "UNIT_PRICE",
            "preis": "UNIT_PRICE",
            "total_price": "LINE_TOTAL",
            "line_total": "LINE_TOTAL",
            "gesamt": "LINE_TOTAL",
            "summe": "LINE_TOTAL",
            "positionssumme": "LINE_TOTAL",
            # Elektrotechnik-spezifisch
            "kabel": "CABLE_TYPE",
            "leitung": "CABLE_TYPE",
            "spannung": "VOLTAGE_RATING",
            "volt": "VOLTAGE_RATING",
            "strom": "AMPERAGE",
            "ampere": "AMPERAGE",
            "installation": "INSTALLATION_TYPE",
            "sicherheit": "SAFETY_CLASS",
            "zertifikat": "CERTIFICATION",
            "energieeffizienz": "ENERGY_EFFICIENCY",
            # SKR03 & Buchung
            "konto": "SKR03_ACCOUNT",
            "kontonummer": "SKR03_ACCOUNT",
            "sachkonto": "SKR03_ACCOUNT",
            "kostenstelle": "COST_CENTER",
            "kategorie": "ELEKTRO_CATEGORY",
            "materialgruppe": "MATERIAL_GROUP",
            "buchungstext": "BOOKING_TEXT",
            # Zahlungsinformationen
            "zahlungsbedingungen": "PAYMENT_TERMS",
            "bank": "BANK_DETAILS",
            "referenz": "REFERENCE_NUMBER",
            "währung": "CURRENCY",
        }

        return field_mapping.get(field.lower(), "PRODUCT_NAME")

    def _remove_overlapping_entities(
        self, entities: list[list[int | str]]
    ) -> list[list[int | str]]:
        """Remove overlapping entity annotations"""

        # Sort by start position
        entities.sort(key=lambda x: x[0])

        non_overlapping = []
        for entity in entities:
            start, end, label = entity

            # Check for overlap with previous entities
            overlap = False
            for prev_start, prev_end, _ in non_overlapping:
                if start < prev_end and end > prev_start:
                    overlap = True
                    break

            if not overlap:
                non_overlapping.append(entity)

        return non_overlapping

    # =================================================================
    # SPEZIALISIERTE MODELLE FÜR VOLLSTÄNDIGE AUTOMATISIERUNG
    # =================================================================

    def train_specialized_extraction_model(
        self,
        training_data_path: Path | str,
        output_model_path: Path | str | None = None,
        epochs: int = 30,
        dropout: float = 0.15,
    ) -> TrainingMetrics:
        """
        MODELL 1: Spezialisierte Datenextraktion für Elektrotechnik-Rechnungen.

        Trainiert ein hochspezialisiertes NER-Modell für zuverlässige Extraktion
        aller steuer- und fibu-relevanten Daten aus deutschen Elektrotechnik-Rechnungen.

        Args:
            training_data_path: Pfad zu spezialisierten Trainingsdaten
            output_model_path: Ausgabepfad für das Modell
            epochs: Anzahl Trainingsepochen (höher für bessere Präzision)
            dropout: Dropout-Rate für Regularisierung

        Returns:
            TrainingMetrics mit detaillierten Performance-Metriken
        """
        training_path = Path(training_data_path)

        if output_model_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_model_path = (
                self.config.models_path / f"elektro_extraction_model_{timestamp}"
            )
        else:
            output_model_path = Path(output_model_path)

        logger.info("🎯 Training Specialized Extraction Model (Modell 1)")
        logger.info(f"📊 Input: {training_path} -> Output: {output_model_path}")

        # Lade optimierte deutsche spaCy-Pipeline
        try:
            self.nlp = spacy.load(
                "de_core_news_lg"
            )  # Größeres Modell für bessere Präzision
            logger.info("✅ Großes deutsches spaCy-Modell geladen")
        except OSError:
            logger.warning("⚠️ Großes Modell nicht verfügbar, verwende Standard-Modell")
            try:
                self.nlp = spacy.load("de_core_news_sm")
            except OSError:
                self.nlp = spacy.blank("de")

        # Konfiguriere NER-Pipeline speziell für Datenextraktion
        if "ner" not in self.nlp.pipe_names:
            ner = self.nlp.add_pipe("ner", last=True)
        else:
            ner = self.nlp.get_pipe("ner")
        # Ensure ner is the correct type before adding labels
        if hasattr(ner, "add_label"):
            for label in self.entity_labels:
                ner.add_label(label)
        else:
            # Remove and re-add NER pipe if misconfigured
            self.nlp.remove_pipe("ner")
            ner = self.nlp.add_pipe("ner", last=True)
            for label in self.entity_labels:
                ner.add_label(label)

        # Lade und prepare Trainingsdaten
        training_examples = self._load_training_data(training_path)
        split_idx = int(
            len(training_examples) * 0.85
        )  # Mehr Trainingsdaten für bessere Extraktion
        train_examples = training_examples[:split_idx]
        val_examples = training_examples[split_idx:]

        logger.info(
            f"📈 Training: {len(train_examples)} examples, Validation: {len(val_examples)} examples"
        )

        # Optimierte Trainingskonfiguration für Datenextraktion
        pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
        other_pipes = [
            pipe for pipe in self.nlp.pipe_names if pipe not in pipe_exceptions
        ]

        start_time = datetime.now()
        best_f1 = 0.0
        patience_counter = 0
        patience_limit = 5

        with self.nlp.disable_pipes(*other_pipes):
            self.nlp.begin_training()

            for epoch in range(epochs):
                losses = {}

                # Shuffle mit seed für reproduzierbare Ergebnisse
                import random

                random.seed(42 + epoch)
                random.shuffle(train_examples)

                # Optimierte Batch-Größen für Elektrotechnik-Daten
                batches = minibatch(train_examples, size=8)

                for batch in batches:
                    examples = []
                    for text, annotations in batch:
                        doc = self.nlp.make_doc(text)
                        example = Example.from_dict(doc, annotations)
                        examples.append(example)

                    self.nlp.update(examples, drop=dropout, losses=losses)

                # Evaluiere nach jeder Epoche
                current_metrics = self._evaluate_model(val_examples)
                current_f1 = current_metrics["f1_score"]

                logger.info(
                    f"📊 Epoch {epoch + 1}/{epochs} | "
                    f"Loss: {losses.get('ner', 0):.4f} | "
                    f"F1: {current_f1:.3f} | "
                    f"Precision: {current_metrics['precision']:.3f} | "
                    f"Recall: {current_metrics['recall']:.3f}"
                )

                # Early Stopping für optimale Performance
                if current_f1 > best_f1:
                    best_f1 = current_f1
                    patience_counter = 0
                    # Speichere bestes Modell
                    output_model_path.mkdir(parents=True, exist_ok=True)
                    self.nlp.to_disk(output_model_path)
                else:
                    patience_counter += 1
                    if patience_counter >= patience_limit:
                        logger.info(
                            f"🛑 Early stopping at epoch {epoch + 1} (best F1: {best_f1:.3f})"
                        )
                        break

        training_time = (datetime.now() - start_time).total_seconds()

        # Finale Evaluation
        final_metrics = self._evaluate_model(val_examples)

        # Berechne Modellgröße
        model_size = sum(
            f.stat().st_size for f in output_model_path.rglob("*") if f.is_file()
        )
        model_size_mb = model_size / (1024 * 1024)

        result = TrainingMetrics(
            total_examples=len(training_examples),
            training_examples=len(train_examples),
            validation_examples=len(val_examples),
            precision=final_metrics["precision"],
            recall=final_metrics["recall"],
            f1_score=final_metrics["f1_score"],
            epochs_trained=epoch + 1,
            training_time_seconds=training_time,
            model_size_mb=model_size_mb,
            model_path=str(output_model_path),
            training_data_path=str(training_path),
        )

        logger.info(
            f"🎉 Extraction Model Training Complete! | "
            f"F1: {final_metrics['f1_score']:.3f} | "
            f"Time: {training_time:.1f}s | "
            f"Size: {model_size_mb:.1f}MB"
        )

        return result

    def train_specialized_classification_model(
        self,
        training_data_path: Path | str,
        output_model_path: Path | str | None = None,
        epochs: int = 25,
        learning_rate: float = 0.001,
    ) -> TrainingMetrics:
        """
        MODELL 2: Spezialisierte SKR03-Vorkontierung für Elektrotechnik.

        Trainiert ein Textcat-Modell für musterbasierte Vorkontierung von
        Elektrotechnik-Artikeln basierend auf SKR03-Kontenrahmen.

        Args:
            training_data_path: Pfad zu SKR03-Klassifizierungs-Trainingsdaten
            output_model_path: Ausgabepfad für das Klassifizierungsmodell
            epochs: Anzahl Trainingsepochen
            learning_rate: Lernrate für das Training

        Returns:
            TrainingMetrics mit Klassifizierungs-Performance
        """
        training_path = Path(training_data_path)

        if output_model_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_model_path = (
                self.config.models_path / f"elektro_skr03_classifier_{timestamp}"
            )
        else:
            output_model_path = Path(output_model_path)

        logger.info("🏷️ Training Specialized SKR03 Classification Model (Modell 2)")
        logger.info(f"📊 Input: {training_path} -> Output: {output_model_path}")

        output_model_path.mkdir(parents=True, exist_ok=True)

        # Erstelle spezialisierte Textcat-Pipeline
        nlp = spacy.blank("de")

        # Füge Textcat mit optimierter Architektur hinzu
        config = {
            "threshold": 0.5,
            "model": {
                "@architectures": "spacy.TextCatBOW.v2",
                "exclusive_classes": True,
                "ngram_size": 1,
                "no_output_layer": False,
            },
        }
        textcat = cast(TextCategorizer, nlp.add_pipe("textcat", config=config))

        # Lade SKR03-Klassifizierungsdaten
        try:
            doc_bin = DocBin().from_disk(training_path)
            training_docs = list(doc_bin.get_docs(nlp.vocab))
        except Exception:
            # Fallback: Lade aus JSONL-Format
            training_docs = self._load_textcat_data_from_jsonl(training_path, nlp)

        # Sammle alle SKR03-Konten als Labels
        all_skr03_labels = set()
        for doc in training_docs:
            all_skr03_labels.update(doc.cats.keys())

        # Füge Labels mit Elektrotechnik-spezifischen SKR03-Konten hinzu
        elektro_skr03_konten = [
            "4400",  # Rohstoffe und Materialien
            "4410",  # Hilfsstoffe
            "4420",  # Betriebsstoffe
            "4430",  # Handelswaren
            "4440",  # Anzahlungen auf Vorräte
            "6300",  # Löhne und Gehälter
            "6400",  # Gesetzliche soziale Aufwendungen
            "6500",  # Altersversorgung
            "6600",  # Sonstige Personalkosten
            "6000",  # Aufwendungen für Roh-, Hilfs- und Betriebsstoffe
        ]

        for konto in elektro_skr03_konten:
            all_skr03_labels.add(konto)

        for label in all_skr03_labels:
            textcat.add_label(label)

        logger.info(
            f"🏷️ Training mit {len(all_skr03_labels)} SKR03-Konten: {sorted(all_skr03_labels)}"
        )

        # Split Training/Validation
        split_idx = int(len(training_docs) * 0.8)
        train_docs = training_docs[:split_idx]
        val_docs = training_docs[split_idx:]

        # Training mit Early Stopping
        nlp.begin_training()
        start_time = datetime.now()
        best_accuracy = 0.0
        patience_counter = 0

        for epoch in range(epochs):
            import random

            random.seed(42 + epoch)
            random.shuffle(train_docs)
            losses = {}

            # Training in optimierten Batches
            batches = minibatch(train_docs, size=16)
            for batch in batches:
                examples = []
                for doc in batch:
                    example = Example.from_dict(doc, {"cats": doc.cats})
                    examples.append(example)

                nlp.update(examples, losses=losses)

            # Evaluiere Klassifizierungsgenauigkeit
            current_accuracy = self._evaluate_textcat_model(nlp, val_docs)

            logger.info(
                f"📈 Epoch {epoch + 1}/{epochs} | "
                f"Loss: {losses.get('textcat', 0):.4f} | "
                f"Accuracy: {current_accuracy:.3f}"
            )

            # Early Stopping
            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                patience_counter = 0
                nlp.to_disk(output_model_path)
            else:
                patience_counter += 1
                if patience_counter >= 3:
                    logger.info(
                        f"🛑 Early stopping at epoch {epoch + 1} (best accuracy: {best_accuracy:.3f})"
                    )
                    break

        training_time = (datetime.now() - start_time).total_seconds()

        # Berechne finale Metriken
        model_size = sum(
            f.stat().st_size for f in output_model_path.rglob("*") if f.is_file()
        )
        model_size_mb = model_size / (1024 * 1024)

        result = TrainingMetrics(
            total_examples=len(training_docs),
            training_examples=len(train_docs),
            validation_examples=len(val_docs),
            precision=best_accuracy,  # Für Textcat verwenden wir Accuracy als Hauptmetrik
            recall=best_accuracy,
            f1_score=best_accuracy,
            epochs_trained=epoch + 1,
            training_time_seconds=training_time,
            model_size_mb=model_size_mb,
            model_path=str(output_model_path),
            training_data_path=str(training_path),
        )

        logger.info(
            f"🎉 SKR03 Classification Model Training Complete! | "
            f"Accuracy: {best_accuracy:.3f} | "
            f"Time: {training_time:.1f}s | "
            f"Size: {model_size_mb:.1f}MB"
        )

        return result

    def _load_textcat_data_from_jsonl(self, jsonl_path: Path, nlp: Any) -> list[Any]:
        """Lade Textcat-Trainingsdaten aus JSONL-Format"""
        docs = []

        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    text = data.get("text", "")
                    label = data.get("label", "")

                    if text and label:
                        doc = nlp.make_doc(text)
                        doc.cats = {label: 1.0}  # Setze nur das korrekte Label auf 1.0
                        docs.append(doc)

                except json.JSONDecodeError:
                    continue

        return docs

    def _evaluate_textcat_model(self, nlp: Any, validation_docs: list[Any]) -> float:
        """Evaluiere Textcat-Modell-Genauigkeit"""
        if not validation_docs:
            return 0.0

        correct = 0
        total = 0

        for doc in validation_docs[:50]:  # Sample für Performance
            predicted_doc = nlp(doc.text)
            predicted_label = max(predicted_doc.cats, key=predicted_doc.cats.get)
            true_label = max(doc.cats, key=doc.cats.get)

            if predicted_label == true_label:
                correct += 1
            total += 1

        return correct / total if total > 0 else 0.0

    def train_model(
        self,
        training_data_path: Path | str,
        output_model_path: Path | str | None = None,
        epochs: int = 20,
        dropout: float = 0.2,
    ) -> TrainingMetrics:
        """
        Train spaCy NER model for German electrical contractor invoices.

        Args:
            training_data_path: Path to JSONL training data
            output_model_path: Output model directory (optional)
            epochs: Number of training epochs
            dropout: Dropout rate for training

        Returns:
            TrainingMetrics with training performance
        """

        training_path = Path(training_data_path)

        if output_model_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_model_path = self.config.models_path / f"elektro_ner_{timestamp}"
        else:
            output_model_path = Path(output_model_path)

        logger.info(f"Training model: {training_path} -> {output_model_path}")

        # Load training data
        training_examples = self._load_training_data(training_path)

        # Split into training and validation
        split_idx = int(len(training_examples) * 0.8)
        train_examples = training_examples[:split_idx]
        val_examples = training_examples[split_idx:]

        logger.info(
            f"Training with {len(train_examples)} examples, "
            f"validating with {len(val_examples)} examples"
        )

        # Disable other pipeline components during training
        pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
        other_pipes = [
            pipe for pipe in self.nlp.pipe_names if pipe not in pipe_exceptions
        ]

        start_time = datetime.now()

        # Training loop
        with self.nlp.disable_pipes(*other_pipes):
            self.nlp.begin_training()

            for epoch in range(epochs):
                losses = {}

                # Shuffle training data
                import random

                random.shuffle(train_examples)

                # Train in batches
                batches = minibatch(train_examples, size=16)

                for batch in batches:
                    examples = []
                    for text, annotations in batch:
                        doc = self.nlp.make_doc(text)
                        example = Example.from_dict(doc, annotations)
                        examples.append(example)

                    self.nlp.update(examples, drop=dropout, losses=losses)

                logger.info(
                    f"Epoch {epoch + 1}/{epochs}, Loss: {losses.get('ner', 0):.3f}"
                )

        training_time = (datetime.now() - start_time).total_seconds()

        # Evaluate model
        metrics = self._evaluate_model(val_examples)

        # Save model
        output_model_path.mkdir(parents=True, exist_ok=True)
        self.nlp.to_disk(output_model_path)

        # Calculate model size
        model_size = sum(
            f.stat().st_size for f in output_model_path.rglob("*") if f.is_file()
        )
        model_size_mb = model_size / (1024 * 1024)

        result = TrainingMetrics(
            total_examples=len(training_examples),
            training_examples=len(train_examples),
            validation_examples=len(val_examples),
            precision=metrics["precision"],
            recall=metrics["recall"],
            f1_score=metrics["f1_score"],
            epochs_trained=epochs,
            training_time_seconds=training_time,
            model_size_mb=model_size_mb,
            model_path=str(output_model_path),
            training_data_path=str(training_path),
        )

        logger.info(
            f"Training complete: F1={metrics['f1_score']:.3f}, "
            f"Time={training_time:.1f}s, Size={model_size_mb:.1f}MB"
        )

        return result

    def _load_training_data(self, jsonl_path: Path) -> list[tuple[str, dict[str, Any]]]:
        """Load training data from JSONL file"""

        examples = []

        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    text = data.get("text", "")
                    entities = data.get("entities", [])

                    if text and entities:
                        annotations = {"entities": entities}
                        examples.append((text, annotations))

                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSONL line: {e}")
                    continue

        logger.info(f"Loaded {len(examples)} training examples from {jsonl_path}")
        return examples

    def _evaluate_model(
        self,
        validation_examples: list[tuple[str, dict[str, Any]]],
    ) -> dict[str, float]:
        """Evaluate model performance on validation set"""

        if not validation_examples:
            return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0}

        # Simple evaluation - count correct entities
        total_predicted = 0
        total_correct = 0
        total_gold = 0

        for text, annotations in validation_examples[:10]:  # Sample for speed
            doc = self.nlp(text)
            predicted_entities = [
                (ent.start_char, ent.end_char, ent.label_) for ent in doc.ents
            ]
            gold_entities = list(annotations["entities"])

            total_predicted += len(predicted_entities)
            total_gold += len(gold_entities)

            # Count matches
            for pred_entity in predicted_entities:
                if pred_entity in gold_entities:
                    total_correct += 1

        # Calculate metrics
        precision = total_correct / total_predicted if total_predicted > 0 else 0.0
        recall = total_correct / total_gold if total_gold > 0 else 0.0
        f1_score = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return {"precision": precision, "recall": recall, "f1_score": f1_score}

    def save_training_metrics(
        self,
        metrics: TrainingMetrics,
        output_path: Path | str | None = None,
    ) -> Path:
        """Save training metrics to JSON file"""

        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.config.models_path / f"training_metrics_{timestamp}.json"
        else:
            output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(asdict(metrics), f, indent=2, ensure_ascii=False)

        logger.info(f"Training metrics saved to: {output_path}")
        return output_path

    def export_textcat_data(
        self,
        processed_data_dir: Path | str,
        output_path: Path | str | None = None,
    ) -> ExportResult:
        """Exportiert Daten speziell für das Text-Klassifizierungs-Training."""
        logger.info(
            f"Exportiere Textcat-Daten von {processed_data_dir} nach {output_path}"
        )

        processed_dir = Path(processed_data_dir)
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = (
                self.config.training_data_path / f"textcat_training_{timestamp}.spacy"
            )
        else:
            output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        data_pairs = []
        all_labels = set()

        for json_file in processed_dir.glob("*.json"):
            try:
                with open(json_file, encoding="utf-8") as f:
                    data = json.load(f)

                for item in data.get("skr03_classifications", []):
                    text = item.get("description")
                    label = item.get("skr03_konto") or item.get("konto")
                    if text and label:
                        data_pairs.append((text, label))
                        all_labels.add(label)
            except Exception as e:
                logger.warning(f"Fehler beim Lesen von {json_file}: {e}")
                continue

        # Erstelle .spacy-Datei
        nlp = spacy.blank("de")
        doc_bin = DocBin()

        for text, label in data_pairs:
            doc = nlp.make_doc(text)
            doc.cats = {lbl: 1.0 if lbl == label else 0.0 for lbl in all_labels}
            doc_bin.add(doc)

        doc_bin.to_disk(output_path)

        return ExportResult(
            str(output_path),
            len(data_pairs),
            len(data_pairs),
            datetime.now().isoformat(),
        )

    def train_textcat_model(
        self,
        training_data_path: Path | str,
        validation_data_path: Path | str | None = None,
        output_path: Path | str | None = None,
        epochs: int = 20,
    ) -> Path:
        """Trainiert ein spaCy Textcat-Modell für SKR03-Klassifikation."""
        logger.info(f"Starte Textcat-Training. Training data: {training_data_path}")

        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.config.models_path / f"elektro_textcat_{timestamp}"
        else:
            output_path = Path(output_path)

        output_path.mkdir(parents=True, exist_ok=True)

        # Erstelle eine neue spaCy-Pipeline für Textcat
        nlp = spacy.blank("de")
        textcat = nlp.add_pipe("textcat")

        # Lade Training-Daten und extrahiere Labels
        doc_bin = DocBin().from_disk(training_data_path)
        training_docs = list(doc_bin.get_docs(nlp.vocab))

        # Sammle alle Labels
        all_labels = set()
        for doc in training_docs:
            all_labels.update(doc.cats.keys())

        # Füge Labels zum Textcat hinzu
        for label in all_labels:
            textcat.add_label(label)

        logger.info(
            f"Training Textcat mit {len(all_labels)} Labels: {sorted(all_labels)}"
        )

        # Trainiere das Modell
        nlp.begin_training()

        for epoch in range(epochs):
            random.shuffle(training_docs)
            losses = {}

            # Training in batches
            batches = minibatch(training_docs, size=8)
            for batch in batches:
                examples = []
                for doc in batch:
                    example = Example.from_dict(doc, {"cats": doc.cats})
                    examples.append(example)

                nlp.update(examples, losses=losses)

            logger.info(
                f"Epoch {epoch+1}/{epochs}, Loss: {losses.get('textcat', 0):.4f}"
            )

        # Speichere das trainierte Modell
        nlp.to_disk(output_path)
        logger.info(
            f"Textcat-Modell erfolgreich trainiert und in {output_path} gespeichert."
        )

        return output_path


# Convenience functions
def export_training_data(
    processed_data_dir: Path | str,
    output_path: Path | str | None = None,
    config: Config | None = None,
) -> ExportResult:
    """Export training data - simplified interface"""
    service = TrainingService(config)
    return service.export_training_data(processed_data_dir, output_path)


def train_elektro_model(
    training_data_path: Path | str,
    output_model_path: Path | str | None = None,
    epochs: int = 20,
    config: Config | None = None,
) -> TrainingMetrics:
    """Train electrical contractor NER model - simplified interface"""
    service = TrainingService(config)
    return service.train_model(training_data_path, output_model_path, epochs)


if __name__ == "__main__":
    # Quick test/demo
    import argparse

    parser = argparse.ArgumentParser(description="Test training service")
    parser.add_argument("command", choices=["export", "train"], help="Command to run")
    parser.add_argument("--input", "-i", required=True, help="Input path")
    parser.add_argument("--output", "-o", help="Output path")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")

    args = parser.parse_args()

    config = Config()
    service = TrainingService(config)

    if args.command == "export":
        result = service.export_training_data(args.input, args.output)
        print(f"✅ Export complete: {result.total_records} examples")
        print(f"📁 Saved: {result.jsonl_path}")

    elif args.command == "train":
        result = service.train_model(args.input, args.output, args.epochs)
        print("✅ Training complete!")
        print(f"🎯 F1 Score: {result.f1_score:.3f}")
        print(f"⏱️  Time: {result.training_time_seconds:.1f}s")
        print(f"💾 Model: {result.model_path}")
