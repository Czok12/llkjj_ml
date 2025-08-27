from llkjj_ml.src.settings_bridge import ConfigType

#!/usr/bin/env python3
"""
LLKJJ ML Pipeline - Refactored Training Service
===============================================

Kompakte, gut strukturierte Training-Services für deutsche Elektrohandwerk-Rechnungen.
Aufgeteilt in spezialisierte Klassen für bessere Wartbarkeit.

Author: LLKJJ ML Pipeline
Version: 2.1.0 (Refactored)
"""

import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import spacy
from pydantic import BaseModel, Field, ValidationInfo, field_validator
from spacy.pipeline import TextCategorizer
from spacy.training import Example
from spacy.util import minibatch

from .settings_bridge import Config

logger = logging.getLogger(__name__)


class TrainingMetrics(BaseModel):
    """
    Training performance metrics for ML models.

    Migrated from dataclass to Pydantic BaseModel for enhanced:
    - Numerical validation for ML metrics
    - Automatic JSON serialization
    - Better logging and debugging
    """

    total_examples: int = Field(
        ..., ge=0, description="Gesamtzahl der Trainingsbeispiele"
    )
    training_examples: int = Field(..., ge=0, description="Anzahl Trainingsbeispiele")
    validation_examples: int = Field(
        ..., ge=0, description="Anzahl Validierungsbeispiele"
    )
    precision: float = Field(
        ..., ge=0.0, le=1.0, description="Precision Score (0.0-1.0)"
    )
    recall: float = Field(..., ge=0.0, le=1.0, description="Recall Score (0.0-1.0)")
    f1_score: float = Field(..., ge=0.0, le=1.0, description="F1 Score (0.0-1.0)")
    epochs_trained: int = Field(..., ge=1, description="Anzahl trainierter Epochen")
    training_time_seconds: float = Field(
        ..., ge=0.0, description="Trainingszeit in Sekunden"
    )

    @field_validator("training_examples", "validation_examples")
    @classmethod
    def validate_example_counts(cls, v: int, info: ValidationInfo) -> int:
        """Validiere dass Trainings-/Validierungsbeispiele <= total_examples"""
        if info.data.get("total_examples") is not None:
            total = info.data["total_examples"]
            if v > total:
                raise ValueError(
                    f"Beispielanzahl ({v}) kann nicht größer als total_examples ({total}) sein"
                )
        return v

    def get_summary(self) -> str:
        """Erstelle eine Zusammenfassung der Trainingsmetriken"""
        return (
            f"Training: {self.training_examples}/{self.total_examples} Beispiele | "
            f"F1: {self.f1_score:.3f} | Precision: {self.precision:.3f} | "
            f"Recall: {self.recall:.3f} | Zeit: {self.training_time_seconds:.1f}s"
        )


class ExportResult(BaseModel):
    """
    Data export result for training pipelines.

    Migrated from dataclass to Pydantic BaseModel for enhanced:
    - Path validation and normalization
    - Export status tracking
    - Better error handling
    """

    jsonl_path: str = Field(..., description="Pfad zur exportierten JSONL-Datei")
    total_records: int = Field(
        ..., ge=0, description="Gesamtzahl exportierter Datensätze"
    )
    skr03_classifications: int = Field(
        ..., ge=0, description="Anzahl SKR03-Klassifizierungen"
    )
    export_timestamp: str = Field(
        ..., description="Zeitstempel des Exports (ISO format)"
    )

    @field_validator("jsonl_path")
    @classmethod
    def validate_jsonl_path(cls, v: str) -> str:
        """Validiere JSONL-Dateipfad"""
        if not v.lower().endswith(".jsonl"):
            raise ValueError(f"Pfad muss eine JSONL-Datei sein: {v}")
        return v

    @field_validator("export_timestamp")
    @classmethod
    def validate_export_timestamp(cls, v: str) -> str:
        """Validiere ISO-Zeitstempel-Format"""
        try:
            datetime.fromisoformat(v.replace("Z", "+00:00"))
            return v
        except ValueError as e:
            raise ValueError(f"Ungültiges Zeitstempel-Format: {v}") from e

    @field_validator("skr03_classifications")
    @classmethod
    def validate_skr03_count(cls, v: int, info: ValidationInfo) -> int:
        """Validiere dass SKR03-Klassifizierungen <= total_records"""
        if info.data.get("total_records") is not None:
            total = info.data["total_records"]
            if v > total:
                raise ValueError(
                    f"SKR03-Klassifizierungen ({v}) können nicht größer als total_records ({total}) sein"
                )
        return v

    def get_summary(self) -> str:
        """Erstelle eine Zusammenfassung des Exports"""
        jsonl_file = Path(self.jsonl_path).name
        return (
            f"Export: {jsonl_file} | {self.total_records} Datensätze | "
            f"{self.skr03_classifications} SKR03-Klassifizierungen"
        )


class BaseTrainer:
    """Base class with shared functionality for all trainers."""

    def __init__(self, config: ConfigType | None = None):
        """Initialize base trainer."""
        self.config = config or Config
        self._setup_logging()
        self._setup_spacy()

    def _setup_logging(self) -> None:
        """Configure logging."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    def _setup_spacy(self) -> None:
        """Setup spaCy with German language model."""
        try:
            self.nlp = spacy.load("de_core_news_sm")
            logger.info("German spaCy model loaded successfully")
        except OSError:
            logger.warning("German spaCy model not found, creating blank model")
            self.nlp = spacy.blank("de")

        # Initialize entity labels list
        self.entity_labels: list[str] = []

    def save_model(
        self, output_path: Path, model_name: str = "spacy_extractor"
    ) -> None:
        """Save trained model to disk."""
        model_path = output_path / model_name
        model_path.mkdir(parents=True, exist_ok=True)
        self.nlp.to_disk(model_path)
        logger.info("✅ Model saved to %s", model_path)

    def save_training_metrics(
        self, metrics: TrainingMetrics, output_path: Path
    ) -> None:
        """Save training metrics to JSON."""
        metrics_file = output_path / "training_metrics.json"
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics.model_dump(), f, indent=2, ensure_ascii=False)
        logger.info("📊 Metrics saved to %s", metrics_file)


class DataExporter(BaseTrainer):
    """Handles export of training data from JSON to spaCy formats."""

    def export_training_data(self, input_path: Path, output_path: Path) -> ExportResult:
        """Export training data from processed JSON files."""
        output_path.mkdir(parents=True, exist_ok=True)

        # Find JSON files
        json_files = list(input_path.glob("*.json"))
        if not json_files:
            logger.warning("No JSON files found in %s", input_path)
            return ExportResult(
                jsonl_path="",
                total_records=0,
                skr03_classifications=0,
                export_timestamp=datetime.now().isoformat(),
            )

        logger.info("📁 Found %d JSON files for export", len(json_files))

        # Process files
        all_examples: list[tuple[str, dict[str, Any]]] = []
        skr03_count = 0

        for json_file in json_files:
            try:
                with open(json_file, encoding="utf-8") as f:
                    data = json.load(f)

                # Extract examples and count SKR03 classifications
                examples = self._extract_examples(data)
                all_examples.extend(examples)

                if isinstance(data, dict) and "skr03_classifications" in data:
                    skr03_count += len(data["skr03_classifications"])

            except (OSError, json.JSONDecodeError) as e:
                logger.error("Error loading %s: %s", json_file, e)
                continue

        if not all_examples:
            logger.warning("No valid examples found")
            return ExportResult(
                jsonl_path="",
                total_records=0,
                skr03_classifications=0,
                export_timestamp=datetime.now().isoformat(),
            )

        # Convert to spaCy format and save
        output_file = output_path / "pipeline_training.jsonl"
        self._save_jsonl(all_examples, output_file)

        logger.info("✅ Exported %d examples to %s", len(all_examples), output_file)

        return ExportResult(
            jsonl_path=str(output_file),
            total_records=len(all_examples),
            skr03_classifications=skr03_count,
            export_timestamp=datetime.now().isoformat(),
        )

    def _extract_examples(
        self, data: dict[str, Any]
    ) -> list[tuple[str, dict[str, Any]]]:
        """Extract training examples from JSON data."""
        examples: list[tuple[str, dict[str, Any]]] = []

        # Get text content
        text = self._get_text_content(data)
        if not text:
            return examples

        # Extract entities and labels
        entities = self._extract_entities(data, text)
        categories = self._extract_categories(data)

        # Create spaCy format
        annotations: dict[str, Any] = {"entities": entities}
        if categories:
            annotations["cats"] = categories

        examples.append((text, annotations))
        return examples

    def _get_text_content(self, data: dict[str, Any]) -> str:
        """Extract text from various possible fields."""
        text_fields = ["raw_text", "text", "content", "ocr_text"]

        for field in text_fields:
            if field in data and data[field]:
                return str(data[field]).strip()

        return ""

    def _extract_entities(
        self, data: dict[str, Any], text: str
    ) -> list[tuple[int, int, str]]:
        """Extract NER entities from data."""
        entities = []

        entity_mapping = {
            # Artikel- und Positionsdaten
            "artikel": "ARTIKEL",
            "artikelnummer": "ARTIKELNUMMER",
            "artikel_nummer": "ARTIKELNUMMER",
            "art_nr": "ARTIKELNUMMER",
            "rechnungsposition": "RECHNUNGSPOSITION",
            "position": "RECHNUNGSPOSITION",
            "pos": "RECHNUNGSPOSITION",
            "menge": "MENGE",
            "einzelpreis": "EINZELPREIS",
            "preis": "EINZELPREIS",
            "gesamtpreis": "GESAMTPREIS",
            "betrag": "GESAMTPREIS",
            "mwst_satz": "MWST_SATZ",
            "mwst": "MWST_SATZ",
            "mehrwertsteuer": "MWST_SATZ",
            "ust_satz": "MWST_SATZ",
            # Rechnungsinformationen
            "rechnungsnummer": "RECHNUNGSNUMMER",
            "rechnung_nr": "RECHNUNGSNUMMER",
            "invoice_number": "RECHNUNGSNUMMER",
            "rechnungsdatum": "RECHNUNGSDATUM",
            "rechnung_datum": "RECHNUNGSDATUM",
            "rg_datum": "RECHNUNGSDATUM",
            "lieferdatum": "LIEFERDATUM",
            "liefer_datum": "LIEFERDATUM",
            "lieferung": "LIEFERDATUM",
            # Parteien und Kundendaten
            "händler": "HÄNDLER",
            "haendler": "HÄNDLER",
            "lieferant": "HÄNDLER",
            "verkäufer": "HÄNDLER",
            "verkaeufer": "HÄNDLER",
            "kunde": "KUNDE",
            "käufer": "KUNDE",
            "kaeufer": "KUNDE",
            "empfänger": "KUNDE",
            "empfaenger": "KUNDE",
            "kundennummer": "KUNDENNUMMER",
            "kunden_nr": "KUNDENNUMMER",
            "kd_nr": "KUNDENNUMMER",
            # Legacy-Support für Rückwärtskompatibilität
            "datum": "RECHNUNGSDATUM",  # Fallback
            "kreditor": "HÄNDLER",  # Alt -> Neu
            "debitor": "KUNDE",  # Alt -> Neu
        }

        for field, label in entity_mapping.items():
            if field in data and data[field]:
                value = str(data[field]).strip()
                if value in text:
                    start = text.find(value)
                    if start >= 0:
                        entities.append((start, start + len(value), label))

        return self._remove_overlapping_entities(entities)

    def _extract_categories(self, data: dict[str, Any]) -> dict[str, float]:
        """Extract text classification categories."""
        categories = {}

        if "skr03_konto" in data:
            account = str(data["skr03_konto"])
            if account:
                categories[f"SKR03_{account}"] = 1.0

        return categories

    def _remove_overlapping_entities(
        self, entities: list[tuple[int, int, str]]
    ) -> list[tuple[int, int, str]]:
        """Remove overlapping entity annotations."""
        if not entities:
            return entities

        sorted_entities = sorted(entities, key=lambda x: x[0])
        result = [sorted_entities[0]]

        for current in sorted_entities[1:]:
            last = result[-1]
            if current[0] >= last[1]:  # No overlap
                result.append(current)
            elif (current[1] - current[0]) > (last[1] - last[0]):  # Keep longer
                result[-1] = current

        return result

    def _save_jsonl(
        self, examples: list[tuple[str, dict[str, Any]]], output_path: Path
    ) -> None:
        """Save examples to JSONL format."""
        with open(output_path, "w", encoding="utf-8") as f:
            for text, annotations in examples:
                example = {"text": text, "annotations": annotations}
                f.write(json.dumps(example, ensure_ascii=False) + "\n")


class NERTrainer(BaseTrainer):
    """Handles Named Entity Recognition training."""

    def train_ner_model(
        self, jsonl_path: Path, output_path: Path, epochs: int = 10
    ) -> TrainingMetrics:
        """Train NER model from JSONL data."""
        logger.info("🚀 Starting NER training for %d epochs", epochs)

        # Load training data
        train_examples = self._load_ner_data(jsonl_path)
        if not train_examples:
            raise ValueError("No valid NER training examples found")

        # Setup NER pipeline
        self._setup_ner_pipeline()

        # Split data
        random.seed(42)
        random.shuffle(train_examples)
        split_idx = int(len(train_examples) * 0.8)
        train_data = train_examples[:split_idx]
        val_data = train_examples[split_idx:]

        logger.info(
            "📊 Training: %d examples, Validation: %d examples",
            len(train_data),
            len(val_data),
        )

        # Training loop
        start_time = datetime.now()

        for epoch in range(epochs):
            losses: dict[str, float] = {}
            random.shuffle(train_data)

            # Training in batches
            batches = minibatch(train_data, size=8)  # type: ignore[no-untyped-call]
            for batch in batches:
                examples = []
                for text, annotations in batch:
                    doc = self.nlp.make_doc(text)
                    example = Example.from_dict(doc, annotations)
                    examples.append(example)

                self.nlp.update(examples, losses=losses)

            logger.info("Epoch %d/%d - Losses: %s", epoch + 1, epochs, losses)

        training_time = (datetime.now() - start_time).total_seconds()

        # Evaluate model
        precision, recall, f1 = self._evaluate_ner_model(val_data)

        # Save model with specific name for NER/Extractor
        self.save_model(output_path, "spacy_extractor")

        metrics = TrainingMetrics(
            total_examples=len(train_examples),
            training_examples=len(train_data),
            validation_examples=len(val_data),
            precision=precision,
            recall=recall,
            f1_score=f1,
            epochs_trained=epochs,
            training_time_seconds=training_time,
        )

        self.save_training_metrics(metrics, output_path)

        logger.info("✅ NER training completed - F1: %.3f", f1)
        return metrics

    def _load_ner_data(self, jsonl_path: Path) -> list[tuple[str, dict[str, Any]]]:
        """Load NER training data from JSONL."""
        examples = []

        try:
            with open(jsonl_path, encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line.strip())
                    if "text" in data and "annotations" in data:
                        if data["annotations"].get("entities"):
                            examples.append((data["text"], data["annotations"]))

        except (OSError, json.JSONDecodeError) as e:
            logger.error("Error loading NER data: %s", e)

        logger.info("📁 Loaded %d NER examples", len(examples))
        return examples

    def _setup_ner_pipeline(self) -> None:
        """Setup NER pipeline with elektro-specific labels."""
        elektro_labels = [
            "ARTIKEL",
            "EINZELPREIS",
            "MENGE",
            "GESAMTPREIS",
            "RECHNUNGSNUMMER",
            "DATUM",
            "KREDITOR",
            "DEBITOR",
        ]

        # Add or get NER pipe
        if "ner" not in self.nlp.pipe_names:
            ner = self.nlp.add_pipe("ner", last=True)
        else:
            ner = self.nlp.get_pipe("ner")

        # Add labels
        for label in elektro_labels:
            ner.add_label(label)  # type: ignore[attr-defined]

        self.entity_labels = elektro_labels
        logger.info("🏷️ Added %d entity labels", len(elektro_labels))

    def _evaluate_ner_model(
        self, validation_data: list[tuple[str, dict[str, Any]]]
    ) -> tuple[float, float, float]:
        """Evaluate NER model performance using spaCy's Scorer.

        This method uses spaCy's industrial-standard Scorer which handles
        token alignment and provides more realistic evaluation metrics
        compared to the previous strict character-level matching.
        """
        if not validation_data:
            return 0.0, 0.0, 0.0

        # Import Scorer and Example from spaCy
        from spacy.scorer import Scorer
        from spacy.training import Example

        # Create Example objects for evaluation
        examples = []
        for text, annotations in validation_data:
            # Create predicted document by processing text through model
            pred_doc = self.nlp(text)

            # Create Example object linking predicted and reference
            # This handles tokenization alignment automatically
            example = Example.from_dict(pred_doc, annotations)
            examples.append(example)

        # Use spaCy's Scorer to evaluate spans
        # This provides industry-standard evaluation with proper alignment
        scorer = Scorer()
        scores = scorer.score_spans(examples, "ents")

        # Extract metrics from scores - these are now more realistic
        precision = scores.get("ents_p", 0.0)
        recall = scores.get("ents_r", 0.0)
        f1_score = scores.get("ents_f", 0.0)

        logger.info(
            "🔍 NER Evaluation - P: %.3f, R: %.3f, F1: %.3f",
            precision,
            recall,
            f1_score,
        )

        return precision, recall, f1_score


class TextCatTrainer(BaseTrainer):
    """Handles Text Classification training."""

    def train_textcat_model(
        self, jsonl_path: Path, output_path: Path, epochs: int = 10
    ) -> TrainingMetrics:
        """Train text classification model."""
        logger.info("🚀 Starting TextCat training for %d epochs", epochs)

        # Load data
        train_examples = self._load_textcat_data(jsonl_path)
        if not train_examples:
            raise ValueError("No valid TextCat training examples found")

        # Setup pipeline
        self._setup_textcat_pipeline(train_examples)

        # Split data
        random.seed(42)
        random.shuffle(train_examples)
        split_idx = int(len(train_examples) * 0.8)
        train_data = train_examples[:split_idx]
        val_data = train_examples[split_idx:]

        logger.info(
            "📊 Training: %d examples, Validation: %d examples",
            len(train_data),
            len(val_data),
        )

        # Training loop
        start_time = datetime.now()

        for epoch in range(epochs):
            losses: dict[str, float] = {}
            random.shuffle(train_data)

            # Training in batches
            batches = minibatch(train_data, size=16)  # type: ignore[no-untyped-call]
            for batch in batches:
                examples = []
                for doc in batch:
                    example = Example.from_dict(doc, {"cats": doc.cats})
                    examples.append(example)

                self.nlp.update(examples, losses=losses)

            logger.info("Epoch %d/%d - Losses: %s", epoch + 1, epochs, losses)

        training_time = (datetime.now() - start_time).total_seconds()

        # Evaluate model
        accuracy = self._evaluate_textcat_model(val_data)

        # Save model with specific name for TextCat/SKR03
        self.save_model(output_path, "spacy_skr03")

        metrics = TrainingMetrics(
            total_examples=len(train_examples),
            training_examples=len(train_data),
            validation_examples=len(val_data),
            precision=accuracy,  # Simplified
            recall=accuracy,
            f1_score=accuracy,
            epochs_trained=epochs,
            training_time_seconds=training_time,
        )

        self.save_training_metrics(metrics, output_path)

        logger.info("✅ TextCat training completed - Accuracy: %.3f", accuracy)
        return metrics

    def _load_textcat_data(self, jsonl_path: Path) -> list[Any]:
        """Load TextCat training data."""
        docs = []

        try:
            with open(jsonl_path, encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line.strip())
                    if "text" in data and "annotations" in data:
                        cats = data["annotations"].get("cats", {})
                        if cats:
                            doc = self.nlp.make_doc(data["text"])
                            doc.cats = cats
                            docs.append(doc)

        except (OSError, json.JSONDecodeError) as e:
            logger.error("Error loading TextCat data: %s", e)

        logger.info("📁 Loaded %d TextCat examples", len(docs))
        return docs

    def _setup_textcat_pipeline(self, train_docs: list[Any]) -> None:
        """Setup text classification pipeline."""
        # Get all categories
        all_categories = set()
        for doc in train_docs:
            all_categories.update(doc.cats.keys())

        # Add textcat pipe
        if "textcat" not in self.nlp.pipe_names:
            textcat = self.nlp.add_pipe("textcat", last=True)
        else:
            textcat = cast(TextCategorizer, self.nlp.get_pipe("textcat"))

        # Add labels
        for category in all_categories:
            textcat.add_label(category)  # type: ignore[attr-defined]

        logger.info("🏷️ Added %d text categories", len(all_categories))

    def _evaluate_textcat_model(self, validation_docs: list[Any]) -> float:
        """Evaluate text classification model."""
        if not validation_docs:
            return 0.0

        correct = 0

        for doc in validation_docs:
            pred_doc = self.nlp(doc.text)

            # Get highest scoring category for each
            true_cat = max(doc.cats.items(), key=lambda x: x[1])[0] if doc.cats else ""
            pred_cat = (
                max(pred_doc.cats.items(), key=lambda x: x[1])[0]
                if pred_doc.cats
                else ""
            )

            if true_cat == pred_cat:
                correct += 1

        return correct / len(validation_docs) if validation_docs else 0.0


class TrainingService:
    """
    Main training service orchestrating different training tasks.
    Backwards compatible with original API.
    """

    def __init__(self, config: ConfigType | None = None):
        """Initialize training service."""
        self.config = config or Config
        self.data_exporter = DataExporter(config)
        self.ner_trainer = NERTrainer(config)
        self.textcat_trainer = TextCatTrainer(config)

    def export_training_data(self, input_path: Path, output_path: Path) -> ExportResult:
        """Export training data (backwards compatible)."""
        return self.data_exporter.export_training_data(input_path, output_path)

    def train_model(
        self,
        jsonl_path: Path,
        output_path: Path,
        epochs: int = 10,
        model_type: str = "ner",
    ) -> TrainingMetrics:
        """Train model (backwards compatible)."""
        if model_type == "ner":
            return self.ner_trainer.train_ner_model(jsonl_path, output_path, epochs)
        elif model_type == "textcat":
            return self.textcat_trainer.train_textcat_model(
                jsonl_path, output_path, epochs
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")


# Convenience functions for CLI compatibility
def export_training_data(input_path: str, output_path: str) -> ExportResult:
    """Export training data from CLI."""
    service = TrainingService()
    return service.export_training_data(Path(input_path), Path(output_path))


def train_elektro_model(
    jsonl_path: str, output_path: str, epochs: int = 10
) -> TrainingMetrics:
    """Train elektro model from CLI."""
    service = TrainingService()
    return service.train_model(Path(jsonl_path), Path(output_path), epochs, "ner")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLKJJ ML Training Service")
    parser.add_argument("command", choices=["export", "train"], help="Command to run")
    parser.add_argument("--input", "-i", required=True, help="Input path")
    parser.add_argument("--output", "-o", required=True, help="Output path")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument(
        "--model-type", choices=["ner", "textcat"], default="ner", help="Model type"
    )

    args = parser.parse_args()

    if args.command == "export":
        export_result = export_training_data(args.input, args.output)
        print(
            f"✅ Exported {export_result.total_records} examples to {export_result.jsonl_path}"
        )
    elif args.command == "train":
        train_result = train_elektro_model(args.input, args.output, args.epochs)
        print(f"✅ Training completed - F1: {train_result.f1_score:.3f}")
