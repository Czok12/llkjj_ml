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
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import spacy  # type: ignore
from spacy.training import Example  # type: ignore
from spacy.util import compounding, minibatch  # type: ignore

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
            ner = self.nlp.add_pipe("ner", last=True)
        else:
            ner = self.nlp.get_pipe("ner")

        # Add Elektrotechnik-specific entity labels
        elektro_labels = [
            "INVOICE_NUMBER",
            "INVOICE_DATE",
            "SUPPLIER_NAME",
            "CUSTOMER_NUMBER",
            "NET_AMOUNT",
            "VAT_AMOUNT",
            "GROSS_AMOUNT",
            "PRODUCT_NAME",
            "PRODUCT_CODE",
            "QUANTITY",
            "UNIT_PRICE",
            "LINE_TOTAL",
            "SKR03_ACCOUNT",
            "ELEKTRO_CATEGORY",
        ]

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
                f.write(json.dumps(example, ensure_ascii=False) + "\\n")

        result = ExportResult(
            jsonl_path=str(output_path),
            total_records=len(training_examples),
            skr03_classifications=skr03_count,
            export_timestamp=datetime.now().isoformat(),
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
        """Map field names to entity labels"""

        field_mapping = {
            "invoice_number": "INVOICE_NUMBER",
            "date": "INVOICE_DATE",
            "invoice_date": "INVOICE_DATE",
            "customer_number": "CUSTOMER_NUMBER",
            "total_amount": "GROSS_AMOUNT",
            "net_total": "NET_AMOUNT",
            "vat_amount": "VAT_AMOUNT",
            "gross_total": "GROSS_AMOUNT",
            "description": "PRODUCT_NAME",
            "artikel": "PRODUCT_NAME",
            "quantity": "QUANTITY",
            "menge": "QUANTITY",
            "unit_price": "UNIT_PRICE",
            "einzelpreis": "UNIT_PRICE",
            "total_price": "LINE_TOTAL",
            "gesamt": "LINE_TOTAL",
        }

        return field_mapping.get(field.lower(), "PRODUCT_NAME")

    def _remove_overlapping_entities(self, entities: list[list]) -> list[list]:
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
                batches = minibatch(train_examples, size=compounding(4.0, 32.0, 1.001))

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
        self, validation_examples: list[tuple[str, dict[str, Any]]]
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
        self, metrics: TrainingMetrics, output_path: Path | str | None = None
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
