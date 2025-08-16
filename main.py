#!/usr/bin/env python3
"""
LLKJJ ML Pipeline - Unified CLI (KISS Architecture)
==================================================

Post-consolidation main entry point using simplified, consolidated services.
Follows KISS principles: Keep It Simple, Stupid.

Replaces 4 redundant entry points with single, streamlined CLI interface.

Author: LLKJJ ML Pipeline
Version: 2.0.0 (Post-Consolidation)
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Import consolidated services
from src.config import Config
from src.pipeline.processor import UnifiedProcessor
from src.trainer import TrainingService


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for CLI"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("logs/llkjj_ml.log")],
    )


def process_pdfs(args: argparse.Namespace) -> None:
    """Process PDF invoices with consolidated pipeline"""
    config = Config()
    processor = UnifiedProcessor(config)

    input_path = Path(args.input)
    output_dir = Path(args.output) if args.output else Path("data/output")

    print(f"🔄 Processing: {input_path}")
    print(f"📁 Output: {output_dir}")

    if input_path.is_file() and input_path.suffix.lower() == ".pdf":
        # Single PDF
        result = processor.process_pdf(input_path)

        print("✅ Processing complete!")
        print(f"📄 Processed: {Path(result.pdf_path).name}")
        print(f"🎯 SKR03 matches: {len(result.skr03_classifications)}")
        print(f"💾 Quality: {result.extraction_quality}")

    elif input_path.is_dir():
        # Batch process directory
        pdf_files = list(input_path.glob("*.pdf"))

        if not pdf_files:
            print(f"❌ No PDF files found in {input_path}")
            return

        print(f"📦 Found {len(pdf_files)} PDF files")

        results = []
        for pdf_file in pdf_files:
            try:
                result = processor.process_pdf(pdf_file)
                results.append(result)

                print(
                    f"  ✅ {pdf_file.name}: {len(result.skr03_classifications)} SKR03, {result.extraction_quality} quality"
                )

            except Exception as e:
                print(f"  ❌ {pdf_file.name}: {e}")
                continue

        # Summary
        total_skr03 = sum(len(r.skr03_classifications) for r in results)
        avg_confidence = (
            sum(r.confidence_score for r in results) / len(results) if results else 0
        )

        print("\n📊 Batch Summary:")
        print(f"   Files processed: {len(results)}/{len(pdf_files)}")
        print(f"   SKR03 classifications: {total_skr03}")
        print(f"   Average confidence: {avg_confidence:.3f}")

    else:
        print(f"❌ Invalid input: {input_path} (must be PDF file or directory)")


def export_training_data(args: argparse.Namespace) -> None:
    """Export processed data for spaCy training"""
    config = Config()
    training_service = TrainingService(config)

    input_dir = Path(args.input)
    output_path = Path(args.output) if args.output else Path("data/training")

    print(f"📤 Exporting training data from: {input_dir}")

    result = training_service.export_training_data(input_dir, output_path)

    print("✅ Export complete!")
    print(f"📊 Training examples: {result.total_records}")
    print(f"🎯 SKR03 classifications: {result.skr03_classifications}")
    print(f"💾 Saved: {result.jsonl_path}")


def train_model(args: argparse.Namespace) -> None:
    """Train spaCy NER model for German electrical invoices"""
    config = Config()
    training_service = TrainingService(config)

    training_data = Path(args.input)
    model_output = Path(args.output) if args.output else Path("output_model")
    epochs = args.epochs

    print(f"🚀 Training model with: {training_data}")
    print(f"⚙️  Epochs: {epochs}")

    result = training_service.train_model(training_data, model_output, epochs)

    print("✅ Training complete!")
    print(f"🎯 F1 Score: {result.f1_score:.3f}")
    print(f"📊 Precision: {result.precision:.3f}")
    print(f"📊 Recall: {result.recall:.3f}")
    print(f"⏱️  Training time: {result.training_time_seconds:.1f}s")
    print(f"� Model saved to: {model_output}")

    print(f"� Metrics saved to: {model_output}/training_metrics.json")


def export_textcat_data(args: argparse.Namespace) -> None:
    """Export textcat training data for SKR03 classification"""
    config = Config()
    training_service = TrainingService(config)

    input_dir = Path(args.input)
    output_dir = Path(args.output) if args.output else Path("data/training/textcat/")

    print(f"📤 Exporting textcat data from: {input_dir}")
    print(f"🎯 Output directory: {output_dir}")

    # Use regular export (supports both NER and TextCat)
    result = training_service.export_training_data(input_dir, output_dir)

    print("✅ Textcat export complete!")
    print(f"💾 Training data saved: {result.jsonl_path}")
    print(f"📊 Total examples: {result.total_records}")


def train_textcat_model(args: argparse.Namespace) -> None:
    """Train spaCy text classification model for SKR03"""
    config = Config()
    training_service = TrainingService(config)

    training_data_dir = Path(args.input)
    model_output = (
        Path(args.output) if args.output else Path("output_model/textcat_model")
    )
    epochs = args.epochs

    print(f"🚀 Training textcat model with: {training_data_dir}")
    print(f"⚙️  Epochs: {epochs}")
    print(f"🎯 Output model: {model_output}")

    # Train textcat model using new API
    result = training_service.train_model(
        training_data_dir, model_output, epochs, "textcat"
    )

    print("✅ Textcat training complete!")
    print(f"📁 Model saved to: {model_output}")
    print(f"🎯 F1 Score: {result.f1_score:.3f}")


def run_pipeline(args: argparse.Namespace) -> None:
    """Run complete PDF→Training pipeline"""
    config = Config()

    input_dir = Path(args.input)
    temp_processed = Path("data/processed/pipeline_temp")
    temp_training = Path("data/training/pipeline_training.jsonl")
    model_output = (
        Path(args.output) if args.output else Path("output_model/pipeline_model")
    )

    print("🔄 Running complete ML pipeline...")
    print(f"📁 Input: {input_dir}")
    print(f"🎯 Final model: {model_output}")

    # Step 1: Process PDFs
    print("\n📄 Step 1: Processing PDFs...")
    processor = UnifiedProcessor(config)

    pdf_files = list(input_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"❌ No PDF files found in {input_dir}")
        return

    temp_processed.mkdir(parents=True, exist_ok=True)

    for pdf_file in pdf_files:
        result = processor.process_pdf(pdf_file)
        print(f"  ✅ {pdf_file.name}: {len(result.skr03_classifications)} SKR03")

    # Step 2: Export training data
    print("\n📤 Step 2: Exporting training data...")
    training_service = TrainingService(config)
    export_result = training_service.export_training_data(temp_processed, temp_training)
    print(f"  ✅ {export_result.total_records} training examples exported")

    # Step 3: Train model
    print("\n🚀 Step 3: Training model...")
    train_result = training_service.train_model(
        temp_training, model_output, args.epochs
    )

    print("\n🎉 Pipeline complete!")
    print(f"🎯 F1 Score: {train_result.f1_score:.3f}")
    print(f"📁 Final model saved to: {model_output}")


def analyze_results(args: argparse.Namespace) -> None:
    """Analyze processing results and model performance"""
    results_dir = Path(args.input)

    print(f"📊 Analyzing results in: {results_dir}")

    # Load all result files
    json_files = list(results_dir.glob("*.json"))

    if not json_files:
        print(f"❌ No JSON result files found in {results_dir}")
        return

    total_docs = len(json_files)
    total_skr03 = 0
    processing_times = []
    quality_scores = []

    for json_file in json_files:
        try:
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)

            # Extract metrics
            skr03_count = len(data.get("skr03_classifications", []))
            total_skr03 += skr03_count

            if "processing_time_ms" in data:
                processing_times.append(
                    data["processing_time_ms"] / 1000
                )  # Convert to seconds

            if "confidence_score" in data:
                quality_scores.append(data["confidence_score"])

        except Exception as e:
            print(f"⚠️  Failed to analyze {json_file.name}: {e}")
            continue

    # Calculate statistics
    avg_processing_time = (
        sum(processing_times) / len(processing_times) if processing_times else 0
    )
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0

    print("\n📈 Analysis Results:")
    print(f"   Documents processed: {total_docs}")
    print(f"   SKR03 classifications: {total_skr03}")
    print(f"   Average SKR03 per doc: {total_skr03/total_docs:.1f}")

    if processing_times:
        print(f"   Average processing time: {avg_processing_time:.2f}s")

    if quality_scores:
        print(f"   Average quality score: {avg_quality:.3f}")


def main() -> None:
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="LLKJJ ML Pipeline - Unified Processing & Training CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Process single PDF:
    python main.py process input.pdf

  Process directory of PDFs:
    python main.py process /path/to/pdfs/ --output /path/to/results/

  Export training data:
    python main.py export /path/to/processed/ --output training.jsonl

  Train model:
    python main.py train training.jsonl --output model/ --epochs 30

  Export textcat training data:
    python main.py export-textcat /path/to/processed/ --output textcat_data/

  Train textcat model:
    python main.py train-textcat textcat_data/ --output textcat_model/ --epochs 30

  Run complete pipeline:
    python main.py pipeline /path/to/pdfs/ --output final_model/

  Analyze results:
    python main.py analyze /path/to/results/
        """,
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Process command
    process_parser = subparsers.add_parser("process", help="Process PDF invoices")
    process_parser.add_argument("input", help="PDF file or directory to process")
    process_parser.add_argument("--output", "-o", help="Output directory (optional)")

    # Export command
    export_parser = subparsers.add_parser("export", help="Export training data")
    export_parser.add_argument(
        "input", help="Directory containing processed JSON files"
    )
    export_parser.add_argument("--output", "-o", help="Output JSONL file (optional)")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train spaCy model")
    train_parser.add_argument("input", help="Training data JSONL file")
    train_parser.add_argument(
        "--output", "-o", help="Output model directory (optional)"
    )
    train_parser.add_argument(
        "--epochs", type=int, default=20, help="Training epochs (default: 20)"
    )

    # Export textcat command
    export_textcat_parser = subparsers.add_parser(
        "export-textcat", help="Export textcat training data"
    )
    export_textcat_parser.add_argument(
        "input", help="Directory containing processed JSON files"
    )
    export_textcat_parser.add_argument(
        "--output", "-o", help="Output directory for textcat data (optional)"
    )

    # Train textcat command
    train_textcat_parser = subparsers.add_parser(
        "train-textcat", help="Train spaCy text classification model"
    )
    train_textcat_parser.add_argument("input", help="Textcat training data directory")
    train_textcat_parser.add_argument(
        "--output", "-o", help="Output textcat model directory (optional)"
    )
    train_textcat_parser.add_argument(
        "--epochs", type=int, default=30, help="Training epochs (default: 30)"
    )

    # Pipeline command
    pipeline_parser = subparsers.add_parser("pipeline", help="Run complete pipeline")
    pipeline_parser.add_argument("input", help="Directory containing PDF files")
    pipeline_parser.add_argument(
        "--output", "-o", help="Final model output directory (optional)"
    )
    pipeline_parser.add_argument(
        "--epochs", type=int, default=20, help="Training epochs (default: 20)"
    )

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze processing results")
    analyze_parser.add_argument("input", help="Directory containing result JSON files")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Setup logging
    setup_logging(args.verbose)

    # Ensure required directories exist
    Path("data/output").mkdir(parents=True, exist_ok=True)
    Path("data/training").mkdir(parents=True, exist_ok=True)
    Path("output_model").mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(parents=True, exist_ok=True)

    # Route to appropriate function
    try:
        if args.command == "process":
            process_pdfs(args)
        elif args.command == "export":
            export_training_data(args)
        elif args.command == "train":
            train_model(args)
        elif args.command == "export-textcat":
            export_textcat_data(args)
        elif args.command == "train-textcat":
            train_textcat_model(args)
        elif args.command == "pipeline":
            run_pipeline(args)
        elif args.command == "analyze":
            analyze_results(args)

    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Pipeline error: {e}", exc_info=True)
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
