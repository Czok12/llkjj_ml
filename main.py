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
from src.processing.modular_processor import ModularProcessor
from src.trainer import TrainingService


def setup_logging(verbose: bool = False, production: bool = False) -> None:
    """Configure logging for CLI"""
    if production:
        # Produktions-Modus: Nur Warnungen und Fehler
        level = logging.WARNING
    else:
        # Development-Modus: Debug oder Info
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

            except (ValueError, FileNotFoundError) as e:
                print(f"  ❌ {pdf_file.name}: Eingabefehler - {e}")
                continue
            except (MemoryError, OSError) as e:
                print(f"  ❌ {pdf_file.name}: Systemfehler - {e}")
                continue
            except Exception as e:
                print(f"  ❌ {pdf_file.name}: Unerwarteter Fehler - {e}")
                logging.error(
                    "Unerwarteter Fehler bei PDF-Verarbeitung: %s", e, exc_info=True
                )
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
    temp_training_dir = Path("data/training")  # Verzeichnis für Export
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
    export_result = training_service.export_training_data(
        temp_processed, temp_training_dir
    )
    print(f"  ✅ {export_result.total_records} training examples exported")

    # Step 3: Train model
    print("\n🚀 Step 3: Training model...")
    train_result = training_service.train_model(
        Path(export_result.jsonl_path), model_output, args.epochs
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

        except (FileNotFoundError, PermissionError) as e:
            print(f"⚠️  Dateizugriff fehlgeschlagen für {json_file.name}: {e}")
            continue
        except (json.JSONDecodeError, KeyError) as e:
            print(f"⚠️  Ungültiges JSON-Format in {json_file.name}: {e}")
            continue
        except Exception as e:
            print(f"⚠️  Unerwarteter Fehler bei Analyse von {json_file.name}: {e}")
            logging.error("Unerwarteter Fehler bei Datei-Analyse: %s", e, exc_info=True)
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


# NEW MODULAR WORKFLOW FUNCTIONS
def run_workflow_1(args: argparse.Namespace) -> None:
    """Run Workflow 1: PDF → Docling → TXT only"""
    config = Config()
    processor = ModularProcessor(config)

    pdf_path = Path(args.input)
    output_path = Path(args.output) if args.output else None

    print("🔄 Workflow 1: PDF → Docling → TXT")
    print(f"📄 Input: {pdf_path}")

    result = processor.workflow_1_pdf_to_docling_txt(pdf_path, output_path)

    print("✅ Workflow 1 completed!")
    print(f"📝 Output TXT: {result['output_txt']}")
    print(f"📊 Text length: {result['text_length']} characters")
    print(f"📋 Tables found: {result['table_count']}")


def run_workflow_2(args: argparse.Namespace) -> None:
    """Run Workflow 2: PDF → Gemini directly"""
    config = Config()
    processor = ModularProcessor(config)

    pdf_path = Path(args.input)
    output_path = Path(args.output) if args.output else None

    print("🔄 Workflow 2: PDF → Gemini direct")
    print(f"📄 Input: {pdf_path}")

    result = processor.workflow_2_pdf_to_gemini_direct(pdf_path, output_path)

    print("✅ Workflow 2 completed!")
    print(f"💾 Output JSON: {result['output_json']}")
    print(f"📊 Raw text: {result['raw_text_length']} characters")
    print(f"🤖 Method: {result['extraction_method']}")


def run_workflow_3(args: argparse.Namespace) -> None:
    """Run Workflow 3: Docling TXT → Gemini processing"""
    config = Config()
    processor = ModularProcessor(config)

    txt_path = Path(args.input)
    output_path = Path(args.output) if args.output else None

    print("🔄 Workflow 3: Docling TXT → Gemini")
    print(f"📝 Input: {txt_path}")

    result = processor.workflow_3_docling_txt_to_gemini(txt_path, output_path)

    print("✅ Workflow 3 completed!")
    print(f"💾 Output JSON: {result['output_json']}")
    print(f"📊 Text processed: {result['text_length']} characters")
    print(f"🤖 Method: {result['processing_method']}")


def run_workflow_4(args: argparse.Namespace) -> None:
    """Run Workflow 4: Complete pipeline"""
    config = Config()
    processor = ModularProcessor(config)

    pdf_path = Path(args.input)
    output_path = Path(args.output) if args.output else None

    print("🔄 Workflow 4: Complete pipeline")
    print(f"📄 Input: {pdf_path}")

    # Verwende existierenden Workflow
    result = processor.workflow_2_gemini_direct(pdf_path, output_path)

    print("✅ Workflow 4 completed!")
    print(f"💾 Output JSON: {result['output_json']}")
    print(f"🎯 Quality: {result['quality_score']}")


def init_database(args: argparse.Namespace) -> None:
    """Initialize database with simple manager"""
    from src.database.simple_manager import create_simple_manager

    db_path = Path(args.path)

    print(f"🔧 Initializing database at: {db_path}")

    db_manager = create_simple_manager(str(db_path))
    results = db_manager.initialize_database()

    if results["success"]:
        print("✅ Database initialization completed!")
        print(f"⏱️  Total time: {results['total_time_seconds']}s")
        print(f"� Database path: {results['db_path']}")
        if "existing_collections" in results:
            print(f"�️  Existing collections: {results['existing_collections']}")
    else:
        print(
            f"❌ Database initialization failed: {results.get('error', 'Unknown error')}"
        )


def show_database_stats(args: argparse.Namespace) -> None:
    """Show database statistics with simple manager"""
    from src.database.simple_manager import create_simple_manager

    db_path = Path(args.path)

    print(f"📊 Database statistics for: {db_path}")

    db_manager = create_simple_manager(str(db_path))
    stats = db_manager.get_database_stats()

    if "error" in stats:
        print(f"❌ Error getting stats: {stats['error']}")
        return

    print(f"📁 Database path: {stats['db_path']}")
    print(f"🗃️  Total collections: {stats['total_collections']}")
    print(f"💓 Heartbeat: {stats['heartbeat']}")

    if stats["collections"]:
        print("\n📋 Collections:")
        for col in stats["collections"]:
            print(f"  • {col['name']}: {col['count']} documents")
    else:
        print("📭 No collections found")

        # Storage info
        storage = stats["database"]["storage"]
        print("\n💾 Storage:")
        print(f"   Path exists: {storage['path_exists']}")
        if storage["path_exists"]:
            print(f"   Total size: {storage['total_size_mb']} MB")

        # Collections info
        collections = stats["database"]["collections"]
        print(f"\n📚 Collections: {len(collections)}")
        for collection in collections:
            print(
                f"   {collection['name']}: {collection.get('count', 'N/A')} documents"
            )

        # Connection pool info
        pool_stats = stats["connection_pool"]
        print("\n🔗 Connection Pool:")
        print(f"   Active connections: {pool_stats['active_connections']}")
        print(f"   Total operations: {pool_stats['total_operations']}")
        print(
            f"   Average response time: {pool_stats['average_response_time_ms']:.2f}ms"
        )

        # Migration history
        migrations = stats["migration_history"]
        successful_migrations = [m for m in migrations if m["success"]]
        print(
            f"\n🔄 Migrations: {len(successful_migrations)}/{len(migrations)} successful"
        )


def backup_database(args: argparse.Namespace) -> None:
    """Create database backup with simple manager"""
    from src.database.simple_manager import create_simple_manager

    db_path = Path(args.path)
    backup_name = args.name

    print(f"💾 Creating backup for: {db_path}")

    db_manager = create_simple_manager(str(db_path))
    result = db_manager.backup_database(backup_name)

    if result["success"]:
        print(f"✅ Backup created: {result['backup_path']}")
        print(f"📦 Backup size: {result['backup_size_mb']:.2f} MB")
    else:
        print(f"❌ Backup failed: {result['error']}")


def optimize_database(args: argparse.Namespace) -> None:
    """Optimize database performance with simple manager"""
    from src.database.simple_manager import create_simple_manager

    db_path = Path(args.path)

    print(f"⚡ Optimizing database: {db_path}")

    db_manager = create_simple_manager(str(db_path))
    results = db_manager.optimize_database()

    if results["success"]:
        print("✅ Database optimization completed!")
        print(f"⏱️  Total time: {results['optimization_time_seconds']}s")
        print("🔧 Performance optimization applied")
    else:
        print(f"❌ Optimization failed: {results.get('error', 'Unknown error')}")


def create_arg_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser"""
    parser = argparse.ArgumentParser(
        description="LLKJJ ML Pipeline - Deutsche Elektrohandwerk Buchhaltungs-KI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  python main.py process rechnung.pdf
  python main.py export data/processed/ --output training.jsonl
  python main.py train training.jsonl --epochs 30
  python main.py database init --path data/vectors
  python main.py workflow1 rechnung.pdf
        """,
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose output (DEBUG logging)"
    )

    parser.add_argument(
        "--production",
        action="store_true",
        help="Production mode (only warnings and errors)",
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

    # Database command
    db_parser = subparsers.add_parser("database", help="Database operations")
    db_subparsers = db_parser.add_subparsers(
        dest="db_command", help="Database commands"
    )

    # Database init
    db_init_parser = db_subparsers.add_parser(
        "init", help="Initialize optimized database"
    )
    db_init_parser.add_argument("--path", default="data/vectors", help="Database path")

    # Database stats
    db_stats_parser = db_subparsers.add_parser("stats", help="Show database statistics")
    db_stats_parser.add_argument("--path", default="data/vectors", help="Database path")

    # Database backup
    db_backup_parser = db_subparsers.add_parser("backup", help="Create database backup")
    db_backup_parser.add_argument(
        "--path", default="data/vectors", help="Database path"
    )
    db_backup_parser.add_argument("--name", help="Backup name (optional)")

    # Database optimize
    db_optimize_parser = db_subparsers.add_parser("optimize", help="Optimize database")
    db_optimize_parser.add_argument(
        "--path", default="data/vectors", help="Database path"
    )

    # NEW MODULAR WORKFLOWS
    # Workflow 1: PDF → Docling → TXT only
    w1_parser = subparsers.add_parser("workflow1", help="PDF → Docling → TXT only")
    w1_parser.add_argument("input", help="PDF file to process")
    w1_parser.add_argument("--output", "-o", help="Output TXT file path (optional)")

    # Workflow 2: PDF → Gemini directly (bypass Docling)
    w2_parser = subparsers.add_parser(
        "workflow2", help="PDF → Gemini direct (bypass Docling)"
    )
    w2_parser.add_argument("input", help="PDF file to process")
    w2_parser.add_argument("--output", "-o", help="Output JSON file path (optional)")

    # Workflow 3: PDF → Docling → Gemini → Classification
    w3_parser = subparsers.add_parser(
        "workflow3", help="PDF → Docling → Gemini → Classification"
    )
    w3_parser.add_argument("input", help="PDF file to process")
    w3_parser.add_argument("--output", "-o", help="Output JSON file path (optional)")

    # Workflow 4: Full Pipeline with all features
    w4_parser = subparsers.add_parser(
        "workflow4", help="Full Pipeline with all features"
    )
    w4_parser.add_argument("input", help="PDF file to process")
    w4_parser.add_argument("--output", "-o", help="Output JSON file path (optional)")

    return parser


def main() -> None:
    """Main CLI entry point"""
    parser = create_arg_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Setup logging
    setup_logging(args.verbose, getattr(args, "production", False))

    try:
        # Route commands
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
        elif args.command == "database":
            # Database sub-commands
            if args.db_command == "init":
                init_database(args)
            elif args.db_command == "stats":
                show_database_stats(args)
            elif args.db_command == "backup":
                backup_database(args)
            elif args.db_command == "optimize":
                optimize_database(args)
            else:
                print("❌ Unknown database command. Use 'database -h' for help.")
        # NEW MODULAR WORKFLOWS
        elif args.command == "workflow1":
            run_workflow_1(args)
        elif args.command == "workflow2":
            run_workflow_2(args)
        elif args.command == "workflow3":
            run_workflow_3(args)
        elif args.command == "workflow4":
            run_workflow_4(args)

    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user")
        sys.exit(1)
    except (ImportError, ModuleNotFoundError) as e:
        logging.error("Modul-Import-Fehler: %s", e, exc_info=True)
        print(f"❌ Abhängigkeitsfehler: {e}")
        sys.exit(1)
    except (ValueError, TypeError) as e:
        logging.error("Konfigurationsfehler: %s", e, exc_info=True)
        print(f"❌ Konfigurationsfehler: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error("Pipeline-Fehler: %s", e, exc_info=True)
        print(f"❌ Unerwarteter Pipeline-Fehler: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
