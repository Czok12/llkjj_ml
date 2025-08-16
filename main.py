#!/usr/bin/env python3
"""
LLKJJ ML Pipeline - Poetry CLI Runner
Haupteinstiegspunkt für die Pipeline-Ausführung
"""

import logging
import os
import sys
from pathlib import Path

import click
from dotenv import load_dotenv

# Füge src zum Python Path hinzu
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config import Config
from src.extraction.docling_invoice_processor import (
    create_accurate_processor,
    create_fast_processor,
)
from src.extraction.gemini_extractor import create_gemini_extractor
from src.pipeline.invoice_processor import InvoiceProcessor
from src.training.data_exporter import DataExporter
from src.training.spacy_trainer import SpacyTrainer

# Lade Umgebungsvariablen
load_dotenv()


@click.group()
@click.option("--debug/--no-debug", default=False, help="Debug-Modus aktivieren")
@click.option("--config-file", default=None, help="Pfad zur Konfigurationsdatei")
def cli(debug: bool, config_file: str):
    """LLKJJ ML Pipeline - Automatisierte Rechnungsverarbeitung"""
    if debug:
        logging.basicConfig(level=logging.DEBUG)  # type: ignore
    else:
        logging.basicConfig(level=logging.INFO)

    if config_file:
        os.environ["CONFIG_FILE"] = config_file


@cli.command()
@click.option("--input-dir", default="./data/input/pdfs", help="PDF Input Verzeichnis")
@click.option("--output-dir", default="./data/output", help="Output Verzeichnis")
@click.option(
    "--mode",
    type=click.Choice(["fast", "accurate"]),
    default="accurate",
    help="Verarbeitungsmodus",
)
@click.option(
    "--with-gemini/--docling-only", default=False, help="Zusätzlich Gemini AI verwenden"
)
def docling(input_dir: str, output_dir: str, mode: str, with_gemini: bool):
    """Verarbeite PDF-Rechnungen mit Docling (strukturelle Dokumentenanalyse)"""

    try:
        click.echo("🔍 Starte Docling-basierte Rechnungsverarbeitung...")  # type: ignore

        # Konfiguration laden
        config = Config()

        # Docling Processor erstellen
        if mode == "fast":
            processor = create_fast_processor()
            click.echo("⚡ Verwende Fast-Modus für schnelle Verarbeitung")
        else:
            processor = create_accurate_processor()
            click.echo("🎯 Verwende Accurate-Modus für höchste Genauigkeit")

        # Optional: Gemini Extractor
        gemini_extractor = None
        if with_gemini:
            api_key = config.google_api_key
            if api_key:
                gemini_extractor = create_gemini_extractor(api_key)
                click.echo("🤖 Gemini AI-Extraktion aktiviert")
            else:
                click.echo("⚠️  Keine Google API Key - nur Docling-Extraktion")

        # Input-Verzeichnis vorbereiten
        input_path = Path(input_dir)
        if not input_path.exists():
            click.echo(f"📁 Erstelle Input-Verzeichnis: {input_path}")
            input_path.mkdir(parents=True, exist_ok=True)

        # PDF-Dateien finden
        pdf_files = list(input_path.glob("*.pdf"))
        if not pdf_files:
            click.echo(f"⚠️  Keine PDF-Dateien in {input_path} gefunden")
            click.echo("💡 Tipp: Lege PDF-Rechnungen in das Input-Verzeichnis")
            return

        click.echo(f"📄 Gefunden: {len(pdf_files)} PDF-Dateien")

        # Output-Verzeichnis vorbereiten
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Batch-Verarbeitung starten
        click.echo("🔄 Starte Batch-Verarbeitung...")
        results = processor.batch_process(input_path, output_path, gemini_extractor)

        click.echo(f"✅ Verarbeitung abgeschlossen: {len(results)} Rechnungen")

        # Zusammenfassung
        if results:
            total_amount = sum(r.gross_amount for r in results if r.gross_amount > 0)
            avg_confidence = sum(r.docling_confidence for r in results) / len(results)
            total_tables = sum(r.tables_found for r in results)

            click.echo(f"💰 Gesamtbetrag: {total_amount:.2f} EUR")
            click.echo(f"🎯 Durchschnittliche Confidence: {avg_confidence:.2f}")
            click.echo(f"📊 Tabellen erkannt: {total_tables}")
            click.echo(f"💾 Ergebnisse gespeichert in: {output_path}")

    except Exception as e:
        click.echo(f"❌ Fehler bei der Docling-Verarbeitung: {e}")
        if os.getenv("DEBUG_MODE", "false").lower() == "true":
            import traceback

            traceback.print_exc()


@cli.command()
@click.option("--input-dir", default="./data/input/pdfs", help="PDF Input Verzeichnis")
@click.option("--output-dir", default="./data/output", help="Output Verzeichnis")
@click.option("--mock/--no-mock", default=False, help="Mock-Modus für Tests")
def process(input_dir: str, output_dir: str, mock: bool) -> None:
    """Verarbeite PDF-Rechnungen"""
    click.echo("🔄 Starte LLKJJ Pipeline...")

    try:
        # Konfiguration laden
        Config()

        # API Key prüfen
        api_key = os.getenv("GOOGLE_API_KEY", "test-key" if mock else None)
        if not api_key and not mock:
            click.echo("❌ GOOGLE_API_KEY nicht gesetzt. Nutze --mock für Tests.")
            return

        # Gemini Extractor erstellen
        create_gemini_extractor(api_key if api_key else "test-key")

        # Pipeline erstellen und ausführen
        processor = InvoiceProcessor()

        # Input-Verzeichnis prüfen
        input_path = Path(input_dir)
        if not input_path.exists():
            click.echo(f"📁 Erstelle Input-Verzeichnis: {input_path}")
            input_path.mkdir(parents=True, exist_ok=True)

        # PDF-Dateien finden
        pdf_files = list(input_path.glob("*.pdf"))
        if not pdf_files:
            click.echo(f"⚠️  Keine PDF-Dateien in {input_path} gefunden")
            click.echo("💡 Tipp: Lege PDF-Rechnungen in das Input-Verzeichnis")
            return

        click.echo(f"📄 Gefunden: {len(pdf_files)} PDF-Dateien")

        # Verarbeitung starten
        results = [processor.process_pdf(pdf) for pdf in pdf_files]

        click.echo(f"✅ Verarbeitung abgeschlossen: {len(results)} Rechnungen")

        # Zusammenfassung
        total_amount = sum(
            getattr(r, "gross_amount", 0)
            for r in results
            if getattr(r, "gross_amount", 0) > 0
        )
        avg_confidence = (
            sum(getattr(r, "confidence_score", 0) for r in results) / len(results)
            if results
            else 0
        )

        click.echo(f"💰 Gesamtbetrag: {total_amount:.2f} EUR")
        click.echo(f"🎯 Durchschnittliche Confidence: {avg_confidence:.2f}")

    except Exception as e:
        click.echo(f"❌ Fehler bei der Verarbeitung: {e}")
        if os.getenv("DEBUG_MODE", "false").lower() == "true":
            import traceback

            traceback.print_exc()


@cli.command()
@click.option(
    "--input-dir", default="./data/training", help="Training Data Verzeichnis"
)
@click.option("--output-dir", default="./models", help="Modell Output Verzeichnis")
@click.option("--iterations", default=100, help="Training Iterationen")
def train(input_dir: str, output_dir: str, iterations: int):
    """Trainiere spaCy-Modell für SKR03-Klassifikation"""
    click.echo("🧠 Starte spaCy Training...")

    try:
        config = Config()
        trainer = SpacyTrainer(config)

        # Training starten
        model_path = trainer.train_from_jsonl(
            jsonl_path=f"{input_dir}/training_data.jsonl",
            output_dir=output_dir,
            iterations=iterations,
        )

        click.echo(f"✅ Training abgeschlossen: {model_path}")

    except Exception as e:
        click.echo(f"❌ Fehler beim Training: {e}")


@cli.command()
@click.option(
    "--input-dir", default="./data/output", help="Extraktionsdaten Verzeichnis"
)
@click.option(
    "--output-file",
    default="./data/training/training_data.jsonl",
    help="JSONL Output Datei",
)
def export(input_dir: str, output_file: str):
    """Exportiere Daten für spaCy-Training"""
    click.echo("📦 Starte Datenexport...")

    try:
        config = Config()
        exporter = DataExporter(config)

        # Export starten
        count = exporter.export_to_jsonl(input_dir, output_file)

        click.echo(f"✅ Export abgeschlossen: {count} Einträge nach {output_file}")

    except Exception as e:
        click.echo(f"❌ Fehler beim Export: {e}")


@cli.command()
def test():
    """Teste die Pipeline mit Mock-Daten"""
    click.echo("🧪 Starte Pipeline-Test...")

    try:
        # Test-Konfiguration
        os.environ["MOCK_GEMINI"] = "true"
        os.environ["DEBUG_MODE"] = "true"

        Config()

        # Mock-Extractor
        extractor = create_gemini_extractor("test-key")

        # Test-Text
        test_text = """
        RECHNUNG RE-2024-TEST
        Conrad Electronic SE
        Datum: 14.08.2024

        1x Installationsleitung NYM-J 3x1,5mm² - 125,00€
        1x SCHUKO Steckdose weiß - 31,78€

        Netto: 156,78€ | USt 19%: 29,79€ | Brutto: 186,57€
        """

        # Test-Extraktion
        result = extractor.extract_from_text(test_text)
        validation = extractor.validate_extraction(result)

        click.echo(f"📄 Test-Rechnung: {result.invoice_number}")
        click.echo(f"🏢 Lieferant: {result.supplier}")
        click.echo(f"💰 Betrag: {result.gross_amount:.2f} EUR")
        click.echo(f"🎯 Confidence: {result.confidence_score:.2f}")
        click.echo(f"✅ Validierung: {'OK' if validation['is_valid'] else 'FEHLER'}")

        if validation["warnings"]:
            click.echo("⚠️  Warnungen:")
            for warning in validation["warnings"]:
                click.echo(f"   - {warning}")

        click.echo("🧪 Test erfolgreich abgeschlossen!")

    except Exception as e:
        click.echo(f"❌ Test fehlgeschlagen: {e}")
        import traceback

        traceback.print_exc()


@cli.command()
def setup():
    """Erstelle notwendige Verzeichnisse und Konfigurationsdateien"""
    click.echo("⚙️  Richte LLKJJ Pipeline ein...")

    # Verzeichnisse erstellen
    dirs = [
        "data/input/pdfs",
        "data/output",
        "data/training",
        "data/vectors",
        "data/raw",
        "data/processed",
        "models",
        "logs",
        "config",
    ]

    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        click.echo(f"📁 {dir_path}")

    # .env erstellen falls nicht vorhanden
    env_path = Path(".env")
    if not env_path.exists():
        env_example = Path(".env.example")
        if env_example.exists():
            import shutil

            shutil.copy(env_example, env_path)
            click.echo("📝 .env Datei erstellt (bitte API-Key eintragen)")
        else:
            click.echo("⚠️  .env.example nicht gefunden")

    click.echo("✅ Setup abgeschlossen!")
    click.echo("")
    click.echo("Nächste Schritte:")
    click.echo("1. GOOGLE_API_KEY in .env eintragen")
    click.echo("2. PDF-Dateien in data/input/pdfs/ ablegen")
    click.echo("3. 'poetry run llkjj-pipeline process' ausführen")


@cli.command()
def info():
    """Zeige System-Informationen"""
    click.echo("📊 LLKJJ Pipeline Information")
    click.echo("=" * 40)

    try:
        Config()

        click.echo(f"🐍 Python: {sys.version}")
        click.echo(f"📦 Poetry: {os.popen('poetry --version').read().strip()}")

        # Prüfe wichtige Verzeichnisse
        dirs_info = {
            "Input PDFs": "data/input/pdfs",
            "Output": "data/output",
            "Training": "data/training",
            "Models": "models",
            "Logs": "logs",
        }

        click.echo("\n📁 Verzeichnisse:")
        for name, path in dirs_info.items():
            exists = "✅" if Path(path).exists() else "❌"
            count = len(list(Path(path).glob("*"))) if Path(path).exists() else 0
            click.echo(f"   {exists} {name}: {path} ({count} Dateien)")

        # API-Key Status
        api_key = os.getenv("GOOGLE_API_KEY")
        api_status = "✅ Gesetzt" if api_key else "❌ Nicht gesetzt"
        click.echo(f"\n🔑 Google AI API Key: {api_status}")

        # spaCy Modell
        try:
            import spacy

            spacy.load("de_core_news_sm")
            spacy_status = "✅ Verfügbar"
        except (ImportError, OSError):
            spacy_status = "❌ Nicht installiert"
        click.echo(f"🧠 spaCy Deutsch-Modell: {spacy_status}")

    except Exception as e:
        click.echo(f"❌ Fehler beim Laden der Informationen: {e}")


if __name__ == "__main__":
    cli()
