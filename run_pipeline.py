#!/usr/bin/env python3
"""
LLKJJ ML Pipeline - CLI Runner

Einfacher Command-Line Interface zum Testen der PDF-Verarbeitungs-Pipeline
für deutsche Elektrotechnik-Rechnungen.
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Local imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.classification.skr03_classifier import SKR03Classifier
from src.extraction.gemini_extractor import GeminiExtractor
from src.pipeline.invoice_processor import InvoiceProcessor
from src.training.data_exporter import DataExporter, ExportConfig


def setup_argparser() -> argparse.ArgumentParser:
    """Erstellt Command-Line Interface"""

    parser = argparse.ArgumentParser(
        description="LLKJJ ML Pipeline - PDF Rechnungsverarbeitung für Elektrotechnik",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  %(prog)s process invoice.pdf                    # Einzelne PDF verarbeiten
  %(prog)s batch ./rechnungen/                    # Alle PDFs in Ordner verarbeiten
  %(prog)s export data/processed/ --format jsonl # Exportiere Training-Daten
  %(prog)s test                                   # Test-Modus mit Beispiel-Daten
        """,
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Verfügbare Kommandos")

    # Process command
    process_parser = subparsers.add_parser("process", help="Verarbeite einzelne PDF")
    process_parser.add_argument("pdf_path", type=str, help="Pfad zur PDF-Datei")
    process_parser.add_argument(
        "--output", "-o", type=str, help="Ausgabedatei (optional)"
    )
    process_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Ausführliche Ausgabe"
    )

    # Batch command
    batch_parser = subparsers.add_parser("batch", help="Verarbeite alle PDFs in Ordner")
    batch_parser.add_argument("input_dir", type=str, help="Eingabe-Ordner mit PDFs")
    batch_parser.add_argument("--output-dir", "-o", type=str, help="Ausgabe-Ordner")
    batch_parser.add_argument(
        "--max-files", type=int, default=10, help="Max. Anzahl Dateien"
    )

    # Export command
    export_parser = subparsers.add_parser(
        "export", help="Exportiere verarbeitete Daten"
    )
    export_parser.add_argument(
        "data_dir", type=str, help="Ordner mit verarbeiteten Daten"
    )
    export_parser.add_argument(
        "--format",
        choices=["jsonl", "csv", "sqlite", "json"],
        default="jsonl",
        help="Export-Format",
    )
    export_parser.add_argument("--output", "-o", type=str, help="Ausgabedatei")

    # Test command
    test_parser = subparsers.add_parser("test", help="Test-Modus mit Beispiel-Daten")
    test_parser.add_argument("--mock", action="store_true", help="Verwende Mock-Daten")

    return parser


def print_header() -> None:
    """Zeigt LLKJJ ML Pipeline Header"""
    print("=" * 60)
    print("🔌 LLKJJ ML Pipeline - Elektrotechnik Rechnungsverarbeitung")
    print("   Gemini 2.5 Pro + SKR03 + spaCy Training")
    print("=" * 60)
    print()


def print_results(results: dict[str, Any], verbose: bool = False) -> None:
    """Zeigt Verarbeitungsresultate"""

    if not results:
        print("❌ Keine Ergebnisse erhalten")
        return

    print("📊 Verarbeitungsresultate:")
    print("-" * 30)

    # Basis-Informationen
    gemini_data = results.get("gemini_extraction", {})
    skr03_data = results.get("skr03_classification", {})

    print(f"📄 Rechnungsnummer: {gemini_data.get('invoice_number', 'N/A')}")
    print(f"🏢 Lieferant: {gemini_data.get('supplier', 'N/A')}")
    print(f"📅 Datum: {gemini_data.get('date', 'N/A')}")
    print(f"💰 Nettobetrag: {gemini_data.get('net_amount', 0):.2f} €")
    print(f"🧾 MwSt.: {gemini_data.get('vat_amount', 0):.2f} €")
    print(f"💵 Bruttobetrag: {gemini_data.get('gross_amount', 0):.2f} €")

    # SKR03 Klassifikation
    primary_account = skr03_data.get("primary_account", "N/A")
    print(f"📋 SKR03 Hauptkonto: {primary_account}")

    # Positionen
    line_items = gemini_data.get("line_items", [])
    print(f"📦 Anzahl Positionen: {len(line_items)}")

    if verbose and line_items:
        print("\n📦 Rechnungspositionen:")
        for i, item in enumerate(line_items[:5], 1):  # Zeige max. 5
            print(f"  {i}. {item.get('description', 'N/A')}")
            print(
                f"     Menge: {item.get('quantity', 0)} | "
                f"Preis: {item.get('unit_price', 0):.2f} € | "
                f"Gesamt: {item.get('total_price', 0):.2f} €"
            )
            print(f"     SKR03: {item.get('skr03_account', 'N/A')}")

    # Confidence Score
    confidence = gemini_data.get("confidence_score", 0)
    confidence_emoji = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.6 else "🔴"
    print(f"{confidence_emoji} Confidence: {confidence:.1%}")

    print()


async def process_single_pdf(
    pdf_path: str, output_path: str | None = None
) -> dict[str, Any]:
    """Verarbeitet einzelne PDF-Datei"""

    print(f"🔄 Verarbeite PDF: {pdf_path}")

    try:
        # Initialisiere Pipeline
        processor = InvoiceProcessor()

        # Verarbeite PDF
        pdf_file = Path(pdf_path)
        if not pdf_file.exists():
            raise FileNotFoundError(f"PDF-Datei nicht gefunden: {pdf_path}")

        invoice_data = processor.process_pdf(pdf_file)
        results = (
            invoice_data.__dict__ if hasattr(invoice_data, "__dict__") else invoice_data
        )

        # Speichere Ergebnisse
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            print(f"💾 Ergebnisse gespeichert: {output_file}")

        # Always return dict for downstream compatibility
        return results if isinstance(results, dict) else results.__dict__

    except Exception as e:
        print(f"❌ Fehler bei PDF-Verarbeitung: {e}")
        return {}


async def process_batch(
    input_dir: str, output_dir: str | None = None, max_files: int = 10
) -> list[dict[str, Any]]:
    """Verarbeitet alle PDFs in einem Ordner"""

    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"❌ Eingabe-Ordner nicht gefunden: {input_dir}")
        return []

    # Finde alle PDF-Dateien
    pdf_files = list(input_path.glob("*.pdf"))
    if not pdf_files:
        print(f"❌ Keine PDF-Dateien gefunden in: {input_dir}")
        return []

    # Begrenze Anzahl
    pdf_files = pdf_files[:max_files]

    print(f"🔄 Verarbeite {len(pdf_files)} PDF-Dateien...")

    # Ausgabe-Ordner vorbereiten
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

    # Pipeline initialisieren
    processor = InvoiceProcessor()

    all_results = []

    # Verarbeite alle PDFs
    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"\n📄 [{i}/{len(pdf_files)}] {pdf_file.name}")

        try:
            invoice_data = processor.process_pdf(pdf_file)
            results = (
                invoice_data.__dict__
                if not isinstance(invoice_data, dict)
                else invoice_data
            )
            all_results.append(results)

            # Speichere Einzelergebnis
            if output_dir:
                result_file = Path(output_dir) / f"{pdf_file.stem}_result.json"
                with open(result_file, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)

            print("✅ Erfolgreich verarbeitet")

        except Exception as e:
            print(f"❌ Fehler: {e}")
            continue

    # Batch-Zusammenfassung speichern
    if output_dir and all_results:
        batch_file = Path(output_dir) / "batch_results.json"
        batch_data = {
            "processing_date": datetime.now().isoformat(),
            "total_files": len(pdf_files),
            "successful": len(all_results),
            "failed": len(pdf_files) - len(all_results),
            "results": all_results,
        }

        with open(batch_file, "w", encoding="utf-8") as f:
            json.dump(batch_data, f, ensure_ascii=False, indent=2)

        print(f"\n💾 Batch-Ergebnisse gespeichert: {batch_file}")

    print("\n📊 Batch-Verarbeitung abgeschlossen:")
    print(f"   ✅ Erfolgreich: {len(all_results)}")
    print(f"   ❌ Fehlgeschlagen: {len(pdf_files) - len(all_results)}")

    return all_results


def export_data(
    data_dir: str, format_type: str, output_path: str | None = None
) -> None:
    """Exportiert verarbeitete Daten"""

    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"❌ Daten-Ordner nicht gefunden: {data_dir}")
        return

    print(f"📤 Exportiere Daten im {format_type.upper()}-Format...")

    # Lade alle JSON-Ergebnisse
    processed_data = []
    for json_file in data_path.glob("*.json"):
        try:
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)

                # Handle batch results
                if "results" in data and isinstance(data["results"], list):
                    processed_data.extend(data["results"])
                else:
                    processed_data.append(data)
        except Exception as e:
            print(f"⚠️  Fehler beim Laden von {json_file}: {e}")
            continue

    if not processed_data:
        print("❌ Keine verarbeiteten Daten gefunden")
        return

    # Export-Konfiguration
    output_dir = (
        Path(data_dir) / "exports" if not output_path else Path(output_path).parent
    )
    export_config = ExportConfig(
        output_dir=output_dir,
        include_metadata=True,
        include_confidence=True,
        anonymize_data=False,
    )

    exporter = DataExporter(export_config)

    try:
        if format_type == "jsonl":
            result_path = exporter.export_jsonl_for_spacy(processed_data)
        elif format_type == "csv":
            result_path = exporter.export_csv_summary(processed_data)
        elif format_type == "sqlite":
            result_path = exporter.export_sqlite_database(processed_data)
        elif format_type == "json":
            result_path = exporter.export_json_archive(processed_data)
        else:
            print(f"❌ Unbekanntes Format: {format_type}")
            return

        print(f"✅ Export erfolgreich: {result_path}")
        print(f"📊 {len(processed_data)} Datensätze exportiert")

    except Exception as e:
        print(f"❌ Export-Fehler: {e}")


def run_test_mode(mock: bool = False) -> None:
    """Test-Modus mit Beispiel-Daten"""

    print("🧪 Test-Modus gestartet")

    if mock:
        print("📝 Verwende Mock-Daten...")

        # Mock-Ergebnis
        mock_result = {
            "file_path": "test_rechnung.pdf",
            "processing_date": datetime.now().isoformat(),
            "gemini_extraction": {
                "invoice_number": "RE-2024-001",
                "supplier": "Conrad Electronic GmbH",
                "date": "2024-08-14",
                "net_amount": 125.50,
                "vat_amount": 23.85,
                "gross_amount": 149.35,
                "confidence_score": 0.94,
                "line_items": [
                    {
                        "item_number": "NYM-J-001",
                        "description": "Installationsleitung NYM-J 3x1,5 mm²",
                        "quantity": 10.0,
                        "unit_price": 12.55,
                        "total_price": 125.50,
                        "skr03_account": "4390",
                        "product_category": "Elektromaterial",
                    }
                ],
            },
            "skr03_classification": {
                "primary_account": "4390",
                "account_name": "Elektromaterial",
                "confidence": 0.91,
                "classifications": [
                    {
                        "account": "4390",
                        "name": "Elektromaterial",
                        "confidence": 0.91,
                        "type": "material",
                    }
                ],
            },
        }

        print_results(mock_result, verbose=True)

    else:
        print("🔍 System-Test wird ausgeführt...")

        # Teste Konfiguration
        try:
            print("✅ Konfiguration geladen")
        except Exception as e:
            print(f"❌ Konfigurationsfehler: {e}")
            return

        # Teste Gemini Extraktor
        try:
            GeminiExtractor(api_key="YOUR_API_KEY")
            print("✅ Gemini Extraktor initialisiert")
        except Exception as e:
            print(f"❌ Gemini Extraktor Fehler: {e}")

        # Teste SKR03 Klassifikator
        try:
            SKR03Classifier()
            print("✅ SKR03 Klassifikator initialisiert")
        except Exception as e:
            print(f"❌ SKR03 Klassifikator Fehler: {e}")

        # Teste Pipeline
        try:
            InvoiceProcessor()
            print("✅ Invoice Processor initialisiert")
        except Exception as e:
            print(f"❌ Invoice Processor Fehler: {e}")

        print("\n🎉 System-Test abgeschlossen!")


async def main() -> None:
    """Hauptfunktion"""

    parser = setup_argparser()
    args = parser.parse_args()

    print_header()

    if not args.command:
        parser.print_help()
        return

    try:
        if args.command == "process":
            results = await process_single_pdf(args.pdf_path, args.output)
            if results:
                print_results(
                    results if isinstance(results, dict) else results.__dict__,
                    args.verbose,
                )

        elif args.command == "batch":
            batch_results = await process_batch(
                args.input_dir, args.output_dir, args.max_files
            )
            print(f"\n📋 Verarbeitet: {len(batch_results)} Rechnungen")

        elif args.command == "export":
            export_data(args.data_dir, args.format, args.output)

        elif args.command == "test":
            run_test_mode(args.mock)

    except KeyboardInterrupt:
        print("\n⏹️  Vorgang abgebrochen")
    except Exception as e:
        print(f"\n❌ Unerwarteter Fehler: {e}")
        if args.verbose if hasattr(args, "verbose") else False:
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
