#!/usr/bin/env python3
"""
CLI Interface für ML Service Package als Test-Harness
======================================================

Eigenständige Kommandozeilen-Schnittstelle für das LLKJJ ML Service Package.
Diese CLI dient als Test-Harness und ermöglicht es, alle ML Service Funktionen
eigenständig zu testen, ohne auf die Haupt-main.py angewiesen zu sein.

Verwendung:
    python -m ml_service.cli process /path/to/document.pdf
    python -m ml_service.cli validate-config
    python -m ml_service.cli test-capabilities

    # Mit Poetry:
    poetry run python -m ml_service.cli process test_pdfs/Sonepar_test3.pdf
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

# ML Service Imports
from ml_service import MLProcessor, MLSettings, ProcessingResult, process_pdf_simple


def setup_logging(verbose: bool = False) -> None:
    """Konfiguriere Logging für CLI."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def print_banner() -> None:
    """Zeige CLI-Banner."""
    print("🚀 LLKJJ ML Service CLI - Test-Harness")
    print("=" * 50)
    print("Eigenständiges ML Package für deutsche Elektrohandwerk-Rechnungen")
    print("Version: 2.1.0 (Post-Konsolidierung)")
    print()


def format_processing_result(result: ProcessingResult) -> dict[str, Any]:
    """Formatiere ProcessingResult für CLI-Ausgabe."""
    return {
        "metadata": {
            "pdf_path": result.pdf_path,
            "processing_time_seconds": result.processing_time_ms / 1000,
            "extraction_quality": result.extraction_quality,
            "confidence_score": result.confidence_score,
            "processing_timestamp": result.processing_timestamp,
        },
        "invoice_data": result.invoice_data,
        "structured_data": result.structured_data,
        "skr03_classifications": result.skr03_classifications,
        "summary": {
            "total_classifications": len(result.skr03_classifications),
            "avg_confidence": result.confidence_score,
            "raw_text_length": len(result.raw_text),
        },
    }


def cmd_process(args: argparse.Namespace) -> int:
    """Verarbeite PDF-Dokument."""
    pdf_path = Path(args.pdf_file)

    if not pdf_path.exists():
        print(f"❌ Fehler: PDF-Datei nicht gefunden: {pdf_path}")
        return 1

    if not pdf_path.suffix.lower() == ".pdf":
        print(f"❌ Fehler: Datei ist keine PDF: {pdf_path}")
        return 1

    print(f"🔍 Verarbeite PDF: {pdf_path}")
    print(f"📊 Dateigröße: {pdf_path.stat().st_size / 1024:.1f} KB")
    print()

    try:
        # Verwende convenience function für einfache Verarbeitung
        if args.simple:
            print("⚡ Verwende einfache Verarbeitung...")
            start_time = time.time()
            result = process_pdf_simple(str(pdf_path))
            elapsed = time.time() - start_time
            print(f"✅ Verarbeitung abgeschlossen in {elapsed:.1f}s")
        else:
            # Verwende vollständigen MLProcessor für erweiterte Kontrolle
            print("🚀 Verwende erweiterte Verarbeitung...")
            settings = MLSettings()

            if args.verbose:
                print("🔧 ML Settings:")
                print(f"   Gemini Model: {settings.gemini_model}")
                print(f"   Cache Enabled: {settings.cache_enabled}")
                print(f"   Vector DB: {settings.vector_db_path}")
                print()

            with MLProcessor(settings) as processor:
                print(
                    f"💡 Verfügbare Capabilities: {', '.join(processor.get_capabilities())}"
                )
                start_time = time.time()
                result = processor.process_pdf(str(pdf_path))
                elapsed = time.time() - start_time
                print(f"✅ Verarbeitung abgeschlossen in {elapsed:.1f}s")

        # Ausgabe formatieren
        if args.json:
            # JSON-Ausgabe für maschinelle Weiterverarbeitung
            output = format_processing_result(result)
            print(json.dumps(output, ensure_ascii=False, indent=2))
        else:
            # Menschenlesbare Ausgabe
            print()
            print("📋 VERARBEITUNGSERGEBNIS:")
            print("=" * 40)
            print(f"📄 Datei: {result.pdf_path}")
            print(f"⏱️  Zeit: {result.processing_time_ms / 1000:.1f}s")
            print(f"🎯 Qualität: {result.extraction_quality}")

            # Rechnungsdaten
            if result.invoice_data:
                print("\n💰 RECHNUNGSDATEN:")
                for key, value in result.invoice_data.items():
                    if value:
                        print(f"   {key}: {value}")

            # Strukturierte Daten
            structured_items = result.structured_data.get("items", [])
            print(f"\n📦 ARTIKEL ({len(structured_items)}):")
            for i, item in enumerate(structured_items[:10], 1):  # Erste 10 anzeigen
                art_nr = item.get("article_number", "N/A")
                desc = (
                    item.get("description", "N/A")[:50] + "..."
                    if len(item.get("description", "")) > 50
                    else item.get("description", "N/A")
                )
                qty = item.get("quantity", "N/A")
                print(f"   {i:2d}. {art_nr} | {desc} | Menge: {qty}")

            if len(structured_items) > 10:
                print(f"   ... und {len(structured_items) - 10} weitere")

            # SKR03-Klassifizierungen
            print(
                f"\n🏷️  SKR03-KLASSIFIZIERUNGEN ({len(result.skr03_classifications)}):"
            )
            for cls in result.skr03_classifications[:10]:  # Erste 10 anzeigen
                pos = cls.get("position", "N/A")
                code = cls.get("skr03_code", "N/A")
                conf = cls.get("confidence", 0)
                method = cls.get("method", "N/A")
                print(f"   {pos:2s}. SKR03 {code} | {conf:.1f}% | {method}")

            if len(result.skr03_classifications) > 10:
                print(f"   ... und {len(result.skr03_classifications) - 10} weitere")

        return 0

    except Exception as e:
        print(f"❌ Fehler bei der Verarbeitung: {str(e)}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


def cmd_validate_config(args: argparse.Namespace) -> int:
    """Validiere ML Service Konfiguration."""
    print("🔧 Validiere ML Service Konfiguration...")
    print()

    try:
        settings = MLSettings()
        validation = settings.validate_configuration()

        print("📋 KONFIGURATION:")
        print("=" * 30)
        print(f"✅ Status: {'Gültig' if validation['valid'] else 'Ungültig'}")
        print(f"🤖 Gemini Model: {settings.gemini_model}")
        print(f"💾 Cache: {'Aktiviert' if settings.cache_enabled else 'Deaktiviert'}")
        print(f"📊 Vector DB: {settings.vector_db_path}")
        print(f"📁 Models Cache: {settings.models_cache_path}")
        print(f"📂 Data Path: {settings.data_path}")
        print()

        if validation["warnings"]:
            print("⚠️  WARNUNGEN:")
            for warning in validation["warnings"]:
                print(f"   • {warning}")
            print()

        if validation.get("errors"):
            print("❌ FEHLER:")
            for error in validation["errors"]:
                print(f"   • {error}")
            print()

        print("🌍 UMGEBUNGSVARIABLEN:")
        env_vars = [
            "GEMINI_API_KEY",
            "ML_GEMINI_MODEL",
            "ML_ENABLE_CACHING",
            "ML_CACHE_DIR",
            "ML_OUTPUT_DIR",
            "ML_VECTOR_DB_PATH",
        ]

        import os

        for var in env_vars:
            value = os.getenv(var)
            status = "✅ Gesetzt" if value else "❌ Nicht gesetzt"
            masked_value = "***" if value and "API_KEY" in var else value
            print(
                f"   {var}: {status}"
                + (f" ({masked_value})" if value and "API_KEY" not in var else "")
            )

        return 0 if validation["valid"] else 1

    except Exception as e:
        print(f"❌ Fehler bei Konfigurationsprüfung: {str(e)}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


def cmd_test_capabilities(args: argparse.Namespace) -> int:
    """Teste ML Service Capabilities."""
    print("🧪 Teste ML Service Capabilities...")
    print()

    try:
        settings = MLSettings()

        with MLProcessor(settings) as processor:
            capabilities = processor.get_capabilities()

            print("💡 VERFÜGBARE CAPABILITIES:")
            print("=" * 35)
            for cap in capabilities:
                print(f"   ✅ {cap}")
            print()

            # Teste verschiedene Funktionen
            tests = [
                ("Processor Initialisierung", lambda: True),
                (
                    "Settings Validierung",
                    lambda: settings.validate_configuration()["valid"],
                ),
                ("Cache Verfügbarkeit", lambda: settings.models_cache_path.exists()),
                ("Data Directory", lambda: settings.data_path.exists()),
            ]

            print("🔬 FUNKTIONALITÄTSTESTS:")
            print("=" * 30)
            all_passed = True

            for test_name, test_func in tests:
                try:
                    result: bool = test_func()  # type: ignore[no-untyped-call]
                    status = "✅ PASS" if result else "❌ FAIL"
                    print(f"   {test_name}: {status}")
                    if not result:
                        all_passed = False
                except Exception as e:
                    print(f"   {test_name}: ❌ ERROR ({str(e)})")
                    all_passed = False

            print()
            overall_status = (
                "✅ Alle Tests bestanden"
                if all_passed
                else "❌ Einige Tests fehlgeschlagen"
            )
            print(f"🎯 GESAMTERGEBNIS: {overall_status}")

            return 0 if all_passed else 1

    except Exception as e:
        print(f"❌ Fehler bei Capability-Tests: {str(e)}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Führe Performance-Benchmark durch."""
    print("⚡ Performance-Benchmark...")
    print()

    if not args.pdf_file:
        print("❌ Fehler: PDF-Datei für Benchmark erforderlich")
        return 1

    pdf_path = Path(args.pdf_file)
    if not pdf_path.exists():
        print(f"❌ Fehler: PDF-Datei nicht gefunden: {pdf_path}")
        return 1

    try:
        MLSettings()
        runs = args.runs or 3

        print(f"🔄 Führe {runs} Benchmark-Läufe durch...")
        print(f"📄 Datei: {pdf_path}")
        print(f"📊 Größe: {pdf_path.stat().st_size / 1024:.1f} KB")
        print()

        times = []
        qualities = []

        for run in range(runs):
            print(f"🏃 Lauf {run + 1}/{runs}...")

            start_time = time.time()
            result = process_pdf_simple(str(pdf_path))
            elapsed = time.time() - start_time

            times.append(elapsed)
            qualities.append(result.extraction_quality)

            structured_items = result.structured_data.get("items", [])
            print(f"   ⏱️  Zeit: {elapsed:.2f}s")
            print(f"   🎯 Qualität: {result.extraction_quality}")
            print(f"   📦 Items: {len(structured_items)}")
            print()

        # Statistiken
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        avg_quality = (
            sum(len(q) for q in qualities) / len(qualities) if qualities else 0
        )

        print("📊 BENCHMARK-ERGEBNISSE:")
        print("=" * 30)
        print(f"   Durchschnitt: {avg_time:.2f}s")
        print(f"   Minimum: {min_time:.2f}s")
        print(f"   Maximum: {max_time:.2f}s")
        print(f"   Ø Qualität: {avg_quality}")
        print(f"   Varianz: {max_time - min_time:.2f}s")

        return 0

    except Exception as e:
        print(f"❌ Fehler beim Benchmark: {str(e)}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


def main() -> int:
    """Haupt-CLI-Funktion."""
    parser = argparse.ArgumentParser(
        description="LLKJJ ML Service CLI - Test-Harness für eigenständiges ML Package",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  %(prog)s process document.pdf              # Verarbeite PDF-Dokument
  %(prog)s process --simple document.pdf    # Einfache Verarbeitung
  %(prog)s process --json document.pdf      # JSON-Ausgabe
  %(prog)s validate-config                  # Prüfe Konfiguration
  %(prog)s test-capabilities                # Teste Funktionen
  %(prog)s benchmark document.pdf           # Performance-Test

Environment-Variablen:
  GEMINI_API_KEY         # Gemini API-Schlüssel (aus root .env)
  ML_GEMINI_MODEL        # Gemini-Modell (default: gemini-2.5-flash)
  ML_ENABLE_CACHING      # Cache aktivieren (true/false)
  ML_CACHE_DIR           # Cache-Verzeichnis
  ML_OUTPUT_DIR          # Output-Verzeichnis
  ML_VECTOR_DB_PATH      # Vector-DB-Pfad
        """,
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Detaillierte Ausgabe"
    )
    parser.add_argument(
        "--no-banner", action="store_true", help="Banner nicht anzeigen"
    )

    subparsers = parser.add_subparsers(dest="command", help="Verfügbare Kommandos")

    # Process command
    process_parser = subparsers.add_parser("process", help="PDF-Dokument verarbeiten")
    process_parser.add_argument("pdf_file", help="Pfad zur PDF-Datei")
    process_parser.add_argument(
        "--simple", action="store_true", help="Einfache Verarbeitung verwenden"
    )
    process_parser.add_argument(
        "--json", action="store_true", help="JSON-Ausgabe für maschinelle Verarbeitung"
    )

    # Validate config command
    subparsers.add_parser("validate-config", help="Konfiguration validieren")

    # Test capabilities command
    subparsers.add_parser("test-capabilities", help="Capabilities testen")

    # Benchmark command
    benchmark_parser = subparsers.add_parser("benchmark", help="Performance-Benchmark")
    benchmark_parser.add_argument("pdf_file", help="Pfad zur PDF-Datei für Benchmark")
    benchmark_parser.add_argument(
        "--runs", type=int, default=3, help="Anzahl Benchmark-Läufe (default: 3)"
    )

    args = parser.parse_args()

    # Banner anzeigen
    if not args.no_banner:
        print_banner()

    # Logging konfigurieren
    setup_logging(args.verbose)

    # Kommando ausführen
    if args.command == "process":
        return cmd_process(args)
    elif args.command == "validate-config":
        return cmd_validate_config(args)
    elif args.command == "test-capabilities":
        return cmd_test_capabilities(args)
    elif args.command == "benchmark":
        return cmd_benchmark(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
