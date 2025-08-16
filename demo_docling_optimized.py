#!/usr/bin/env python3
"""
Demo-Skript für den optimierten Docling-Extractor
Zeigt die verschiedenen Konfigurationen und deren Performance für deutsche Rechnungen
"""

import sys
from pathlib import Path

# Prüfe ob die notwendigen Module verfügbar sind
try:
    # from docling_optimized_extractor import OptimizedDoclingExtractor
    # FIX: Module not found. If needed, create src/docling_optimized_extractor.py or update import.
    print(
        "❌ Docling-Optimized-Extractor nicht gefunden. Bitte Modul erstellen oder Import korrigieren."
    )
except ImportError as e:
    print(f"❌ Import-Fehler: {e}")
    print("Stelle sicher, dass Docling installiert ist: pip install docling")
    sys.exit(1)


def demo_single_file() -> None:
    """Demo: Einzelne PDF-Datei mit verschiedenen Konfigurationen testen"""

    # Verfügbare Test-PDFs suchen
    test_dirs = ["test_pdfs", "data/input/pdfs", "."]
    test_pdf = None

    for test_dir in test_dirs:
        test_path = Path(test_dir)
        if test_path.exists():
            pdf_files = list(test_path.glob("*.pdf"))
            if pdf_files:
                test_pdf = pdf_files[0]
                break

    if not test_pdf:
        print("❌ Keine Test-PDF gefunden!")
        print("Lege eine PDF-Datei in einen der folgenden Ordner:")
        for test_dir in test_dirs:
            print(f"   - {test_dir}/")
        return

    print(f"🔍 Demo mit Datei: {test_pdf.name}")
    print("=" * 60)

    # extractor = OptimizedDoclingExtractor("demo_docling_output")

    # Teste verschiedene Konfigurationen
    configs = ["easyocr_optimized", "tesseract_auto", "tesseract_de"]

    for config in configs:
        print(f"\n📋 Teste Konfiguration: {config}")
        print("-" * 40)
        print(
            "❌ Docling-Optimized-Extractor nicht verfügbar. Demo kann nicht ausgeführt werden."
        )


def demo_batch_processing() -> None:
    """Demo: Batch-Verarbeitung von mehreren PDF-Dateien"""

    # Test-Ordner suchen
    test_dirs = ["test_pdfs", "data/input/pdfs"]
    test_dir = None

    for dir_path in test_dirs:
        path = Path(dir_path)
        if path.exists() and list(path.glob("*.pdf")):
            test_dir = path
            break

    if not test_dir:
        print("❌ Kein Ordner mit PDF-Dateien gefunden!")
        print("Erstelle einen 'test_pdfs' Ordner mit PDF-Dateien für die Demo")
        return

    print(f"📁 Demo Batch-Verarbeitung: {test_dir}")
    print("=" * 60)

    # extractor = OptimizedDoclingExtractor("demo_docling_batch")

    # Batch-Verarbeitung mit optimierter EasyOCR Konfiguration
    print("🚀 Starte Batch-Verarbeitung mit EasyOCR Optimierung...")

    print(
        "❌ Docling-Optimized-Extractor nicht verfügbar. Batch-Demo kann nicht ausgeführt werden."
    )


def demo_configuration_comparison() -> None:
    """Demo: Vergleich verschiedener Docling-Konfigurationen"""

    # Test-Ordner suchen
    test_dirs = ["test_pdfs", "data/input/pdfs"]
    test_dir = None

    for dir_path in test_dirs:
        path = Path(dir_path)
        if path.exists() and list(path.glob("*.pdf")):
            test_dir = path
            break

    if not test_dir:
        print("❌ Kein Ordner mit PDF-Dateien gefunden für Konfigurationsvergleich!")
        return

    print("🔄 Demo Konfigurationsvergleich")
    print("=" * 60)

    # extractor = OptimizedDoclingExtractor("demo_docling_comparison")

    print(
        "❌ Docling-Optimized-Extractor nicht verfügbar. Konfigurationsvergleich kann nicht ausgeführt werden."
    )


def show_docling_best_practices() -> None:
    """Zeige Docling Best Practices für deutsche Rechnungen"""

    print("📚 Docling Best Practices für deutsche Rechnungen")
    print("=" * 60)

    best_practices = [
        "🔧 OCR-Engine: EasyOCR mit 'de' für deutsche Texte",
        "📋 TableFormer: ACCURATE Modus für präzise Tabellenstruktur",
        "🔗 Cell Matching: Aktiviert für bessere Zellenzuordnung",
        "⚡ Accelerator: AUTO-Modus mit 8 Threads für Performance",
        "🎯 Confidence: 0.5 Schwellenwert für OCR-Erkennung",
        "📄 Force OCR: Nur bei Bedarf, Standard ist False",
        "🔍 Bitmap Threshold: 0.05 für optimale Bilderkennung",
        "🏃 GPU: Aktiviert falls verfügbar für EasyOCR",
    ]

    for practice in best_practices:
        print(f"   {practice}")

    print("\n📝 Verfügbare Konfigurationen:")
    print("   • easyocr_optimized: Deutsche OCR-Optimierung (Empfohlen)")
    print("   • tesseract_auto: Automatische Spracherkennung")
    print("   • tesseract_de: Explizit deutsche Tesseract-Konfiguration")

    print("\n🚀 Verwendung:")
    print(
        "   python docling_optimized_extractor.py /pfad/zu/pdfs --config easyocr_optimized"
    )
    print("   python docling_optimized_extractor.py /pfad/zu/pdfs --compare")


def main() -> None:
    """Hauptfunktion für das Demo"""

    print("🎯 Docling Optimized Extractor - Demo")
    print("Optimierte Konfigurationen für deutsche Rechnungsverarbeitung")
    print("=" * 70)

    # Best Practices anzeigen
    show_docling_best_practices()

    print("\n" + "=" * 70)

    # Demo-Menü
    def run_all_demos() -> None:
        """Führt alle Demos nacheinander aus"""
        demo_single_file()
        demo_batch_processing()
        demo_configuration_comparison()

    demos = [
        ("1", "Einzeldatei-Test", demo_single_file),
        ("2", "Batch-Verarbeitung", demo_batch_processing),
        ("3", "Konfigurationsvergleich", demo_configuration_comparison),
        ("4", "Alle Demos", run_all_demos),
    ]

    print("\n🎮 Verfügbare Demos:")
    for key, name, _ in demos:
        print(f"   {key}. {name}")

    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        choice = input("\nWähle Demo (1-4, Enter für alle): ").strip() or "4"

    # Demo ausführen
    for key, name, demo_func in demos:
        if choice == key:
            print(f"\n🚀 Starte: {name}")
            print("=" * 60)
            demo_func()
            break
    else:
        print(f"❌ Ungültige Auswahl: {choice}")
        print("Verfügbare Optionen: 1, 2, 3, 4")


if __name__ == "__main__":
    main()
