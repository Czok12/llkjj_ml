#!/usr/bin/env python3
"""
RAG-System Population Script
===========================

Verarbeitet alle verfügbaren PDF-Dateien und befüllt das ChromaDB RAG-System
mit klassifizierten Artikeldaten für bessere zukünftige Klassifizierungen.

Autor: LLKJJ ML Pipeline Team
Version: 1.0.0
"""

import logging
import sys
import time
from pathlib import Path
from typing import Any

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.pipeline.processor import UnifiedProcessor
from src.settings_bridge import Config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/rag_population.log", encoding="utf-8"),
    ],
)

logger = logging.getLogger(__name__)


def populate_rag_system(
    pdf_directory: Path, config: Config | None = None
) -> dict[str, Any]:
    """
    Verarbeitet alle PDFs im angegebenen Verzeichnis und befüllt das RAG-System.

    Args:
        pdf_directory: Verzeichnis mit PDF-Dateien
        config: Optional Config object

    Returns:
        Dictionary mit Verarbeitungsstatistiken
    """
    if config is None:
        config = Config()

    processor = UnifiedProcessor(config)

    # Find all PDF files
    pdf_files = list(pdf_directory.glob("*.pdf"))

    if not pdf_files:
        logger.warning("Keine PDF-Dateien in %s gefunden", pdf_directory)
        return {"processed": 0, "successful": 0, "failed": 0, "errors": []}

    logger.info("🚀 Starte RAG-System Population mit %d PDF-Dateien", len(pdf_files))

    stats: dict[str, Any] = {
        "processed": 0,
        "successful": 0,
        "failed": 0,
        "errors": [],
        "total_items": 0,
        "total_processing_time": 0,
    }

    start_time = time.time()

    for pdf_file in pdf_files:
        try:
            logger.info("📄 Verarbeite: %s", pdf_file.name)

            # Process PDF
            result = processor.process_pdf(pdf_file)

            stats["processed"] += 1
            stats["total_processing_time"] += result.processing_time_ms

            # Check if processing was successful
            if result.skr03_classifications:
                stats["successful"] += 1
                stats["total_items"] += len(result.skr03_classifications)

                logger.info(
                    "✅ %s erfolgreich verarbeitet: %d Artikel, Konfidenz: %.1f%%",
                    pdf_file.name,
                    len(result.skr03_classifications),
                    result.confidence_score * 100,
                )
            else:
                stats["failed"] += 1
                stats["errors"].append(f"{pdf_file.name}: Keine Artikel klassifiziert")
                logger.warning("⚠️ %s: Keine Artikel klassifiziert", pdf_file.name)

        except Exception as e:
            stats["failed"] += 1
            error_msg = f"{pdf_file.name}: {str(e)}"
            stats["errors"].append(error_msg)
            logger.error("❌ Fehler bei %s: %s", pdf_file.name, str(e))
            continue

    # Cleanup processor
    processor.cleanup()

    total_time = time.time() - start_time

    # Print summary
    logger.info("=" * 60)
    logger.info("🎯 RAG-System Population abgeschlossen!")
    logger.info("📊 Statistiken:")
    logger.info("   • Verarbeitet: %d/%d PDFs", stats["successful"], stats["processed"])
    logger.info("   • Artikel gesamt: %d", stats["total_items"])
    logger.info("   • Verarbeitungszeit: %.1f Sekunden", total_time)
    logger.info(
        "   • Durchschnitt pro PDF: %.1f Sekunden",
        total_time / len(pdf_files) if pdf_files else 0,
    )

    if stats["failed"] > 0:
        logger.warning("⚠️ %d PDFs fehlgeschlagen:", stats["failed"])
        for error in stats["errors"]:
            logger.warning("   • %s", error)

    return stats


def main() -> int:
    """Main entry point"""
    # Setup paths
    project_root = Path(__file__).parent.parent
    pdf_directory = project_root / "test_pdfs"

    if not pdf_directory.exists():
        logger.error("PDF-Verzeichnis nicht gefunden: %s", pdf_directory)
        return 1

    # Ensure logs directory exists
    logs_dir = project_root / "logs"
    logs_dir.mkdir(exist_ok=True)

    try:
        # Run population
        stats = populate_rag_system(pdf_directory)

        # Check results
        if stats["successful"] == 0:
            logger.error("❌ Keine PDFs erfolgreich verarbeitet!")
            return 1
        elif stats["failed"] > 0:
            logger.warning(
                "⚠️ Einige PDFs fehlgeschlagen, aber %d erfolgreich", stats["successful"]
            )
            return 0
        else:
            logger.info("✅ Alle PDFs erfolgreich verarbeitet!")
            return 0

    except Exception as e:
        logger.error("❌ Kritischer Fehler: %s", str(e))
        return 1


if __name__ == "__main__":
    sys.exit(main())
