#!/usr/bin/env python3
"""
Optimierter Docling PDF-Extraktor für deutsche Rechnungen (Version 2.0)
=========================================================================

Dieses Skript implementiert die neuesten Docling Best Practices für die
Verarbeitung deutscher Elektro-Großhandel Rechnungen (speziell Sonepar Format).

Features:
- TableFormer mit 95.4% Genauigkeit für komplexe Tabellen
- Optimierte deutsche OCR mit EasyOCR
- OTSL (Output Table Structure Language) für 2x schnellere Tabellenverarbeitung
- Automatische Hardware-Acceleration (CPU/MPS/CUDA)
- Strukturierte JSON- und Markdown-Ausgabe
- Batch-Verarbeitung aller PDFs in einem Ordner

Verwendung:
    poetry run python docling_best_practices.py [OPTIONEN]

Author: LLKJJ ML Pipeline
Version: 2.0.0 (Docling 2.44.0 optimiert)
"""

import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    EasyOcrOptions,
    PdfPipelineOptions,
    TableFormerMode,
    TesseractCliOcrOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption

# Logging Setup
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class ExtractionStats:
    """Statistiken für die PDF-Extraktion"""

    file_name: str
    processing_time: float
    total_pages: int
    tables_found: int
    text_blocks_found: int
    ocr_used: bool
    table_former_mode: str
    success: bool
    error_message: str | None = None


@dataclass
class DocumentAnalysis:
    """Strukturierte Analyse eines Dokuments"""

    file_info: dict[str, Any]
    extraction_stats: ExtractionStats
    content: dict[str, Any]
    tables: list[dict[str, Any]]
    metadata: dict[str, Any]


class DoclingBestPracticesExtractor:
    """
    Optimierter Docling-Extraktor für deutsche Rechnungen.

    Implementiert die neuesten Best Practices:
    - TableFormer ACCURATE Modus für 95.4% Tabellengenauigkeit
    - Deutsche OCR-Optimierung mit EasyOCR
    - Automatische Hardware-Acceleration
    - OTSL für 2x schnellere Tabellenverarbeitung
    """

    def __init__(
        self,
        input_folder: str = "data/input/pdfs",
        output_folder: str = "data/output",
        use_tesseract: bool = False,
        force_full_page_ocr: bool = False,
        images_scale: float = 2.0,
        enable_debug: bool = False,
    ):
        """
        Initialisiert den optimierten Docling-Extraktor.

        Args:
            input_folder: Ordner mit PDF-Dateien
            output_folder: Ausgabeordner für Ergebnisse
            use_tesseract: Verwende Tesseract statt EasyOCR
            force_full_page_ocr: Erzwinge vollständige Seitenerkennnng
            images_scale: Bildskala für bessere OCR (Standard: 2.0)
            enable_debug: Debug-Modus aktivieren
        """
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.use_tesseract = use_tesseract
        self.force_full_page_ocr = force_full_page_ocr
        self.images_scale = images_scale
        self.enable_debug = enable_debug

        # Erstelle Output-Ordner
        self.output_folder.mkdir(parents=True, exist_ok=True)

        # Konfiguriere Docling Pipeline
        self.converter = self._setup_docling_pipeline()

        logger.info("Docling Best Practices Extractor initialisiert")
        logger.info(f"Input: {self.input_folder}")
        logger.info(f"Output: {self.output_folder}")
        logger.info(f"OCR Engine: {'Tesseract' if use_tesseract else 'EasyOCR'}")
        logger.info("TableFormer: ACCURATE Modus aktiviert")

    def _setup_docling_pipeline(self) -> DocumentConverter:
        """
        Konfiguriert die optimierte Docling-Pipeline basierend auf Best Practices.

        Returns:
            Konfigurierter DocumentConverter
        """
        # Pipeline-Optionen für deutsche Rechnungen optimiert
        pipeline_options = PdfPipelineOptions()

        # === OCR-Konfiguration für deutsche Texte ===
        pipeline_options.do_ocr = True

        if self.use_tesseract:
            # Tesseract für deutsche Texte
            pipeline_options.ocr_options = TesseractCliOcrOptions(
                lang=["deu"],  # Deutsche Sprache
                force_full_page_ocr=self.force_full_page_ocr,
            )
            logger.info("OCR: Tesseract CLI mit deutscher Sprache")
        else:
            # EasyOCR für deutsche Texte (Standard)
            pipeline_options.ocr_options = EasyOcrOptions(
                lang=["de"],  # Deutsche Sprache
                force_full_page_ocr=self.force_full_page_ocr,
                confidence_threshold=0.5,
                use_gpu=None,  # Automatische GPU-Erkennung
            )
            logger.info("OCR: EasyOCR mit deutscher Sprache")

        # === TableFormer-Konfiguration (95.4% Genauigkeit) ===
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
        pipeline_options.table_structure_options.do_cell_matching = True
        logger.info("TableFormer: ACCURATE Modus für optimale Tabellengenauigkeit")

        # === Bildverarbeitung ===
        pipeline_options.generate_page_images = True
        pipeline_options.generate_picture_images = True
        pipeline_options.images_scale = self.images_scale
        logger.info(f"Bildverarbeitung: Skala {self.images_scale} für bessere OCR")

        # === Hardware-Acceleration ===
        pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=4,
            device=AcceleratorDevice.AUTO,  # Automatische Hardware-Erkennung
        )
        logger.info("Hardware-Acceleration: AUTO (CPU/MPS/CUDA)")

        # === Debug-Optionen ===
        if self.enable_debug:
            pipeline_options.generate_parsed_pages = True
            logger.info("Debug-Modus: Aktiviert")

        # Erstelle DocumentConverter
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

        return converter

    def extract_document_content(self, pdf_path: Path) -> DocumentAnalysis:
        """
        Extrahiert Inhalte aus einem PDF-Dokument mit optimierter Pipeline.

        Args:
            pdf_path: Pfad zur PDF-Datei

        Returns:
            DocumentAnalysis mit strukturierten Ergebnissen
        """
        start_time = time.time()

        try:
            logger.info(f"Verarbeite: {pdf_path.name}")

            # Dokument konvertieren
            conv_result = self.converter.convert(pdf_path)
            doc = conv_result.document

            processing_time = time.time() - start_time

            # === Basis-Statistiken ===
            stats = ExtractionStats(
                file_name=pdf_path.name,
                processing_time=processing_time,
                total_pages=len(doc.pages),
                tables_found=len(doc.tables),
                text_blocks_found=len(
                    [item for item in doc.iterate_items() if hasattr(item[0], "text")]
                ),
                ocr_used=True,
                table_former_mode="ACCURATE",
                success=True,
            )

            # === Inhalt extrahieren ===
            content: dict[str, object] = {
                "full_text": doc.export_to_markdown(),
                "pages": [],
            }

            # Seiten-Details
            for page_no, page in doc.pages.items():
                page_content: dict[str, object] = {
                    "page_number": page_no,
                    "size": {"width": page.size.width, "height": page.size.height},
                    "elements": [],
                }

                # Text-Elemente extrahieren
                for element, _level in doc.iterate_items():
                    if hasattr(element, "page_no") and element.page_no == page_no:  # type: ignore
                        if hasattr(element, "text"):
                            elements = page_content["elements"]
                            if isinstance(elements, list):
                                elements.append(
                                    {
                                        "type": "text",
                                        "content": element.text,  # type: ignore
                                        "label": (
                                            element.label.value  # type: ignore
                                            if hasattr(element, "label") and element.label  # type: ignore
                                            else "text"
                                        ),
                                    }
                                )

                pages = content["pages"]
                if isinstance(pages, list):
                    pages.append(page_content)

            # === Tabellen extrahieren ===
            tables = []
            for i, table in enumerate(doc.tables):
                try:
                    # Tabelle als DataFrame extrahieren
                    df = table.export_to_dataframe()

                    table_data = {
                        "table_id": i + 1,
                        "rows": len(df),
                        "columns": len(df.columns),
                        "data": df.to_dict("records"),
                        "html": table.export_to_html(doc),
                        "markdown": df.to_markdown(index=False),
                    }
                    tables.append(table_data)

                except Exception as e:
                    logger.warning(f"Fehler bei Tabelle {i+1}: {e}")
                    tables.append(
                        {"table_id": i + 1, "error": str(e), "raw_data": str(table)}
                    )

            # === Metadata ===
            metadata = {
                "extraction_timestamp": datetime.now().isoformat(),
                "docling_version": "2.44.0",
                "table_former_mode": "ACCURATE",
                "ocr_engine": "Tesseract" if self.use_tesseract else "EasyOCR",
                "language": "de",
                "images_scale": self.images_scale,
            }

            return DocumentAnalysis(
                file_info={
                    "filename": pdf_path.name,
                    "path": str(pdf_path),
                    "size_bytes": pdf_path.stat().st_size,
                },
                extraction_stats=stats,
                content=content,
                tables=tables,
                metadata=metadata,
            )

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Fehler bei {pdf_path.name}: {e}")

            # Fehler-Statistiken
            stats = ExtractionStats(
                file_name=pdf_path.name,
                processing_time=processing_time,
                total_pages=0,
                tables_found=0,
                text_blocks_found=0,
                ocr_used=True,
                table_former_mode="ACCURATE",
                success=False,
                error_message=str(e),
            )

            return DocumentAnalysis(
                file_info={
                    "filename": pdf_path.name,
                    "path": str(pdf_path),
                    "size_bytes": pdf_path.stat().st_size if pdf_path.exists() else 0,
                },
                extraction_stats=stats,
                content={"error": str(e)},
                tables=[],
                metadata={"error": str(e)},
            )

    def save_results(self, analysis: DocumentAnalysis) -> None:
        """
        Speichert die Extraktionsergebnisse in verschiedenen Formaten.

        Args:
            analysis: DocumentAnalysis-Objekt
        """
        base_name = Path(analysis.file_info["filename"]).stem

        # === JSON-Ausgabe (strukturiert) ===
        json_path = self.output_folder / f"{base_name}_docling_best_practices.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(asdict(analysis), f, ensure_ascii=False, indent=2)

        # === Markdown-Ausgabe (menschenlesbar) ===
        md_path = self.output_folder / f"{base_name}_docling_best_practices.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(
                f"# Docling Best Practices Extraction: {analysis.file_info['filename']}\n\n"
            )
            f.write("## Statistiken\n")
            f.write(
                f"- **Verarbeitungszeit:** {analysis.extraction_stats.processing_time:.2f}s\n"
            )
            f.write(f"- **Seiten:** {analysis.extraction_stats.total_pages}\n")
            f.write(f"- **Tabellen:** {analysis.extraction_stats.tables_found}\n")
            f.write(
                f"- **Textblöcke:** {analysis.extraction_stats.text_blocks_found}\n"
            )
            f.write(
                f"- **TableFormer:** {analysis.extraction_stats.table_former_mode}\n"
            )
            f.write(
                f"- **OCR:** {'✅' if analysis.extraction_stats.ocr_used else '❌'}\n\n"
            )

            if analysis.extraction_stats.success:
                f.write("## Volltext\n\n")
                f.write(analysis.content.get("full_text", "Kein Text extrahiert"))
                f.write("\n\n")

                if analysis.tables:
                    f.write("## Tabellen\n\n")
                    for table in analysis.tables:
                        if "markdown" in table:
                            f.write(f"### Tabelle {table['table_id']}\n\n")
                            f.write(table["markdown"])
                            f.write("\n\n")
            else:
                f.write(f"## ❌ Fehler\n\n{analysis.extraction_stats.error_message}\n")

        logger.info(f"Ergebnisse gespeichert: {base_name}")

    def process_all_pdfs(self) -> list[DocumentAnalysis]:
        """
        Verarbeitet alle PDF-Dateien im Input-Ordner.

        Returns:
            Liste aller DocumentAnalysis-Objekte
        """
        pdf_files = list(self.input_folder.glob("*.pdf"))

        if not pdf_files:
            logger.warning(f"Keine PDF-Dateien in {self.input_folder} gefunden")
            return []

        logger.info(f"Gefundene PDF-Dateien: {len(pdf_files)}")

        results = []
        total_start_time = time.time()

        for pdf_file in pdf_files:
            analysis = self.extract_document_content(pdf_file)
            self.save_results(analysis)
            results.append(analysis)

        total_time = time.time() - total_start_time

        # === Zusammenfassung ===
        successful = [r for r in results if r.extraction_stats.success]
        failed = [r for r in results if not r.extraction_stats.success]

        logger.info("=" * 60)
        logger.info("DOCLING BEST PRACTICES EXTRACTION - ZUSAMMENFASSUNG")
        logger.info("=" * 60)
        logger.info(f"Gesamt verarbeitet: {len(results)}")
        logger.info(f"Erfolgreich: {len(successful)}")
        logger.info(f"Fehlgeschlagen: {len(failed)}")
        logger.info(f"Gesamtzeit: {total_time:.2f}s")
        logger.info(f"Durchschnitt: {total_time/len(results):.2f}s pro PDF")

        if successful:
            avg_tables = sum(r.extraction_stats.tables_found for r in successful) / len(
                successful
            )
            avg_text_blocks = sum(
                r.extraction_stats.text_blocks_found for r in successful
            ) / len(successful)
            logger.info(f"Durchschnitt Tabellen: {avg_tables:.1f}")
            logger.info(f"Durchschnitt Textblöcke: {avg_text_blocks:.1f}")

        # Speichere Zusammenfassung
        self._save_summary(results, total_time)

        return results

    def _save_summary(self, results: list[DocumentAnalysis], total_time: float) -> None:
        """Speichert eine Zusammenfassung aller Extraktionen."""
        summary = {
            "extraction_summary": {
                "timestamp": datetime.now().isoformat(),
                "total_files": len(results),
                "successful": len([r for r in results if r.extraction_stats.success]),
                "failed": len([r for r in results if not r.extraction_stats.success]),
                "total_processing_time": total_time,
                "average_time_per_file": total_time / len(results) if results else 0,
                "configuration": {
                    "docling_version": "2.44.0",
                    "table_former_mode": "ACCURATE",
                    "ocr_engine": "Tesseract" if self.use_tesseract else "EasyOCR",
                    "language": "de",
                    "images_scale": self.images_scale,
                },
            },
            "file_results": [
                {
                    "filename": r.file_info["filename"],
                    "success": r.extraction_stats.success,
                    "processing_time": r.extraction_stats.processing_time,
                    "tables_found": r.extraction_stats.tables_found,
                    "text_blocks": r.extraction_stats.text_blocks_found,
                    "error": r.extraction_stats.error_message,
                }
                for r in results
            ],
        }

        summary_path = self.output_folder / "docling_best_practices_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        logger.info(f"Zusammenfassung gespeichert: {summary_path}")


def main() -> None:
    """Hauptfunktion für Kommandozeilen-Ausführung."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Optimierter Docling PDF-Extraktor für deutsche Rechnungen",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  poetry run python docling_best_practices.py
  poetry run python docling_best_practices.py --input data/input/pdfs --output data/output
  poetry run python docling_best_practices.py --tesseract --force-ocr --debug

Best Practices:
  - TableFormer ACCURATE Modus für 95.4% Tabellengenauigkeit
  - EasyOCR mit deutscher Sprache (Standard)
  - Automatische Hardware-Acceleration (CPU/MPS/CUDA)
  - OTSL für 2x schnellere Tabellenverarbeitung
        """,
    )

    parser.add_argument(
        "--input",
        "-i",
        default="test_pdfs",
        help="Input-Ordner mit PDF-Dateien (Standard: test_pdfs)",
    )

    parser.add_argument(
        "--output",
        "-o",
        default="demo_output",
        help="Output-Ordner für Ergebnisse (Standard: demo_output)",
    )

    parser.add_argument(
        "--tesseract", action="store_true", help="Verwende Tesseract statt EasyOCR"
    )

    parser.add_argument(
        "--force-ocr", action="store_true", help="Erzwinge vollständige Seitenerkennung"
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=2.0,
        help="Bildskala für bessere OCR (Standard: 2.0)",
    )

    parser.add_argument("--debug", action="store_true", help="Debug-Modus aktivieren")

    args = parser.parse_args()

    # Initialisiere Extraktor
    extractor = DoclingBestPracticesExtractor(
        input_folder=args.input,
        output_folder=args.output,
        use_tesseract=args.tesseract,
        force_full_page_ocr=args.force_ocr,
        images_scale=args.scale,
        enable_debug=args.debug,
    )

    # Verarbeite alle PDFs
    results = extractor.process_all_pdfs()

    if results:
        logger.info("✅ Docling Best Practices Extraction abgeschlossen!")
    else:
        logger.error("❌ Keine PDFs verarbeitet!")


if __name__ == "__main__":
    main()
