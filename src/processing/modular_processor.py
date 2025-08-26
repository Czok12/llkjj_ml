#!/usr/bin/env python3
"""
LLKJJ ML Pipeline - Modular Processing Service
==============================================

Provides 4 separate processing workflows as requested:
1. PDF → Docling → TXT only
2. PDF → Gemini directly (bypass Docling)
3. Docling TXT → Gemini processing
4. Complete pipeline: PDF → Docling → Gemini → Output

This modular approach allows flexible testing and debugging of individual components.

Author: LLKJJ ML Pipeline
Version: 1.0.0 (Modular Architecture)
"""

import json
import logging
from pathlib import Path
from typing import Any

from docling.document_converter import DocumentConverter

from src.extraction.extractor import DataExtractor
from src.processing.classifier import DataClassifier
from src.settings_bridge import Config
from src.skr03_manager import lade_skr03_manager

logger = logging.getLogger(__name__)


class ModularProcessor:
    """
    Modular PDF processing service with 4 separate workflows.

    Allows testing and debugging individual components of the pipeline:
    - Docling extraction only
    - Gemini enhancement only
    - Text processing only
    - Complete pipeline
    """

    def __init__(self, config: Config | None = None) -> None:
        """Initialize modular processor with config"""
        self.config = config or Config()
        self.document_converter = DocumentConverter()

        # Initialize SKR03 Manager for classification
        self.skr03_manager = lade_skr03_manager()

        # Initialize components as needed
        self._gemini_model: Any | None = None
        self._extractor: DataExtractor | None = None
        self._classifier: DataClassifier | None = None

        logger.info(
            "ModularProcessor initialized with SKR03Manager and 4 separate workflows"
        )

    @property
    def gemini_model(self) -> Any | None:
        """Lazy load Gemini model when needed"""
        if self._gemini_model is None:
            try:
                from google import genai

                if self.config.google_api_key:
                    client = genai.Client(api_key=self.config.google_api_key)
                    self._gemini_model = client
                    logger.info("✅ Gemini client loaded: %s", self.config.gemini_model)
                else:
                    logger.warning("⚠️ No Google API key - Gemini disabled")
            except ImportError:
                logger.warning("⚠️ Google GenAI not installed - Gemini disabled")
            except Exception as e:
                logger.warning("⚠️ Failed to load Gemini: %s", e)
        return self._gemini_model

    @property
    def extractor(self) -> DataExtractor:
        """Lazy load extractor when needed"""
        if self._extractor is None:
            self._extractor = DataExtractor(self.config)
        return self._extractor

    @property
    def classifier(self) -> DataClassifier:
        """Lazy load classifier when needed"""
        if self._classifier is None:
            self._classifier = DataClassifier(skr03_manager=self.skr03_manager)
        return self._classifier

    def workflow_1_pdf_to_docling_txt(
        self, pdf_path: Path, output_path: Path | None = None
    ) -> dict[str, Any]:
        """
        Workflow 1: PDF → Docling → TXT only

        Pure Docling extraction without any AI enhancement.
        Saves raw text output to file.

        Args:
            pdf_path: Path to PDF file
            output_path: Optional output path for TXT file

        Returns:
            Dictionary with extraction results and file paths
        """
        logger.info("🔄 Workflow 1: PDF → Docling → TXT (reine Extraktion)")

        try:
            # Extract with Docling only
            conv_result = self.document_converter.convert(str(pdf_path))

            # Extract raw text
            raw_text = conv_result.document.export_to_markdown()

            # Extract tables
            tables = []
            for table in conv_result.document.tables:
                table_data = self._parse_table_basic(table)
                if table_data:
                    tables.append(table_data)

            # Prepare output path
            if output_path is None:
                output_path = Path("data/output") / f"{pdf_path.stem}_docling.txt"

            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save raw text to file
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(raw_text)

            result = {
                "workflow": "pdf_to_docling_txt",
                "pdf_file": str(pdf_path),
                "output_txt": str(output_path),
                "raw_text": raw_text,
                "tables": tables,
                "page_count": len(conv_result.document.pages),
                "text_length": len(raw_text),
                "table_count": len(tables),
            }

            logger.info(
                "✅ Workflow 1 abgeschlossen: %s → %s (%d Zeichen, %d Tabellen)",
                pdf_path.name,
                output_path.name,
                len(raw_text),
                len(tables),
            )

            return result

        except Exception as e:
            logger.error("❌ Workflow 1 fehlgeschlagen: %s", e)
            raise

    def workflow_2_pdf_to_gemini_direct(
        self, pdf_path: Path, output_path: Path | None = None
    ) -> dict[str, Any]:
        """
        Workflow 2: PDF → Gemini directly (bypass Docling)

        Direct Gemini processing of PDF without Docling preprocessing.
        Uses basic PDF text extraction then enhances with Gemini.

        Args:
            pdf_path: Path to PDF file
            output_path: Optional output path for JSON result

        Returns:
            Dictionary with Gemini-enhanced results
        """
        logger.info("🔄 Workflow 2: PDF → Gemini direkt (bypass Docling)")

        if self.gemini_model is None:
            raise ValueError(
                "Gemini model not available - check API key and installation"
            )

        try:
            # Basic PDF text extraction using PyMuPDF (faster and more reliable)
            import importlib
            from typing import Any

            # Import dynamically so mypy won't attempt to analyze the external
            # PyMuPDF package (which has no stubs/py.typed marker here).
            fitz: Any = importlib.import_module("fitz")

            text_content = ""
            with fitz.open(pdf_path) as pdf_doc:
                for page_num in range(fitz.open(pdf_path).page_count):
                    page = pdf_doc[page_num]
                    text_content += page.get_text("text") + "\n"

            if not text_content.strip():
                raise ValueError("No text extracted from PDF")

            # Enhance with Gemini directly
            enhanced_data = self._enhance_with_gemini_direct(text_content)

            # Prepare output path
            if output_path is None:
                output_path = (
                    Path("data/output") / f"{pdf_path.stem}_gemini_direct.json"
                )

            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save enhanced data
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(enhanced_data, f, ensure_ascii=False, indent=2)

            result = {
                "workflow": "pdf_to_gemini_direct",
                "pdf_file": str(pdf_path),
                "output_json": str(output_path),
                "raw_text_length": len(text_content),
                "enhanced_data": enhanced_data,
                "extraction_method": "PyPDF2 + Gemini",
            }

            logger.info(
                "✅ Workflow 2 abgeschlossen: %s → %s (Gemini direkt)",
                pdf_path.name,
                output_path.name,
            )

            return result

        except Exception as e:
            logger.error("❌ Workflow 2 fehlgeschlagen: %s", e)
            raise

    def workflow_3_docling_txt_to_gemini(
        self, txt_path: Path, output_path: Path | None = None
    ) -> dict[str, Any]:
        """
        Workflow 3: Docling TXT → Gemini processing

        Takes pre-extracted Docling text and enhances it with Gemini.
        Useful for testing Gemini enhancement on known good text.

        Args:
            txt_path: Path to Docling-extracted TXT file
            output_path: Optional output path for enhanced JSON result

        Returns:
            Dictionary with Gemini-enhanced results
        """
        logger.info("🔄 Workflow 3: Docling TXT → Gemini (Text-Enhancement)")

        if self.gemini_model is None:
            raise ValueError(
                "Gemini model not available - check API key and installation"
            )

        try:
            # Load Docling text
            with open(txt_path, encoding="utf-8") as f:
                docling_text = f.read()

            if not docling_text.strip():
                raise ValueError("Empty or invalid text file")

            # Enhance with Gemini
            enhanced_data = self._enhance_with_gemini_direct(docling_text)

            # Prepare output path
            if output_path is None:
                output_path = (
                    Path("data/output") / f"{txt_path.stem}_gemini_enhanced.json"
                )

            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save enhanced data
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(enhanced_data, f, ensure_ascii=False, indent=2)

            result = {
                "workflow": "docling_txt_to_gemini",
                "txt_file": str(txt_path),
                "output_json": str(output_path),
                "text_length": len(docling_text),
                "enhanced_data": enhanced_data,
                "processing_method": "Docling TXT + Gemini",
            }

            logger.info(
                "✅ Workflow 3 abgeschlossen: %s → %s (Text enhanced)",
                txt_path.name,
                output_path.name,
            )

            return result

        except Exception as e:
            logger.error("❌ Workflow 3 fehlgeschlagen: %s", e)
            raise

    def workflow_4_complete_pipeline(
        self, pdf_path: Path, output_path: Path | None = None
    ) -> dict[str, Any]:
        """
        Workflow 4: Complete pipeline PDF → Docling → Gemini → Output

        Full processing pipeline with all components:
        - Docling extraction
        - Gemini enhancement
        - SKR03 classification
        - Complete structured output

        Args:
            pdf_path: Path to PDF file
            output_path: Optional output path for complete JSON result

        Returns:
            Dictionary with complete processing results
        """
        logger.info(
            "🔄 Workflow 4: Complete Pipeline (PDF → Docling → Gemini → Output)"
        )

        try:
            # Phase 1: Docling extraction
            extraction_result = self.extractor.process_pdf(pdf_path)

            # Phase 2: SKR03 classification
            line_items = extraction_result.get("line_items", [])

            if line_items:
                # Pass both line_items and structured_data to classifier
                structured_data = extraction_result.get("structured_data", {})
                skr03_results = self.classifier.process_classifications(
                    line_items, structured_data
                )
            else:
                skr03_results = []

            # Prepare output path
            if output_path is None:
                output_path = (
                    Path("data/output") / f"{pdf_path.stem}_complete_pipeline.json"
                )

            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Combine all results
            complete_result = {
                "workflow": "complete_pipeline",
                "pdf_file": str(pdf_path),
                "output_json": str(output_path),
                "extraction_result": extraction_result,
                "skr03_classifications": skr03_results,
                "summary": {
                    "line_items_found": len(line_items),
                    "skr03_classifications": len(skr03_results),
                    "has_gemini_enhancement": "gemini"
                    in extraction_result.get("metadata", {}),
                    "text_length": len(extraction_result.get("raw_text", "")),
                    "table_count": len(extraction_result.get("tables", [])),
                },
            }

            # Save complete result
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(complete_result, f, ensure_ascii=False, indent=2)

            logger.info(
                "✅ Workflow 4 abgeschlossen: %s → %s (%d Positionen, %d SKR03)",
                pdf_path.name,
                output_path.name,
                len(line_items),
                len(skr03_results),
            )

            return complete_result

        except Exception as e:
            logger.error("❌ Workflow 4 fehlgeschlagen: %s", e)
            raise

    def _parse_table_basic(self, table: Any) -> dict[str, Any] | None:
        """Basic table parsing for Workflow 1"""
        try:
            rows = []
            if hasattr(table, "data") and table.data:
                for row in table.data:
                    if hasattr(row, "cells"):
                        row_data = [
                            cell.text.strip() if hasattr(cell, "text") else str(cell)
                            for cell in row.cells
                        ]
                        if any(row_data):
                            rows.append(row_data)

            if not rows:
                return None

            return {
                "headers": rows[0] if rows else [],
                "rows": rows[1:] if len(rows) > 1 else [],
                "row_count": len(rows) - 1 if len(rows) > 1 else 0,
            }

        except Exception:
            return None

    def _enhance_with_gemini_direct(self, text_content: str) -> dict[str, Any]:
        """Direct Gemini enhancement of text content"""
        try:
            # Load prompt template
            prompt_file = (
                Path(__file__).parent.parent / "config" / "gemini_extraction_prompt.txt"
            )

            try:
                with open(prompt_file, encoding="utf-8") as f:
                    prompt_template = f.read()
                prompt = prompt_template.format(rechnungstext=text_content[:3000])
            except FileNotFoundError:
                # Fallback prompt
                prompt = f"""
Du bist ein deutschsprachiger Experte für Elektrotechnik-Rechnungen und SKR03-Buchhaltung.

AUFGABE: Analysiere diese deutsche Elektrotechnik-Rechnung und extrahiere strukturierte Daten.

RECHNUNGSTEXT:
{text_content[:3000]}

EXTRAHIERE FOLGENDE DATEN (als valides JSON):
{{
  "rechnung": {{
    "nummer": "string",
    "datum": "YYYY-MM-DD",
    "betrag_netto": "number",
    "betrag_brutto": "number"
  }},
  "lieferant": {{
    "name": "string",
    "adresse": "string"
  }},
  "positionen": [
    {{
      "beschreibung": "deutsche Artikelbeschreibung",
      "menge": "number",
      "einzelpreis": "number",
      "gesamtpreis": "number"
    }}
  ]
}}

WICHTIG: Nur valides JSON zurückgeben.
"""

            # Call Gemini
            if self.gemini_model is None:
                raise ValueError("Gemini model is not available")

            response = self.gemini_model.models.generate_content(
                model=self.config.gemini_model,
                contents=prompt,
            )
            response_text = (
                response.text if hasattr(response, "text") else str(response)
            )

            # Parse JSON response
            try:
                # Clean response text (remove markdown if present)
                clean_text = response_text.strip()
                if clean_text.startswith("```json"):
                    clean_text = clean_text[7:]  # Remove ```json
                if clean_text.endswith("```"):
                    clean_text = clean_text[:-3]  # Remove ```
                clean_text = clean_text.strip()

                enhanced_data: dict[str, Any] = json.loads(clean_text)
                enhanced_data["enhancement_method"] = "gemini_direct"
                enhanced_data["model_used"] = self.config.gemini_model
                return enhanced_data
            except json.JSONDecodeError as jde:
                logger.warning(
                    "⚠️ Gemini response is not valid JSON: %s", str(jde)[:100]
                )
                return {
                    "enhancement_method": "gemini_direct_raw",
                    "model_used": self.config.gemini_model,
                    "raw_response": response_text[:1000],  # Limit size
                    "error": f"Invalid JSON response: {str(jde)[:100]}",
                }

        except Exception as e:
            logger.error("❌ Gemini enhancement failed: %s", e)
            return {
                "enhancement_method": "gemini_direct_failed",
                "error": str(e),
                "fallback_data": {"rechnung": {}, "lieferant": {}, "positionen": []},
            }
