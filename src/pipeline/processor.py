#!/usr/bin/env python3
"""
LLKJJ ML Pipeline - Slim Orchestrator (KISS Architecture)
========================================================

Schlanker Orchestrator der die spezialisierten Module koordiniert:
- DataExtractor: PDF-Extraktion und Datengewinnung
- DataClassifier: SKR03-Klassifizierung und RAG-System
- QualityAssessor: Qualitätsbewertung und Konfidenz-Scores

Folgt dem Single Responsibility Principle für bessere Wartbarkeit.

Autor: LLKJJ ML Pipeline Team
Version: 2.1.0 (Post-Konsolidierung Refactoring)
"""

import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer

try:
    import google.generativeai as genai
except ImportError:
    genai = None  # type: ignore[assignment]

from src.config import Config
from src.extraction import DataExtractor
from src.processing import DataClassifier, QualityAssessor

logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """Unified result from PDF processing pipeline"""

    # Source information
    pdf_path: str
    processing_timestamp: str

    # Extraction results
    raw_text: str
    structured_data: dict[str, Any]

    # Classification results
    invoice_data: dict[str, Any]
    skr03_classifications: list[dict[str, Any]]

    # Performance metrics
    processing_time_ms: int
    ocr_time_ms: int
    classification_time_ms: int

    # Quality indicators
    confidence_score: float
    extraction_quality: str  # "high", "medium", "low"


class UnifiedProcessor:
    """
    Schlanker Orchestrator für die modulare PDF-zu-SKR03-Pipeline.

    Koordiniert die spezialisierten Module:
    - DataExtractor für PDF-Verarbeitung und Extraktion
    - DataClassifier für SKR03-Klassifizierung und RAG-System
    - QualityAssessor für Qualitätsbewertung

    Folgt KISS-Prinzipien durch klare Verantwortungstrennung.
    """

    # Type annotations for instance attributes
    config: Config
    client: Any  # chromadb.Client
    invoice_collection: Any | None
    embedding_model: SentenceTransformer | None
    extractor: Any  # DataExtractor
    classifier: Any  # DataClassifier
    quality_assessor: Any  # QualityAssessor

    def __init__(self, cfg: Config | None = None):
        """Initialize unified processor with modular components"""
        self.config = cfg or Config()
        self._setup_logging()
        self._setup_vector_db()

        # Initialize Gemini model if API key is available
        gemini_model = self._setup_gemini_model()

        # Initialize specialized modules
        self.extractor = DataExtractor(gemini_model=gemini_model)
        self.classifier = DataClassifier(vector_store=self.invoice_collection)
        self.quality_assessor = QualityAssessor()

    def _setup_logging(self) -> None:
        """Configure logging for the processor"""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    def _setup_vector_db(self) -> None:
        """Setup ChromaDB and embedding model for RAG system"""
        try:
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=str(self.config.vector_db_path)
            )

            # Get or create collection for invoices
            self.invoice_collection = self.client.get_or_create_collection(
                name="invoice_line_items",
                metadata={
                    "description": "German electrical contractor invoice items for SKR03 classification"
                },
            )

            # Initialize embedding model for semantic search
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

            logger.info("✅ ChromaDB und Embedding-Model initialisiert")

        except Exception as e:
            logger.error("❌ Fehler bei Vektordatenbank-Setup: %s", e)
            self.invoice_collection = None
            self.embedding_model = None

    def _setup_gemini_model(self) -> Any | None:
        """Setup Gemini AI model for extraction enhancement"""
        try:
            if not genai:
                logger.warning(
                    "⚠️ Google Generative AI nicht installiert - Gemini deaktiviert"
                )
                return None

            if not self.config.google_api_key:
                logger.warning("⚠️ Keine Google API Key gefunden - Gemini deaktiviert")
                return None

            # Configure and initialize Gemini
            # Note: Google AI library lacks proper type stubs
            genai.configure(api_key=self.config.google_api_key)  # type: ignore[attr-defined]
            model = genai.GenerativeModel(self.config.gemini_model)  # type: ignore[attr-defined]

            logger.info("✅ Gemini %s initialisiert", self.config.gemini_model)
            return model

        except Exception as e:
            logger.warning("⚠️ Gemini-Setup fehlgeschlagen: %s", e)
            return None

    def process_pdf(self, pdf_path: str | Path) -> ProcessingResult:
        """
        Hauptfunktion: Verarbeitet eine PDF-Datei durch die modulare Pipeline.

        Args:
            pdf_path: Pfad zur PDF-Datei

        Returns:
            ProcessingResult mit allen Extraktions- und Klassifizierungsergebnissen
        """
        start_time = time.time()
        pdf_path = Path(pdf_path)

        logger.info("🚀 Starte modulare PDF-Verarbeitung: %s", pdf_path.name)

        try:
            # Phase 1: Extraktion mit DataExtractor
            extraction_start = time.time()
            extraction_result = self.extractor.process_pdf(pdf_path)
            ocr_time_ms = int((time.time() - extraction_start) * 1000)

            logger.info("✅ Extraktion abgeschlossen (%d ms)", ocr_time_ms)

            # Phase 2: Klassifizierung mit DataClassifier
            classification_start = time.time()
            classifications = self.classifier.process_classifications(
                extraction_result["line_items"], extraction_result["structured_data"]
            )
            classification_time_ms = int((time.time() - classification_start) * 1000)

            logger.info(
                "✅ Klassifizierung abgeschlossen (%d ms)", classification_time_ms
            )

            # Phase 3: Qualitätsbewertung mit QualityAssessor
            confidence_score = self.quality_assessor.calculate_confidence(
                extraction_result["structured_data"], classifications
            )
            quality_assessment = self.quality_assessor.assess_quality(confidence_score)

            logger.info(
                "✅ Qualitätsbewertung: %s (%.1f%%)",
                quality_assessment,
                confidence_score * 100,
            )

            # Ergebnis zusammenstellen
            processing_time_ms = int((time.time() - start_time) * 1000)

            proc_result = ProcessingResult(
                pdf_path=str(pdf_path),
                processing_timestamp=datetime.now().isoformat(),
                raw_text=extraction_result["raw_text"],
                structured_data=extraction_result["structured_data"],
                invoice_data=extraction_result["invoice_data"],
                skr03_classifications=classifications,
                processing_time_ms=processing_time_ms,
                ocr_time_ms=ocr_time_ms,
                classification_time_ms=classification_time_ms,
                confidence_score=confidence_score,
                extraction_quality=quality_assessment,
            )

            # Phase 4: Vektorisierung für RAG-System (optional)
            if self.invoice_collection and self.embedding_model:
                self._vectorize_and_store(proc_result)

            logger.info(
                "🎉 PDF-Verarbeitung erfolgreich abgeschlossen: %d Positionen, %.1f%% Konfidenz",
                len(classifications),
                confidence_score * 100,
            )

            return proc_result

        except Exception as e:
            logger.error("❌ Fehler bei PDF-Verarbeitung: %s", e)
            raise

    def _vectorize_and_store(self, proc_result: ProcessingResult) -> None:
        """Vektorisiert Rechnungspositionen und speichert sie in ChromaDB."""
        if not self.invoice_collection or not self.embedding_model:
            logger.warning(
                "Vektordatenbank nicht initialisiert. Überspringe Speicherung."
            )
            return

        items = proc_result.skr03_classifications
        if not items:
            return

        documents_to_embed = []
        metadatas_to_store = []
        ids_to_add = []

        for i, item in enumerate(items):
            # Der Text, der in einen Vektor umgewandelt wird (Kontext ist König!)
            content = f"Lieferant: {proc_result.invoice_data.get('supplier', '')} | Artikel: {item.get('description', '')}"
            documents_to_embed.append(content)

            # Die Metadaten, die wir abrufen wollen
            metadata = {
                "supplier": str(proc_result.invoice_data.get("supplier", "")),
                "description": str(item.get("description", "")),
                "skr03_konto": str(item.get("skr03_konto", "N/A")),
                "amount": float(str(item.get("amount", "0")).replace(",", ".")),
                "pdf_path": str(proc_result.pdf_path),
            }
            metadatas_to_store.append(metadata)

            # Eindeutige ID
            ids_to_add.append(f"{Path(proc_result.pdf_path).stem}_{i}")

        try:
            # Vektoren erstellen
            embeddings = self.embedding_model.encode(documents_to_embed).tolist()

            # In ChromaDB speichern
            self.invoice_collection.add(
                embeddings=embeddings,
                documents=documents_to_embed,
                metadatas=metadatas_to_store,
                ids=ids_to_add,
            )
            logger.info(
                "%d Positionen für '%s' in Vektordatenbank gespeichert.",
                len(items),
                Path(proc_result.pdf_path).name,
            )
        except Exception as e:
            logger.error("Fehler beim Speichern in Vektordatenbank: %s", e)

    def save_result(
        self, proc_result: ProcessingResult, output_path: Path | None = None
    ) -> Path:
        """Save processing result to JSON file"""

        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pdf_name = Path(proc_result.pdf_path).stem
            output_path = (
                self.config.data_processed_path / f"{pdf_name}_{timestamp}.json"
            )

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save as JSON
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(asdict(proc_result), f, indent=2, ensure_ascii=False)

        logger.info("Results saved to: %s", output_path)
        return output_path


# Convenience functions for backward compatibility
def create_unified_processor(cfg: Config | None = None) -> UnifiedProcessor:
    """Factory function to create unified processor"""
    return UnifiedProcessor(cfg)


def process_single_pdf(
    pdf_path: str | Path, cfg: Config | None = None
) -> ProcessingResult:
    """Process a single PDF file - simplified interface"""
    processor = UnifiedProcessor(cfg)
    return processor.process_pdf(pdf_path)


if __name__ == "__main__":
    # Quick test/demo
    import argparse

    parser = argparse.ArgumentParser(description="Test unified processor")
    parser.add_argument("pdf_path", help="Path to PDF file")
    parser.add_argument("--output", "-o", help="Output JSON file path")

    args = parser.parse_args()

    # Process PDF
    config = Config()
    proc = UnifiedProcessor(config)
    final_result = proc.process_pdf(args.pdf_path)

    # Save result
    output_file_path: Path | None = None
    if args.output:
        output_file_path = Path(args.output)

    saved_path = proc.save_result(final_result, output_file_path)

    print("✅ Processing complete!")
    print(f"📄 PDF: {final_result.pdf_path}")
    print(f"⏱️  Time: {final_result.processing_time_ms}ms")
    print(f"🎯 Confidence: {final_result.confidence_score:.1%}")
    print(f"📊 Positions: {len(final_result.skr03_classifications)}")
    print(f"💾 Saved: {saved_path}")
