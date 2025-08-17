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
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import chromadb
from pydantic import BaseModel, Field, field_validator
from sentence_transformers import SentenceTransformer

try:
    import google.genai as genai
except ImportError:
    genai = None  # type: ignore[assignment]

from src.config import Config
from src.extraction import DataExtractor
from src.processing import DataClassifier, QualityAssessor
from src.skr03_manager import lade_skr03_manager

logger = logging.getLogger(__name__)


class ResourceManager:
    """
    Singleton Resource Manager für schwere ML-Modelle und DB-Verbindungen.
    Verhindert Memory-Leaks durch mehrfache Initialisierung.
    """

    _instance = None
    _embedding_model = None
    _chromadb_client = None
    _skr03_manager = None
    _docling_processor = None  # Add DoclingProcessor singleton

    def __new__(cls) -> "ResourceManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_embedding_model(self) -> SentenceTransformer:
        """Lazy-loaded Singleton SentenceTransformer"""
        if self._embedding_model is None:
            logger.info("🔄 Lade SentenceTransformer (Singleton)...")
            self._embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("✅ SentenceTransformer geladen")
        return self._embedding_model

    def get_chromadb_client(self, db_path: str) -> Any:
        """Lazy-loaded Singleton ChromaDB Client"""
        if self._chromadb_client is None:
            logger.info("🔄 Lade ChromaDB Client (Singleton)...")
            # Deaktiviere ChromaDB Telemetry für saubere Logs
            os.environ["ANONYMIZED_TELEMETRY"] = "False"
            self._chromadb_client = chromadb.PersistentClient(path=db_path)
            logger.info("✅ ChromaDB Client geladen")
        return self._chromadb_client

    def get_skr03_manager(self) -> Any:
        """Lazy-loaded Singleton SKR03 Manager"""
        if self._skr03_manager is None:
            logger.info("🔄 Lade SKR03Manager (Singleton)...")
            self._skr03_manager = lade_skr03_manager()
            logger.info("✅ SKR03Manager geladen")
        return self._skr03_manager

    def get_docling_processor(self) -> Any:
        """Lazy-loaded Singleton DoclingProcessor"""
        if self._docling_processor is None:
            logger.info("🔄 Lade DoclingProcessor (Singleton)...")
            from src.extraction.docling_processor import AdvancedDoclingProcessor

            self._docling_processor = AdvancedDoclingProcessor(
                use_gpu=True,
                ocr_engine="rapid",
                table_mode="accurate",
                german_optimized=True,
            )
            logger.info("✅ DoclingProcessor geladen")
        return self._docling_processor

    def cleanup(self) -> None:
        """Explicitly cleanup all resources to prevent memory leaks"""
        import gc

        import torch

        logger.info("🧹 Bereinige ResourceManager...")

        # Cleanup ChromaDB client
        if self._chromadb_client:
            try:
                # ChromaDB hat keine explizite close(), aber reset das Objekt
                self._chromadb_client = None
            except Exception as e:
                logger.warning(f"ChromaDB cleanup warning: {e}")

        # Cleanup embedding model with explicit torch cache clear
        if self._embedding_model:
            try:
                # Clear CUDA cache if available
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Delete embedding model
                del self._embedding_model
                self._embedding_model = None
            except Exception as e:
                logger.warning(f"Embedding model cleanup warning: {e}")

        # Cleanup Docling processor
        if self._docling_processor:
            try:
                # Delete docling processor
                del self._docling_processor
                self._docling_processor = None
            except Exception as e:
                logger.warning(f"DoclingProcessor cleanup warning: {e}")

        # Cleanup SKR03 manager
        if self._skr03_manager:
            try:
                del self._skr03_manager
                self._skr03_manager = None
            except Exception as e:
                logger.warning(f"SKR03 manager cleanup warning: {e}")

        # Force garbage collection multiple times
        for i in range(5):
            collected = gc.collect()
            if collected > 0:
                logger.info(f"GC Pass {i+1}: {collected} objects collected")

        logger.info("✅ ResourceManager bereinigt")


# Global Resource Manager Instance
_resource_manager = ResourceManager()


class ProcessingResult(BaseModel):
    """
    Unified result from PDF processing pipeline.

    Migrated from dataclass to Pydantic BaseModel for enhanced:
    - Type safety and runtime validation
    - Automatic serialization/deserialization
    - German invoice processing optimization
    """

    # Source information
    pdf_path: str = Field(..., description="Pfad zur verarbeiteten PDF-Datei")
    processing_timestamp: str = Field(
        ..., description="Zeitstempel der Verarbeitung (ISO format)"
    )

    # Extraction results
    raw_text: str = Field(..., description="Extrahierter Rohtext aus PDF")
    structured_data: dict[str, Any] = Field(
        default_factory=dict, description="Strukturierte Daten aus Docling"
    )

    # Classification results
    invoice_data: dict[str, Any] = Field(
        default_factory=dict, description="Klassifizierte Rechnungsdaten"
    )
    skr03_classifications: list[dict[str, Any]] = Field(
        default_factory=list, description="SKR03-Kontierungen"
    )

    # Performance metrics
    processing_time_ms: int = Field(
        ..., ge=0, description="Gesamtverarbeitungszeit in Millisekunden"
    )
    ocr_time_ms: int = Field(
        ..., ge=0, description="OCR-Verarbeitungszeit in Millisekunden"
    )
    classification_time_ms: int = Field(
        ..., ge=0, description="Klassifizierungszeit in Millisekunden"
    )

    # Quality indicators
    confidence_score: float = Field(
        ..., ge=0.0, le=1.0, description="Konfidenz-Score (0.0-1.0)"
    )
    extraction_quality: Literal["high", "medium", "low"] = Field(
        ..., description="Qualitätsbewertung der Extraktion"
    )

    @field_validator("processing_timestamp")
    @classmethod
    def validate_timestamp_format(cls, v: str) -> str:
        """Validiere ISO-Zeitstempel-Format"""
        try:
            datetime.fromisoformat(v.replace("Z", "+00:00"))
            return v
        except ValueError as e:
            raise ValueError(f"Ungültiges Zeitstempel-Format: {v}") from e

    @field_validator("pdf_path")
    @classmethod
    def validate_pdf_path(cls, v: str) -> str:
        """Validiere PDF-Dateipfad"""
        if not v.lower().endswith(".pdf"):
            raise ValueError(f"Pfad muss eine PDF-Datei sein: {v}")
        return v

    def to_dict(self) -> dict[str, Any]:
        """Konvertiere zu Dictionary für Rückwärtskompatibilität"""
        return self.model_dump()

    def get_summary(self) -> str:
        """Erstelle eine Zusammenfassung der Verarbeitungsergebnisse"""
        return (
            f"PDF: {Path(self.pdf_path).name} | "
            f"Qualität: {self.extraction_quality} | "
            f"Konfidenz: {self.confidence_score:.2f} | "
            f"Zeit: {self.processing_time_ms}ms"
        )


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
        """Initialize unified processor with modular components using singleton resources"""
        self.config = cfg or Config()
        self._setup_logging()
        self._setup_vector_db()

        # Use singleton resources to prevent memory leaks
        self.skr03_manager = _resource_manager.get_skr03_manager()

        # Initialize Gemini model if API key is available
        gemini_model = self._setup_gemini_model()

        # Initialize components with config and dependencies
        self.extractor = DataExtractor(gemini_model=gemini_model, config=self.config)
        self.classifier = DataClassifier(
            skr03_manager=self.skr03_manager, vector_store=self.invoice_collection
        )
        self.quality_assessor = QualityAssessor()

    def cleanup(self) -> None:
        """Cleanup resources - call this after processing to free memory"""
        import gc

        logger.info("🧹 Cleanup UnifiedProcessor resources...")

        # Explicit cleanup of all components
        if hasattr(self, "extractor") and self.extractor:
            try:
                # Clear any cached models in extractor
                if hasattr(self.extractor, "cleanup"):
                    self.extractor.cleanup()
                del self.extractor
            except Exception as e:
                logger.warning(f"Extractor cleanup warning: {e}")

        if hasattr(self, "classifier") and self.classifier:
            try:
                del self.classifier
            except Exception as e:
                logger.warning(f"Classifier cleanup warning: {e}")

        if hasattr(self, "quality_assessor") and self.quality_assessor:
            try:
                del self.quality_assessor
            except Exception as e:
                logger.warning(f"Quality assessor cleanup warning: {e}")

        if hasattr(self, "invoice_collection") and self.invoice_collection:
            try:
                del self.invoice_collection
            except Exception as e:
                logger.warning(f"Invoice collection cleanup warning: {e}")

        if hasattr(self, "embedding_model") and self.embedding_model:
            try:
                del self.embedding_model
            except Exception as e:
                logger.warning(f"Embedding model cleanup warning: {e}")

        if hasattr(self, "client") and self.client:
            try:
                del self.client
            except Exception as e:
                logger.warning(f"Client cleanup warning: {e}")

        # Clear all references
        self.extractor = None
        self.classifier = None
        self.quality_assessor = None
        self.invoice_collection = None
        self.embedding_model = None
        self.client = None

        # Force garbage collection
        for i in range(3):
            collected = gc.collect()
            if collected > 0:
                logger.info(
                    f"UnifiedProcessor GC Pass {i+1}: {collected} objects collected"
                )

        logger.info("✅ UnifiedProcessor cleanup complete")

    def _setup_logging(self) -> None:
        """Configure logging for the processor"""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    def _setup_vector_db(self) -> None:
        """Setup ChromaDB and embedding model for RAG system using singletons"""
        try:
            # Use singleton ChromaDB client
            self.client = _resource_manager.get_chromadb_client(
                str(self.config.vector_db_path)
            )

            # Get or create collection for invoices
            self.invoice_collection = self.client.get_or_create_collection(
                name="invoice_line_items",
                metadata={
                    "description": "German electrical contractor invoice items for SKR03 classification"
                },
            )

            # Use singleton embedding model
            self.embedding_model = _resource_manager.get_embedding_model()

            logger.info("✅ ChromaDB und Embedding-Model initialisiert (Singleton)")

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

            # Create Gemini client with API key
            client = genai.Client(api_key=self.config.google_api_key)

            logger.info("✅ Gemini Client initialisiert: %s", self.config.gemini_model)
            return client

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
            # Fix: Ensure line_items are accessible in structured_data for classification
            structured_data_with_items = extraction_result["structured_data"].copy()
            structured_data_with_items["line_items"] = extraction_result["line_items"]

            classifications = self.classifier.process_classifications(
                extraction_result["line_items"], structured_data_with_items
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
                "supplier": str(proc_result.invoice_data.get("supplier", "Unknown")),
                "description": str(item.get("description", "")),
                "skr03_account": str(
                    item.get("skr03_konto", "N/A")
                ),  # FIXED: Use skr03_account for consistency
                "category": str(
                    item.get("category", "Unbekannt")
                ),  # ADDED: Category field
                "confidence": float(
                    item.get("confidence", 0.0)
                ),  # FIXED: Add confidence field for RAG querying
                "amount": self._parse_safe_float(item.get("amount", "0")),
                "pdf_path": str(proc_result.pdf_path),
            }
            metadatas_to_store.append(metadata)

            # Eindeutige ID
            ids_to_add.append(f"{Path(proc_result.pdf_path).stem}_{i}")

        try:
            # Prüfe auf existierende IDs um Warnungen zu vermeiden
            existing_ids: list[str] = []
            if ids_to_add:
                try:
                    existing_items = self.invoice_collection.get(ids=ids_to_add)
                    existing_ids = existing_items["ids"] if existing_items else []
                except Exception:
                    # Falls get() fehlschlägt, fahre normal fort
                    existing_ids = []

            # Nur neue Items hinzufügen (oder upsert verwenden)
            if existing_ids:
                logger.debug(
                    "Verwende upsert für %d existierende Items", len(existing_ids)
                )
                # Vektoren erstellen
                embeddings = self.embedding_model.encode(documents_to_embed).tolist()

                # Upsert statt add um Duplikate zu vermeiden
                self.invoice_collection.upsert(
                    embeddings=embeddings,
                    documents=documents_to_embed,
                    metadatas=metadatas_to_store,
                    ids=ids_to_add,
                )
            else:
                # Vektoren erstellen
                embeddings = self.embedding_model.encode(documents_to_embed).tolist()

                # Normale Addition für neue Items
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

        # Save as JSON using Pydantic's model_dump
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(proc_result.model_dump(), f, indent=2, ensure_ascii=False)

        logger.info("Results saved to: %s", output_path)
        return output_path

    def _parse_safe_float(self, value: str | float | int) -> float:
        """
        Sichere Float-Konvertierung mit Fallback.

        Args:
            value: Wert zur Konvertierung

        Returns:
            Float-Wert oder 0.0 bei Fehlern
        """
        if isinstance(value, int | float):
            return float(value)

        try:
            # String-Bereinigung
            clean_value = str(value).strip()

            # Leer oder None
            if not clean_value or clean_value.lower() in ["none", "null", ""]:
                return 0.0

            # Nur Zahlen und Komma/Punkt enthalten
            import re

            # Entferne alle nicht-numerischen Zeichen außer Komma und Punkt
            clean_value = re.sub(r"[^\d,.-]", "", clean_value)

            # Deutsche Notation: Komma als Dezimaltrennzeichen
            clean_value = clean_value.replace(",", ".")

            # Multiple Punkte handhaben (z.B. "1.234.56" -> "1234.56")
            if clean_value.count(".") > 1:
                parts = clean_value.split(".")
                clean_value = "".join(parts[:-1]) + "." + parts[-1]

            return float(clean_value)

        except (ValueError, AttributeError):
            logger.warning(
                f"Konnte Wert nicht zu Float konvertieren: '{value}', verwende 0.0"
            )
            return 0.0


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
