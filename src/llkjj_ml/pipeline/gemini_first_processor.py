#!/usr/bin/env python3
"""
LLKJJ ML Pipeline - Gemini-First Processor
==========================================

NEUE STANDARD-PIPELINE: PDF → Gemini direkt (ohne Docling)
- Gemini analysiert und extrahiert alle wichtigen Daten direkt aus PDF
- Vollständige Annotation mit Labels und SKR03-Kontierungen
- Kein Fallback auf Docling - bei Gemini-Fehler wird Error-Log ausgegeben

ALTERNATIVE: Docling-Methode nur noch bei expliziter Anfrage verfügbar

Autor: LLKJJ ML Pipeline Team
Version: 4.0.0 (Gemini-First Architecture)
Datum: 18. August 2025
"""

import json
import logging
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer

# Gemini AI Integration
GENAI_AVAILABLE: bool = False
try:
    from google import genai
    GENAI_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    genai = None  # type: ignore[assignment]

from ..intelligence.context_classifier import (
    ContextAwareClassifier,
)
from ..intelligence.feedback_learning import FeedbackLearningEngine
from ..models.gemini_schemas import create_validation_report, validate_gemini_response
from ..models.processing_result import ProcessingResult
from ..monitoring.performance_metrics import PerformanceMonitor
from ..processing.spacy_corrector import SpacyAnnotationCorrector
from ..settings_bridge import ConfigBridge, config_instance
from ..skr03_manager import lade_skr03_manager
from ..training_data_persistence import TrainingDataPersistence
from ..utils.resource_manager import get_resource_manager

logger = logging.getLogger(__name__)


class GeminiDirectProcessor:
    """
    Gemini-First Processor für direkte PDF-Analyse ohne Docling.

    NEUE STANDARD-PIPELINE:
    PDF → Gemini AI → strukturierte Daten → SKR03-Klassifizierung → Ausgabe

    Eigenschaften:
    - Direkte PDF-Analyse durch Gemini 2.5 Pro
    - Vollständige Extraktion und Annotation in einem Schritt
    - Deutsche Elektrotechnik-optimierte Prompts
    - RAG-Integration für intelligente Klassifizierung
    - Kein Fallback - Error-Logging bei Fehlern
    """

    def __init__(self, config: ConfigBridge | None = None) -> None:
        """Initialize Gemini-First processor"""
        self.config = config or config_instance

        # SKR03 Manager für deutsche Buchhaltung
        self.skr03_manager = lade_skr03_manager()

        # Gemini AI Client
        self._gemini_client: Any = None

        # ChromaDB für RAG-System
        self._chromadb_client: Any = None
        self._embedding_model: SentenceTransformer | None = None
        self._invoice_collection: Any = None

        # Quality Assessor
        self._quality_assessor: Any = None

        # 🎯 A2: Training Data Persistence (Strategic TODO)
        self.training_persistence = TrainingDataPersistence(self.config)

        # 📊 Performance Monitoring für Business Metrics
        self.performance_monitor = PerformanceMonitor(self.config)

        # 🧠 Context-Aware Classifier für intelligente Klassifizierung
        self.context_classifier = ContextAwareClassifier()

        # 🔄 Feedback Learning Engine für kontinuierliche Verbesserung
        self.feedback_engine = FeedbackLearningEngine(self.config)

        logger.info(
            "✅ GeminiDirectProcessor initialisiert (v4.0.0 + Training Data Collection + Performance Monitoring + Context Intelligence + Feedback Learning)"
        )

    @property
    def gemini_client(self) -> Any:
        """Lazy-loaded Gemini Client"""
        if self._gemini_client is None:
            if not self.config.google_api_key:
                raise ValueError(
                    "❌ Google API Key erforderlich für Gemini-First Pipeline. "
                    "Setze GEMINI_API_KEY Environment Variable."
                )

            try:
                if genai is not None:
                    self._gemini_client = genai.Client(
                        api_key=self.config.google_api_key
                    )
                    logger.info("✅ Gemini 2.5 Pro Client geladen")
                else:
                    raise ImportError("genai module not available")
            except Exception as e:
                logger.error("❌ Gemini Client konnte nicht geladen werden: %s", e)
                raise ValueError(
                    f"Gemini-First Pipeline erfordert funktionsfähigen Gemini Client. "
                    f"Fehler: {e}"
                ) from e

        return self._gemini_client

    @property
    def embedding_model(self) -> SentenceTransformer:
        """Lazy-loaded Embedding Model für RAG"""
        if self._embedding_model is None:
            logger.info("🔄 Lade SentenceTransformer für RAG...")
            self._embedding_model = SentenceTransformer("all-MiniLM-L12-v2")
            logger.info("✅ Embedding Model geladen")
        return self._embedding_model

    @property
    def chromadb_client(self) -> Any:
        """Lazy-loaded ChromaDB Client"""
        if self._chromadb_client is None:
            logger.info("🔄 Initialisiere ChromaDB für RAG...")
            os.environ["ANONYMIZED_TELEMETRY"] = "False"
            db_path = str(self.config.vector_db_path or "data/vectors")
            self._chromadb_client = chromadb.PersistentClient(path=db_path)

            # Collection für Rechnungen
            try:
                self._invoice_collection = self._chromadb_client.get_collection(
                    "invoices"
                )
                logger.info("✅ ChromaDB Collection 'invoices' geladen")
            except ValueError:
                self._invoice_collection = self._chromadb_client.create_collection(
                    "invoices"
                )
                logger.info("✅ ChromaDB Collection 'invoices' erstellt")

        return self._chromadb_client

    @property
    def quality_assessor(self) -> Any:
        """Lazy-loaded Quality Assessor"""
        if self._quality_assessor is None:
            logger.info("🔄 Initialisiere QualityAssessor...")
            from ..processing.quality import QualityAssessor

            self._quality_assessor = QualityAssessor()
        return self._quality_assessor

    def process_pdf_gemini_first(self, pdf_path: str | Path) -> ProcessingResult:
        """
        NEUE STANDARD-METHODE: PDF → Gemini direkt mit robustem Error Handling

        Gemini analysiert PDF direkt und extrahiert:
        - Rechnungskopf (Lieferant, Kunde, Datum, Nummer)
        - Alle Positionen mit Artikelnummer, Beschreibung, Menge, Preise
        - SKR03-Klassifizierungsvorschläge
        - Annotierte Labels für spaCy-Training
        - Qualitätsbewertung

        Args:
            pdf_path: Pfad zur PDF-Datei

        Returns:
            ProcessingResult mit vollständig strukturierten und annotierten Daten

        Raises:
            FileNotFoundError: PDF-Datei nicht gefunden
            ValueError: Ungültige PDF-Datei oder Konfiguration
            RuntimeError: Gemini API-Fehler oder Pipeline-Fehler
        """
        start_time = time.time()
        validated_path = self._validate_pdf_input(pdf_path)

        try:
            pdf_content = self._prepare_pdf_for_gemini_analysis(validated_path)
            gemini_result = self._execute_gemini_analysis(pdf_content, validated_path)
            enhanced_classifications = self._enhance_with_rag_classification(
                gemini_result
            )
            quality_data = self._assess_quality_and_generate_annotations(
                gemini_result, enhanced_classifications
            )
            processing_result = self._build_complete_processing_result(
                validated_path,
                gemini_result,
                enhanced_classifications,
                quality_data,
                start_time,
            )
            self._handle_training_data_and_metrics(processing_result, validated_path)
            return processing_result

        except Exception as e:
            self._handle_processing_error(e, validated_path, start_time)
            raise

    def _validate_pdf_input(self, pdf_path: str | Path) -> Path:
        """
        Validates PDF input file and returns validated Path object.

        Args:
            pdf_path: Raw input path to PDF file

        Returns:
            Validated Path object

        Raises:
            FileNotFoundError: PDF file does not exist
            ValueError: Invalid PDF file or configuration
        """
        from ..utils.german_errors import GermanErrorMessages

        pdf_path = Path(pdf_path)

        # Initiale Validierung
        if not pdf_path.exists():
            raise FileNotFoundError(GermanErrorMessages.pdf_not_found(pdf_path))

        if not pdf_path.suffix.lower() == ".pdf":
            raise ValueError(GermanErrorMessages.pdf_invalid_format(pdf_path))

        if pdf_path.stat().st_size == 0:
            raise ValueError(GermanErrorMessages.pdf_empty(pdf_path))

        logger.info(
            "🚀 GEMINI-FIRST Pipeline startet: %s (%.1f KB)",
            pdf_path.name,
            pdf_path.stat().st_size / 1024,
        )

        # Resource Manager für robustes Memory-Management (nicht kritisch bei Fehler)
        try:
            get_resource_manager()
        except Exception as rm_error:  # pragma: no cover - Schutzpfad
            logger.warning(
                "⚠️ Resource Manager Initialisierung fehlgeschlagen (fahre fort): %s",
                rm_error,
            )

        return pdf_path

    def _prepare_pdf_for_gemini_analysis(self, pdf_path: Path) -> bytes:
        """
        Reads and validates PDF content for Gemini analysis.

        Args:
            pdf_path: Validated PDF path

        Returns:
            PDF content as bytes

        Raises:
            ValueError: PDF file too large or empty
            RuntimeError: File reading error
        """
        logger.info("📤 Sende PDF an Gemini 2.5 Pro für direkte Analyse...")

        # PDF-Datei für Gemini vorbereiten mit Validierung
        try:
            with open(pdf_path, "rb") as pdf_file:
                pdf_content = pdf_file.read()

            if len(pdf_content) == 0:
                raise ValueError("PDF-Datei ist leer oder konnte nicht gelesen werden")

            if len(pdf_content) > 20 * 1024 * 1024:  # 20MB Limit
                raise ValueError(
                    f"PDF-Datei zu groß: {len(pdf_content) / 1024 / 1024:.1f}MB (Max: 20MB)"
                )

            return pdf_content

        except OSError as e:
            raise RuntimeError(f"Fehler beim Lesen der PDF-Datei: {e}") from e

    def _execute_gemini_analysis(self, pdf_content: bytes, pdf_path: Path) -> dict[str, Any]:
        """
        Executes Gemini analysis on PDF content with retry logic.

        Args:
            pdf_content: PDF file content as bytes
            pdf_path: PDF file path for context

        Returns:
            Structured analysis result from Gemini

        Raises:
            RuntimeError: Gemini analysis failed
        """
        # Optimierter Prompt für deutsche Elektrotechnik-Rechnungen
        analysis_prompt = self._get_german_elektro_analysis_prompt()

        # Gemini-Analyse durchführen mit Retry-Logic
        gemini_start = time.time()
        try:
            structured_result = self._analyze_pdf_with_gemini_robust(
                pdf_content, analysis_prompt, pdf_path
            )
        except Exception as e:
            raise RuntimeError(f"Gemini-Analyse fehlgeschlagen: {e}") from e

        gemini_time_ms = int((time.time() - gemini_start) * 1000)
        logger.info("✅ Gemini-Analyse abgeschlossen (%d ms)", gemini_time_ms)

        return structured_result

    def _enhance_with_rag_classification(self, structured_result: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Enhances Gemini results with RAG-based SKR03 classification.

        Args:
            structured_result: Raw Gemini analysis results

        Returns:
            Enhanced classifications with SKR03 codes

        Note:
            Uses fallback classification if RAG enhancement fails
        """
        # Phase 2: SKR03-Klassifizierung mit RAG-Enhancement
        classification_start = time.time()
        try:
            enhanced_classifications = self._enhance_with_rag_system_robust(
                structured_result.get("line_items", [])
            )
        except Exception as e:
            logger.warning(
                "⚠️ RAG-Enhancement fehlgeschlagen, verwende Basis-Klassifizierung: %s",
                e,
            )
            # Fallback auf Basis-Klassifizierung
            enhanced_classifications = self._fallback_skr03_classification(
                structured_result.get("line_items", [])
            )

        classification_time_ms = int((time.time() - classification_start) * 1000)
        logger.info(
            "✅ SKR03-Klassifizierung abgeschlossen (%d ms)", classification_time_ms
        )

        return enhanced_classifications

    def _assess_quality_and_generate_annotations(
        self, structured_result: dict[str, Any], enhanced_classifications: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Assesses processing quality and generates spaCy training annotations.

        Args:
            structured_result: Gemini analysis results
            enhanced_classifications: Enhanced SKR03 classifications

        Returns:
            Dictionary containing quality metrics and training annotations

        Note:
            Uses fallback values if assessment fails
        """
        # Phase 3: Qualitätsbewertung mit Error Handling
        try:
            confidence_score = self.quality_assessor.calculate_confidence(
                structured_result, enhanced_classifications
            )
            quality_level = self.quality_assessor.assess_quality(confidence_score)
        except Exception as e:
            logger.warning(
                "⚠️ Qualitätsbewertung fehlgeschlagen, verwende Standard-Werte: %s",
                e,
            )
            confidence_score = 0.75  # Standardwert
            quality_level = "medium"

        # Phase 4: Annotation für spaCy-Training mit Error Handling
        try:
            training_annotations = self._generate_spacy_annotations_robust(
                structured_result, enhanced_classifications
            )
            # Method always returns list[dict[str, Any]], no type check needed
        except Exception as e:
            logger.warning(
                "⚠️ spaCy-Annotation fehlgeschlagen, verwende leere Annotations: %s",
                e,
            )
            training_annotations = []

        return {
            "confidence_score": confidence_score,
            "quality_level": quality_level,
            "training_annotations": training_annotations,
        }

    def _build_complete_processing_result(
        self,
        pdf_path: Path,
    gemini_result: dict[str, Any],
    enhanced_classifications: list[dict[str, Any]],
    quality_data: dict[str, Any],
        start_time: float,
    ) -> ProcessingResult:
        """
        Builds complete ProcessingResult with all pipeline data.

        Args:
            pdf_path: Validated PDF file path
            gemini_result: Raw Gemini analysis results
            enhanced_classifications: Enhanced SKR03 classifications
            quality_data: Quality metrics and training annotations
            start_time: Pipeline start time

        Returns:
            Complete ProcessingResult object

        Raises:
            Warning: Logs if no line items extracted
        """
        processing_time_ms = int((time.time() - start_time) * 1000)

        self._validate_extracted_data(gemini_result)
        self._store_in_rag_system_if_enabled(gemini_result, pdf_path)
        result = self._create_processing_result_object(
            pdf_path,
            gemini_result,
            enhanced_classifications,
            quality_data,
            processing_time_ms,
        )
        self._log_pipeline_success(enhanced_classifications, quality_data)

        return result

    def _validate_extracted_data(self, gemini_result: dict[str, Any]) -> None:
        """
        Validates extracted data and logs warnings if incomplete.

        Args:
            gemini_result: Raw Gemini analysis results
        """
        line_items = gemini_result.get("line_items", [])
        if not line_items:
            logger.warning("⚠️ Keine Rechnungspositionen extrahiert")

    def _store_in_rag_system_if_enabled(
        self,
    gemini_result: dict[str, Any],
        pdf_path: Path,
    ) -> None:
        """
        Stores results in RAG system for future improvements if enabled.

        Args:
            gemini_result: Gemini analysis results
            pdf_path: PDF file path
        """
        try:
            self._store_in_rag_system_robust(gemini_result, pdf_path)
        except Exception as e:
            logger.warning("⚠️ RAG-Speicherung fehlgeschlagen (nicht kritisch): %s", e)

    def _create_processing_result_object(
        self,
        pdf_path: Path,
    gemini_result: dict[str, Any],
    enhanced_classifications: list[dict[str, Any]],
    quality_data: dict[str, Any],
        processing_time_ms: int,
    ) -> ProcessingResult:
        """
        Creates ProcessingResult object from pipeline data.

        Args:
            pdf_path: PDF file path
            gemini_result: Gemini analysis results
            enhanced_classifications: Enhanced classifications
            quality_data: Quality metrics and annotations
            processing_time_ms: Total processing time

        Returns:
            ProcessingResult object
        """
        return ProcessingResult.from_gemini_analysis(
            pdf_path=str(pdf_path),
            gemini_result=gemini_result,
            skr03_classifications=enhanced_classifications,
            confidence_score=quality_data["confidence_score"],
            quality_level=quality_data["quality_level"],
            training_annotations=quality_data["training_annotations"],
            processing_time_ms=processing_time_ms,
            gemini_time_ms=0,  # Will be set by calling method
            classification_time_ms=0,  # Will be set by calling method
            gemini_model=self.config.gemini_model,
        )

    def _log_pipeline_success(
        self,
        enhanced_classifications: list[dict[str, Any]],
        quality_data: dict[str, Any],
    ) -> None:
        """
        Logs successful pipeline completion statistics.

        Args:
            enhanced_classifications: Enhanced SKR03 classifications
            quality_data: Quality metrics
        """
        logger.info(
            "🎉 GEMINI-FIRST Pipeline erfolgreich: %d Positionen, %.1f%% Konfidenz, %s Qualität",
            len(enhanced_classifications),
            quality_data["confidence_score"] * 100,
            quality_data["quality_level"],
        )

    def _handle_training_data_and_metrics(
        self,
        result: ProcessingResult,
        pdf_path: Path,
    ) -> None:
        """
        Handles training data persistence and performance metrics recording.

        Args:
            result: Complete processing result
            pdf_path: PDF file path

        Note:
            Failures in training data persistence or metrics recording do not fail the main pipeline
        """
        # 🎯 A2: Training Data Persistence - Collect valuable training data
        try:
            persistence_stats = self.training_persistence.persist_training_data(
                pdf_path=str(pdf_path),
                validated_gemini_result=result.structured_data,
                skr03_classifications=result.skr03_classifications,
                processing_result=result,
            )
            logger.info("✅ Training data persisted: %s", persistence_stats)
        except Exception as training_error:
            # Training data persistence should not fail the main pipeline
            logger.warning("⚠️ Training data persistence failed: %s", training_error)

        # 📊 Performance Monitoring - Record business metrics
        try:
            self.performance_monitor.record_processing_metrics(
                result=result,
                gemini_time_ms=result.gemini_time_ms,
                classification_time_ms=result.classification_time_ms,
                cache_hit=False,  # Not using cache in this standard processor
                error_occurred=False,
            )
            logger.debug("📊 Performance metrics recorded successfully")
        except Exception as metrics_error:
            # Metrics recording should not fail the main pipeline
            logger.warning("⚠️ Performance metrics recording failed: %s", metrics_error)

    def _handle_processing_error(
        self,
        error: Exception,
        pdf_path: Path,
        start_time: float,
    ) -> ProcessingResult:
        """
        Handles processing errors with detailed logging and metrics recording.

        Args:
            error: The exception that occurred
            pdf_path: PDF file path for context
            start_time: Processing start time

        Returns:
            ProcessingResult with error information
        """
    error_context = self._build_error_context(error, pdf_path, start_time)
        self._log_detailed_error(error_context)
        self._write_error_log_detailed(pdf_path, error, error_context)
        self._record_error_metrics(error, error_context)

        # Return error result instead of re-raising
        return ProcessingResult(
            pdf_path=str(pdf_path),
            processing_timestamp=datetime.now(UTC).isoformat(),
            processing_method="gemini_first",
            processing_time_ms=int((time.time() - start_time) * 1000),
            confidence_score=0.0,
            extraction_quality="poor",
            skr03_classifications=[],
        )

    def _build_error_context(
        self,
        error: Exception,
        pdf_path: Path,
        start_time: float,
    ) -> dict[str, Any]:
        """
        Builds comprehensive error context for logging and metrics.

        Args:
            error: The exception that occurred
            pdf_path: PDF file path
            start_time: Processing start time

        Returns:
            Error context dictionary
        """
        return {
            "pdf_path": str(pdf_path),
            "pdf_size_bytes": pdf_path.stat().st_size if pdf_path.exists() else 0,
            "processing_time_ms": int((time.time() - start_time) * 1000),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "gemini_model": self.config.gemini_model,
        }

    def _log_detailed_error(self, error_context: dict[str, Any]) -> None:
        """
        Logs detailed error information with context.

        Args:
            error_context: Error context dictionary
        """
        logger.error(
            "❌ GEMINI-FIRST Pipeline fehlgeschlagen: %s",
            error_context,
            exc_info=True,
        )

    def _record_error_metrics(self, error: Exception, error_context: dict[str, Any]) -> None:
        """
        Records error metrics for monitoring and analysis.

        Args:
            error: The exception that occurred
            error_context: Error context information
        """
        try:
            # Create a minimal ProcessingResult for error recording
            error_result = ProcessingResult(
                pdf_path=str(error_context["pdf_path"]),
                processing_timestamp=datetime.now().isoformat() + "Z",
                processing_method="gemini_first",
                structured_data={},
                skr03_classifications=[],
                confidence_score=0.0,
                extraction_quality="poor",
                training_annotations=[],
                processing_time_ms=int(str(error_context["processing_time_ms"])),
                gemini_time_ms=0,
                classification_time_ms=0,
            )

            self.performance_monitor.record_processing_metrics(
                result=error_result,
                gemini_time_ms=0,
                classification_time_ms=0,
                cache_hit=False,
                error_occurred=True,
                error_message=str(error),
            )
            logger.debug("📊 Error metrics recorded successfully")
        except Exception as metrics_error:
            logger.warning("⚠️ Error metrics recording failed: %s", metrics_error)

    def _re_raise_with_context(self, error: Exception, pdf_path: Path) -> None:
        """
        Re-raises exception with improved context and error message.

        Args:
            error: Original exception
            pdf_path: PDF file path for context

        Raises:
            Original exception or RuntimeError with context
        """
        # Exception re-raise mit besserer Nachricht
        if isinstance(error, FileNotFoundError | ValueError):
            # Input-Validation Errors direkt weiterreichen
            raise error
        else:
            # Alle anderen Errors als RuntimeError mit Kontext
            raise RuntimeError(
                f"GEMINI-FIRST Pipeline fehlgeschlagen für {pdf_path.name}: {error}"
            ) from error

    def _get_german_elektro_analysis_prompt(self) -> str:
        """
        Optimierter Prompt für deutsche Elektrotechnik-Rechnungen.
        Instruiert Gemini für perfekte Extraktion und SKR03-Klassifizierung.
        """
        return """
# AUFGABE: Deutsche Elektrotechnik-Rechnung analysieren und strukturiert extrahieren

Du bist ein Experte für deutsche Elektrotechnik-Rechnungen und SKR03-Buchhaltung.
Analysiere die PDF-Rechnung und extrahiere ALLE Daten strukturiert.

## AUSGABE-FORMAT (JSON):

```json
{
    "invoice_header": {
        "lieferant": "Firmenname des Lieferanten",
        "kundennummer": "Kundennummer",
        "rechnungsnummer": "Rechnungsnummer",
        "rechnungsdatum": "YYYY-MM-DD",
        "lieferdatum": "YYYY-MM-DD",
        "zahlungsziel": "Zahlungsziel in Tagen"
    },
    "line_items": [
        {
            "position": 1,
            "artikelnummer": "Artikelnummer/SKU",
            "beschreibung": "Vollständige Artikelbeschreibung",
            "marke": "Herstellermarke (GIRA, Hager, Siemens, etc.)",
            "menge": 5.0,
            "einheit": "Stk",
            "einzelpreis": 12.50,
            "gesamtpreis": 62.50,
            "elektro_kategorie": "Installation/Schalterprogramm/Kabel/etc.",
            "skr03_vorschlag": "3400",
            "skr03_beschreibung": "Wareneingang Elektromaterial"
        }
    ],
    "totals": {
        "nettosumme": 1234.56,
        "mwst_betrag": 234.56,
        "mwst_satz": 19.0,
        "bruttosumme": 1469.12
    },
    "zusatzinfos": {
        "zahlungsart": "Überweisung/Lastschrift/etc.",
        "lieferadresse": "Adresse falls abweichend",
        "projekt": "Projektnummer falls vorhanden"
    }
}
```

## WICHTIGE ELEKTROTECHNIK-KATEGORIEN:
- Kabel und Leitungen (NYM, NYY, etc.)
- Schalterprogramme (GIRA, Berker, Jung)
- Installationsmaterial (Dosen, Rohre, Klemmen)
- Sicherungstechnik (Sicherungen, FI-Schalter)
- Beleuchtung (LEDs, Leuchten)
- KNX/Smart Home Komponenten
- Werkzeuge und Verbrauchsmaterial

## SKR03-KONTEN (Vorschläge):
- 3400: Wareneingang Elektromaterial
- 4930: Büro- und Verbrauchsmaterial
- 4985: Werkzeuge und Geräte
- 0200: Technische Anlagen (bei hochwertigen Geräten)

EXTRAHIERE ALLE sichtbaren Positionen vollständig und präzise!
"""

    def _analyze_pdf_with_gemini(
        self,
        pdf_content: bytes,
        prompt: str,
    ) -> dict[str, Any]:
        """
        Führt die direkte PDF-Analyse mit Gemini durch.
        """
        try:
            # PDF an Gemini senden für echte Analyse
            logger.info("🔄 Gemini analysiert PDF-Inhalt...")

            # Echte Gemini-API-Integration mit PDF Upload
            try:
                import base64

                # PDF als base64 encodieren
                pdf_base64 = base64.b64encode(pdf_content).decode("utf-8")

                # API-Call an Gemini mit korrektem Format
                response = self.gemini_client.models.generate_content(
                    model=self.config.gemini_model,
                    contents=[
                        {
                            "role": "user",
                            "parts": [
                                {"text": prompt},
                                {
                                    "inline_data": {
                                        "mime_type": "application/pdf",
                                        "data": pdf_base64,
                                    }
                                },
                            ],
                        }
                    ],
                )

                # Response verarbeiten
                if response and hasattr(response, "text"):
                    response_text = response.text
                    logger.info(
                        "📄 Gemini Response erhalten (%d Zeichen)", len(response_text)
                    )

                    # JSON aus Response extrahieren
                    try:
                        # Suche nach JSON in der Response
                        import re

                        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
                        if json_match:
                            json_str = json_match.group()
                            result: dict[str, Any] = json.loads(json_str)
                            logger.info(
                                "✅ Gemini-Analyse erfolgreich - Produktionsdaten von %s",
                                self.config.gemini_model,
                            )
                            return result
                        else:
                            logger.error(
                                "❌ Kein JSON in Gemini Response gefunden. Response: %s",
                                response_text[:500],
                            )
                            raise ValueError(
                                "Gemini API Response enthält kein gültiges JSON"
                            )
                    except json.JSONDecodeError as e:
                        logger.error(
                            "❌ JSON-Parsing-Fehler in Gemini Response: %s. Response: %s",
                            e,
                            response_text[:500],
                        )
                        raise ValueError(
                            f"Gemini API Response JSON ist ungültig: {e}"
                        ) from e
                else:
                    logger.error("❌ Leere oder ungültige Gemini Response erhalten")
                    raise ValueError("Gemini API Response ist leer oder ungültig")

            except Exception as api_error:
                logger.error("❌ Gemini API-Fehler: %s", api_error)
                raise ValueError(
                    f"Gemini API-Aufruf fehlgeschlagen: {api_error}"
                ) from api_error

        except Exception as e:
            logger.error("❌ Gemini-PDF-Analyse fehlgeschlagen: %s", e)
            raise

    def _enhance_with_rag_system(
        self,
        line_items: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        🧠 Verbessert SKR03-Klassifizierung durch RAG-System + Context-Aware Intelligence.
        """
        logger.info("🔄 SKR03-Enhancement mit RAG-System + Context Intelligence...")

        enhanced_items = []

        for item in line_items:
            # RAG-basierte Verbesserung der Klassifizierung
            enhanced_item = item.copy()

            # 1. SKR03Manager für intelligente Klassifizierung nutzen
            if self.skr03_manager:
                artikel_text = f"{item.get('beschreibung', '')} {item.get('marke', '')}"
                kategorie, konto, konfidenz, keywords = (
                    self.skr03_manager.klassifiziere_artikel(artikel_text)
                )

                enhanced_item.update(
                    {
                        "skr03_konto": konto,
                        "skr03_kategorie": kategorie,
                        "klassifizierungs_konfidenz": konfidenz,
                        "matched_keywords": keywords,
                    }
                )

            # 2. 🧠 Context-Aware Intelligence Enhancement
            try:
                context_enhanced_item = self.context_classifier.classify_with_context(
                    item=enhanced_item,
                    all_items=line_items,  # Für Multi-Position-Context
                )
                enhanced_item.update(context_enhanced_item)

                # 3. 🔄 Feedback Learning Enhancement
                feedback_enhanced_item = self.feedback_engine.apply_feedback_boost(
                    classification=enhanced_item,
                    supplier_context={
                        "supplier": item.get("lieferant", ""),
                        "invoice_date": item.get("datum"),
                        "total_amount": item.get("gesamtbetrag", 0),
                    },
                )
                enhanced_items.append(feedback_enhanced_item)

                logger.debug(
                    "Feedback-Enhanced: %s -> %s (Supplier: %s) [Boost: +%.2f]",
                    item.get("beschreibung", "")[:30],
                    feedback_enhanced_item.get("skr03_konto", ""),
                    item.get("lieferant", "")[:15],
                    feedback_enhanced_item.get("feedback_boost", 0.0),
                )

            except Exception as context_error:
                logger.warning(
                    "Context-Enhancement fehler für %s: %s",
                    item.get("beschreibung", "")[:30],
                    str(context_error),
                )
                enhanced_items.append(
                    enhanced_item
                )  # Fallback: Standard RAG-Enhancement

        logger.info(
            "✅ RAG + Context Enhancement abgeschlossen für %d Positionen",
            len(enhanced_items),
        )
        return enhanced_items

    def _generate_spacy_annotations(
        self,
        structured_data: dict[str, Any],
        classifications: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Generiert spaCy-Training-Annotationen aus den extrahierten Daten.
        """
        logger.info("🔄 Generiere spaCy-Training-Annotationen...")

        # Annotations für NER-Training
        ner_annotations = []

        # Textcat-Annotations für SKR03-Klassifizierung
        textcat_annotations = []

        for item in classifications:
            # NER-Entities extrahieren
            entities = []
            if item.get("artikelnummer"):
                entities.append(("ARTIKELNUMMER", item["artikelnummer"]))
            if item.get("beschreibung"):
                entities.append(("ARTIKEL", item["beschreibung"]))
            if item.get("einzelpreis"):
                entities.append(("EINZELPREIS", str(item["einzelpreis"])))
            if item.get("menge"):
                entities.append(("MENGE", str(item["menge"])))

            ner_annotations.append(
                {
                    "text": f"{item.get('beschreibung', '')} {item.get('artikelnummer', '')}",
                    "entities": entities,
                }
            )

            # TextCat für SKR03-Klassifizierung
            if item.get("skr03_kategorie"):
                textcat_annotations.append(
                    {
                        "text": item.get("beschreibung", ""),
                        "cats": {item["skr03_kategorie"]: 1.0},
                    }
                )

        annotations = {
            "ner_annotations": ner_annotations,
            "textcat_annotations": textcat_annotations,
            "total_entities": len(ner_annotations),
            "total_categories": len(textcat_annotations),
        }

        logger.info(
            "✅ spaCy-Annotationen generiert: %d NER, %d TextCat",
            len(ner_annotations),
            len(textcat_annotations),
        )

        return annotations

    def _store_in_rag_system(self, result: dict[str, Any]) -> None:
        """
        Speichert Ergebnisse im RAG-System für zukünftige Verbesserungen.
        """
        try:
            if not self._invoice_collection:
                _ = self.chromadb_client  # Initialize collection

            items = result.get("skr03_classifications", [])
            if not items:
                return

            logger.info("🔄 Speichere %d Positionen im RAG-System...", len(items))

            # Vektorisierung und Speicherung
            for i, item in enumerate(items):
                doc_text = f"{item.get('beschreibung', '')} {item.get('marke', '')}"
                embedding = self.embedding_model.encode([doc_text])[0]

                metadata = {
                    "skr03_account": item.get("skr03_konto", "3400"),
                    "category": item.get("skr03_kategorie", "wareneingang_elektro"),
                    "confidence": item.get("klassifizierungs_konfidenz", 0.0),
                    "source": "gemini_first_pipeline",
                }

                self._invoice_collection.add(
                    embeddings=[embedding.tolist()],
                    documents=[doc_text],
                    metadatas=[metadata],
                    ids=[f"gemini_first_{result['timestamp']}_{i}"],
                )

            logger.info("✅ RAG-System aktualisiert")

        except Exception as e:
            logger.warning("⚠️ RAG-Speicherung fehlgeschlagen: %s", e)

    # === ROBUST HELPER METHODS (Hybrid Implementation) ===

    def _analyze_pdf_with_gemini_robust(
        self,
        pdf_content: bytes,
        analysis_prompt: str,
        pdf_path: Path,
    ) -> dict[str, Any]:
        """
        Robuste Gemini-Analyse mit Retry-Logic und detailliertem Error Handling.

        Args:
            pdf_content: PDF-Binärdaten
            analysis_prompt: Gemini-Prompt für Analyse
            pdf_path: Pfad für Logging

        Returns:
            Strukturierte Gemini-Analyseergebnisse

        Raises:
            RuntimeError: Bei Gemini API-Fehlern nach allen Retry-Versuchen
        """
        max_retries = 3
        retry_delay = 2.0

        for attempt in range(max_retries):
            try:
                logger.info(f"🔄 Gemini-Analyse Versuch {attempt + 1}/{max_retries}...")

                # Fallback auf bestehende Methode
                raw_result = self._analyze_pdf_with_gemini(pdf_content, analysis_prompt)

                # 🎯 NEUE PYDANTIC-VALIDIERUNG (A1 TODO-Item)
                logger.info("🔍 Validiere Gemini-Response mit Pydantic...")

                # DEBUG: Only log in development mode, never in production
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "Gemini response validation starting - response length: %d chars",
                        len(str(raw_result)),
                    )

                validated_result, validation_errors = validate_gemini_response(
                    raw_result
                )

                if validated_result:
                    logger.info("✅ Gemini-Response erfolgreich validiert!")
                    # Konvertiere zurück zu dict für Kompatibilität
                    result: dict[str, Any] = validated_result.model_dump()
                else:
                    # Validierung fehlgeschlagen - detaillierte Fehlerausgabe
                    error_report = create_validation_report(validation_errors)
                    logger.error("❌ Gemini-Response-Validierung fehlgeschlagen:")
                    logger.error(error_report)
                    raise ValueError(
                        f"Pydantic-Validierung fehlgeschlagen: {len(validation_errors)} Fehler"
                    )

                # Legacy-Validierung für Rückwärtskompatibilität
                if not isinstance(result, dict):
                    raise ValueError("Gemini-Antwort ist kein Dictionary")

                if not result.get("line_items"):
                    logger.warning("⚠️ Gemini extrahierte keine Rechnungspositionen")

                logger.info("✅ Gemini-Analyse mit Pydantic-Validierung erfolgreich")
                return result

            except Exception as e:
                logger.warning(
                    f"⚠️ Gemini-Analyse Versuch {attempt + 1} fehlgeschlagen: {e}"
                )

                if attempt == max_retries - 1:
                    # Letzter Versuch - Exception weiterreichen
                    raise RuntimeError(
                        f"Gemini-Analyse nach {max_retries} Versuchen fehlgeschlagen: {e}"
                    ) from e

                # Exponential backoff
                import time

                time.sleep(retry_delay * (2**attempt))

        # Sollte nie erreicht werden
        raise RuntimeError("Unerwarteter Fehler in Gemini-Retry-Logic")

    def _enhance_with_rag_system_robust(
        self,
        line_items: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Robuste RAG-Enhancement mit Fallback auf Basis-Klassifizierung.

        Args:
            line_items: Rechnungspositionen zur Klassifizierung

        Returns:
            Enhanced classifications mit SKR03-Kontierungen
        """
        try:
            # Fallback auf bestehende Methode
            return self._enhance_with_rag_system(line_items)

        except Exception as e:
            logger.warning(f"⚠️ RAG-Enhancement fehlgeschlagen, verwende Fallback: {e}")
            return self._fallback_skr03_classification(line_items)

    def _fallback_skr03_classification(
        self,
        line_items: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Fallback-Klassifizierung ohne RAG-System bei Fehlern.

        Args:
            line_items: Rechnungspositionen

        Returns:
            Basis-SKR03-Klassifizierungen
        """
        classifications = []

        for i, item in enumerate(line_items):
            description = item.get("beschreibung", "").lower()

            # Einfache regelbasierte Klassifizierung
            skr03_account = "3400"  # Standard: Wareneingang
            category = "wareneingang_elektro_allgemein"
            confidence = 0.6

            # Einfache Kategorisierung
            if any(
                keyword in description
                for keyword in ["schalter", "steckdose", "rahmen"]
            ):
                skr03_account = "3400"
                category = "installationsmaterial"
                confidence = 0.7
            elif any(
                keyword in description for keyword in ["kabel", "leitung", "draht"]
            ):
                skr03_account = "3400"
                category = "kabel_leitungen"
                confidence = 0.7
            elif any(keyword in description for keyword in ["lampe", "leuchte", "led"]):
                skr03_account = "3400"
                category = "beleuchtung"
                confidence = 0.7

            classification = {
                "position": i + 1,
                "description": item.get("beschreibung", ""),
                "amount": item.get("gesamtpreis", 0.0),
                "skr03_account": skr03_account,
                "skr03_category": category,
                "confidence": confidence,
                "source": "fallback_classification",
            }

            classifications.append(classification)

        logger.info(f"✅ Fallback-Klassifizierung: {len(classifications)} Positionen")
        return classifications

    def _generate_spacy_annotations_robust(
        self,
        structured_result: dict[str, Any],
        classifications: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Robuste spaCy-Annotation-Generierung mit Position-Korrektur.

        **VERBESSERUNG aus Gemini-Vorschlag:**
        - Verwendet SpacyAnnotationCorrector für exakte Positionen
        - Korrigiert Character-Offsets für bessere Training-Qualität
        - Validiert Annotationen gegen Original-Text

        Args:
            structured_result: Gemini-Ergebnisse
            classifications: SKR03-Klassifizierungen

        Returns:
            Position-korrigierte spaCy-Training-Annotationen als Liste
        """
        try:
            # Fallback auf bestehende Methode für rohe Annotationen
            annotations_dict = self._generate_spacy_annotations(
                structured_result, classifications
            )

            # Konvertiere dict zu list für ProcessingResult-Kompatibilität
            raw_annotations = []

            # NER Annotations
            for annotation in annotations_dict.get("ner", []):
                raw_annotations.append(
                    {
                        "type": "ner",
                        "text": annotation.get("text", ""),
                        "label": annotation.get("label", ""),
                        "start_char": annotation.get("start_char", 0),
                        "end_char": annotation.get("end_char", 0),
                    }
                )

            # TextCat Annotations
            for annotation in annotations_dict.get("textcat", []):
                raw_annotations.append(
                    {
                        "type": "textcat",
                        "text": annotation.get("text", ""),
                        "label": annotation.get("label", ""),
                        "score": annotation.get("score", 0.0),
                    }
                )

            # **POSITION-KORREKTUR ANWENDEN**
            if raw_annotations:
                # Extrahiere vollständigen Text für Position-Korrektur
                full_text = self._extract_full_text_from_result(structured_result)

                # Initialisiere spaCy Corrector
                corrector = SpacyAnnotationCorrector()

                # Korrigiere nur NER-Annotationen (TextCat braucht keine Positionen)
                ner_annotations = [
                    ann for ann in raw_annotations if ann.get("type") == "ner"
                ]
                textcat_annotations = [
                    ann for ann in raw_annotations if ann.get("type") == "textcat"
                ]

                if ner_annotations and full_text:
                    logger.info(
                        f"🔧 Korrigiere {len(ner_annotations)} NER-Annotationen..."
                    )

                    # Position-Korrektur anwenden
                    corrected_ner = corrector.correct_batch_annotations(
                        full_text, ner_annotations
                    )

                    # Validierung durchführen
                    validation_report = corrector.validate_annotations(
                        corrected_ner, full_text
                    )

                    logger.info(
                        f"✅ Position-Korrektur: {validation_report['valid_count']}/{len(ner_annotations)} "
                        f"valide Annotationen, Qualität: {validation_report['quality_score']:.2f}"
                    )

                    # Kombiniere korrigierte und TextCat-Annotationen
                    final_annotations = corrected_ner + textcat_annotations
                else:
                    # Fallback ohne Korrektur
                    final_annotations = raw_annotations
                    logger.warning(
                        "⚠️ Keine Text-Korrektur möglich - verwende rohe Annotationen"
                    )
            else:
                final_annotations = raw_annotations

            logger.info(
                f"✅ spaCy-Annotationen erstellt: {len(final_annotations)} Einträge"
            )
            return final_annotations

        except Exception as e:
            logger.warning(f"⚠️ spaCy-Annotation fehlgeschlagen: {e}")
            return []

    def _extract_full_text_from_result(self, structured_result: dict[str, Any]) -> str:
        """
        Extrahiert vollständigen Text aus Gemini-Ergebnissen für Position-Korrektur.

        Args:
            structured_result: Gemini-Analyseergebnisse

        Returns:
            Vollständiger Text oder leerer String
        """
        try:
            # Suche nach vollständigem Text in verschiedenen Feldern
            full_text: str = structured_result.get("raw_text", "")

            if not full_text:
                # Fallback: Konstruiere Text aus Rechnungsheader und Items
                header = structured_result.get("invoice_header", {})
                items = structured_result.get("line_items", [])

                text_parts = []

                # Header-Informationen
                if header.get("lieferant"):
                    text_parts.append(f"Lieferant: {header['lieferant']}")
                if header.get("rechnungsnummer"):
                    text_parts.append(f"Rechnung: {header['rechnungsnummer']}")

                # Item-Beschreibungen
                for item in items:
                    if item.get("beschreibung"):
                        text_parts.append(item["beschreibung"])

                full_text = "\n".join(text_parts)

            return full_text

        except Exception as e:
            logger.warning(
                f"⚠️ Text-Extraktion für Position-Korrektur fehlgeschlagen: {e}"
            )
            return ""

    def _store_in_rag_system_robust(
        self,
        structured_result: dict[str, Any],
        pdf_path: Path,
    ) -> None:
        """
        Robuste RAG-Speicherung mit Error Handling.

        Args:
            structured_result: Strukturierte Analyseergebnisse
            pdf_path: PDF-Pfad für Metadaten
        """
        try:
            # Erstelle temporäres dict im erwarteten Format für bestehende Methode
            temp_result = {
                "timestamp": datetime.now().isoformat(),
                "skr03_classifications": [],
                "line_items": structured_result.get("line_items", []),
            }

            self._store_in_rag_system(temp_result)
            logger.info("✅ RAG-Speicherung erfolgreich")

        except Exception as e:
            logger.warning(f"⚠️ RAG-Speicherung nicht kritisch, fortfahren: {e}")

    def _write_error_log_detailed(
        self,
        pdf_path: Path,
        error: Exception,
        context: dict[str, Any],
    ) -> None:
        """
        Detailliertes Error-Logging mit Kontext für Debugging.

        Args:
            pdf_path: PDF-Pfad
            error: Exception-Objekt
            context: Zusätzlicher Fehlerkontext
        """
        try:
            log_dir = Path("logs/gemini_errors")
            log_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir / f"gemini_error_{timestamp}_{pdf_path.stem}.log"

            # Erweiterte Error-Details
            error_details = {
                "timestamp": datetime.now().isoformat(),
                "pdf_file": str(pdf_path),
                "pipeline": "gemini_first",
                "error_type": type(error).__name__,
                "error_message": str(error),
                "error_chain": str(error.__cause__) if error.__cause__ else None,
                "context": context,
                "gemini_model": self.config.gemini_model,
                "api_key_available": bool(self.config.google_api_key),
                "resource_usage": get_resource_manager().get_memory_report(),
            }

            with open(log_file, "w", encoding="utf-8") as f:
                json.dump(error_details, f, indent=2, ensure_ascii=False)

            logger.info(f"📝 Detailliertes Error-Log erstellt: {log_file}")

        except Exception as log_error:
            logger.error(f"❌ Error-Logging fehlgeschlagen: {log_error}")

    def _write_error_log(self, pdf_path: Path, error_message: str) -> None:
        """
        Schreibt detailliertes Error-Log bei Gemini-Fehlern.
        """
        try:
            log_dir = Path("logs/gemini_errors")
            log_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir / f"gemini_error_{timestamp}_{pdf_path.stem}.log"

            error_details = {
                "timestamp": datetime.now().isoformat(),
                "pdf_file": str(pdf_path),
                "pipeline": "gemini_first",
                "error_message": error_message,
                "gemini_model": self.config.gemini_model,
                "api_key_available": bool(self.config.google_api_key),
            }

            with open(log_file, "w", encoding="utf-8") as f:
                f.write("LLKJJ ML Pipeline - Gemini-First Error Log\n")
                f.write("=" * 50 + "\n\n")
                f.write(json.dumps(error_details, indent=2, ensure_ascii=False))
                f.write("\n\nKEIN FALLBACK AUF DOCLING - Fehler muss behoben werden!\n")

            logger.error("❌ Error-Log geschrieben: %s", log_file)

        except Exception as log_error:
            logger.error("❌ Konnte Error-Log nicht schreiben: %s", log_error)


class DoclingAlternativeProcessor:
    """
    Alternative Docling-basierte Verarbeitung.
    Nur noch bei expliziter Anfrage verfügbar!
    """

    def __init__(self, config: ConfigBridge | None = None):
        """Initialize Docling alternative processor"""
        self.config = config or config_instance
        logger.info(
            "DoclingAlternativeProcessor initialisiert (nur bei expliziter Anfrage)"
        )

    def process_pdf_with_docling(self, pdf_path: str | Path) -> dict[str, Any]:
        """
        Alternative Docling-Verarbeitung (explizit angefordert).

        Args:
            pdf_path: Pfad zur PDF-Datei

        Returns:
            Docling-basierte Extraktionsergebnisse
        """
        logger.info("🔄 ALTERNATIVE: Docling-Verarbeitung explizit angefordert")

        # Importiere den bestehenden UnifiedProcessor für Docling-Verarbeitung
        from .unified_processor import UnifiedProcessor

        processor = UnifiedProcessor(self.config)
        result = processor.process_pdf(pdf_path)

        # Konvertiere ProcessingResult zu dict für Konsistenz
        result_dict = {
            "processing_method": "docling_alternative",
            "pdf_path": result.pdf_path,
            "timestamp": result.processing_timestamp,
            "raw_text": result.raw_text,
            "structured_data": result.structured_data,
            "invoice_header": result.invoice_header,
            "skr03_classifications": result.skr03_classifications,
            "processing_time_ms": result.processing_time_ms,
            "ocr_time_ms": result.ocr_time_ms,
            "classification_time_ms": result.classification_time_ms,
            "confidence_score": result.confidence_score,
            "extraction_quality": result.extraction_quality,
        }

        logger.info("✅ ALTERNATIVE: Docling-Verarbeitung abgeschlossen")
        return result_dict
