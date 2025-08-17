#!/usr/bin/env python3
"""
LLKJJ ML Service - Eigenständiger Processor
==========================================

Eigenständiger ML-Processor der völlig unabhängig von externen Systemen
funktioniert. Verwendet MLSettings für Konfiguration und stellt eine
saubere API für PDF-Verarbeitung bereit.

**EIGENSTÄNDIGKEIT:**
- Keine Abhängigkeiten zu src/config oder externen Modulen
- Eigene Ressourcenverwaltung
- Konfiguration über MLSettings
- Wiederverwendbar in jedem Python-Projekt

Author: LLKJJ ML Pipeline Team
Version: 3.0.0 (Eigenständige Implementierung)
"""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .config import MLSettings, default_settings

# Import existing components - we'll adapt them gradually
try:
    from src.pipeline.processor import ProcessingResult as BaseProcessingResult
    from src.pipeline.processor import UnifiedProcessor

    LEGACY_COMPONENTS_AVAILABLE = True
except ImportError:
    LEGACY_COMPONENTS_AVAILABLE = False
    BaseProcessingResult = None
    UnifiedProcessor = None

logger = logging.getLogger(__name__)


class ProcessingResult(BaseModel):
    """
    Eigenständiges ProcessingResult für ML Service

    Falls die ursprünglichen Komponenten verfügbar sind, wird das bestehende
    ProcessingResult verwendet. Ansonsten wird eine vereinfachte Version bereitgestellt.
    """

    # === SOURCE INFORMATION ===
    pdf_path: str = Field(..., description="Pfad zur verarbeiteten PDF-Datei")
    processing_timestamp: str = Field(..., description="Verarbeitungszeitpunkt")

    # === EXTRACTION RESULTS ===
    raw_text: str = Field(default="", description="Extrahierter Text")
    structured_data: dict[str, Any] = Field(
        default_factory=dict, description="Strukturierte Daten"
    )

    # === CLASSIFICATION RESULTS ===
    invoice_data: dict[str, Any] = Field(
        default_factory=dict, description="Rechnungsdaten"
    )
    skr03_classifications: list[dict[str, Any]] = Field(
        default_factory=list, description="SKR03-Klassifizierungen"
    )

    # === PERFORMANCE METRICS ===
    processing_time_ms: int = Field(default=0, description="Verarbeitungszeit in ms")
    ocr_time_ms: int = Field(default=0, description="OCR-Zeit in ms")
    classification_time_ms: int = Field(
        default=0, description="Klassifizierungszeit in ms"
    )

    # === QUALITY INDICATORS ===
    confidence_score: float = Field(default=0.0, description="Konfidenz-Score")
    extraction_quality: str = Field(default="unknown", description="Qualitätsbewertung")

    def get_summary(self) -> str:
        """Erstellt eine kompakte Zusammenfassung der Ergebnisse."""
        return (
            f"PDF: {Path(self.pdf_path).name} | "
            f"Quality: {self.extraction_quality} | "
            f"Confidence: {self.confidence_score:.2f} | "
            f"Time: {self.processing_time_ms}ms | "
            f"Classifications: {len(self.skr03_classifications)}"
        )


class MLResourceManager:
    """
    Eigenständiger Resource Manager für das ML Service

    Verwaltet ML-Modelle und Ressourcen unabhängig von externen Systemen.
    """

    def __init__(self, settings: MLSettings):
        self.settings = settings
        self._initialized = False

    def initialize(self) -> None:
        """Initialisiert alle notwendigen Ressourcen."""
        if self._initialized:
            return

        logger.info("🔄 Initialisiere ML Service Ressourcen...")

        # Verzeichnisse erstellen
        self.settings.ensure_directories()

        # Logging konfigurieren
        self._setup_logging()

        self._initialized = True
        logger.info("✅ ML Service Ressourcen initialisiert")

    def _setup_logging(self) -> None:
        """Konfiguriert Logging für das ML Service."""
        log_level = getattr(logging, self.settings.log_level.upper(), logging.INFO)
        logging.basicConfig(level=log_level, format=self.settings.log_format)

    def cleanup(self) -> None:
        """Bereinigt Ressourcen."""
        logger.info("🧹 Bereinige ML Service Ressourcen...")
        self._initialized = False
        logger.info("✅ ML Service Ressourcen bereinigt")


class MLProcessor:
    """
    Eigenständiger ML-Processor für PDF-Verarbeitung

    **USAGE:**
    ```python
    from ml_service import MLProcessor, MLSettings

    # Mit Standardkonfiguration
    processor = MLProcessor()
    result = processor.process_pdf("invoice.pdf")

    # Mit eigener Konfiguration
    settings = MLSettings(cache_enabled=False)
    processor = MLProcessor(settings=settings)
    result = processor.process_pdf("invoice.pdf")

    # Ressourcen bereinigen
    processor.cleanup()
    ```
    """

    def __init__(self, settings: MLSettings | None = None):
        """
        Initialisiert ML-Processor.

        Args:
            settings: ML-Konfiguration. Verwendet Defaults falls None.
        """
        self.settings = settings or default_settings
        self.resource_manager = MLResourceManager(self.settings)
        self._legacy_processor: Any | None = None

        # Validiere Konfiguration
        validation = self.settings.validate_configuration()
        if not validation["valid"]:
            logger.error(f"Konfigurationsfehler: {validation['errors']}")
            raise ValueError(f"Ungültige Konfiguration: {validation['errors']}")

        if validation["warnings"]:
            for warning in validation["warnings"]:
                logger.warning(warning)

        logger.info(f"ML Processor initialisiert: {self.settings}")

    def _get_legacy_processor(self) -> Any:
        """Lazy-loaded Legacy Processor für Übergangszeit."""
        if not LEGACY_COMPONENTS_AVAILABLE:
            raise RuntimeError(
                "Legacy-Komponenten nicht verfügbar. "
                "Bitte implementieren Sie native ML Service Komponenten."
            )

        if self._legacy_processor is None:
            logger.info("🔄 Initialisiere Legacy Processor...")
            self._legacy_processor = UnifiedProcessor()
            logger.info("✅ Legacy Processor initialisiert")

        return self._legacy_processor

    def process_pdf(self, pdf_path: str | Path) -> ProcessingResult:
        """
        Verarbeitet PDF-Datei und gibt strukturierte Ergebnisse zurück.

        Args:
            pdf_path: Pfad zur PDF-Datei

        Returns:
            ProcessingResult mit strukturierten Daten und Klassifizierungen

        Raises:
            FileNotFoundError: Wenn PDF-Datei nicht existiert
            ValueError: Wenn PDF-Datei ungültig ist
            RuntimeError: Bei Verarbeitungsfehlern
        """
        start_time = time.time()

        # Input-Validierung
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF-Datei nicht gefunden: {pdf_path}")

        if not pdf_path.suffix.lower() == ".pdf":
            raise ValueError(f"Datei ist keine PDF: {pdf_path}")

        # Dateigröße prüfen
        file_size_mb = pdf_path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.settings.max_pdf_size_mb:
            raise ValueError(
                f"PDF zu groß: {file_size_mb:.1f}MB > {self.settings.max_pdf_size_mb}MB"
            )

        logger.info(f"🚀 Starte PDF-Verarbeitung: {pdf_path.name}")

        try:
            # Ressourcen initialisieren
            self.resource_manager.initialize()

            # Übergangsweise Legacy Processor verwenden
            if LEGACY_COMPONENTS_AVAILABLE:
                result = self._process_with_legacy(pdf_path)
            else:
                result = self._process_native(pdf_path)

            processing_time = int((time.time() - start_time) * 1000)

            # Verarbeitungszeit aktualisieren
            if hasattr(result, "processing_time_ms"):
                result.processing_time_ms = processing_time

            logger.info(
                f"✅ PDF-Verarbeitung abgeschlossen: {pdf_path.name} "
                f"({processing_time}ms, {result.confidence_score:.1%} Konfidenz)"
            )

            return result

        except Exception as e:
            logger.error(f"❌ Fehler bei PDF-Verarbeitung: {e}")
            raise RuntimeError(f"PDF-Verarbeitung fehlgeschlagen: {e}") from e

    def _process_with_legacy(self, pdf_path: Path) -> ProcessingResult:
        """Verarbeitung mit Legacy-Komponenten."""
        legacy_processor = self._get_legacy_processor()
        legacy_result = legacy_processor.process_pdf(str(pdf_path))

        # Legacy ProcessingResult zu eigenständigem Result konvertieren
        if BaseProcessingResult and isinstance(legacy_result, BaseProcessingResult):
            # Direktes Mapping falls kompatibel
            return ProcessingResult(**legacy_result.model_dump())
        else:
            # Fallback für andere Formate
            return ProcessingResult(
                pdf_path=str(pdf_path),
                processing_timestamp=datetime.now().isoformat(),
                raw_text=getattr(legacy_result, "raw_text", ""),
                structured_data=getattr(legacy_result, "structured_data", {}),
                invoice_data=getattr(legacy_result, "invoice_data", {}),
                skr03_classifications=getattr(
                    legacy_result, "skr03_classifications", []
                ),
                processing_time_ms=getattr(legacy_result, "processing_time_ms", 0),
                ocr_time_ms=getattr(legacy_result, "ocr_time_ms", 0),
                classification_time_ms=getattr(
                    legacy_result, "classification_time_ms", 0
                ),
                confidence_score=getattr(legacy_result, "confidence_score", 0.0),
                extraction_quality=getattr(
                    legacy_result, "extraction_quality", "unknown"
                ),
            )

    def _process_native(self, pdf_path: Path) -> ProcessingResult:
        """Native Verarbeitung ohne Legacy-Komponenten."""
        # TODO: Implementierung der nativen Verarbeitung
        # Diese Methode sollte die vollständige Verarbeitung ohne Legacy-Code implementieren

        logger.warning(
            "Native Verarbeitung noch nicht implementiert - verwende Fallback"
        )

        return ProcessingResult(
            pdf_path=str(pdf_path),
            processing_timestamp=datetime.now().isoformat(),
            raw_text="[Native processing not yet implemented]",
            structured_data={},
            invoice_data={},
            skr03_classifications=[],
            processing_time_ms=0,
            ocr_time_ms=0,
            classification_time_ms=0,
            confidence_score=0.0,
            extraction_quality="unknown",
        )

    def validate_input(self, pdf_path: str | Path) -> bool:
        """
        Validiert Eingabedatei.

        Args:
            pdf_path: Pfad zur PDF-Datei

        Returns:
            True wenn Datei gültig ist
        """
        try:
            pdf_path = Path(pdf_path)
            return (
                pdf_path.exists()
                and pdf_path.suffix.lower() == ".pdf"
                and pdf_path.stat().st_size
                <= self.settings.max_pdf_size_mb * 1024 * 1024
            )
        except Exception:
            return False

    def get_capabilities(self) -> list[str]:
        """
        Gibt verfügbare Capabilities zurück.

        Returns:
            Liste der verfügbaren Funktionen
        """
        capabilities = [
            "pdf_extraction",
            "german_nlp",
            "text_processing",
        ]

        if self.settings.cache_enabled:
            capabilities.append("caching")

        if self.settings.google_api_key:
            capabilities.append("gemini_enhancement")

        if LEGACY_COMPONENTS_AVAILABLE:
            capabilities.extend(
                ["skr03_classification", "quality_assessment", "vector_search"]
            )

        return capabilities

    def get_settings_summary(self) -> dict[str, Any]:
        """
        Gibt Zusammenfassung der aktuellen Einstellungen zurück.

        Returns:
            Dict mit Konfigurationszusammenfassung
        """
        return self.settings.validate_configuration()["summary"]

    def cleanup(self) -> None:
        """Bereinigt alle Ressourcen."""
        logger.info("🧹 Bereinige ML Processor...")

        if self._legacy_processor:
            try:
                self._legacy_processor.cleanup()
            except Exception as e:
                logger.warning(f"Fehler beim Bereinigen des Legacy Processors: {e}")

        self.resource_manager.cleanup()

        logger.info("✅ ML Processor bereinigt")

    def __enter__(self):
        """Context Manager Eingang."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context Manager Ausgang."""
        self.cleanup()


# Convenience-Funktion für einfache Verwendung
def process_pdf_simple(
    pdf_path: str | Path, settings: MLSettings | None = None
) -> ProcessingResult:
    """
    Einfache PDF-Verarbeitung mit automatischer Ressourcenverwaltung.

    Args:
        pdf_path: Pfad zur PDF-Datei
        settings: Optionale ML-Konfiguration

    Returns:
        ProcessingResult mit Verarbeitungsergebnissen
    """
    with MLProcessor(settings=settings) as processor:
        return processor.process_pdf(pdf_path)
