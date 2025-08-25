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
    from src.models.processing_result import ProcessingResult as BaseProcessingResult
    from src.pipeline.unified_processor import UnifiedProcessor

    LEGACY_COMPONENTS_AVAILABLE = True
except ImportError:
    LEGACY_COMPONENTS_AVAILABLE = False
    from typing import TYPE_CHECKING

    if TYPE_CHECKING:
        from src.models.processing_result import (
            ProcessingResult as BaseProcessingResult,
        )
        from src.pipeline.unified_processor import UnifiedProcessor
    else:
        BaseProcessingResult = None  # type: ignore[misc,assignment]
        UnifiedProcessor = None  # type: ignore[misc,assignment]

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


# MLResourceManager removed in v2.0.0 final cleanup - use dependency injection instead


class MLProcessor:
    """
    REMOVED: Legacy MLProcessor has been removed in v2.0.0 final cleanup
    
    ⚠️ This class has been completely replaced by llkjj_ml.MLPlugin (v2.0).
    
    Use llkjj_ml.MLPlugin with Repository-Pattern and dependency injection instead.
    See llkjj_ml_plugin_v2.py for the new stateless implementation.
    """
    
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "MLProcessor has been removed in v2.0.0. "
            "Use llkjj_ml.MLPlugin (v2.0) with Repository-Pattern instead."
        )


# DEPRECATED: Legacy convenience function
def process_pdf_simple(pdf_path: str, config: dict | None = None) -> dict:
    """
    REMOVED: Legacy process_pdf_simple has been removed in v2.0.0 final cleanup
    
    ⚠️ This function has been completely replaced by llkjj_ml.MLPlugin (v2.0).
    
    Use MLPlugin.process_invoice_pdf() with Repository-Pattern instead.
    """
    raise RuntimeError(
        "process_pdf_simple() has been removed in v2.0.0. "
        "Use MLPlugin (v2.0).process_invoice_pdf() with Repository-Pattern instead."
    )
