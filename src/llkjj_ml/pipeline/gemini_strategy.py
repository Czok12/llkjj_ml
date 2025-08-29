#!/usr/bin/env python3
"""
LLKJJ ML Pipeline - Gemini Strategy Implementation
================================================

Konkrete Strategy-Implementation für Gemini-First Processing.
Wrapper um bestehenden GeminiDirectProcessor mit Strategy-Pattern.

Autor: LLKJJ ML Pipeline Team
Version: 4.2.0 (Strategy Pattern)
Datum: 18. August 2025
"""

import logging
from pathlib import Path
from typing import Any

from ..models.processing_result import ProcessingResult
from ..pipeline.gemini_first_processor import GeminiDirectProcessor
from ..pipeline.processing_strategy import ProcessingStrategy
from ..settings_bridge import ConfigBridge

logger = logging.getLogger(__name__)


class GeminiStrategy(ProcessingStrategy):
    """
    Strategy-Implementation für Gemini-First Processing.

    Phase 1 (AKTUELL): Gemini AI als produktive Intelligence-Engine
    - Wrapper um GeminiDirectProcessor
    - Einheitliche Strategy-API
    - Training Data Collection für Phase 2
    """

    def __init__(self, config: ConfigBridge):
        super().__init__(config)
        self._processor: GeminiDirectProcessor | None = None
        logger.info("🎯 GeminiStrategy initialisiert (Phase 1: AI-powered)")

    @property
    def processor(self) -> GeminiDirectProcessor:
        """Lazy-loaded GeminiDirectProcessor."""
        if self._processor is None:
            self._processor = GeminiDirectProcessor(self.config)
        return self._processor

    def process_pdf(self, pdf_path: str | Path) -> ProcessingResult:
        """
        Verarbeite PDF mit Gemini-First Strategy.

        Args:
            pdf_path: Pfad zur PDF-Datei

        Returns:
            ProcessingResult mit Gemini-Analysis
        """
        pdf_path = Path(pdf_path)

        # Gemeinsame Validierung
        self.validate_pdf(pdf_path)

        # Strategy-spezifisches Logging
        self.log_processing_start(pdf_path)

        # Delegate to GeminiDirectProcessor
        result = self.processor.process_pdf_gemini_first(pdf_path)

        # Strategy-spezifisches End-Logging
        self.log_processing_end(pdf_path, result)

        return result

    def get_strategy_info(self) -> dict[str, Any]:
        """Gemini Strategy Information."""
        return {
            "name": "GeminiStrategy",
            "version": "4.2.0",
            "phase": "Phase 1: AI-powered",
            "engine": "Gemini 2.5 Pro",
            "capabilities": [
                "direct_pdf_analysis",
                "structured_extraction",
                "skr03_classification",
                "training_data_collection",
                "pydantic_validation",
                "rag_enhancement",
            ],
            "dependencies": ["google.genai", "chromadb", "sentence_transformers"],
            "performance": {
                "avg_processing_time_ms": 5000,
                "supports_async": True,
                "supports_batch": True,
                "supports_caching": True,
            },
            "training_data_collection": True,
            "transition_ready": True,  # Bereit für Phase 2 Transition
        }

    def is_available(self) -> bool:
        """
        Prüft Gemini Strategy Verfügbarkeit.

        Returns:
            True wenn Gemini API verfügbar
        """
        try:
            # Check API Key
            if not self.config.google_api_key:
                logger.warning("❌ GeminiStrategy: Google API Key nicht konfiguriert")
                return False

            # Check if processor can be initialized
            _ = self.processor
            logger.debug("✅ GeminiStrategy verfügbar")
            return True

        except Exception as e:
            logger.warning("❌ GeminiStrategy nicht verfügbar: %s", e)
            return False

    def get_training_data_stats(self) -> dict[str, Any]:
        """
        Statistiken zur Training Data Collection.

        Returns:
            Dict mit Training-Daten-Statistiken
        """
        try:
            # Access training persistence service

            # Return aggregated stats (placeholder implementation)
            return {
                "total_processed_documents": "unknown",  # Would need persistent tracking
                "spacy_annotations_collected": "unknown",
                "rag_embeddings_stored": "unknown",
                "audit_entries": "unknown",
                "collection_active": True,
                "phase_2_readiness": "collecting_data",
            }

        except Exception as e:
            logger.warning("❌ Training data stats nicht verfügbar: %s", e)
            return {"collection_active": False, "error": str(e)}
