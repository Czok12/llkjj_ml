#!/usr/bin/env python3
"""
LLKJJ ML Pipeline - Processing Strategy Pattern
==============================================

Abstrakte Strategy-Pattern-Implementation für nahtlose Transition zwischen
verschiedenen Processing-Engines (Gemini → spaCy/RAG).

Vorbereitung für Phase 2: Lokale Autonomie nach Gemini-Training-Phase.

Autor: LLKJJ ML Pipeline Team
Version: 4.2.0 (Strategy Pattern)
Datum: 18. August 2025
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ..models.processing_result import ProcessingResult
from ..settings_bridge import ConfigBridge

logger = logging.getLogger(__name__)


class ProcessingStrategy(ABC):
    """
    Abstrakte Basis-Klasse für alle Processing-Engines.

    Ermöglicht nahtlose Transition zwischen:
    - GeminiStrategy (Phase 1: AI-powered)
    - SpacyRagStrategy (Phase 2: Local autonomous)
    - HybridStrategy (Phase 3: Best-of-both)
    """

    def __init__(self, config: ConfigBridge):
        self.config = config
        self.name = self.__class__.__name__
        logger.debug("🔧 Strategy initialisiert: %s", self.name)

    @abstractmethod
    def process_pdf(self, pdf_path: str | Path) -> ProcessingResult:
        """
        Verarbeite eine einzelne PDF-Datei.

        Args:
            pdf_path: Pfad zur PDF-Datei

        Returns:
            ProcessingResult mit einheitlicher Struktur

        Raises:
            FileNotFoundError: PDF nicht gefunden
            ValueError: Ungültige PDF oder Konfiguration
            RuntimeError: Processing-Fehler
        """
        pass

    @abstractmethod
    def get_strategy_info(self) -> dict[str, Any]:
        """
        Liefert Informationen über die Strategy.

        Returns:
            Dict mit Strategy-Metadaten (name, version, capabilities)
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        Prüft, ob die Strategy verfügbar ist.

        Returns:
            True wenn Strategy funktionsfähig, False sonst
        """
        pass

    def validate_pdf(self, pdf_path: Path) -> None:
        """
        Gemeinsame PDF-Validierung für alle Strategies.

        Args:
            pdf_path: Pfad zur PDF-Datei

        Raises:
            FileNotFoundError: PDF nicht gefunden
            ValueError: Ungültige PDF-Datei
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF-Datei nicht gefunden: {pdf_path}")

        if not pdf_path.suffix.lower() == ".pdf":
            raise ValueError(f"Nur PDF-Dateien werden unterstützt: {pdf_path.suffix}")

        if pdf_path.stat().st_size == 0:
            raise ValueError(f"PDF-Datei ist leer: {pdf_path}")

        # Größe-Check
        max_size_bytes = self.config.max_pdf_size_mb * 1024 * 1024
        if pdf_path.stat().st_size > max_size_bytes:
            raise ValueError(
                f"PDF-Datei zu groß: {pdf_path.stat().st_size / 1024 / 1024:.1f}MB "
                f"(Max: {self.config.max_pdf_size_mb}MB)"
            )

    def log_processing_start(self, pdf_path: Path) -> None:
        """Einheitliches Logging für Processing-Start."""
        logger.info(
            "🚀 %s Processing startet: %s (%.1f KB)",
            self.name,
            pdf_path.name,
            pdf_path.stat().st_size / 1024,
        )

    def log_processing_end(self, pdf_path: Path, result: ProcessingResult) -> None:
        """Einheitliches Logging für Processing-Ende."""
        logger.info(
            "✅ %s Processing abgeschlossen: %s -> %d SKR03, %.1f%% Konfidenz, %s Qualität",
            self.name,
            pdf_path.name,
            len(result.skr03_classifications),
            result.confidence_score * 100,
            result.extraction_quality,
        )
