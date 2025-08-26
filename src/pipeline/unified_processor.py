#!/usr/bin/env python3
"""
LLKJJ ML Pipeline - Unified Processor (Strategy Pattern)
=======================================================

Einheitlicher Processor mit automatischer Strategy-Auswahl.
Ermöglicht nahtlose Transition zwischen Processing-Engines.

Strategien:
- GeminiStrategy (Phase 1: AKTUELL)
- SpacyRagStrategy (Phase 2: PLACEHOLDER)
- HybridStrategy (Phase 3: ZUKUNFT)

Autor: LLKJJ ML Pipeline Team
Version: 4.2.0 (Strategy Pattern)
Datum: 18. August 2025
"""

import logging
from pathlib import Path
from typing import Any, Literal

from src.models.processing_result import ProcessingResult
from src.pipeline.gemini_strategy import GeminiStrategy
from src.pipeline.processing_strategy import ProcessingStrategy
from src.pipeline.spacy_rag_strategy import SpacyRagStrategy
from src.settings_bridge import Config

logger = logging.getLogger(__name__)

StrategyType = Literal["gemini", "spacy_rag", "hybrid", "auto"]


class UnifiedProcessor:
    """
    Einheitlicher Processor mit Strategy-Pattern für nahtlose Engine-Transition.

    Features:
    - Automatische Strategy-Auswahl basierend auf Verfügbarkeit
    - Explizite Strategy-Auswahl für Tests/Vergleiche
    - Fallback-Mechanismus bei Strategy-Fehlern
    - Einheitliche API für alle Processing-Engines
    """

    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self._strategies: dict[str, ProcessingStrategy] = {}
        self._initialize_strategies()

        logger.info("🔧 UnifiedProcessor initialisiert mit Strategy-Pattern")

    def _initialize_strategies(self) -> None:
        """Initialisiere alle verfügbaren Strategies."""
        try:
            # GeminiStrategy (Phase 1)
            self._strategies["gemini"] = GeminiStrategy(self.config)
            logger.debug("✅ GeminiStrategy geladen")
        except Exception as e:
            logger.warning("❌ GeminiStrategy konnte nicht geladen werden: %s", e)

        try:
            # SpacyRagStrategy (Phase 2 Placeholder)
            self._strategies["spacy_rag"] = SpacyRagStrategy(self.config)
            logger.debug("🔮 SpacyRagStrategy geladen (Placeholder)")
        except Exception as e:
            logger.warning("❌ SpacyRagStrategy konnte nicht geladen werden: %s", e)

        # Log verfügbare Strategies
        available = [
            name
            for name, strategy in self._strategies.items()
            if strategy.is_available()
        ]
        logger.info("📊 Verfügbare Strategies: %s", available)

    def get_available_strategies(self) -> list[str]:
        """
        Liste aller verfügbaren Strategies.

        Returns:
            Liste der verfügbaren Strategy-Namen
        """
        return [
            name
            for name, strategy in self._strategies.items()
            if strategy.is_available()
        ]

    def get_default_strategy(self) -> str:
        """
        Ermittelt die beste verfügbare Default-Strategy.

        Priorität:
        1. spacy_rag (Phase 2) - wenn verfügbar und trainiert
        2. gemini (Phase 1) - aktueller Standard
        3. Fallback zu erster verfügbarer Strategy

        Returns:
            Name der Default-Strategy
        """
        available = self.get_available_strategies()

        if not available:
            raise RuntimeError("❌ Keine Processing-Strategy verfügbar!")

        # Phase 2: SpaCy RAG bevorzugt (wenn verfügbar und trainiert)
        if "spacy_rag" in available:
            spacy_strategy = self._strategies["spacy_rag"]
            # Type-safe check für SpacyRagStrategy-specific method
            if hasattr(spacy_strategy, "check_training_data_readiness"):
                training_status = spacy_strategy.check_training_data_readiness()
                if training_status.get("training_data_sufficient", False):
                    logger.info(
                        "🎯 Default Strategy: spacy_rag (Phase 2: Local autonomous)"
                    )
                    return "spacy_rag"

        # Phase 1: Gemini bevorzugt (aktueller Standard)
        if "gemini" in available:
            logger.info("🎯 Default Strategy: gemini (Phase 1: AI-powered)")
            return "gemini"

        # Fallback zu erster verfügbarer Strategy
        default = available[0]
        logger.warning("⚠️ Fallback zu Strategy: %s", default)
        return default

    def process_pdf(
        self, pdf_path: str | Path, strategy: StrategyType = "auto"
    ) -> ProcessingResult:
        """
        Verarbeite PDF mit ausgewählter Strategy.

        Args:
            pdf_path: Pfad zur PDF-Datei
            strategy: Strategy-Auswahl ("auto", "gemini", "spacy_rag", "hybrid")

        Returns:
            ProcessingResult mit einheitlicher Struktur

        Raises:
            ValueError: Ungültige Strategy
            RuntimeError: Keine Strategy verfügbar
        """
        pdf_path = Path(pdf_path)

        # Strategy-Auswahl
        if strategy == "auto":
            selected_strategy = self.get_default_strategy()
        else:
            selected_strategy = strategy

        # Strategy-Validierung
        if selected_strategy not in self._strategies:
            available = list(self._strategies.keys())
            raise ValueError(
                f"Unbekannte Strategy: {selected_strategy}. " f"Verfügbar: {available}"
            )

        strategy_instance = self._strategies[selected_strategy]

        if not strategy_instance.is_available():
            # Fallback zu Default-Strategy
            fallback_strategy = self.get_default_strategy()
            if fallback_strategy != selected_strategy:
                logger.warning(
                    "⚠️ Strategy %s nicht verfügbar, Fallback zu %s",
                    selected_strategy,
                    fallback_strategy,
                )
                strategy_instance = self._strategies[fallback_strategy]
            else:
                raise RuntimeError(f"Strategy {selected_strategy} nicht verfügbar!")

        # Processing mit ausgewählter Strategy
        logger.info("🚀 Processing mit %s Strategy", strategy_instance.name)

        try:
            result = strategy_instance.process_pdf(pdf_path)

            # Strategy-Info zu Result hinzufügen
            result.processing_method = selected_strategy  # type: ignore[assignment]

            logger.info("✅ %s Strategy erfolgreich", strategy_instance.name)
            return result

        except Exception as e:
            logger.error("❌ %s Strategy fehlgeschlagen: %s", strategy_instance.name, e)

            # Fallback-Mechanismus
            if (
                selected_strategy != "gemini"
                and "gemini" in self.get_available_strategies()
            ):
                logger.warning("🔄 Fallback zu GeminiStrategy...")
                fallback_result = self._strategies["gemini"].process_pdf(pdf_path)
                fallback_result.processing_method = "gemini_fallback"
                return fallback_result
            else:
                raise

    def get_strategy_info(self, strategy_name: str | None = None) -> dict[str, Any]:
        """
        Informationen über Strategy(s).

        Args:
            strategy_name: Spezifische Strategy oder None für alle

        Returns:
            Dict mit Strategy-Informationen
        """
        if strategy_name:
            if strategy_name not in self._strategies:
                raise ValueError(f"Unbekannte Strategy: {strategy_name}")

            strategy = self._strategies[strategy_name]
            info = strategy.get_strategy_info()
            info["is_available"] = strategy.is_available()
            return info

        # Alle Strategies
        all_info = {}
        for name, strategy in self._strategies.items():
            info = strategy.get_strategy_info()
            info["is_available"] = strategy.is_available()
            all_info[name] = info

        # Unified Processor Info hinzufügen
        all_info["unified_processor"] = {
            "version": "4.2.0",
            "default_strategy": self.get_default_strategy(),
            "available_strategies": self.get_available_strategies(),
            "total_strategies": len(self._strategies),
            "strategy_pattern": "enabled",
        }

        return all_info

    def compare_strategies(self, pdf_path: str | Path) -> dict[str, Any]:
        """
        Vergleiche verfügbare Strategies auf derselben PDF.

        Nützlich für:
        - Performance-Benchmarking
        - Genauigkeits-Vergleiche
        - Strategy-Auswahl-Entscheidungen

        Args:
            pdf_path: Test-PDF für Vergleich

        Returns:
            Dict mit Vergleichsergebnissen
        """
        pdf_path = Path(pdf_path)
        available_strategies = self.get_available_strategies()

        if len(available_strategies) < 2:
            logger.warning("⚠️ Weniger als 2 Strategies verfügbar für Vergleich")
            return {"error": "Mindestens 2 Strategies für Vergleich erforderlich"}

        comparison_results = {}

        for strategy_name in available_strategies:
            try:
                logger.info("🔄 Teste %s Strategy...", strategy_name)
                result = self.process_pdf(pdf_path, strategy_name)  # type: ignore[arg-type]

                comparison_results[strategy_name] = {
                    "success": True,
                    "result": result,
                    "processing_time_ms": result.processing_time_ms,
                    "confidence_score": result.confidence_score,
                    "extraction_quality": result.extraction_quality,
                    "skr03_classifications": len(result.skr03_classifications),
                    "total_line_items": len(result.line_items),
                }

            except Exception as e:
                logger.error("❌ %s Strategy fehlgeschlagen: %s", strategy_name, e)
                comparison_results[strategy_name] = {"success": False, "error": str(e)}

        # Zusammenfassung
        successful_strategies: list[str] = [
            k for k, v in comparison_results.items() if v.get("success")
        ]

        summary: dict[str, Any] = {
            "comparison_results": comparison_results,
            "summary": {
                "total_strategies_tested": len(available_strategies),
                "successful_strategies": len(successful_strategies),
                "fastest_strategy": None,
                "highest_confidence": None,
                "most_classifications": None,
            },
        }

        # Best-Performance-Analyse
        if successful_strategies:
            # Fastest
            fastest: str = min(
                successful_strategies,
                key=lambda x: comparison_results[x]["processing_time_ms"],  # type: ignore[arg-type, return-value]
            )
            summary["summary"]["fastest_strategy"] = fastest

            # Highest confidence
            highest_conf: str = max(
                successful_strategies,
                key=lambda x: comparison_results[x]["confidence_score"],  # type: ignore[arg-type, return-value]
            )
            summary["summary"]["highest_confidence"] = highest_conf

            # Most classifications
            most_class = max(
                successful_strategies,
                key=lambda x: comparison_results[x]["skr03_classifications"],  # type: ignore[arg-type, return-value]
            )
            summary["summary"]["most_classifications"] = most_class

        return summary
