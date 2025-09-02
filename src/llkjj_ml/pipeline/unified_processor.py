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
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from llkjj_ml.models.processing_result import ProcessingResult
from llkjj_ml.optimization import batch_memory_optimizer
from llkjj_ml.settings_bridge import ConfigBridge, config_instance

from .gemini_strategy import GeminiStrategy
from .processing_strategy import ProcessingStrategy
from .spacy_rag_strategy import SpacyRagStrategy

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

    def __init__(self, config: ConfigBridge | None = None):
        self.config = config or config_instance
        self._strategies: dict[str, ProcessingStrategy] = {}
        self._memory_optimizer = batch_memory_optimizer.get_global_optimizer()
        self._initialize_strategies()

        logger.info(
            "🔧 UnifiedProcessor initialisiert mit Strategy-Pattern und Memory-Optimization"
        )

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

        # Processing mit ausgewählter Strategy und Memory-Optimization
        logger.info("🚀 Processing mit %s Strategy", strategy_instance.name)

        # Memory-optimized processing
        with self._memory_optimizer.memory_managed_processing():
            try:
                result = strategy_instance.process_pdf(pdf_path)

                # Strategy-Info zu Result hinzufügen
                result.processing_method = selected_strategy  # type: ignore[assignment]

                logger.info("✅ %s Strategy erfolgreich", strategy_instance.name)
                return result

            except Exception as e:
                logger.error(
                    "❌ %s Strategy fehlgeschlagen: %s", strategy_instance.name, e
                )

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

    def process_batch_optimized(
        self, pdf_paths: list[str | Path], strategy: StrategyType = "auto"
    ) -> list[ProcessingResult]:
        """
        Memory-optimized batch processing of multiple PDFs.

        Args:
            pdf_paths: List of PDF file paths
            strategy: Processing strategy to use

        Returns:
            List of ProcessingResults
        """
        batch_config = self._prepare_optimized_batch_configuration(pdf_paths)
        results = self._execute_memory_managed_batch_processing(batch_config, strategy)

        self._log_batch_optimization_completion(results)
        return results

    def _prepare_optimized_batch_configuration(
        self, pdf_paths: list[str | Path]
    ) -> dict:
        """
        Prepares optimized batch configuration based on memory constraints.

        Args:
            pdf_paths: List of PDF file paths

        Returns:
            Dictionary containing batch configuration parameters
        """
        pdf_paths = [Path(p) for p in pdf_paths]

        # Get optimal batch size based on memory
        optimal_batch_size, batch_sizes = self._memory_optimizer.suggest_batch_size(
            item_count=len(pdf_paths),
            memory_per_item_mb=50.0,  # Conservative estimate for PDF processing
        )

        logger.info(
            f"📊 Batch processing {len(pdf_paths)} PDFs in {len(batch_sizes)} batches"
        )

        return {
            "pdf_paths": pdf_paths,
            "optimal_batch_size": optimal_batch_size,
            "batch_sizes": batch_sizes,
            "total_pdfs": len(pdf_paths),
        }

    def _execute_memory_managed_batch_processing(
        self, batch_config: dict, strategy: StrategyType
    ) -> list[ProcessingResult]:
        """
        Executes batch processing with memory management between batches.

        Args:
            batch_config: Batch configuration parameters
            strategy: Processing strategy to use

        Returns:
            List of all processing results
        """
        results = []
        start_idx = 0

        for batch_num, batch_size in enumerate(batch_config["batch_sizes"], 1):
            batch_paths = batch_config["pdf_paths"][start_idx : start_idx + batch_size]

            logger.info(
                f"🔄 Processing batch {batch_num}/{len(batch_config['batch_sizes'])} ({len(batch_paths)} PDFs)"
            )

            # Process current batch with memory management
            batch_results = self._process_single_batch_with_memory_context(
                batch_paths, strategy
            )
            results.extend(batch_results)

            start_idx += batch_size

            # Optimize memory between batches if not the last batch
            if batch_num < len(batch_config["batch_sizes"]):
                self._optimize_inter_batch_memory()

        return results

    def _process_single_batch_with_memory_context(
        self, batch_paths: list[Path], strategy: StrategyType
    ) -> list[ProcessingResult]:
        """
        Processes a single batch within memory-managed context.

        Args:
            batch_paths: List of PDF paths for current batch
            strategy: Processing strategy to use

        Returns:
            List of processing results for this batch
        """
        # Memory-managed batch processing
        with self._memory_optimizer.memory_managed_processing():
            batch_results = []
            for pdf_path in batch_paths:
                try:
                    result = self.process_pdf(pdf_path, strategy)
                    batch_results.append(result)
                except Exception as e:
                    logger.error(f"❌ Failed to process {pdf_path}: {e}")
                    # Create error result with required fields
                    error_result = ProcessingResult(
                        pdf_path=str(pdf_path),
                        processing_timestamp=datetime.now(datetime.UTC).isoformat()
                        + "Z",
                        processing_method="gemini_first",
                        processing_time_ms=0,
                        confidence_score=0.0,
                        extraction_quality="poor",
                        skr03_classifications=[],
                    )
                    batch_results.append(error_result)

            return batch_results

    def _optimize_inter_batch_memory(self) -> None:
        """
        Optimizes memory usage between batch processing iterations.

        Note:
            Performs memory cleanup and logs optimization results
        """
        optimization_result = self._memory_optimizer.optimize_memory()
        logger.debug(
            f"🗑️ Inter-batch cleanup: {optimization_result.memory_freed_mb:.1f}MB freed"
        )

    def _log_batch_optimization_completion(
        self, results: list[ProcessingResult]
    ) -> None:
        """
        Logs completion statistics for optimized batch processing.

        Args:
            results: List of all processing results
        """
        logger.info(f"✅ Batch processing completed: {len(results)} results")

    def get_memory_status(self) -> dict[str, Any]:
        """
        Get current memory status and optimization information.

        Returns:
            Dict with memory status and optimization history
        """
        memory_status = self._memory_optimizer.get_memory_status()
        optimization_history = self._memory_optimizer.get_optimization_history()

        return {
            "memory_status": {
                "total_memory_gb": memory_status.total_memory_gb,
                "available_memory_gb": memory_status.available_memory_gb,
                "used_memory_gb": memory_status.used_memory_gb,
                "memory_percent": memory_status.memory_percent,
                "recommended_batch_size": memory_status.recommended_batch_size,
            },
            "optimization_history": {
                "total_optimizations": len(optimization_history),
                "total_memory_freed_mb": sum(
                    opt.memory_freed_mb for opt in optimization_history
                ),
                "average_optimization_time_ms": (
                    sum(opt.optimization_time_ms for opt in optimization_history)
                    / len(optimization_history)
                    if optimization_history
                    else 0
                ),
                "last_optimization": optimization_history[-1].__dict__
                if optimization_history
                else None,
            },
        }

    def cleanup(self) -> None:
        """
        Bereinigt Ressourcen und führt Memory-Optimization durch.

        Diese Methode wird von Tests erwartet für saubere Ressourcenbereinigung.
        """
        try:
            # Strategien bereinigen
            for strategy in self._strategies.values():
                if hasattr(strategy, "cleanup"):
                    strategy.cleanup()

            # Memory-Optimizer bereinigen
            if self._memory_optimizer:
                self._memory_optimizer.optimize_memory()

            logger.debug("🧹 UnifiedProcessor cleanup abgeschlossen")

        except Exception as e:
            logger.warning(f"⚠️ UnifiedProcessor cleanup Fehler: {e}")
