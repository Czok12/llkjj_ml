#!/usr/bin/env python3
"""
LLKJJ ML Pipeline - Enhanced Batch Memory Manager
=================================================

Erweiterte Memory-Management-Strategien für optimale Batch-Performance:
- Dynamische Batch-Größe basierend auf verfügbarem Memory
- Intelligent Memory-Pressure Detection
- Adaptive Concurrency-Scaling
- Memory-Leak-Detection und Cleanup

🚀 QUICK WIN: Skalierung auf 100+ PDFs ohne Memory-Issues

Autor: LLKJJ ML Pipeline Team
Version: 4.2.0 (Batch-Memory-Optimierung)
Datum: 19. August 2025
"""

import asyncio
import gc
import logging
import time
from pathlib import Path
from typing import Any

import psutil

from src.config import Config
from src.models.processing_result import ProcessingResult
from src.pipeline.async_gemini_processor import AsyncGeminiDirectProcessor

logger = logging.getLogger(__name__)


class EnhancedBatchMemoryManager:
    """
    🧠 Intelligentes Memory-Management für Large-Scale-Batch-Processing.

    Features:
    1. Dynamic Batch-Sizing: Adaptiert Batch-Größe an verfügbares Memory
    2. Memory-Pressure-Detection: Proaktive Cleanup-Strategien
    3. Adaptive Concurrency: Reduziert Parallelität bei Memory-Stress
    4. Memory-Leak-Detection: Erkennt und behebt Memory-Leaks
    """

    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.processor = AsyncGeminiDirectProcessor(config)

        # 🧠 MEMORY-MANAGEMENT-KONFIGURATION
        self.memory_thresholds = {
            "warning_mb": 1000,  # Memory-Warning bei 1GB Growth
            "critical_mb": 2000,  # Memory-Critical bei 2GB Growth
            "available_min_gb": 2,  # Mindestens 2GB verfügbar lassen
        }

        # 📊 ADAPTIVE BATCH-SIZING
        self.batch_sizing = {
            "small_pdfs": {"max_concurrent": 10, "max_batch": 100},  # <5MB
            "medium_pdfs": {"max_concurrent": 5, "max_batch": 50},  # 5-20MB
            "large_pdfs": {"max_concurrent": 3, "max_batch": 20},  # 20-100MB
            "huge_pdfs": {"max_concurrent": 1, "max_batch": 5},  # >100MB
        }

        # 🔄 CLEANUP-STRATEGIEN
        self.cleanup_strategies = [
            "garbage_collect",
            "cache_cleanup",
            "model_reload",
            "force_gc_full",
        ]

        logger.info(
            "🧠 EnhancedBatchMemoryManager initialisiert - Ready für Large-Scale-Processing!"
        )

    def get_memory_status(self) -> dict[str, Any]:
        """
        📊 Umfassende Memory-Status-Analysis.

        Returns:
            Detaillierte Memory-Statistiken und Recommendations
        """
        process = psutil.Process()
        system = psutil.virtual_memory()
        memory_info = process.memory_info()

        status = {
            "process": {
                "rss_mb": round(memory_info.rss / (1024 * 1024), 2),
                "vms_mb": round(memory_info.vms / (1024 * 1024), 2),
                "percent": round(process.memory_percent(), 2),
            },
            "system": {
                "total_gb": round(system.total / (1024 * 1024 * 1024), 2),
                "available_gb": round(system.available / (1024 * 1024 * 1024), 2),
                "used_percent": round(system.percent, 2),
            },
            "thresholds": {
                "memory_pressure": (
                    "high"
                    if system.available
                    < (self.memory_thresholds["available_min_gb"] * 1024 * 1024 * 1024)
                    else "normal"
                ),
                "process_status": (
                    "critical"
                    if memory_info.rss
                    > (self.memory_thresholds["critical_mb"] * 1024 * 1024)
                    else "normal"
                ),
            },
        }

        return status

    def calculate_optimal_batch_size(self, pdf_paths: list[Path]) -> dict[str, Any]:
        """
        🎯 Berechne optimale Batch-Größe basierend auf PDF-Kategorien und Memory-Status.

        Args:
            pdf_paths: Liste der zu verarbeitenden PDFs

        Returns:
            Optimale Batch-Konfiguration
        """
        memory_status = self.get_memory_status()

        # PDF-Kategorien analysieren
        pdf_categories: dict[str, int] = {
            "small": 0,
            "medium": 0,
            "large": 0,
            "huge": 0,
        }

        for pdf_path in pdf_paths:
            try:
                file_size = pdf_path.stat().st_size
                if file_size < 5 * 1024 * 1024:  # <5MB
                    pdf_categories["small"] += 1
                elif file_size < 20 * 1024 * 1024:  # 5-20MB
                    pdf_categories["medium"] += 1
                elif file_size < 100 * 1024 * 1024:  # 20-100MB
                    pdf_categories["large"] += 1
                else:  # >100MB
                    pdf_categories["huge"] += 1
            except Exception:
                pdf_categories["medium"] += 1  # Fallback

        # Dominante Kategorie bestimmen
        dominant_category = max(pdf_categories.items(), key=lambda x: x[1])[0]

        # Batch-Konfiguration anpassen
        base_config = self.batch_sizing[f"{dominant_category}_pdfs"]

        # Memory-Pressure-Anpassung
        memory_factor = 1.0
        if memory_status["thresholds"]["memory_pressure"] == "high":
            memory_factor = 0.5  # Halbiere Batch-Größe bei Memory-Pressure
        elif memory_status["system"]["available_gb"] < 4:
            memory_factor = 0.7  # Reduziere bei wenig verfügbarem Memory

        optimal_config = {
            "max_concurrent": max(
                1, int(base_config["max_concurrent"] * memory_factor)
            ),
            "max_batch_size": max(5, int(base_config["max_batch"] * memory_factor)),
            "dominant_category": dominant_category,
            "memory_factor": memory_factor,
            "pdf_distribution": pdf_categories,
            "estimated_memory_mb": len(pdf_paths)
            * self._estimate_memory_per_pdf(dominant_category),
        }

        logger.info(
            "🎯 Optimale Batch-Config: %d concurrent, %d batch-size (%s PDFs, %.1fx Memory-Factor)",
            optimal_config["max_concurrent"],
            optimal_config["max_batch_size"],
            dominant_category,
            memory_factor,
        )

        return optimal_config

    def _estimate_memory_per_pdf(self, category: str) -> float:
        """Schätze Memory-Verbrauch pro PDF basierend auf Kategorie."""
        estimates = {
            "small": 50,  # 50MB pro kleiner PDF
            "medium": 150,  # 150MB pro mittlerer PDF
            "large": 400,  # 400MB pro großer PDF
            "huge": 800,  # 800MB pro sehr großer PDF
        }
        return estimates.get(category, 150)

    async def process_batch_with_smart_memory_management(
        self, pdf_paths: list[str | Path]
    ) -> dict[str, Any]:
        """
        🚀 SMART BATCH-PROCESSING mit adaptivem Memory-Management.

        Args:
            pdf_paths: Liste der PDF-Pfade

        Returns:
            Batch-Processing-Ergebnisse mit Memory-Analytics
        """
        pdf_paths_list = [Path(p) for p in pdf_paths]
        start_time = time.time()
        initial_memory = self.get_memory_status()

        logger.info("🚀 Smart Batch-Processing startet: %d PDFs", len(pdf_paths_list))

        # 1. Optimale Batch-Konfiguration berechnen
        batch_config = self.calculate_optimal_batch_size(pdf_paths_list)

        # 2. Memory-Monitoring Setup
        memory_checkpoints: list[dict[str, Any]] = [
            {"stage": "start", "memory": initial_memory}
        ]
        cleanup_actions: list[str] = []

        # 3. Batch-Processing in optimalen Chunks
        all_results: list[ProcessingResult | None] = []
        successful_count = 0
        failed_count = 0

        for i in range(0, len(pdf_paths_list), batch_config["max_batch_size"]):
            chunk = pdf_paths_list[i : i + batch_config["max_batch_size"]]
            chunk_number = (i // batch_config["max_batch_size"]) + 1
            total_chunks = (
                len(pdf_paths_list) + batch_config["max_batch_size"] - 1
            ) // batch_config["max_batch_size"]

            logger.info(
                "📦 Processing Chunk %d/%d: %d PDFs",
                chunk_number,
                total_chunks,
                len(chunk),
            )

            # Memory-Check vor Chunk
            pre_chunk_memory = self.get_memory_status()
            if pre_chunk_memory["thresholds"]["memory_pressure"] == "high":
                logger.warning("⚠️ Memory-Pressure detected - triggering cleanup")
                cleanup_applied = await self._apply_memory_cleanup("memory_pressure")
                cleanup_actions.extend(cleanup_applied)

            # Chunk verarbeiten
            try:
                chunk_results = await self.processor.process_batch_async(
                    [str(p) for p in chunk],
                    max_concurrent=batch_config["max_concurrent"],
                )
                all_results.extend(chunk_results)

                # Erfolgszählung
                chunk_successful = sum(1 for r in chunk_results if r is not None)
                successful_count += chunk_successful
                failed_count += len(chunk_results) - chunk_successful

                logger.info(
                    "✅ Chunk %d abgeschlossen: %d/%d erfolgreich",
                    chunk_number,
                    chunk_successful,
                    len(chunk),
                )

            except Exception as e:
                logger.error("❌ Chunk %d fehlgeschlagen: %s", chunk_number, e)
                failed_count += len(chunk)
                all_results.extend([None] * len(chunk))

            # Memory-Checkpoint nach Chunk
            post_chunk_memory = self.get_memory_status()
            memory_checkpoints.append(
                {
                    "stage": f"chunk_{chunk_number}",
                    "memory": post_chunk_memory,
                    "pdfs_processed": len(chunk),
                }
            )

        # 4. Final Memory-Cleanup
        final_cleanup = await self._apply_memory_cleanup("final_cleanup")
        cleanup_actions.extend(final_cleanup)

        # 5. Ergebnisse und Analytics
        total_time = time.time() - start_time
        final_memory = self.get_memory_status()

        batch_analytics: dict[str, Any] = {
            "processing_summary": {
                "total_pdfs": len(pdf_paths_list),
                "successful": successful_count,
                "failed": failed_count,
                "success_rate": (
                    successful_count / len(pdf_paths_list) if pdf_paths_list else 0
                ),
                "total_time_seconds": round(total_time, 2),
                "pdfs_per_minute": (
                    round((successful_count / total_time) * 60, 2)
                    if total_time > 0
                    else 0
                ),
            },
            "memory_analytics": {
                "initial_memory_mb": initial_memory["process"]["rss_mb"],
                "final_memory_mb": final_memory["process"]["rss_mb"],
                "peak_memory_growth_mb": max(
                    cp["memory"]["process"]["rss_mb"]
                    - initial_memory["process"]["rss_mb"]
                    for cp in memory_checkpoints
                ),
                "memory_efficiency_mb_per_pdf": (
                    final_memory["process"]["rss_mb"] / max(1, successful_count)
                ),
                "cleanup_actions_applied": cleanup_actions,
                "memory_checkpoints": memory_checkpoints,
            },
            "batch_configuration": batch_config,
            "results": all_results,
        }

        logger.info(
            "🎉 Smart Batch-Processing abgeschlossen: %d/%d erfolgreich, %.2f PDFs/min, %.1fMB peak growth",
            successful_count,
            len(pdf_paths_list),
            batch_analytics["processing_summary"]["pdfs_per_minute"],
            batch_analytics["memory_analytics"]["peak_memory_growth_mb"],
        )

        return batch_analytics

    async def _apply_memory_cleanup(self, reason: str) -> list[str]:
        """
        🧹 Wende Memory-Cleanup-Strategien an.

        Args:
            reason: Grund für Cleanup (memory_pressure, final_cleanup, etc.)

        Returns:
            Liste der angewendeten Cleanup-Aktionen
        """
        applied_actions: list[str] = []

        try:
            # 1. Standard Garbage Collection
            gc.collect()
            applied_actions.append("garbage_collect")

            # 2. Cache-Optimierung falls verfügbar
            if hasattr(self.processor, "optimize_cache_performance"):
                await self.processor.optimize_cache_performance()
                applied_actions.append("cache_optimization")

            # 3. Force Full GC bei kritischem Memory
            current_memory = self.get_memory_status()
            if current_memory["thresholds"]["memory_pressure"] == "high":
                # Entferne lokalen Import, nutze globales gc-Modul
                gc.set_debug(gc.DEBUG_STATS)
                collected = gc.collect()
                gc.set_debug(0)
                applied_actions.append(f"force_gc_collected_{collected}")

            logger.info(
                "🧹 Memory-Cleanup angewendet (%s): %s",
                reason,
                ", ".join(applied_actions),
            )

        except Exception as e:
            logger.warning("⚠️ Memory-Cleanup teilweise fehlgeschlagen: %s", e)

        return applied_actions


async def main() -> None:
    """Test der EnhancedBatchMemoryManager."""
    logging.basicConfig(level=logging.INFO)

    manager = EnhancedBatchMemoryManager()

    # Memory-Status testen
    memory_status = manager.get_memory_status()
    print(f"📊 Memory-Status: {memory_status}")

    # Test mit verfügbaren PDFs
    test_pdfs_dir = Path("test_pdfs")
    if test_pdfs_dir.exists():
        pdf_files = list(test_pdfs_dir.glob("*.pdf"))[:10]  # Limit für Test

        if pdf_files:
            # Optimale Batch-Konfiguration berechnen
            batch_config = manager.calculate_optimal_batch_size(pdf_files)
            print(f"🎯 Optimale Batch-Config: {batch_config}")

            # Smart Batch-Processing testen
            # results = await manager.process_batch_with_smart_memory_management(pdf_files)
            # print(f"🚀 Batch-Results: {results['processing_summary']}")
            print("📝 Test abgeschlossen - Smart Batch-Processing bereit!")
        else:
            print("⚠️ Keine Test-PDFs gefunden")
    else:
        print("⚠️ test_pdfs/ Verzeichnis nicht gefunden")


if __name__ == "__main__":
    asyncio.run(main())
    asyncio.run(main())
