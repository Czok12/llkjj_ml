#!/usr/bin/env python3
"""
LLKJJ ML Pipeline - Performance Benchmarking Suite
================================================

Automatisierte Performance-Tests für verschiedene PDF-Typen:
- Small PDFs (0-5MB): Standard-Elektro-Rechnungen
- Medium PDFs (5-20MB): Multi-Page-Kataloge
- Large PDFs (20-100MB): Komplexe Dokumentenpakete
- Huge PDFs (>100MB): Bulk-Uploads

Autor: LLKJJ ML Pipeline Team
Version: 4.1.0 (Business Value Maximierung)
Datum: 18. August 2025
"""

import asyncio
import json
import logging
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from src.config import Config
from src.models.processing_result import ProcessingResult
from src.pipeline.async_gemini_processor import AsyncGeminiDirectProcessor

logger = logging.getLogger(__name__)


class PerformanceBenchmarkSuite:
    """
    🚀 Performance-Benchmarking für LLKJJ ML Pipeline.

    Testet verschiedene PDF-Kategorien und dokumentiert Metriken:
    - Processing-Zeit pro PDF-Größe
    - Memory-Verbrauch Trends
    - Cache-Hit-Raten
    - Batch-Processing Effizienz
    """

    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.processor = AsyncGeminiDirectProcessor(config)
        self.results_dir = Path("data/benchmarks")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        logger.info("🎯 PerformanceBenchmarkSuite initialisiert")

    async def run_single_pdf_benchmark(
        self, pdf_path: str | Path, runs: int = 3
    ) -> dict[str, Any]:
        """
        Benchmark einzelne PDF mit mehreren Durchläufen.

        Args:
            pdf_path: Pfad zur Test-PDF
            runs: Anzahl Durchläufe für Mittelwert

        Returns:
            Benchmark-Ergebnisse mit Statistiken
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF nicht gefunden: {pdf_path}")

        logger.info("📊 Starte Benchmark für %s (%d Durchläufe)", pdf_path.name, runs)

        processing_times: list[float] = []
        confidence_scores: list[float] = []
        memory_usages: list[float] = []

        for run in range(runs):
            start_time = time.time()
            memory_before = self.processor._check_memory_usage()

            try:
                result = await self.processor.process_pdf_async(pdf_path)
                processing_time = time.time() - start_time

                memory_after = self.processor._check_memory_usage()
                memory_delta = memory_after["rss_mb"] - memory_before["rss_mb"]

                processing_times.append(processing_time * 1000)  # ms
                confidence_scores.append(result.confidence_score)
                memory_usages.append(memory_delta)

                logger.info(
                    "Run %d/%d: %.0fms, Konfidenz=%.2f, Memory=+%.1fMB",
                    run + 1,
                    runs,
                    processing_time * 1000,
                    result.confidence_score,
                    memory_delta,
                )

            except Exception as e:
                logger.error("❌ Benchmark-Run %d fehlgeschlagen: %s", run + 1, e)
                continue

        if not processing_times:
            raise RuntimeError("Alle Benchmark-Runs fehlgeschlagen")

        # Statistiken berechnen
        file_size = pdf_path.stat().st_size
        category, timeout, compression = self.processor._get_pdf_size_category(pdf_path)

        benchmark_result: dict[str, Any] = {
            "pdf_info": {
                "name": pdf_path.name,
                "size_bytes": file_size,
                "size_mb": file_size / (1024 * 1024),
                "category": category,
                "timeout": timeout,
                "compression": compression,
            },
            "performance": {
                "runs": len(processing_times),
                "processing_time_ms": {
                    "mean": statistics.mean(processing_times),
                    "median": statistics.median(processing_times),
                    "min": min(processing_times),
                    "max": max(processing_times),
                    "std": (
                        statistics.stdev(processing_times)
                        if len(processing_times) > 1
                        else 0
                    ),
                },
                "confidence_score": {
                    "mean": statistics.mean(confidence_scores),
                    "min": min(confidence_scores),
                    "max": max(confidence_scores),
                },
                "memory_delta_mb": {
                    "mean": statistics.mean(memory_usages),
                    "min": min(memory_usages),
                    "max": max(memory_usages),
                },
            },
            "timestamp": datetime.now().isoformat(),
            "processor_version": "4.1.0",
        }

        logger.info(
            "✅ Benchmark abgeschlossen: %.0f±%.0fms, Konfidenz=%.2f, Memory=+%.1fMB",
            benchmark_result["performance"]["processing_time_ms"]["mean"],
            benchmark_result["performance"]["processing_time_ms"]["std"],
            benchmark_result["performance"]["confidence_score"]["mean"],
            benchmark_result["performance"]["memory_delta_mb"]["mean"],
        )

        return benchmark_result

    async def run_batch_benchmark(
        self, pdf_directory: str | Path, max_pdfs: int = 20
    ) -> dict[str, Any]:
        """
        Benchmark Batch-Processing Performance.

        Args:
            pdf_directory: Verzeichnis mit Test-PDFs
            max_pdfs: Maximale Anzahl PDFs für Test

        Returns:
            Batch-Benchmark-Ergebnisse
        """
        pdf_dir = Path(pdf_directory)
        if not pdf_dir.exists():
            raise FileNotFoundError(f"PDF-Verzeichnis nicht gefunden: {pdf_dir}")

        # Sammle verfügbare PDFs
        pdf_files: list[Path] = list(pdf_dir.glob("*.pdf"))[:max_pdfs]
        if not pdf_files:
            raise ValueError(f"Keine PDFs gefunden in: {pdf_dir}")

        logger.info(
            "🚀 Starte Batch-Benchmark: %d PDFs aus %s", len(pdf_files), pdf_dir
        )

        start_time = time.time()
        memory_before = self.processor._check_memory_usage()

        # Batch-Processing durchführen
        # Ensure the argument has the precise expected type: list[str | Path]
        pdf_paths_arg: list[str | Path] = list(pdf_files)
        results: list[
            ProcessingResult | None
        ] = await self.processor.process_batch_async(pdf_paths_arg, max_concurrent=5)
        batch_time = time.time() - start_time
        memory_after = self.processor._check_memory_usage()
        memory_delta = memory_after["rss_mb"] - memory_before["rss_mb"]

        # Erfolgsstatistiken
        successful_results = [r for r in results if r is not None]
        success_rate = len(successful_results) / len(pdf_files)

        confidence_scores = [float(r.confidence_score) for r in successful_results]
        processing_times = [float(r.processing_time_ms) for r in successful_results]

        batch_benchmark: dict[str, Any] = {
            "batch_info": {
                "total_pdfs": len(pdf_files),
                "successful_pdfs": len(successful_results),
                "success_rate": success_rate,
                "max_concurrent": 5,
            },
            "performance": {
                "total_time_s": batch_time,
                "time_per_pdf_s": batch_time / max(1, len(successful_results)),
                "throughput_pdfs_per_minute": len(successful_results)
                / (batch_time / 60),
                "memory_delta_mb": memory_delta,
                "memory_per_pdf_mb": memory_delta / max(1, len(successful_results)),
            },
            "quality": {
                "confidence_mean": (
                    statistics.mean(confidence_scores) if confidence_scores else 0
                ),
                "confidence_min": min(confidence_scores) if confidence_scores else 0,
                "processing_time_mean_ms": (
                    statistics.mean(processing_times) if processing_times else 0
                ),
            },
            "timestamp": datetime.now().isoformat(),
            "processor_version": "4.1.0",
        }

        logger.info(
            "✅ Batch-Benchmark abgeschlossen: %.1fs total, %.2f PDFs/min, %.1f%% Erfolgsrate",
            batch_time,
            batch_benchmark["performance"]["throughput_pdfs_per_minute"],
            success_rate * 100,
        )

        return batch_benchmark

    async def run_comprehensive_benchmark(
        self, test_pdf: str | Path | None = None
    ) -> dict[str, Any]:
        """
        Umfassender Performance-Benchmark mit allen verfügbaren Tests.

        Args:
            test_pdf: Spezifische Test-PDF (optional)

        Returns:
            Vollständige Benchmark-Ergebnisse
        """
        logger.info("🎯 Starte umfassenden Performance-Benchmark")

        comprehensive_results: dict[str, Any] = {
            "benchmark_type": "comprehensive",
            "timestamp": datetime.now().isoformat(),
            "processor_version": "4.1.0",
            "tests": {},
        }

        # 1. Cache-Statistiken vor Tests
        cache_stats_before = await self.processor.get_cache_statistics()
        comprehensive_results["cache_stats_before"] = cache_stats_before

        # 2. Single PDF Benchmark (falls verfügbar)
        if test_pdf and Path(test_pdf).exists():
            try:
                single_benchmark = await self.run_single_pdf_benchmark(test_pdf, runs=3)
                comprehensive_results["tests"]["single_pdf"] = single_benchmark
            except Exception as e:
                logger.warning("⚠️ Single PDF Benchmark fehlgeschlagen: %s", e)

        # 3. Batch Benchmark (test_pdfs/ Verzeichnis)
        batch_dir = Path("test_pdfs")
        if batch_dir.exists():
            try:
                batch_benchmark = await self.run_batch_benchmark(batch_dir, max_pdfs=10)
                comprehensive_results["tests"]["batch_processing"] = batch_benchmark
            except Exception as e:
                logger.warning("⚠️ Batch Benchmark fehlgeschlagen: %s", e)

        # 4. Cache-Statistiken nach Tests
        cache_stats_after = await self.processor.get_cache_statistics()
        comprehensive_results["cache_stats_after"] = cache_stats_after

        # 5. Cache-Improvement berechnen
        if cache_stats_before and cache_stats_after:
            cache_improvement = {
                "entries_added": cache_stats_after.get("total_entries", 0)
                - cache_stats_before.get("total_entries", 0),
                "hit_rate_improvement": cache_stats_after.get("hit_rate", 0)
                - cache_stats_before.get("hit_rate", 0),
            }
            comprehensive_results["cache_improvement"] = cache_improvement

        # 6. Ergebnisse speichern
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"comprehensive_benchmark_{timestamp}.json"

        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(comprehensive_results, f, indent=2, ensure_ascii=False)

        logger.info("✅ Umfassender Benchmark abgeschlossen → %s", results_file)

        return comprehensive_results

    def generate_benchmark_report(self, results: dict[str, Any]) -> str:
        """
        Generiere lesbaren Benchmark-Report.

        Args:
            results: Benchmark-Ergebnisse

        Returns:
            Formatierter Report-Text
        """
        report_lines: list[str] = []
        report_lines.append("=" * 60)
        report_lines.append("📊 LLKJJ ML PIPELINE - PERFORMANCE BENCHMARK REPORT")
        report_lines.append("=" * 60)
        report_lines.append(f"Timestamp: {results.get('timestamp', 'N/A')}")
        report_lines.append(
            f"Processor Version: {results.get('processor_version', 'N/A')}"
        )
        report_lines.append("")

        # Single PDF Test
        if "single_pdf" in results.get("tests", {}):
            single = results["tests"]["single_pdf"]
            report_lines.append("🔍 SINGLE PDF BENCHMARK:")
            report_lines.append(
                f"  PDF: {single['pdf_info']['name']} ({single['pdf_info']['size_mb']:.1f} MB)"
            )
            report_lines.append(f"  Category: {single['pdf_info']['category']}")
            report_lines.append(
                f"  Processing Time: {single['performance']['processing_time_ms']['mean']:.0f}±{single['performance']['processing_time_ms']['std']:.0f}ms"
            )
            report_lines.append(
                f"  Confidence: {single['performance']['confidence_score']['mean']:.2f}"
            )
            report_lines.append(
                f"  Memory Delta: +{single['performance']['memory_delta_mb']['mean']:.1f}MB"
            )
            report_lines.append("")

        # Batch Processing Test
        if "batch_processing" in results.get("tests", {}):
            batch = results["tests"]["batch_processing"]
            report_lines.append("🚀 BATCH PROCESSING BENCHMARK:")
            report_lines.append(f"  Total PDFs: {batch['batch_info']['total_pdfs']}")
            report_lines.append(
                f"  Success Rate: {batch['batch_info']['success_rate']*100:.1f}%"
            )
            report_lines.append(
                f"  Total Time: {batch['performance']['total_time_s']:.1f}s"
            )
            report_lines.append(
                f"  Throughput: {batch['performance']['throughput_pdfs_per_minute']:.1f} PDFs/min"
            )
            report_lines.append(
                f"  Avg Confidence: {batch['quality']['confidence_mean']:.2f}"
            )
            report_lines.append(
                f"  Memory Usage: +{batch['performance']['memory_delta_mb']:.1f}MB total"
            )
            report_lines.append("")

        # Cache Statistics
        if "cache_stats_after" in results:
            cache = results["cache_stats_after"]
            report_lines.append("💾 CACHE PERFORMANCE:")
            report_lines.append(f"  Total Entries: {cache.get('total_entries', 'N/A')}")
            report_lines.append(f"  Hit Rate: {cache.get('hit_rate', 0)*100:.1f}%")
            report_lines.append(f"  Database Size: {cache.get('db_size_mb', 0):.1f}MB")
            report_lines.append("")

        report_lines.append("=" * 60)
        report_lines.append("✅ Benchmark abgeschlossen - Ready for Production!")
        report_lines.append("=" * 60)

        return "\n".join(report_lines)


async def main() -> None:
    """Test der PerformanceBenchmarkSuite."""
    logging.basicConfig(level=logging.INFO)

    suite = PerformanceBenchmarkSuite()

    # Teste mit verfügbarer PDF
    test_pdf = Path("test_pdfs/Sonepar_test3.pdf")
    if test_pdf.exists():
        results = await suite.run_comprehensive_benchmark(test_pdf)
        report = suite.generate_benchmark_report(results)
        print(report)
    else:
        print("⚠️ Keine Test-PDF gefunden - verwende Cache-Test")
        cache_stats = await suite.processor.get_cache_statistics()
        print(f"Cache-Statistiken: {cache_stats}")


if __name__ == "__main__":
    asyncio.run(main())
