#!/usr/bin/env python3
"""
LLKJJ ML Pipeline - Performance Benchmark Test
=============================================

🏁 Performance-Test der optimierten Batch-Verarbeitung

Vergleicht:
- Baseline: Einzelverarbeitung
- Optimiert: Batch-Verarbeitung mit Cache-Warming
- Ziel: <5s Processing-Zeit pro PDF

Autor: LLKJJ Performance Team
Version: 1.0.0
Datum: 19. August 2025
"""

import asyncio
import time
from pathlib import Path

from src.pipeline.optimized_batch_processor import (
    OptimizedBatchProcessor,
    run_performance_benchmark,
)


async def test_performance_optimization():
    """
    🏁 PERFORMANCE BENCHMARK TEST

    Testet die Performance-Verbesserungen der optimierten Pipeline.
    """
    print("🚀 LLKJJ Performance Optimization Test gestartet...")
    print("=" * 60)

    # Test-PDFs sammeln
    test_pdf_dir = Path("test_pdfs")
    if not test_pdf_dir.exists():
        print("❌ Fehler: test_pdfs Verzeichnis nicht gefunden")
        return

    test_pdfs = list(test_pdf_dir.glob("*.pdf"))
    if not test_pdfs:
        print("❌ Fehler: Keine Test-PDFs gefunden")
        return

    print(f"📄 Gefundene Test-PDFs: {len(test_pdfs)}")
    for pdf in test_pdfs[:5]:  # Zeige nur erste 5
        size_mb = pdf.stat().st_size / (1024 * 1024)
        print(f"   - {pdf.name} ({size_mb:.2f} MB)")

    if len(test_pdfs) > 5:
        print(f"   ... und {len(test_pdfs) - 5} weitere")

    print("\n🎯 Performance-Benchmark wird ausgeführt...")

    # Benchmark ausführen (limitiert auf erste 3 PDFs für schnellen Test)
    benchmark_pdfs = test_pdfs[:3]

    try:
        benchmark_results = await run_performance_benchmark(benchmark_pdfs)

        print("\n🏆 BENCHMARK ERGEBNISSE:")
        print("=" * 60)

        # Grundlegende Metriken
        single = benchmark_results["single_processing"]
        batch = benchmark_results["batch_processing"]
        stats = benchmark_results["performance_stats"]

        print("📊 Einzelverarbeitung:")
        print(f"   - Durchschnitt: {single['avg_time_per_pdf_s']:.2f}s pro PDF")
        print(
            f"   - Gesamt: {single['total_time_s']:.2f}s für {single['pdfs_tested']} PDFs"
        )

        print("\n🚀 Batch-Verarbeitung:")
        print(f"   - Durchschnitt: {batch['avg_time_per_pdf_s']:.2f}s pro PDF")
        print(
            f"   - Gesamt: {batch['total_time_s']:.2f}s für {batch['pdfs_processed']} PDFs"
        )

        improvement = batch["improvement_factor"]
        if improvement > 1:
            print(f"\n✅ PERFORMANCE-VERBESSERUNG: {improvement:.1f}x schneller!")

            # Berechne Ziel-Erreichung
            target_time = 5.0  # Ziel: <5s pro PDF
            if batch["avg_time_per_pdf_s"] < target_time:
                print(
                    f"🎯 ZIEL ERREICHT: {batch['avg_time_per_pdf_s']:.2f}s < {target_time}s"
                )
            else:
                deficit = batch["avg_time_per_pdf_s"] - target_time
                print(f"🎯 Ziel verfehlt: {deficit:.2f}s über 5s-Ziel")
        else:
            print(f"\n⚠️ Keine Verbesserung: {improvement:.1f}x")

        print("\n📈 System-Metriken:")
        print(f"   - Rate-Limit: {stats['current_rate_limit']} concurrent calls")
        print(f"   - API-Latenz: {stats['avg_api_latency_s']:.2f}s")
        print(f"   - Erfolgsquote: {stats['success_rate']:.1%}")

        # Empfehlungen basierend auf Ergebnissen
        print("\n💡 EMPFEHLUNGEN:")
        if batch["avg_time_per_pdf_s"] > 10:
            print("   - PDF-Größe reduzieren oder komprimieren")
            print("   - Rate-Limiting weiter optimieren")
        elif batch["avg_time_per_pdf_s"] > 5:
            print("   - Cache-Warming für häufige Lieferanten erweitern")
            print("   - Gemini API-Prompts optimieren")
        else:
            print("   - Performance-Ziel erreicht! 🎉")
            print("   - Produktions-Deployment empfohlen")

    except Exception as e:
        print(f"❌ Benchmark fehlgeschlagen: {e}")
        return

    print("\n" + "=" * 60)
    print("🏁 Performance-Test abgeschlossen")


async def test_single_pdf_optimized():
    """
    🚀 Test der optimierten Pipeline mit einem einzelnen PDF
    """
    print("\n🔧 Test der optimierten Pipeline mit einzelnem PDF...")

    test_pdf_dir = Path("test_pdfs")
    test_pdfs = list(test_pdf_dir.glob("*.pdf"))

    if not test_pdfs:
        print("❌ Keine Test-PDFs gefunden")
        return

    # Nimm das erste PDF für schnellen Test
    test_pdf = test_pdfs[0]
    print(f"📄 Test-PDF: {test_pdf.name}")

    # Erstelle optimierten Processor
    processor = OptimizedBatchProcessor()

    start_time = time.time()

    try:
        # Teste Cache-Warming
        print("🔥 Cache-Warming...")
        await processor.cache_warm_frequent_suppliers()

        # Teste einzelne PDF-Verarbeitung
        print("📊 PDF-Verarbeitung...")
        result = await processor.process_pdf_async(test_pdf)

        processing_time = time.time() - start_time

        print("\n✅ Verarbeitung erfolgreich!")
        print(f"   - Zeit: {processing_time:.2f}s")
        print(f"   - Konfidenz: {result.confidence_score:.3f}")
        print(f"   - Qualität: {result.extraction_quality}")
        print(f"   - SKR03-Klassifizierungen: {len(result.skr03_classifications)}")

        if processing_time < 5.0:
            print("🎯 Ziel erreicht: <5s Verarbeitungszeit!")
        else:
            print(f"⚠️ Ziel verfehlt: {processing_time:.2f}s > 5s")

    except Exception as e:
        print(f"❌ Test fehlgeschlagen: {e}")


if __name__ == "__main__":
    print("🚀 LLKJJ Performance Optimization Test Suite")
    print("=" * 60)

    # Führe Tests aus
    asyncio.run(test_single_pdf_optimized())
    asyncio.run(test_performance_optimization())
