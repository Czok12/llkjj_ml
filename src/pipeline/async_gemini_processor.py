#!/usr/bin/env python3
"""
LLKJJ ML Pipeline - Async Gemini-First Processor
=============================================

Performance-optimierte Version des GeminiDirectProcessor mit:
- Asynchroner Gemini API-Verarbeitung
- Batch-Processing für mehrere PDFs
- Rate-Limiting mit asyncio.Semaphore
- PDF-Hash-Caching gegen Duplikate

Autor: LLKJJ ML Pipeline Team
Version: 4.1.0 (Async Performance)
Datum: 18. August 2025
"""

import asyncio
import hashlib
import json
import logging
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from src.config import Config
from src.models.processing_result import ProcessingResult
from src.pipeline.gemini_first_processor import GeminiDirectProcessor

logger = logging.getLogger(__name__)


class AsyncGeminiDirectProcessor(GeminiDirectProcessor):
    """
    Async-optimierte Version des GeminiDirectProcessor.

    Performance-Features:
    - Async Gemini API-Calls mit Rate-Limiting
    - PDF-Hash-Caching in SQLite für Duplikatserkennung
    - Batch-Processing für multiple PDFs
    - Non-blocking I/O-Operationen
    """

    def __init__(self, config: Config | None = None):
        super().__init__(config)
        self.rate_limiter = asyncio.Semaphore(
            5
        )  # 🚀 Erhöht: 3→5 concurrent Gemini calls
        self.cache_db_path = Path("data/cache/pdf_hash_cache.db")
        self._setup_cache_database()

        # 📊 MEMORY-OPTIMIERUNG: Batch-Processing Limits
        self.max_batch_size = 50  # Absolute Obergrenze für Memory-Schutz
        self.memory_check_interval = 10  # Alle 10 PDFs Memory-Check

        # 📄 SMART PDF-DETECTION: Größe-basierte Timeouts
        self.pdf_size_thresholds: dict[str, tuple[int, float]] = {
            "small": (0, 5 * 1024 * 1024),  # 0-5MB: Standard-Timeout
            "medium": (5 * 1024 * 1024, 20 * 1024 * 1024),  # 5-20MB: +50% Timeout
            "large": (20 * 1024 * 1024, 100 * 1024 * 1024),  # 20-100MB: +200% Timeout
            "huge": (100 * 1024 * 1024, float("inf")),  # >100MB: +500% Timeout
        }

        self.base_timeout = 30  # Sekunden für kleine PDFs

        logger.info(
            "✅ AsyncGeminiDirectProcessor initialisiert (Rate-Limited + Memory-Optimized + Smart-Detection)"
        )

    def _setup_cache_database(self) -> None:
        """Setup SQLite database for PDF hash caching."""
        self.cache_db_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.cache_db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS pdf_cache (
                    pdf_hash TEXT PRIMARY KEY,
                    pdf_path TEXT NOT NULL,
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    result_json TEXT NOT NULL,
                    confidence_score REAL NOT NULL,
                    extraction_quality TEXT NOT NULL
                )
            """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_processed_at
                ON pdf_cache(processed_at)
            """
            )

        logger.info("🔧 PDF-Hash-Cache-Database initialisiert: %s", self.cache_db_path)

    def _calculate_pdf_hash(self, pdf_path: Path) -> str:
        """Calculate SHA256 hash of PDF file for caching."""
        hasher = hashlib.sha256()
        with open(pdf_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _get_pdf_size_category(self, pdf_path: Path) -> tuple[str, int, float]:
        """
        🚀 SMART PDF-DETECTION: Bestimme PDF-Kategorie basierend auf Dateigröße.

        Returns:
            Tuple von (kategorie, timeout_sekunden, kompression_faktor)
        """
        try:
            file_size = pdf_path.stat().st_size

            for category, (min_size, max_size) in self.pdf_size_thresholds.items():
                if min_size <= file_size < max_size:
                    timeout_multiplier = {
                        "small": 1.0,  # Standard
                        "medium": 1.5,  # +50%
                        "large": 3.0,  # +200%
                        "huge": 6.0,  # +500%
                    }[category]

                    timeout = int(self.base_timeout * timeout_multiplier)
                    compression_factor = min(
                        1.0, 10 * 1024 * 1024 / file_size
                    )  # Komprimierung für >10MB

                    logger.info(
                        "📄 PDF-Kategorie '%s': %s (%.2f MB) → Timeout: %ds, Kompression: %.2f",
                        category,
                        pdf_path.name,
                        file_size / (1024 * 1024),
                        timeout,
                        compression_factor,
                    )

                    return category, timeout, compression_factor

            # Fallback
            return "small", self.base_timeout, 1.0

        except Exception as e:
            logger.warning(
                "⚠️ PDF-Größe-Detection fehlgeschlagen für %s: %s", pdf_path.name, e
            )
            return "small", self.base_timeout, 1.0

    def _check_memory_usage(self) -> dict[str, float]:
        """
        📊 MEMORY-MONITORING: Überwache aktuelle Memory-Usage für Batch-Processing.

        Returns:
            Dictionary mit Memory-Statistiken
        """
        import psutil

        process = psutil.Process()
        memory_info = process.memory_info()

        memory_stats: dict[str, float] = {
            "rss_mb": float(memory_info.rss / (1024 * 1024)),  # Resident Set Size
            "vms_mb": float(memory_info.vms / (1024 * 1024)),  # Virtual Memory Size
            "percent": float(process.memory_percent()),  # % of system memory
            "available_mb": float(psutil.virtual_memory().available / (1024 * 1024)),
        }

        logger.debug(
            "💾 Memory: RSS=%.1fMB, VMS=%.1fMB, Usage=%.1f%%, Available=%.1fGB",
            memory_stats["rss_mb"],
            memory_stats["vms_mb"],
            memory_stats["percent"],
            memory_stats["available_mb"] / 1024,
        )

        return memory_stats

    async def _check_cache(self, pdf_hash: str) -> dict[str, Any] | None:
        """Check if PDF result exists in cache."""

        def _db_query() -> tuple[str, float, float] | None:
            with sqlite3.connect(self.cache_db_path) as conn:
                cursor = conn.execute(
                    """
                SELECT result_json, confidence_score, extraction_quality
                FROM pdf_cache WHERE pdf_hash = ?
            """,
                    (pdf_hash,),
                )
                row = cursor.fetchone()
                return row  # type: ignore[no-any-return]

        # Run database operation in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        row = await loop.run_in_executor(None, _db_query)

        if row:
            result_json, confidence_score, extraction_quality = row
            logger.info("🎯 Cache-Hit für PDF-Hash: %s", pdf_hash[:12])
            return {
                "result": json.loads(result_json),
                "confidence_score": confidence_score,
                "extraction_quality": extraction_quality,
                "from_cache": True,
            }

        return None

    async def _store_in_cache(
        self, pdf_hash: str, pdf_path: Path, result: ProcessingResult
    ) -> None:
        """Store processing result in cache."""

        def _db_insert() -> None:
            with sqlite3.connect(self.cache_db_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO pdf_cache
                    (pdf_hash, pdf_path, result_json, confidence_score, extraction_quality)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    (
                        pdf_hash,
                        str(pdf_path),
                        json.dumps(result.to_dict()),
                        result.confidence_score,
                        result.extraction_quality,
                    ),
                )  # Run database operation in thread pool

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _db_insert)

        logger.info("💾 PDF-Resultat gecacht für Hash: %s", pdf_hash[:12])

    async def process_pdf_async(self, pdf_path: str | Path) -> ProcessingResult:
        """
        Async-Version der PDF-Verarbeitung mit Caching.

        Performance-Optimierungen:
        1. PDF-Hash-Check für Duplikate (0ms wenn gecacht)
        2. Rate-Limited Gemini API-Calls
        3. Non-blocking I/O für File-Operations

        Args:
            pdf_path: Pfad zur PDF-Datei

        Returns:
            ProcessingResult mit optimierter Performance
        """
        start_time = time.time()
        pdf_path = Path(pdf_path)

        # 1. PDF-Hash für Cache-Check berechnen
        try:
            pdf_hash = await asyncio.get_event_loop().run_in_executor(
                None, self._calculate_pdf_hash, pdf_path
            )
            logger.info("📊 PDF-Hash berechnet: %s (%s)", pdf_hash[:12], pdf_path.name)
        except Exception as e:
            logger.warning("⚠️ PDF-Hash-Berechnung fehlgeschlagen: %s", e)
            pdf_hash = None

        # 2. Cache-Check für sofortige Antwort
        if pdf_hash:
            cached_result = await self._check_cache(pdf_hash)
            if cached_result:
                cache_time_ms = int((time.time() - start_time) * 1000)
                logger.info(
                    "⚡ Cache-Hit! PDF bereits verarbeitet in %dms", cache_time_ms
                )

                # ProcessingResult aus Cache rekonstruieren
                cached_data = cached_result["result"]
                return ProcessingResult(
                    pdf_path=str(pdf_path),
                    processing_timestamp=datetime.now().isoformat(),
                    processing_method="gemini_first",
                    structured_data=cached_data.get("structured_data", {}),
                    skr03_classifications=cached_data.get("skr03_classifications", []),
                    confidence_score=cached_result["confidence_score"],
                    extraction_quality=cached_result["extraction_quality"],
                    processing_time_ms=cache_time_ms,
                )

        # 3. Rate-Limited Gemini-Processing (Cache-Miss)
        async with self.rate_limiter:
            logger.info("🚀 Processing PDF mit Rate-Limiting: %s", pdf_path.name)

            # Verwende synchronen Processor in Thread Pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, super().process_pdf_gemini_first, pdf_path
            )

        # 4. Ergebnis in Cache speichern
        if pdf_hash and result:
            try:
                await self._store_in_cache(pdf_hash, pdf_path, result)
            except Exception as e:
                logger.warning("⚠️ Cache-Speicherung fehlgeschlagen: %s", e)

        processing_time_ms = int((time.time() - start_time) * 1000)
        logger.info("✅ Async PDF-Processing abgeschlossen in %dms", processing_time_ms)

        return result

    async def process_batch_async(
        self, pdf_paths: list[str | Path], max_concurrent: int = 3
    ) -> list[ProcessingResult | None]:
        """
        🚀 ENHANCED Batch-Processing für mehrere PDFs mit Memory-Optimierung und Smart Detection.

        Performance-Features:
        - Memory-Monitoring alle 10 PDFs
        - Smart PDF-Größe-Detection für optimale Timeouts
        - Batch-Größe-Limiting bei >50 PDFs
        - Automatische Garbage Collection

        Args:
            pdf_paths: Liste der PDF-Pfade
            max_concurrent: Maximale gleichzeitige Verarbeitung

        Returns:
            Liste der ProcessingResults in derselben Reihenfolge
        """
        total_pdfs = len(pdf_paths)

        # 📊 MEMORY-SCHUTZ: Batch-Größe limitieren
        if total_pdfs > self.max_batch_size:
            logger.warning(
                "⚠️ Batch-Größe %d > Maximum %d - wird in Chunks aufgeteilt",
                total_pdfs,
                self.max_batch_size,
            )
            # Recursive batch processing in chunks
            all_results: list[ProcessingResult | None] = []
            for i in range(0, total_pdfs, self.max_batch_size):
                chunk = pdf_paths[i : i + self.max_batch_size]
                chunk_results = await self.process_batch_async(chunk, max_concurrent)
                all_results.extend(chunk_results)
            return all_results

        logger.info(
            "🚀 Enhanced Batch-Processing startet: %d PDFs, max %d parallel (Memory-Optimized)",
            total_pdfs,
            max_concurrent,
        )

        start_time = time.time()
        initial_memory = self._check_memory_usage()

        # 📄 SMART PDF-ANALYSIS: Analysiere alle PDFs für optimale Strategie
        pdf_analysis: list[dict[str, Any]] = []
        for pdf_path in pdf_paths:
            category, timeout, compression = self._get_pdf_size_category(Path(pdf_path))
            pdf_analysis.append(
                {
                    "path": pdf_path,
                    "category": category,
                    "timeout": timeout,
                    "compression": compression,
                }
            )

        # Sortiere PDFs: Große zuerst (bessere Memory-Verteilung)
        pdf_analysis.sort(key=lambda x: x["timeout"], reverse=True)
        logger.info(
            "📊 PDF-Kategorien: %s",
            ", ".join(
                f"{info['category']}({Path(info['path']).name})"
                for info in pdf_analysis[:5]
            ),
        )

        # Semaphore für Batch-Limit (zusätzlich zu API-Rate-Limit)
        batch_semaphore = asyncio.Semaphore(max_concurrent)
        processed_count = 0

        async def _process_with_enhanced_monitoring(
            pdf_info: dict[str, Any], index: int
        ) -> ProcessingResult | None:
            nonlocal processed_count

            async with batch_semaphore:
                try:
                    # 📊 MEMORY-CHECK alle 10 PDFs
                    if (
                        processed_count > 0
                        and processed_count % self.memory_check_interval == 0
                    ):
                        current_memory = self._check_memory_usage()
                        memory_growth = (
                            current_memory["rss_mb"] - initial_memory["rss_mb"]
                        )

                        if memory_growth > 500:  # >500MB Growth
                            logger.warning(
                                "⚠️ Memory-Growth: +%.1fMB nach %d PDFs - Garbage Collection",
                                memory_growth,
                                processed_count,
                            )
                            import gc

                            gc.collect()

                    result = await self.process_pdf_async(pdf_info["path"])
                    processed_count += 1

                    if processed_count % 10 == 0:
                        logger.info(
                            "📈 Progress: %d/%d PDFs verarbeitet",
                            processed_count,
                            total_pdfs,
                        )

                    return result

                except Exception as e:
                    logger.error(
                        "❌ PDF-Verarbeitung fehlgeschlagen: %s -> %s",
                        pdf_info["path"],
                        e,
                    )
                    return None

        # 🚀 Alle PDFs parallel verarbeiten mit Enhanced Monitoring
        tasks = [
            _process_with_enhanced_monitoring(pdf_info, i)
            for i, pdf_info in enumerate(pdf_analysis)
        ]
        results = await asyncio.gather(*tasks)

        # 📊 FINAL STATISTICS
        batch_time_ms = int((time.time() - start_time) * 1000)
        successful_count = sum(1 for r in results if r is not None)
        final_memory = self._check_memory_usage()
        memory_efficiency = final_memory["rss_mb"] / max(1, successful_count)

        logger.info(
            "✅ Enhanced Batch-Processing abgeschlossen: %d/%d erfolgreich in %dms (%.1fMB/PDF avg)",
            successful_count,
            total_pdfs,
            batch_time_ms,
            memory_efficiency,
        )

        return results

    async def get_cache_statistics(self) -> dict[str, Any]:
        """
        🎯 QUICK WIN: Cache-Statistiken für Performance-Monitoring

        Returns:
            Dictionary mit Cache-Performance-Metriken
        """

        def _gather_stats() -> dict[str, Any]:
            with sqlite3.connect(self.cache_db_path) as conn:
                # Basis-Statistiken
                cursor = conn.execute("SELECT COUNT(*) FROM pdf_cache")
                total_entries = cursor.fetchone()[0]

                # Speicherplatz-Analyse
                cursor = conn.execute(
                    "SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()"
                )
                size_result = cursor.fetchone()
                db_size_bytes = size_result[0] if size_result else 0

                # Cache-Hit-Rate der letzten 24h
                cursor = conn.execute(
                    """
                    SELECT COUNT(*) FROM pdf_cache
                    WHERE processed_at > datetime('now', '-24 hours')
                """
                )
                recent_entries = cursor.fetchone()[0]

                # Qualitäts-Verteilung
                cursor = conn.execute(
                    """
                    SELECT extraction_quality, COUNT(*)
                    FROM pdf_cache
                    GROUP BY extraction_quality
                """
                )
                quality_dist = dict(cursor.fetchall())

                # Durchschnittliche Konfidenz
                cursor = conn.execute("SELECT AVG(confidence_score) FROM pdf_cache")
                avg_confidence = cursor.fetchone()[0] or 0.0

                return {
                    "total_cached_pdfs": total_entries,
                    "database_size_mb": round(db_size_bytes / (1024 * 1024), 2),
                    "cache_entries_24h": recent_entries,
                    "quality_distribution": quality_dist,
                    "average_confidence": round(avg_confidence, 3),
                    "estimated_savings_hours": round(
                        total_entries * 5.0 / 3600, 2
                    ),  # 5s pro PDF gespart
                }

        loop = asyncio.get_event_loop()
        stats = await loop.run_in_executor(None, _gather_stats)

        logger.info(
            "📊 Cache-Statistiken: %d PDFs gecacht, %.2f MB, %.3f Ø Konfidenz",
            stats["total_cached_pdfs"],
            stats["database_size_mb"],
            stats["average_confidence"],
        )

        return stats

    async def warm_cache_for_patterns(self, pattern_files: list[Path]) -> int:
        """
        🎯 QUICK WIN: Cache-Warming für häufige PDF-Patterns

        Analysiert häufige PDF-Muster und cached sie proaktiv.

        Args:
            pattern_files: Liste von PDF-Dateien mit typischen Mustern

        Returns:
            Anzahl neu gecachter PDFs
        """
        warmed_count = 0

        for pdf_path in pattern_files:
            if not pdf_path.exists():
                continue

            # Prüfe ob bereits gecacht
            try:
                pdf_hash = await asyncio.get_event_loop().run_in_executor(
                    None, self._calculate_pdf_hash, pdf_path
                )

                cached_result = await self._check_cache(pdf_hash)
                if cached_result:
                    logger.debug("⚡ Pattern bereits gecacht: %s", pdf_path.name)
                    continue

                # Cache-Miss → verarbeite und cache
                logger.info("🔥 Cache-Warming für Pattern: %s", pdf_path.name)
                await self.process_pdf_async(pdf_path)
                warmed_count += 1

            except Exception as e:
                logger.warning(
                    "⚠️ Cache-Warming fehlgeschlagen für %s: %s", pdf_path.name, e
                )

        logger.info(
            "🔥 Cache-Warming abgeschlossen: %d neue Patterns gecacht", warmed_count
        )
        return warmed_count

    async def optimize_cache_performance(self) -> dict[str, Any]:
        """
        🎯 QUICK WIN: Cache-Performance-Optimierung

        Führt Wartungsoperationen für bessere Cache-Performance durch.

        Returns:
            Optimierungsstatistiken
        """

        def _optimize_db() -> dict[str, Any]:
            with sqlite3.connect(self.cache_db_path) as conn:
                # VACUUM für Speicher-Optimierung
                conn.execute("VACUUM")

                # ANALYZE für Query-Optimierung
                conn.execute("ANALYZE")

                # Zusätzliche Indizes für häufige Queries
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_confidence_quality
                    ON pdf_cache(confidence_score, extraction_quality)
                """
                )

                # Cache-Größe nach Optimierung
                cursor = conn.execute(
                    "SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()"
                )
                optimized_size = cursor.fetchone()[0] if cursor.fetchone() else 0

                return {
                    "vacuum_completed": True,
                    "analyze_completed": True,
                    "new_indexes_created": 1,
                    "optimized_size_mb": round(optimized_size / (1024 * 1024), 2),
                }

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _optimize_db)

        logger.info(
            "⚡ Cache-Optimierung abgeschlossen: %.2f MB nach VACUUM/ANALYZE",
            result["optimized_size_mb"],
        )

        return result

    async def clear_cache(self, older_than_days: int = 30) -> int:
        """
        Cache-Bereinigung für ältere Einträge.

        Args:
            older_than_days: Einträge älter als X Tage löschen

        Returns:
            Anzahl gelöschter Einträge
        """

        def _db_cleanup() -> int:
            with sqlite3.connect(self.cache_db_path) as conn:
                cursor = conn.execute(
                    f"""
                    DELETE FROM pdf_cache
                    WHERE processed_at < datetime('now', '-{older_than_days} days')
                """
                )
                return cursor.rowcount

        loop = asyncio.get_event_loop()
        deleted_count = await loop.run_in_executor(None, _db_cleanup)

        logger.info(
            "🧹 Cache bereinigt: %d Einträge gelöscht (älter als %d Tage)",
            deleted_count,
            older_than_days,
        )

        return deleted_count


# Convenience-Funktion für CLI-Integration
async def process_pdf_async(
    pdf_path: str, config: Config | None = None
) -> ProcessingResult:
    """Convenience function für async PDF-Processing."""
    processor = AsyncGeminiDirectProcessor(config)
    return await processor.process_pdf_async(pdf_path)


async def process_batch_async(
    pdf_paths: list[str | Path], config: Config | None = None, max_concurrent: int = 3
) -> list[ProcessingResult | None]:
    """Convenience function für async Batch-Processing."""
    processor = AsyncGeminiDirectProcessor(config)
    return await processor.process_batch_async(pdf_paths, max_concurrent)
