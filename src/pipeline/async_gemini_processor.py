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
        self.rate_limiter = asyncio.Semaphore(3)  # Max 3 concurrent Gemini calls
        self.cache_db_path = Path("data/cache/pdf_hash_cache.db")
        self._setup_cache_database()

        logger.info(
            "✅ AsyncGeminiDirectProcessor initialisiert (Rate-Limited + Cached)"
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
        Batch-Processing für mehrere PDFs mit optimaler Parallelisierung.

        Args:
            pdf_paths: Liste der PDF-Pfade
            max_concurrent: Maximale gleichzeitige Verarbeitung

        Returns:
            Liste der ProcessingResults in derselben Reihenfolge
        """
        logger.info(
            "🚀 Batch-Processing startet: %d PDFs, max %d parallel",
            len(pdf_paths),
            max_concurrent,
        )

        start_time = time.time()

        # Semaphore für Batch-Limit (zusätzlich zu API-Rate-Limit)
        batch_semaphore = asyncio.Semaphore(max_concurrent)

        async def _process_with_semaphore(
            pdf_path: str | Path,
        ) -> ProcessingResult | None:
            async with batch_semaphore:
                try:
                    return await self.process_pdf_async(pdf_path)
                except Exception as e:
                    logger.error(
                        "❌ PDF-Verarbeitung fehlgeschlagen: %s -> %s", pdf_path, e
                    )
                    return None

        # Alle PDFs parallel verarbeiten
        tasks = [_process_with_semaphore(pdf_path) for pdf_path in pdf_paths]
        results = await asyncio.gather(*tasks)

        batch_time_ms = int((time.time() - start_time) * 1000)
        successful_count = sum(1 for r in results if r is not None)

        logger.info(
            "✅ Batch-Processing abgeschlossen: %d/%d erfolgreich in %dms",
            successful_count,
            len(pdf_paths),
            batch_time_ms,
        )

        return results

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
