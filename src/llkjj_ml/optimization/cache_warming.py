#!/usr/bin/env python3
"""
LLKJJ ML Pipeline - Intelligentes Cache-Warming System
=====================================================

Performance-Optimierung für häufige Lieferanten mit:
- Supplier-spezifische Cache-Prioritäten
- Intelligente Pattern-Erkennung
- Predictive Cache-Warming basierend auf Nutzungsmustern
- Business Logic für deutschen Elektrohandel

Autor: LLKJJ ML Pipeline Team
Version: 4.2.0 (Cache Warming Optimization)
Datum: 19. August 2025
"""

import asyncio
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from ..settings_bridge import ConfigBridge, config_instance

logger = logging.getLogger(__name__)


class IntelligentCacheWarming:
    """
    Intelligentes Cache-Warming System für LLKJJ ML Pipeline.

    Features:
    - Supplier-spezifische Priorisierung (Sonepar, Amazon, FAMO)
    - Pattern-basierte Cache-Erwärmung
    - Usage-History-Analyse für Predictions
    - Business Logic für deutsche Elektrotechnik-UGs
    """

    def __init__(self, config: ConfigBridge | None = None):
        self.config = config or config_instance
        self.cache_warming_db = Path("data/cache/cache_warming.db")
        self.supplier_priorities = {
            # 🎯 Deutsche Elektrotechnik-Lieferanten Priorisierung
            "sonepar": {"priority": 10, "pattern_weight": 3.0},  # Höchste Priorität
            "amazon": {"priority": 8, "pattern_weight": 2.5},  # Verbrauchsmaterial
            "famo": {"priority": 9, "pattern_weight": 2.8},  # Installation/Werkzeug
            "würth": {"priority": 7, "pattern_weight": 2.0},  # Befestigungsmaterial
            "gira": {"priority": 6, "pattern_weight": 1.8},  # Elektroinstallation
            "hager": {"priority": 6, "pattern_weight": 1.8},  # Elektroinstallation
            "siemens": {"priority": 5, "pattern_weight": 1.5},  # Industrie-Komponenten
        }
        self._setup_warming_database()
        logger.info("🔥 IntelligentCacheWarming initialisiert")

    def _setup_warming_database(self) -> None:
        """Setup SQLite database für Cache-Warming Metriken."""
        self.cache_warming_db.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.cache_warming_db) as conn:
            # Usage Pattern Tracking
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS usage_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    supplier_name TEXT NOT NULL,
                    pdf_pattern TEXT NOT NULL,
                    usage_count INTEGER DEFAULT 1,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    average_processing_time REAL DEFAULT 0.0,
                    cache_hit_rate REAL DEFAULT 0.0,
                    business_value_score REAL DEFAULT 0.0
                )
            """
            )

            # Cache Warming History
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS warming_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    warming_session_id TEXT NOT NULL,
                    supplier_name TEXT NOT NULL,
                    files_warmed INTEGER DEFAULT 0,
                    cache_hits_generated INTEGER DEFAULT 0,
                    processing_time_saved REAL DEFAULT 0.0,
                    session_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Performance Metrics
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS performance_impact (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    supplier_name TEXT NOT NULL,
                    before_warming_time REAL NOT NULL,
                    after_warming_time REAL NOT NULL,
                    performance_improvement REAL NOT NULL,
                    cache_efficiency REAL NOT NULL,
                    measured_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Indizes für Performance
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_supplier_patterns ON usage_patterns(supplier_name)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_warming_sessions ON warming_history(warming_session_id)"
            )

        logger.info(
            "🔧 Cache-Warming-Database initialisiert: %s", self.cache_warming_db
        )

    async def analyze_supplier_patterns(
        self, pdf_directory: Path
    ) -> dict[str, list[Path]]:
        """
        Analysiert PDF-Dateien und ordnet sie Lieferanten zu.

        Args:
            pdf_directory: Verzeichnis mit PDF-Dateien zum Analysieren

        Returns:
            Dictionary: {supplier_name: [pdf_paths]}
        """
        supplier_files: dict[str, list[Path]] = {}

        if not pdf_directory.exists():
            logger.warning("⚠️ PDF-Verzeichnis nicht gefunden: %s", pdf_directory)
            return supplier_files

        pdf_files = list(pdf_directory.glob("*.pdf"))
        logger.info("📊 Analysiere %d PDF-Dateien für Supplier-Pattern", len(pdf_files))

        for pdf_path in pdf_files:
            filename_lower = pdf_path.name.lower()

            # 🎯 Pattern-Erkennung für deutsche Elektrotechnik-Lieferanten
            detected_supplier = None

            # Exact-Match-Patterns (hohe Konfidenz)
            if "sonepar" in filename_lower:
                detected_supplier = "sonepar"
            elif "amazon" in filename_lower:
                detected_supplier = "amazon"
            elif "famo" in filename_lower:
                detected_supplier = "famo"
            elif "würth" in filename_lower or "wuerth" in filename_lower:
                detected_supplier = "würth"
            elif "gira" in filename_lower:
                detected_supplier = "gira"
            elif "hager" in filename_lower:
                detected_supplier = "hager"
            elif "siemens" in filename_lower:
                detected_supplier = "siemens"

            # Fallback: Pattern-basierte Erkennung
            if not detected_supplier:
                # Elektrotechnik-spezifische Patterns
                if any(
                    pattern in filename_lower
                    for pattern in ["elektro", "installation", "schaltschrank"]
                ):
                    detected_supplier = "elektro_general"
                elif any(
                    pattern in filename_lower
                    for pattern in ["werkzeug", "befestigung", "schrauben"]
                ):
                    detected_supplier = "werkzeug_general"
                else:
                    detected_supplier = "unknown"

            # Supplier-Zuordnung
            if detected_supplier not in supplier_files:
                supplier_files[detected_supplier] = []
            supplier_files[detected_supplier].append(pdf_path)

            logger.debug("📄 %s → %s", pdf_path.name, detected_supplier)

        # 📊 Statistics-Logging
        for supplier, files in supplier_files.items():
            priority = self.supplier_priorities.get(supplier, {"priority": 1})[
                "priority"
            ]
            logger.info(
                "🏢 %s: %d PDFs (Priorität: %d)", supplier.upper(), len(files), priority
            )

        return supplier_files

    async def execute_intelligent_warming(
        self,
        supplier_files: dict[str, list[Path]],
        async_processor: Any,
        max_files_per_supplier: int = 10,
    ) -> dict[str, Any]:
        """
        Führt intelligentes Cache-Warming basierend auf Supplier-Prioritäten durch.

        Args:
            supplier_files: Dictionary mit Supplier → PDF-Dateien
            async_processor: AsyncGeminiDirectProcessor instance
            max_files_per_supplier: Maximum PDFs pro Lieferant

        Returns:
            Warming-Statistiken und Performance-Metriken
        """
        warming_session_id = f"warming_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        warming_start_time = datetime.now()

        total_files_warmed = 0
        total_cache_hits = 0
        supplier_results = {}

        # 🎯 Sortierung nach Business-Priorität
        sorted_suppliers = sorted(
            supplier_files.items(),
            key=lambda x: self.supplier_priorities.get(x[0], {"priority": 1})[
                "priority"
            ],
            reverse=True,
        )

        logger.info(
            "🔥 Intelligentes Cache-Warming startet für %d Suppliers",
            len(sorted_suppliers),
        )

        for supplier_name, pdf_files in sorted_suppliers:
            supplier_priority = self.supplier_priorities.get(
                supplier_name, {"priority": 1}
            )

            # Performance-Messung vor Warming
            before_warming_time = datetime.now()

            # Limitierung pro Supplier basierend auf Priorität
            priority_factor = supplier_priority["priority"] / 10.0  # 0.1 bis 1.0
            max_files = min(
                max_files_per_supplier, int(len(pdf_files) * priority_factor) + 1
            )

            selected_files = pdf_files[:max_files]

            logger.info(
                "🏢 Warming %s: %d/%d PDFs (Priorität: %d)",
                supplier_name.upper(),
                len(selected_files),
                len(pdf_files),
                supplier_priority["priority"],
            )

            # Cache-Warming für ausgewählte Dateien
            supplier_warming_count = 0
            supplier_cache_hits = 0

            for pdf_path in selected_files:
                try:
                    # Prüfe ob bereits gecacht
                    pdf_hash = await asyncio.get_event_loop().run_in_executor(
                        None, async_processor._calculate_pdf_hash, pdf_path
                    )

                    cached_result = await async_processor._check_cache(pdf_hash)
                    if cached_result:
                        supplier_cache_hits += 1
                        logger.debug("⚡ Cache-Hit: %s", pdf_path.name)
                        continue

                    # Cache-Miss → verarbeite und cache
                    logger.info("🔥 Warming: %s", pdf_path.name)
                    result = await async_processor.process_pdf_async(pdf_path)

                    if result:
                        supplier_warming_count += 1
                        total_files_warmed += 1

                except Exception as e:
                    logger.warning(
                        "⚠️ Warming fehlgeschlagen für %s: %s", pdf_path.name, e
                    )

            # Performance-Messung nach Warming
            after_warming_time = datetime.now()
            warming_duration = (
                after_warming_time - before_warming_time
            ).total_seconds()

            # Supplier-Ergebnisse speichern
            supplier_results[supplier_name] = {
                "files_processed": supplier_warming_count,
                "cache_hits": supplier_cache_hits,
                "total_files": len(pdf_files),
                "priority": supplier_priority["priority"],
                "warming_time_seconds": warming_duration,
                "efficiency": (
                    supplier_cache_hits / len(selected_files) if selected_files else 0.0
                ),
            }

            total_cache_hits += supplier_cache_hits

            # Persistiere Warming-History
            await self._record_warming_session(
                warming_session_id,
                supplier_name,
                supplier_warming_count,
                supplier_cache_hits,
                warming_duration,
            )

        # 📊 Gesamtstatistik
        total_warming_time = (datetime.now() - warming_start_time).total_seconds()

        warming_statistics = {
            "session_id": warming_session_id,
            "total_files_warmed": total_files_warmed,
            "total_cache_hits": total_cache_hits,
            "total_suppliers": len(sorted_suppliers),
            "warming_time_seconds": total_warming_time,
            "supplier_results": supplier_results,
            "efficiency_overall": (
                total_cache_hits / sum(len(files) for files in supplier_files.values())
                if supplier_files
                else 0.0
            ),
            "performance_improvement_estimate": total_files_warmed
            * 15.0,  # 15s pro PDF gespart
        }

        logger.info(
            "✅ Intelligentes Cache-Warming abgeschlossen: %d Dateien erwärmt, %d Cache-Hits, %.1fs",
            total_files_warmed,
            total_cache_hits,
            total_warming_time,
        )

        return warming_statistics

    async def _record_warming_session(
        self,
        session_id: str,
        supplier_name: str,
        files_warmed: int,
        cache_hits: int,
        processing_time: float,
    ) -> None:
        """Zeichnet Warming-Session für Analytics auf."""

        def _db_insert() -> None:
            with sqlite3.connect(self.cache_warming_db) as conn:
                conn.execute(
                    """
                    INSERT INTO warming_history
                    (warming_session_id, supplier_name, files_warmed, cache_hits_generated, processing_time_saved)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    (
                        session_id,
                        supplier_name,
                        files_warmed,
                        cache_hits,
                        processing_time,
                    ),
                )

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _db_insert)

    async def get_warming_analytics(self) -> dict[str, Any]:
        """
        Liefert Analytics für Cache-Warming Performance.

        Returns:
            Dictionary mit Warming-Metriken und Business-Intelligence
        """

        def _gather_analytics() -> dict[str, Any]:
            with sqlite3.connect(self.cache_warming_db) as conn:
                # Top-Performance Suppliers
                cursor = conn.execute(
                    """
                    SELECT supplier_name,
                           SUM(files_warmed) as total_warmed,
                           SUM(cache_hits_generated) as total_hits,
                           AVG(processing_time_saved) as avg_time_saved
                    FROM warming_history
                    GROUP BY supplier_name
                    ORDER BY total_warmed DESC
                """
                )
                top_suppliers = [
                    dict(zip([col[0] for col in cursor.description], row, strict=False))
                    for row in cursor.fetchall()
                ]

                # Recent Performance Trends
                cursor = conn.execute(
                    """
                    SELECT DATE(session_timestamp) as warming_date,
                           COUNT(*) as sessions,
                           SUM(files_warmed) as daily_warmed,
                           SUM(processing_time_saved) as daily_time_saved
                    FROM warming_history
                    WHERE session_timestamp > datetime('now', '-30 days')
                    GROUP BY DATE(session_timestamp)
                    ORDER BY warming_date DESC
                """
                )
                daily_trends = [
                    dict(zip([col[0] for col in cursor.description], row, strict=False))
                    for row in cursor.fetchall()
                ]

                # Overall Statistics
                cursor = conn.execute(
                    """
                    SELECT COUNT(*) as total_sessions,
                           SUM(files_warmed) as total_files_warmed,
                           SUM(cache_hits_generated) as total_cache_hits,
                           SUM(processing_time_saved) as total_time_saved
                    FROM warming_history
                """
                )
                overall_stats = dict(
                    zip(
                        [col[0] for col in cursor.description],
                        cursor.fetchone(),
                        strict=False,
                    )
                )

                return {
                    "top_suppliers": top_suppliers,
                    "daily_trends": daily_trends,
                    "overall_statistics": overall_stats,
                    "supplier_priorities": self.supplier_priorities,
                    "analysis_timestamp": datetime.now().isoformat(),
                }

        loop = asyncio.get_event_loop()
        analytics = await loop.run_in_executor(None, _gather_analytics)

        logger.info(
            "📊 Warming-Analytics: %d Sessions, %d Dateien erwärmt, %.1f Stunden gespart",
            analytics["overall_statistics"]["total_sessions"],
            analytics["overall_statistics"]["total_files_warmed"],
            analytics["overall_statistics"]["total_time_saved"] / 3600.0,
        )

        return analytics

    async def predictive_warming_recommendations(
        self, days_history: int = 30
    ) -> list[dict[str, Any]]:
        """
        Generiert predictive Empfehlungen für Cache-Warming basierend auf Usage-Patterns.

        Args:
            days_history: Anzahl Tage für historische Analyse

        Returns:
            Liste mit Warming-Empfehlungen sortiert nach Business-Impact
        """

        def _analyze_patterns() -> list[dict[str, Any]]:
            with sqlite3.connect(self.cache_warming_db) as conn:
                # Analyse der häufigsten Supplier-Pattern
                cursor = conn.execute(
                    f"""
                    SELECT supplier_name,
                           COUNT(*) as frequency,
                           AVG(processing_time_saved) as avg_time_saved,
                           MAX(session_timestamp) as last_warming
                    FROM warming_history
                    WHERE session_timestamp > datetime('now', '-{days_history} days')
                    GROUP BY supplier_name
                    HAVING frequency >= 2
                    ORDER BY frequency DESC, avg_time_saved DESC
                """
                )

                patterns = []
                for row in cursor.fetchall():
                    supplier_name, frequency, avg_time_saved, last_warming = row

                    # Business-Impact-Score berechnen
                    priority = self.supplier_priorities.get(
                        supplier_name, {"priority": 1}
                    )["priority"]
                    recency_factor = self._calculate_recency_factor(last_warming)

                    business_impact = (
                        (frequency * 0.4)
                        + (priority * 0.3)
                        + (avg_time_saved * 0.2)
                        + (recency_factor * 0.1)
                    )

                    patterns.append(
                        {
                            "supplier_name": supplier_name,
                            "frequency": frequency,
                            "avg_time_saved": round(avg_time_saved, 2),
                            "last_warming": last_warming,
                            "priority": priority,
                            "business_impact_score": round(business_impact, 2),
                            "recommendation": self._generate_recommendation(
                                supplier_name, frequency, avg_time_saved
                            ),
                        }
                    )

                return sorted(
                    patterns, key=lambda x: x["business_impact_score"], reverse=True
                )

        loop = asyncio.get_event_loop()
        recommendations = await loop.run_in_executor(None, _analyze_patterns)

        logger.info(
            "🎯 Generiert %d predictive Warming-Empfehlungen", len(recommendations)
        )

        return recommendations

    def _calculate_recency_factor(self, last_warming: str) -> float:
        """Berechnet Recency-Faktor für Business-Impact-Score."""
        try:
            last_date = datetime.fromisoformat(last_warming.replace("Z", "+00:00"))
            days_ago = (datetime.now() - last_date).days
            return max(0.1, 1.0 - (days_ago / 30.0))  # Linear decay über 30 Tage
        except Exception:
            return 0.1  # Fallback für ungültige Datumsformate

    def _generate_recommendation(
        self, supplier_name: str, frequency: int, avg_time_saved: float
    ) -> str:
        """Generiert spezifische Empfehlung basierend auf Metriken."""
        if frequency >= 5 and avg_time_saved >= 10.0:
            return f"HIGH PRIORITY: {supplier_name} häufig genutzt mit hohem Zeitgewinn - tägliches Warming empfohlen"
        elif frequency >= 3:
            return f"MEDIUM PRIORITY: {supplier_name} regelmäßig genutzt - wöchentliches Warming empfohlen"
        elif avg_time_saved >= 15.0:
            return f"PERFORMANCE FOCUS: {supplier_name} langsame Verarbeitung - proaktives Warming empfohlen"
        else:
            return f"LOW PRIORITY: {supplier_name} gelegentliche Nutzung - bedarfsbasiertes Warming"


# Convenience-Funktionen für CLI-Integration
async def warm_cache_intelligent(
    pdf_directory: Path, async_processor: Any, config: ConfigBridge | None = None
) -> dict[str, Any]:
    """Convenience function für intelligentes Cache-Warming."""
    warming_system = IntelligentCacheWarming(config)

    # Analysiere Supplier-Pattern
    supplier_files = await warming_system.analyze_supplier_patterns(pdf_directory)

    # Führe intelligentes Warming durch
    return await warming_system.execute_intelligent_warming(
        supplier_files, async_processor
    )


async def get_warming_recommendations(
    config: ConfigBridge | None = None,
) -> list[dict[str, Any]]:
    """Convenience function für Warming-Empfehlungen."""
    warming_system = IntelligentCacheWarming(config)
    return await warming_system.predictive_warming_recommendations()
