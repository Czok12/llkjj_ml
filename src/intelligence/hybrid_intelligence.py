#!/usr/bin/env python3
"""
LLKJJ ML Pipeline - Hybrid Intelligence Mode
============================================

🤖 Intelligenter Hybrid-Modus zwischen lokalen Modellen und Gemini

Features:
- Confidence-Threshold-Strategy: Lokale Modelle für >0.8, Gemini für <0.8
- Cost-Optimization-Logic: Gemini nur für schwierige Klassifizierungen
- Performance-Comparison-Framework: Lokale vs. Gemini Genauigkeit
- Seamless-Fallback: Transparenter Wechsel bei lokalen Modell-Fehlern

Autor: LLKJJ ML Pipeline Team
Version: 1.0.0 (Hybrid Intelligence)
Datum: 18. August 2025
"""

import logging
import sqlite3
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from src.config import Config

logger = logging.getLogger(__name__)


class ProcessingMode(Enum):
    """Processing-Modi für Hybrid Intelligence."""

    LOCAL_ONLY = "local_only"
    GEMINI_ONLY = "gemini_only"
    HYBRID_AUTO = "hybrid_auto"
    COST_OPTIMIZED = "cost_optimized"


class ClassificationSource(Enum):
    """Quelle der Klassifizierung."""

    LOCAL_SPACY = "local_spacy"
    LOCAL_CONTEXT_AWARE = "local_context_aware"
    GEMINI_DIRECT = "gemini_direct"
    HYBRID_COMBINED = "hybrid_combined"


class HybridResult(BaseModel):
    """Ergebnis der Hybrid-Intelligence-Klassifizierung."""

    classification: dict[str, Any] = Field(..., description="Finale Klassifizierung")
    source: ClassificationSource = Field(..., description="Verwendete Quelle")
    confidence: float = Field(..., description="Finale Konfidenz")
    local_confidence: float = Field(default=0.0, description="Lokale Modell-Konfidenz")
    gemini_confidence: float = Field(default=0.0, description="Gemini-Konfidenz")
    processing_time_ms: float = Field(..., description="Verarbeitungszeit")
    cost_estimate: float = Field(default=0.0, description="Geschätzte API-Kosten")
    fallback_triggered: bool = Field(default=False, description="Fallback verwendet")
    performance_comparison: dict[str, Any] = Field(default_factory=dict)


class PerformanceMetrics(BaseModel):
    """Performance-Metriken für Vergleich lokaler vs. Gemini-Modelle."""

    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    source: ClassificationSource = Field(...)
    confidence: float = Field(...)
    accuracy: float = Field(default=0.0, description="Gemessene Genauigkeit")
    processing_time_ms: float = Field(...)
    cost: float = Field(default=0.0, description="API-Kosten")
    success: bool = Field(default=True)
    error_message: str = Field(default="")


class HybridIntelligenceEngine:
    """
    🧠 Hybrid Intelligence Engine für optimale Kosten-Genauigkeits-Balance

    Entscheidet intelligent zwischen lokalen Modellen und Gemini basierend auf
    Konfidenz, Kosten und Performance-Metriken.
    """

    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.db_path: Path = Path("data") / "training" / "hybrid_intelligence.db"

        # Threshold-Konfiguration
        self.confidence_threshold = 0.8  # Schwellwert für lokale vs. Gemini
        self.cost_threshold_daily = 10.0  # Max. €10 Gemini-Kosten pro Tag
        self.fallback_enabled = True

        # Performance-Tracking
        self._init_database()

        logger.info("🧠 Hybrid Intelligence Engine initialisiert")

    def _init_database(self) -> None:
        """Initialisiert Performance-Tracking-Datenbank."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    source TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    accuracy REAL DEFAULT 0.0,
                    processing_time_ms REAL NOT NULL,
                    cost REAL DEFAULT 0.0,
                    success BOOLEAN NOT NULL,
                    error_message TEXT DEFAULT '',
                    document_type TEXT DEFAULT '',
                    supplier TEXT DEFAULT ''
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cost_tracking (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    gemini_requests INTEGER DEFAULT 0,
                    total_cost REAL DEFAULT 0.0,
                    cost_threshold_exceeded BOOLEAN DEFAULT FALSE
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS model_comparison (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    local_result TEXT,
                    gemini_result TEXT,
                    local_confidence REAL,
                    gemini_confidence REAL,
                    user_feedback TEXT DEFAULT '',
                    correct_classification TEXT DEFAULT ''
                )
            """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_performance_timestamp
                ON performance_metrics(timestamp)
            """
            )

            conn.commit()

    def classify_intelligent(
        self,
        text: str,
        mode: ProcessingMode = ProcessingMode.HYBRID_AUTO,
        document_type: str = "invoice",
        supplier: str = "",
    ) -> HybridResult:
        """
        🎯 Intelligente Klassifizierung mit Hybrid-Strategie

        Args:
            text: Zu klassifizierender Text
            mode: Processing-Modus
            document_type: Dokumenttyp für Kontext
            supplier: Lieferant für Kontext

        Returns:
            HybridResult: Optimale Klassifizierung
        """
        start_time = datetime.now()

        try:
            # Kosten-Check für heutigen Tag
            daily_cost = self._get_daily_gemini_cost()
            cost_limit_reached = daily_cost >= self.cost_threshold_daily

            # Entscheidungslogik basierend auf Modus
            if mode == ProcessingMode.LOCAL_ONLY or cost_limit_reached:
                return self._classify_local_only(text, start_time, supplier)

            elif mode == ProcessingMode.GEMINI_ONLY:
                return self._classify_gemini_only(text, start_time, supplier)

            elif mode == ProcessingMode.HYBRID_AUTO:
                return self._classify_hybrid_auto(text, start_time, supplier)

            elif mode == ProcessingMode.COST_OPTIMIZED:
                return self._classify_cost_optimized(
                    text, start_time, supplier, daily_cost
                )

        except Exception as e:
            logger.error("❌ Hybrid Intelligence Klassifizierung fehlgeschlagen: %s", e)

            # Fallback auf lokale Klassifizierung
            return self._emergency_fallback(text, start_time, str(e))

    def _classify_local_only(
        self, text: str, start_time: datetime, supplier: str
    ) -> HybridResult:
        """🏠 Lokale Klassifizierung mit Context-Aware-Logik."""
        try:
            # Simuliere lokale Klassifizierung (würde in Realität Context-Aware-Classifier verwenden)
            local_confidence = 0.85  # Simulierte lokale Konfidenz
            local_classification = {
                "skr03_account": "3400",  # Default Elektro
                "description": text[:100],
                "supplier": supplier,
                "confidence": local_confidence,
            }

            processing_time = (datetime.now() - start_time).total_seconds() * 1000

            # Metriken loggen
            self._log_performance_metrics(
                PerformanceMetrics(
                    source=ClassificationSource.LOCAL_CONTEXT_AWARE,
                    confidence=local_confidence,
                    processing_time_ms=processing_time,
                    cost=0.0,  # Lokale Modelle kosten nichts
                    success=True,
                )
            )

            return HybridResult(
                classification=local_classification,
                source=ClassificationSource.LOCAL_CONTEXT_AWARE,
                confidence=local_confidence,
                local_confidence=local_confidence,
                processing_time_ms=processing_time,
                cost_estimate=0.0,
            )

        except Exception as e:
            logger.error("❌ Lokale Klassifizierung fehlgeschlagen: %s", e)
            return self._emergency_fallback(text, start_time, str(e))

    def _classify_gemini_only(
        self, text: str, start_time: datetime, supplier: str
    ) -> HybridResult:
        """☁️ Gemini-only Klassifizierung."""
        try:
            # Simuliere Gemini-Klassifizierung (würde in Realität GeminiDirectProcessor verwenden)
            gemini_confidence = 0.95  # Gemini typisch höhere Konfidenz
            gemini_classification = {
                "skr03_account": "3400",
                "description": text[:100],
                "supplier": supplier,
                "confidence": gemini_confidence,
            }

            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            estimated_cost = 0.002  # ~2 Cent pro Request

            # Kosten tracken
            self._track_gemini_cost(estimated_cost)

            # Metriken loggen
            self._log_performance_metrics(
                PerformanceMetrics(
                    source=ClassificationSource.GEMINI_DIRECT,
                    confidence=gemini_confidence,
                    processing_time_ms=processing_time,
                    cost=estimated_cost,
                    success=True,
                )
            )

            return HybridResult(
                classification=gemini_classification,
                source=ClassificationSource.GEMINI_DIRECT,
                confidence=gemini_confidence,
                gemini_confidence=gemini_confidence,
                processing_time_ms=processing_time,
                cost_estimate=estimated_cost,
            )

        except Exception as e:
            logger.error("❌ Gemini-Klassifizierung fehlgeschlagen: %s", e)
            return self._emergency_fallback(text, start_time, str(e))

    def _classify_hybrid_auto(
        self, text: str, start_time: datetime, supplier: str
    ) -> HybridResult:
        """🤖 Automatischer Hybrid-Modus basierend auf lokaler Konfidenz."""
        try:
            # Zuerst lokale Klassifizierung versuchen
            local_result = self._classify_local_only(text, start_time, supplier)

            # Entscheidung basierend auf lokaler Konfidenz
            if local_result.local_confidence >= self.confidence_threshold:
                # Lokale Konfidenz ausreichend
                logger.info(
                    "💚 Lokale Konfidenz ausreichend: %.3f",
                    local_result.local_confidence,
                )
                return local_result

            else:
                # Lokale Konfidenz zu niedrig → Gemini verwenden
                logger.info(
                    "🟡 Lokale Konfidenz zu niedrig: %.3f → Gemini",
                    local_result.local_confidence,
                )

                gemini_result = self._classify_gemini_only(text, start_time, supplier)

                # Hybrid-Ergebnis kombinieren
                return HybridResult(
                    classification=gemini_result.classification,
                    source=ClassificationSource.HYBRID_COMBINED,
                    confidence=gemini_result.confidence,
                    local_confidence=local_result.local_confidence,
                    gemini_confidence=gemini_result.confidence,
                    processing_time_ms=local_result.processing_time_ms
                    + gemini_result.processing_time_ms,
                    cost_estimate=gemini_result.cost_estimate,
                    fallback_triggered=True,
                    performance_comparison={
                        "local_was_insufficient": True,
                        "confidence_gap": gemini_result.confidence
                        - local_result.local_confidence,
                    },
                )

        except Exception as e:
            logger.error("❌ Hybrid-Auto Klassifizierung fehlgeschlagen: %s", e)
            return self._emergency_fallback(text, start_time, str(e))

    def _classify_cost_optimized(
        self, text: str, start_time: datetime, supplier: str, daily_cost: float
    ) -> HybridResult:
        """💰 Kosten-optimierte Klassifizierung."""
        try:
            # Budget-basierte Entscheidung
            remaining_budget = max(0, self.cost_threshold_daily - daily_cost)
            estimated_gemini_cost = 0.002

            if remaining_budget < estimated_gemini_cost:
                # Budget erschöpft → nur lokale Modelle
                logger.info("💸 Budget erschöpft, verwende lokale Modelle")
                return self._classify_local_only(text, start_time, supplier)

            # Lokale Klassifizierung zuerst versuchen
            local_result = self._classify_local_only(text, start_time, supplier)

            # Strengerer Threshold für Cost-Optimization
            cost_optimized_threshold = self.confidence_threshold + 0.1

            if local_result.local_confidence >= cost_optimized_threshold:
                return local_result
            else:
                # Gemini nur bei wirklich unsicheren Fällen
                logger.info("📊 Cost-Optimization: Gemini für schwierigen Fall")
                return self._classify_gemini_only(text, start_time, supplier)

        except Exception as e:
            logger.error("❌ Cost-Optimized Klassifizierung fehlgeschlagen: %s", e)
            return self._emergency_fallback(text, start_time, str(e))

    def _emergency_fallback(
        self, text: str, start_time: datetime, error: str
    ) -> HybridResult:
        """🆘 Notfall-Fallback bei kritischen Fehlern."""
        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        fallback_classification: dict[str, Any] = {
            "skr03_account": "3400",  # Safe default für Elektro
            "description": text[:100],
            "confidence": 0.5,  # Niedrige Konfidenz für Fallback
            "fallback_reason": error,
        }

        return HybridResult(
            classification=fallback_classification,
            source=ClassificationSource.LOCAL_CONTEXT_AWARE,
            confidence=0.5,
            local_confidence=0.5,
            processing_time_ms=processing_time,
            cost_estimate=0.0,
            fallback_triggered=True,
            performance_comparison={"emergency_fallback": True, "error": error},
        )

    def _get_daily_gemini_cost(self) -> float:
        """💰 Gibt heutige Gemini-Kosten zurück."""
        today = datetime.now().strftime("%Y-%m-%d")

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT total_cost FROM cost_tracking WHERE date = ?
                """,
                    (today,),
                )
                result = cursor.fetchone()
                return result[0] if result else 0.0

        except Exception:
            return 0.0

    def _track_gemini_cost(self, cost: float) -> None:
        """📊 Trackt Gemini-Kosten für Budget-Management."""
        today = datetime.now().strftime("%Y-%m-%d")

        try:
            with sqlite3.connect(self.db_path) as conn:
                # Aktualisiere oder erstelle Tageseintrag
                conn.execute(
                    """
                    INSERT OR REPLACE INTO cost_tracking (date, gemini_requests, total_cost)
                    VALUES (?,
                            COALESCE((SELECT gemini_requests FROM cost_tracking WHERE date = ?), 0) + 1,
                            COALESCE((SELECT total_cost FROM cost_tracking WHERE date = ?), 0) + ?)
                """,
                    (today, today, today, cost),
                )
                conn.commit()

        except Exception as e:
            logger.warning("⚠️ Konnte Kosten nicht tracken: %s", e)

    def _log_performance_metrics(self, metrics: PerformanceMetrics) -> None:
        """📈 Loggt Performance-Metriken für Analyse."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO performance_metrics
                    (timestamp, source, confidence, processing_time_ms, cost, success, error_message)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        metrics.timestamp,
                        metrics.source.value,
                        metrics.confidence,
                        metrics.processing_time_ms,
                        metrics.cost,
                        metrics.success,
                        metrics.error_message,
                    ),
                )
                conn.commit()

        except Exception as e:
            logger.warning("⚠️ Konnte Metriken nicht loggen: %s", e)

    def get_performance_comparison(self, days_back: int = 7) -> dict[str, Any]:
        """
        📊 Performance-Vergleich lokale vs. Gemini-Modelle

        Args:
            days_back: Tage zurückblicken

        Returns:
            dict: Detaillierter Performance-Vergleich
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Vergleich nach Quelle
                cursor = conn.execute(
                    f"""
                    SELECT
                        source,
                        COUNT(*) as requests,
                        AVG(confidence) as avg_confidence,
                        AVG(processing_time_ms) as avg_processing_time,
                        SUM(cost) as total_cost,
                        AVG(accuracy) as avg_accuracy
                    FROM performance_metrics
                    WHERE datetime(timestamp) >= datetime('now', '-{days_back} days')
                    AND success = 1
                    GROUP BY source
                """
                )

                source_stats = {}
                for row in cursor.fetchall():
                    source_stats[row[0]] = {
                        "requests": row[1],
                        "avg_confidence": round(row[2], 3),
                        "avg_processing_time_ms": round(row[3], 2),
                        "total_cost": round(row[4], 4),
                        "avg_accuracy": round(row[5], 3),
                    }

                # Kosten-Übersicht
                cursor = conn.execute(
                    f"""
                    SELECT date, total_cost, gemini_requests
                    FROM cost_tracking
                    WHERE datetime(date) >= datetime('now', '-{days_back} days')
                    ORDER BY date DESC
                """
                )

                daily_costs = [
                    {"date": row[0], "cost": row[1], "requests": row[2]}
                    for row in cursor.fetchall()
                ]

                return {
                    "source_comparison": source_stats,
                    "daily_costs": daily_costs,
                    "cost_efficiency": self._calculate_cost_efficiency(source_stats),
                    "recommendations": self._generate_optimization_recommendations(
                        source_stats
                    ),
                    "period_days": days_back,
                }

        except Exception as e:
            logger.error("❌ Performance-Vergleich fehlgeschlagen: %s", e)
            return {"error": str(e)}

    def _calculate_cost_efficiency(
        self, source_stats: dict[str, Any]
    ) -> dict[str, Any]:
        """💡 Berechnet Kosten-Effizienz-Metriken."""
        local_sources = ["local_spacy", "local_context_aware"]
        gemini_sources = ["gemini_direct"]

        local_total = sum(
            source_stats.get(source, {}).get("requests", 0) for source in local_sources
        )
        gemini_total = sum(
            source_stats.get(source, {}).get("requests", 0) for source in gemini_sources
        )

        total_requests = local_total + gemini_total
        local_percentage = (
            (local_total / total_requests * 100) if total_requests > 0 else 0
        )

        total_cost = sum(stats.get("total_cost", 0) for stats in source_stats.values())

        return {
            "local_percentage": round(local_percentage, 1),
            "cost_per_request": (
                round(total_cost / total_requests, 4) if total_requests > 0 else 0
            ),
            "total_cost": round(total_cost, 2),
            "cost_savings_vs_gemini_only": round(
                (local_percentage / 100) * 0.002 * total_requests, 2
            ),
        }

    def _generate_optimization_recommendations(
        self, source_stats: dict[str, Any]
    ) -> list[str]:
        """🎯 Generiert Optimierungs-Empfehlungen."""
        recommendations = []

        # Konfidenz-Analyse
        local_confidence = source_stats.get("local_context_aware", {}).get(
            "avg_confidence", 0
        )
        gemini_confidence = source_stats.get("gemini_direct", {}).get(
            "avg_confidence", 0
        )

        if local_confidence > 0.85:
            recommendations.append(
                "🚀 Lokale Modelle zeigen hohe Konfidenz - Threshold erhöhen"
            )

        if gemini_confidence - local_confidence > 0.15:
            recommendations.append(
                "📚 Große Konfidenz-Lücke - lokale Modelle trainieren"
            )

        # Kosten-Analyse
        total_cost = sum(stats.get("total_cost", 0) for stats in source_stats.values())
        if total_cost > self.cost_threshold_daily * 0.8:
            recommendations.append(
                "💰 Kosten-Threshold bald erreicht - mehr lokale Klassifizierung"
            )

        # Performance-Analyse
        local_time = source_stats.get("local_context_aware", {}).get(
            "avg_processing_time_ms", 0
        )
        gemini_time = source_stats.get("gemini_direct", {}).get(
            "avg_processing_time_ms", 0
        )

        if local_time < gemini_time * 0.5:
            recommendations.append("⚡ Lokale Modelle deutlich schneller - bevorzugen")

        return recommendations or ["✅ System läuft optimal - keine Änderungen nötig"]

    def update_confidence_threshold(self, new_threshold: float) -> dict[str, Any]:
        """⚙️ Aktualisiert Konfidenz-Schwellwert basierend auf Performance."""
        old_threshold = self.confidence_threshold
        self.confidence_threshold = max(
            0.1, min(0.95, new_threshold)
        )  # Sicherheitsgrenzen

        logger.info(
            "⚙️ Konfidenz-Threshold aktualisiert: %.3f → %.3f",
            old_threshold,
            self.confidence_threshold,
        )

        return {
            "old_threshold": old_threshold,
            "new_threshold": self.confidence_threshold,
            "change": self.confidence_threshold - old_threshold,
            "impact_estimate": f"~{abs(self.confidence_threshold - old_threshold) * 100:.1f}% Shift in Local/Gemini Balance",
        }


# Convenience Functions
def create_hybrid_engine(config: Config | None = None) -> HybridIntelligenceEngine:
    """Factory-Function für HybridIntelligenceEngine."""
    return HybridIntelligenceEngine(config)


def classify_with_hybrid_intelligence(
    text: str,
    mode: ProcessingMode = ProcessingMode.HYBRID_AUTO,
    supplier: str = "",
) -> HybridResult:
    """
    Einfache Hybrid-Intelligence-Klassifizierung.

    Args:
        text: Zu klassifizierender Text
        mode: Processing-Modus
        supplier: Lieferant für Kontext

    Returns:
        HybridResult: Optimale Klassifizierung
    """
    engine = create_hybrid_engine()
    return engine.classify_intelligent(text, mode, supplier=supplier)
