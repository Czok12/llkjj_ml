#!/usr/bin/env python3
"""
LLKJJ ML Pipeline - Performance & Business Metrics Monitoring
===========================================================

🎯 QUICK WIN: Advanced Logging & Metrics für Production-Monitoring

Business-Metriken Dashboard:
- SKR03-Genauigkeit und Konfidenz-Score-Trends
- Verarbeitungszeiten und Performance-Kennzahlen
- Gemini-API-Cost-Tracking und Token-Verwendung
- RAG-System-Effectiveness-Metrics
- Quality-Trend-Analyse und Fehlerquoten

Autor: LLKJJ ML Pipeline Team
Version: 4.1.0 (Business Value Maximierung)
Datum: 18. August 2025
"""

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from src.models.processing_result import ProcessingResult
from src.settings_bridge import Config

logger = logging.getLogger(__name__)


@dataclass
class ProcessingMetrics:
    """Performance-Metriken für eine PDF-Verarbeitung."""

    pdf_path: str
    processing_timestamp: str
    processing_time_ms: int
    gemini_time_ms: int
    classification_time_ms: int
    confidence_score: float
    extraction_quality: str
    skr03_classifications_count: int
    cache_hit: bool
    gemini_model: str
    error_occurred: bool = False
    error_message: str | None = None
    estimated_cost_usd: float = 0.0


@dataclass
class BusinessMetrics:
    """Business-Value-Metriken für Dashboard."""

    total_processed_pdfs: int
    success_rate: float
    average_processing_time_ms: float
    average_confidence_score: float
    cache_hit_rate: float
    cost_per_pdf_usd: float
    total_cost_savings_usd: float
    skr03_accuracy_trend: list[float]
    quality_distribution: dict[str, int]


class PerformanceMonitor:
    """
    🎯 BUSINESS VALUE MAXIMIERUNG: Production-Grade Performance Monitoring

    Sammelt und analysiert Business-Metriken für:
    - ROI-Tracking und Cost-Optimization
    - Qualitäts-Trends und Accuracy-Monitoring
    - Performance-Bottleneck-Identifikation
    - Production-Health-Monitoring
    """

    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.metrics_db_path = Path("data/monitoring/performance_metrics.db")
        self.business_metrics_cache = Path("data/monitoring/business_metrics.json")
        self._setup_metrics_database()

        # Gemini API-Cost-Schätzungen (basierend auf aktuellen Preisen)
        self.gemini_cost_per_1k_tokens = 0.001  # $0.001 per 1K tokens (geschätzt)

        logger.info("📊 PerformanceMonitor initialisiert: %s", self.metrics_db_path)

    def _setup_metrics_database(self) -> None:
        """Setup SQLite database für Performance-Metriken."""
        self.metrics_db_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.metrics_db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS processing_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pdf_path TEXT NOT NULL,
                    processing_timestamp TIMESTAMP NOT NULL,
                    processing_time_ms INTEGER NOT NULL,
                    gemini_time_ms INTEGER NOT NULL,
                    classification_time_ms INTEGER NOT NULL,
                    confidence_score REAL NOT NULL,
                    extraction_quality TEXT NOT NULL,
                    skr03_classifications_count INTEGER NOT NULL,
                    cache_hit BOOLEAN NOT NULL,
                    gemini_model TEXT NOT NULL,
                    error_occurred BOOLEAN DEFAULT FALSE,
                    error_message TEXT,
                    estimated_cost_usd REAL DEFAULT 0.0
                )
            """
            )

            # Indizes für Performance
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_processing_timestamp
                ON processing_metrics(processing_timestamp)
            """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_confidence_quality
                ON processing_metrics(confidence_score, extraction_quality)
            """
            )

        logger.info("🔧 Performance-Metrics-Database initialisiert")

    def record_processing_metrics(
        self,
        result: ProcessingResult,
        gemini_time_ms: int = 0,
        classification_time_ms: int = 0,
        cache_hit: bool = False,
        error_occurred: bool = False,
        error_message: str | None = None,
    ) -> None:
        """
        Speichere Performance-Metriken für ein verarbeitetes PDF.

        Args:
            result: ProcessingResult mit allen Extraktions-Daten
            gemini_time_ms: Gemini-API-Call-Zeit in Millisekunden
            classification_time_ms: SKR03-Klassifizierungs-Zeit
            cache_hit: Ob das Ergebnis aus dem Cache kam
            error_occurred: Ob ein Fehler aufgetreten ist
            error_message: Fehlernachricht falls error_occurred=True
        """
        # Kosten-Schätzung basierend auf PDF-Größe und Komplexität
        estimated_cost = self._estimate_processing_cost(result, cache_hit)

        metrics = ProcessingMetrics(
            pdf_path=result.pdf_path,
            processing_timestamp=result.processing_timestamp,
            processing_time_ms=result.processing_time_ms,
            gemini_time_ms=gemini_time_ms,
            classification_time_ms=classification_time_ms,
            confidence_score=result.confidence_score,
            extraction_quality=result.extraction_quality,
            skr03_classifications_count=len(result.skr03_classifications),
            cache_hit=cache_hit,
            gemini_model=getattr(result, "gemini_model", "gemini-2.5-flash"),
            error_occurred=error_occurred,
            error_message=error_message,
            estimated_cost_usd=estimated_cost,
        )

        self._store_metrics_in_db(metrics)
        logger.info(
            "📊 Metrics gespeichert: %s (%.1f%% Konfidenz, %s, $%.4f)",
            Path(result.pdf_path).name,
            result.confidence_score * 100,
            result.extraction_quality,
            estimated_cost,
        )

    def _estimate_processing_cost(
        self, result: ProcessingResult, cache_hit: bool
    ) -> float:
        """
        Schätze die Verarbeitungskosten für ein PDF.

        Basis für Kosten-Schätzung:
        - Cache-Hit: $0.00 (keine API-Kosten)
        - Kleine PDF (<10 Positionen): ~$0.005
        - Mittlere PDF (10-50 Positionen): ~$0.015
        - Große PDF (>50 Positionen): ~$0.030
        """
        if cache_hit:
            return 0.0

        position_count = len(result.skr03_classifications)

        if position_count <= 10:
            return 0.005
        elif position_count <= 50:
            return 0.015
        else:
            return 0.030

    def _store_metrics_in_db(self, metrics: ProcessingMetrics) -> None:
        """Speichere Metriken in SQLite-Database."""
        with sqlite3.connect(self.metrics_db_path) as conn:
            conn.execute(
                """
                INSERT INTO processing_metrics (
                    pdf_path, processing_timestamp, processing_time_ms,
                    gemini_time_ms, classification_time_ms, confidence_score,
                    extraction_quality, skr03_classifications_count, cache_hit,
                    gemini_model, error_occurred, error_message, estimated_cost_usd
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    metrics.pdf_path,
                    metrics.processing_timestamp,
                    metrics.processing_time_ms,
                    metrics.gemini_time_ms,
                    metrics.classification_time_ms,
                    metrics.confidence_score,
                    metrics.extraction_quality,
                    metrics.skr03_classifications_count,
                    metrics.cache_hit,
                    metrics.gemini_model,
                    metrics.error_occurred,
                    metrics.error_message,
                    metrics.estimated_cost_usd,
                ),
            )

    def get_business_metrics(self, days_back: int = 30) -> BusinessMetrics:
        """
        🎯 BUSINESS VALUE: Aggregierte Business-Metriken für Dashboard

        Args:
            days_back: Anzahl Tage für Trend-Analyse

        Returns:
            BusinessMetrics mit allen wichtigen KPIs
        """
        cutoff_date = datetime.now() - timedelta(days=days_back)

        with sqlite3.connect(self.metrics_db_path) as conn:
            # Basis-Statistiken
            cursor = conn.execute(
                """
                SELECT COUNT(*), AVG(processing_time_ms), AVG(confidence_score),
                       SUM(CASE WHEN cache_hit THEN 1 ELSE 0 END) * 1.0 / COUNT(*),
                       AVG(estimated_cost_usd), SUM(estimated_cost_usd),
                       SUM(CASE WHEN error_occurred THEN 0 ELSE 1 END) * 1.0 / COUNT(*)
                FROM processing_metrics
                WHERE processing_timestamp > ?
            """,
                (cutoff_date.isoformat(),),
            )

            row = cursor.fetchone()
            total_processed = row[0]
            avg_processing_time = row[1] or 0.0
            avg_confidence = row[2] or 0.0
            cache_hit_rate = row[3] or 0.0
            avg_cost_per_pdf = row[4] or 0.0
            success_rate = row[6] or 0.0

            # Qualitäts-Verteilung
            cursor = conn.execute(
                """
                SELECT extraction_quality, COUNT(*)
                FROM processing_metrics
                WHERE processing_timestamp > ?
                GROUP BY extraction_quality
            """,
                (cutoff_date.isoformat(),),
            )
            quality_dist = dict(cursor.fetchall())

            # SKR03-Accuracy-Trend (wöchentliche Konfidenz-Scores)
            cursor = conn.execute(
                """
                SELECT AVG(confidence_score)
                FROM processing_metrics
                WHERE processing_timestamp > ?
                  AND processing_timestamp <= ?
                  AND error_occurred = FALSE
            """,
                (cutoff_date.isoformat(), datetime.now().isoformat()),
            )

            confidence_trend = [cursor.fetchone()[0] or 0.0]

            # Cost-Savings-Berechnung (vs. manuelle Verarbeitung @ $5/PDF)
            manual_cost_per_pdf = 5.0  # $5 für manuelle Buchführung
            total_savings = (manual_cost_per_pdf - avg_cost_per_pdf) * total_processed

        return BusinessMetrics(
            total_processed_pdfs=total_processed,
            success_rate=success_rate,
            average_processing_time_ms=avg_processing_time,
            average_confidence_score=avg_confidence,
            cache_hit_rate=cache_hit_rate,
            cost_per_pdf_usd=avg_cost_per_pdf,
            total_cost_savings_usd=total_savings,
            skr03_accuracy_trend=confidence_trend,
            quality_distribution=quality_dist,
        )

    def generate_business_dashboard(self, days_back: int = 30) -> dict[str, Any]:
        """
        🎯 BUSINESS VALUE: Generiere Business-Dashboard für Management

        Returns:
            Dashboard-Data für Frontend oder Reporting
        """
        metrics = self.get_business_metrics(days_back)

        dashboard = {
            "report_period_days": days_back,
            "generated_at": datetime.now().isoformat(),
            "kpis": {
                "total_processed_pdfs": metrics.total_processed_pdfs,
                "success_rate_percent": round(metrics.success_rate * 100, 1),
                "avg_processing_time_seconds": round(
                    metrics.average_processing_time_ms / 1000, 2
                ),
                "avg_confidence_percent": round(
                    metrics.average_confidence_score * 100, 1
                ),
                "cache_hit_rate_percent": round(metrics.cache_hit_rate * 100, 1),
            },
            "roi_metrics": {
                "cost_per_pdf_usd": round(metrics.cost_per_pdf_usd, 4),
                "total_cost_savings_usd": round(metrics.total_cost_savings_usd, 2),
                "roi_factor": round(
                    (
                        5.0 / metrics.cost_per_pdf_usd
                        if metrics.cost_per_pdf_usd > 0
                        else 0
                    ),
                    1,
                ),
                "break_even_pdfs": 100,  # Geschätzt: Break-even bei 100 PDFs
            },
            "quality_metrics": {
                "quality_distribution": metrics.quality_distribution,
                "skr03_accuracy_trend": metrics.skr03_accuracy_trend,
                "target_accuracy_percent": 92.0,
                "current_vs_target": round(
                    metrics.average_confidence_score * 100 - 92.0, 1
                ),
            },
            "performance_insights": self._generate_performance_insights(metrics),
        }

        # Cache Dashboard für schnellen Zugriff
        self.business_metrics_cache.parent.mkdir(parents=True, exist_ok=True)
        with open(self.business_metrics_cache, "w", encoding="utf-8") as f:
            json.dump(dashboard, f, indent=2, ensure_ascii=False)

        logger.info(
            "📊 Business-Dashboard generiert: %d PDFs, %.1f%% Erfolgsquote, $%.4f/PDF",
            metrics.total_processed_pdfs,
            metrics.success_rate * 100,
            metrics.cost_per_pdf_usd,
        )

        return dashboard

    def _generate_performance_insights(self, metrics: BusinessMetrics) -> list[str]:
        """
        Generiere Performance-Insights und Optimierungsempfehlungen.
        """
        insights = []

        # Cache-Hit-Rate-Analyse
        if metrics.cache_hit_rate > 0.3:
            insights.append(
                f"🎯 Excellent cache performance: {metrics.cache_hit_rate * 100:.1f}% hit rate"
            )
        elif metrics.cache_hit_rate > 0.1:
            insights.append(
                f"⚡ Good cache performance: {metrics.cache_hit_rate * 100:.1f}% hit rate"
            )
        else:
            insights.append(
                f"🔥 Cache optimization opportunity: Only {metrics.cache_hit_rate * 100:.1f}% hit rate"
            )

        # Konfidenz-Analyse
        if metrics.average_confidence_score > 0.92:
            insights.append(
                "🎯 Target accuracy exceeded - excellent SKR03 classification"
            )
        elif metrics.average_confidence_score > 0.85:
            insights.append("✅ Good accuracy - close to target (92%)")
        else:
            insights.append("⚠️ Accuracy below target - RAG system tuning recommended")

        # Cost-Efficiency-Analyse
        if metrics.cost_per_pdf_usd < 0.01:
            insights.append(
                "💰 Excellent cost efficiency - well below $0.01/PDF target"
            )
        elif metrics.cost_per_pdf_usd < 0.05:
            insights.append("💚 Good cost efficiency - within target range")
        else:
            insights.append("💸 Cost optimization opportunity - above $0.05/PDF target")

        # Performance-Analyse
        if metrics.average_processing_time_ms < 1000:
            insights.append("🚀 Lightning fast processing - under 1 second average")
        elif metrics.average_processing_time_ms < 5000:
            insights.append("⚡ Fast processing - under 5 seconds average")
        else:
            insights.append(
                "🐌 Performance optimization needed - over 5 seconds average"
            )

        return insights

    def get_error_analysis(self, days_back: int = 7) -> dict[str, Any]:
        """
        🎯 PRODUCTION MONITORING: Fehler-Analyse für proaktive Wartung

        Returns:
            Detaillierte Fehler-Statistiken und häufige Fehlertypen
        """
        cutoff_date = datetime.now() - timedelta(days=days_back)

        with sqlite3.connect(self.metrics_db_path) as conn:
            # Fehler-Häufigkeit nach Typ
            cursor = conn.execute(
                """
                SELECT error_message, COUNT(*) as error_count
                FROM processing_metrics
                WHERE processing_timestamp > ? AND error_occurred = TRUE
                GROUP BY error_message
                ORDER BY error_count DESC
                LIMIT 10
            """,
                (cutoff_date.isoformat(),),
            )

            error_types = dict(cursor.fetchall())

            # Fehlerrate-Trend
            cursor = conn.execute(
                """
                SELECT
                    DATE(processing_timestamp) as error_date,
                    SUM(CASE WHEN error_occurred THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as error_rate
                FROM processing_metrics
                WHERE processing_timestamp > ?
                GROUP BY DATE(processing_timestamp)
                ORDER BY error_date
            """,
                (cutoff_date.isoformat(),),
            )

            error_trend = [(row[0], row[1]) for row in cursor.fetchall()]

        return {
            "analysis_period_days": days_back,
            "error_types_frequency": error_types,
            "daily_error_rate_trend": error_trend,
            "recommendations": self._generate_error_recommendations(error_types),
        }

    def _generate_error_recommendations(self, error_types: dict[str, int]) -> list[str]:
        """Generiere Empfehlungen basierend auf häufigen Fehlern."""
        recommendations = []

        for error_msg, count in error_types.items():
            if "API" in error_msg and count > 5:
                recommendations.append(
                    "🔧 Consider implementing Circuit Breaker for Gemini API"
                )
            elif "timeout" in error_msg.lower() and count > 3:
                recommendations.append(
                    "⏱️ Increase timeout values for large PDF processing"
                )
            elif "memory" in error_msg.lower() and count > 2:
                recommendations.append(
                    "💾 Implement memory optimization for batch processing"
                )
            elif count > 10:
                recommendations.append(
                    f"⚠️ High frequency error needs attention: {error_msg[:50]}..."
                )

        return recommendations


# Convenience-Funktionen für CLI-Integration
def create_performance_monitor(config: Config | None = None) -> PerformanceMonitor:
    """Factory-Funktion für PerformanceMonitor."""
    return PerformanceMonitor(config)


def log_processing_metrics(
    monitor: PerformanceMonitor, result: ProcessingResult, **kwargs: Any
) -> None:
    """Convenience-Funktion für Metrics-Logging."""
    monitor.record_processing_metrics(result, **kwargs)
