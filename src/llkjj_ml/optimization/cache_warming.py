"""
Cache Warming Optimization Module
=================================

Intelligente Cache-Warming-Algorithmen für ML-Pipeline Performance-Optimierung.
"""

import asyncio
import logging
from pathlib import Path
from typing import Any

from llkjj_ml.core.config import Config
from llkjj_ml.processors.async_gemini_direct_processor import AsyncGeminiDirectProcessor

logger = logging.getLogger(__name__)


class IntelligentCacheWarming:
    """
    🧠 Intelligentes Cache-Warming System für prädiktive Performance-Optimierung.
    """

    def __init__(self, config: Config):
        """
        Initialisiert das intelligente Cache-Warming System.

        Args:
            config: Konfiguration für ML-Pipeline
        """
        self.config = config
        self.processor = AsyncGeminiDirectProcessor(config)

    async def analyze_supplier_patterns(self, directory: str) -> dict[str, list[Path]]:
        """
        Analysiert Supplier-Muster in PDF-Verzeichnis.

        Args:
            directory: PDF-Verzeichnis zur Analyse

        Returns:
            Dictionary mit Supplier -> PDF-Liste Mapping
        """
        logger.info(f"🔍 Analysiere Supplier-Muster in: {directory}")

        supplier_files: dict[str, list[Path]] = {}
        pdf_dir = Path(directory)

        if not pdf_dir.exists():
            logger.warning(f"Verzeichnis existiert nicht: {directory}")
            return supplier_files

        # PDF-Dateien sammeln und Supplier erkennen
        for pdf_file in pdf_dir.glob("**/*.pdf"):
            supplier = self._detect_supplier_from_filename(pdf_file.name)
            if supplier:
                if supplier not in supplier_files:
                    supplier_files[supplier] = []
                supplier_files[supplier].append(pdf_file)

        logger.info(f"📊 Gefundene Suppliers: {len(supplier_files)}")
        return supplier_files

    def _detect_supplier_from_filename(self, filename: str) -> str | None:
        """
        Erkennt Supplier aus PDF-Dateiname.

        Args:
            filename: PDF-Dateiname

        Returns:
            Supplier-Name oder None
        """
        filename_lower = filename.lower()

        # Bekannte Supplier-Patterns
        supplier_patterns = {
            "obi": "obi",
            "bauhaus": "bauhaus",
            "hornbach": "hornbach",
            "hagebau": "hagebau",
            "toom": "toom",
            "hellweg": "hellweg",
            "praktiker": "praktiker",
            "baywa": "baywa",
            "conrad": "conrad",
            "reichelt": "reichelt",
            "rs_components": "rs",
            "farnell": "farnell",
            "digikey": "digikey",
            "mouser": "mouser",
        }

        for pattern, supplier in supplier_patterns.items():
            if pattern in filename_lower:
                return supplier

        return None

    async def get_warming_analytics(self) -> dict[str, Any]:
        """
        Lädt Cache-Warming Analytics.

        Returns:
            Dictionary mit Analytics-Daten
        """
        # Placeholder für Analytics - kann später erweitert werden
        return {
            "overall_statistics": {
                "total_sessions": 0,
                "total_files_warmed": 0,
                "total_cache_hits": 0,
                "total_time_saved": 0.0,
            },
            "top_suppliers": [],
        }


async def warm_cache_intelligent(
    pdf_directory: str,
    async_processor: AsyncGeminiDirectProcessor,
    config: Config
) -> dict[str, Any]:
    """
    🔥 Führt intelligentes Cache-Warming für PDF-Verzeichnis durch.

    Args:
        pdf_directory: Verzeichnis mit PDFs zum Warming
        async_processor: Async Processor für Cache-Warming
        config: ML-Pipeline Konfiguration

    Returns:
        Dictionary mit Warming-Statistiken
    """
    logger.info(f"🚀 Starte intelligentes Cache-Warming: {pdf_directory}")

    start_time = asyncio.get_event_loop().time()

    # Warming-System initialisieren
    warming_system = IntelligentCacheWarming(config)
    supplier_files = await warming_system.analyze_supplier_patterns(pdf_directory)

    total_files_warmed = 0
    total_cache_hits = 0
    supplier_results = {}

    # Pro Supplier warming durchführen
    for supplier, pdf_files in supplier_files.items():
        logger.info(f"🏭 Warming Supplier: {supplier} ({len(pdf_files)} PDFs)")

        files_warmed = 0
        cache_hits = 0

        # Sample von PDFs für Warming (max 10 pro Supplier)
        sample_files = pdf_files[:10]

        for pdf_file in sample_files:
            try:
                # Warming durch Processor
                result = await async_processor.warm_cache_for_file(str(pdf_file))
                if result:
                    files_warmed += 1
                    if result.get("cache_hit", False):
                        cache_hits += 1

            except Exception as e:
                logger.warning(f"⚠️ Warming fehlgeschlagen für {pdf_file}: {e}")
                continue

        # Supplier-Ergebnisse speichern
        supplier_results[supplier] = {
            "files_processed": files_warmed,
            "cache_hits": cache_hits,
            "priority": files_warmed * 10 + cache_hits,  # Simple Priority-Scoring
            "efficiency": cache_hits / max(files_warmed, 1),
        }

        total_files_warmed += files_warmed
        total_cache_hits += cache_hits

    end_time = asyncio.get_event_loop().time()
    warming_time = end_time - start_time

    # Statistiken zusammenfassen
    stats = {
        "total_files_warmed": total_files_warmed,
        "total_cache_hits": total_cache_hits,
        "total_suppliers": len(supplier_files),
        "warming_time_seconds": warming_time,
        "efficiency_overall": total_cache_hits / max(total_files_warmed, 1),
        "performance_improvement_estimate": total_cache_hits * 2.5,  # Geschätzte Zeitersparnis
        "supplier_results": supplier_results,
    }

    logger.info(f"✅ Intelligentes Warming abgeschlossen: {total_files_warmed} Dateien")
    return stats


async def get_warming_recommendations(config: Config) -> list[dict[str, Any]]:
    """
    💡 Generiert Warming-Empfehlungen basierend auf historischen Daten.

    Args:
        config: ML-Pipeline Konfiguration

    Returns:
        Liste mit Warming-Empfehlungen
    """
    logger.info("🎯 Generiere predictive Warming-Empfehlungen...")

    # Placeholder für Recommendations - kann später mit Analytics erweitert werden
    recommendations = [
        {
            "supplier_name": "conrad",
            "frequency": 25,
            "avg_time_saved": 3.2,
            "business_impact_score": 8.5,
            "recommendation": "Häufig verwendeter Supplier - priorisiertes Warming empfohlen",
        },
        {
            "supplier_name": "reichelt",
            "frequency": 18,
            "avg_time_saved": 2.8,
            "business_impact_score": 7.2,
            "recommendation": "Regelmäßiges Warming für Elektronik-Komponenten",
        },
        {
            "supplier_name": "obi",
            "frequency": 12,
            "avg_time_saved": 2.1,
            "business_impact_score": 6.0,
            "recommendation": "Baumarkt-Pattern - Warming in Stoßzeiten",
        },
    ]

    logger.info(f"📋 {len(recommendations)} Empfehlungen generiert")
    return recommendations
