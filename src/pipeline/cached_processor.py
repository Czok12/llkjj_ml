"""
Integration des Caching-Systems in den LLKJJ ML Pipeline
======================================================

Dieses Modul integriert das zweistufige Caching-System in den
UnifiedProcessor und AsyncUnifiedProcessor für optimale Performance.
"""

import asyncio
import logging
from pathlib import Path
from typing import Any

from src.caching import EmbeddingCache, SKR03Cache
from src.pipeline.async_processor import AsyncUnifiedProcessor
from src.pipeline.processor import ProcessingResult, UnifiedProcessor

logger = logging.getLogger(__name__)


class CachedUnifiedProcessor(UnifiedProcessor):
    """
    UnifiedProcessor mit integriertem Caching für optimale Performance.

    Features:
    - SKR03-Klassifizierung mit Cache
    - Embedding-Cache für ML-Models
    - Automatische Cache-Verwaltung
    """

    def __init__(
        self,
        cache_dir: Path | None = None,
        enable_skr03_cache: bool = True,
        enable_embedding_cache: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        # Cache-Verzeichnis festlegen
        if cache_dir is None:
            cache_dir = Path("data/cache")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Caches initialisieren
        self.skr03_cache = None
        self.embedding_cache = None

        if enable_skr03_cache:
            self.skr03_cache = SKR03Cache(
                db_path=self.cache_dir / "skr03_cache.db",
                max_memory_size=1000,
                default_ttl_hours=24,
            )

        if enable_embedding_cache:
            self.embedding_cache = EmbeddingCache(
                cache_dir=self.cache_dir / "embeddings",
                max_memory_size=500,
                default_ttl_hours=168,  # 1 Woche
            )

        logger.info("CachedUnifiedProcessor initialisiert mit Caching-Support")

    def process_invoice_with_cache(self, invoice: Any) -> dict[str, Any]:
        """
        Verarbeitet Rechnung mit Cache-Optimierung.

        Args:
            invoice: InvoiceData-Objekt

        Returns:
            Verarbeitungsergebnis mit Cache-Statistiken
        """
        result = {"invoice": invoice, "cache_stats": {}}

        # SKR03-Klassifizierung mit Cache
        if (
            self.skr03_cache
            and hasattr(invoice, "lieferant")
            and hasattr(invoice, "beschreibung")
        ):
            cache_key = self.skr03_cache.generate_key(
                beschreibung=invoice.beschreibung, lieferant=invoice.lieferant
            )

            # Cache lookup
            cached_classification = self.skr03_cache.get_classification(cache_key)
            if cached_classification:
                result["skr03_classification"] = cached_classification
                result["cache_stats"]["skr03_cache_hit"] = True
                logger.debug("SKR03-Klassifizierung aus Cache geladen")
            else:
                # Normale Verarbeitung
                classification = self._process_skr03_classification(invoice)
                if classification:
                    # Im Cache speichern
                    self.skr03_cache.set_classification(cache_key, classification)
                    result["skr03_classification"] = classification
                    result["cache_stats"]["skr03_cache_hit"] = False

        # Weitere Verarbeitung - delegate to parent class
        # result.update(self.process_invoice(invoice))  # Method not available

        # Cache-Statistiken hinzufügen
        if self.skr03_cache:
            result["cache_stats"]["skr03_stats"] = self.skr03_cache.get_stats()
        if self.embedding_cache:
            result["cache_stats"]["embedding_stats"] = self.embedding_cache.get_stats()

        return result

    def _process_skr03_classification(
        self,
        invoice: Any,  # noqa: ARG002, ANN401
    ) -> dict[str, Any] | None:
        """
        Führt SKR03-Klassifizierung durch (ohne Cache).

        Args:
            invoice: InvoiceData-Objekt

        Returns:
            Klassifizierungsergebnis oder None
        """
        # Hier würde die normale SKR03-Klassifizierung stattfinden
        # Das ist ein Placeholder für die Integration mit dem bestehenden System
        _ = invoice  # Placeholder for future use
        try:
            # Beispiel-Implementation
            # from src.processing.classifier import DataClassifier

            # classifier = DataClassifier()  # noqa: ERA001
            # result = classifier.classify_transaction(  # Method not available
            #     company_info=invoice.lieferant,
            #     description=invoice.beschreibung,
            #     amount=getattr(invoice, "betrag", 0.0),
            # )
            result = {"skr03_konto": "4400", "confidence": 0.8}  # Placeholder

            return {
                "konto": result.get("skr03_konto"),
                "confidence": result.get("confidence", 0.0),
                "kategorie": result.get("kategorie"),
                "timestamp": result.get("timestamp"),
            }
        except (ImportError, AttributeError) as e:
            logger.error("Fehler bei SKR03-Klassifizierung: %s", e)
            return None

    def get_cache_statistics(self) -> dict[str, Any]:
        """
        Holt umfassende Cache-Statistiken.

        Returns:
            Dictionary mit Cache-Statistiken
        """
        stats = {}

        if self.skr03_cache:
            stats["skr03"] = self.skr03_cache.get_stats()

        if self.embedding_cache:
            stats["embeddings"] = self.embedding_cache.get_stats()

        return stats

    def cleanup_caches(self) -> dict[str, int]:
        """
        Bereinigt abgelaufene Cache-Einträge.

        Returns:
            Dictionary mit Anzahl bereinigter Einträge
        """
        cleanup_stats = {}

        if self.skr03_cache:
            cleanup_stats["skr03_expired"] = self.skr03_cache.cleanup_expired()

        if self.embedding_cache:
            # Embedding cache hat eigene cleanup-Logik
            cleanup_stats["embeddings_saved"] = 1
            self.embedding_cache.save_cache()

        return cleanup_stats


class CachedAsyncUnifiedProcessor(AsyncUnifiedProcessor):
    """
    AsyncUnifiedProcessor mit integriertem Caching für optimale Performance.

    Erweitert die asynchrone Verarbeitung um Cache-Funktionalität.
    """

    def __init__(
        self,
        cache_dir: Path | None = None,
        enable_skr03_cache: bool = True,
        enable_embedding_cache: bool = True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        # Cache-Verzeichnis festlegen
        if cache_dir is None:
            cache_dir = Path("data/cache")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Caches initialisieren
        self.skr03_cache = None
        self.embedding_cache = None

        if enable_skr03_cache:
            self.skr03_cache = SKR03Cache(
                db_path=self.cache_dir / "skr03_cache.db",
                max_memory_size=1000,
                default_ttl_hours=24,
            )

        if enable_embedding_cache:
            self.embedding_cache = EmbeddingCache(
                cache_dir=self.cache_dir / "embeddings",
                max_memory_size=500,
                default_ttl_hours=168,
            )

        logger.info("CachedAsyncUnifiedProcessor initialisiert mit Caching-Support")

    async def process_pdf_batch_with_cache(
        self, pdf_paths: list[Path], max_concurrent: int = 3
    ) -> dict[str, Any]:
        """
        Verarbeitet PDF-Batch asynchron mit Cache-Optimierung.

        Args:
            pdf_paths: Liste der PDF-Pfade
            max_concurrent: Maximale gleichzeitige Verarbeitungen

        Returns:
            Batch-Verarbeitungsergebnis mit Cache-Statistiken
        """
        # Semaphore für Concurrency Control
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_single_pdf(pdf_path: Path) -> dict[str, Any]:
            async with semaphore:
                return await self._process_pdf_with_cache(pdf_path)

        # Alle PDFs parallel verarbeiten
        tasks = [process_single_pdf(pdf_path) for pdf_path in pdf_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Ergebnisse zusammenfassen
        successful_results = []
        error_results = []
        total_cache_hits = 0

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_results.append(
                    {"pdf_path": str(pdf_paths[i]), "error": str(result)}
                )
            else:
                successful_results.append(result)
                if isinstance(result, dict) and result.get("cache_stats", {}).get(
                    "skr03_cache_hit"
                ):
                    total_cache_hits += 1

        return {
            "total_processed": len(pdf_paths),
            "successful": len(successful_results),
            "errors": len(error_results),
            "results": successful_results,
            "error_details": error_results,
            "cache_performance": {
                "total_cache_hits": total_cache_hits,
                "cache_hit_rate": (
                    total_cache_hits / len(successful_results)
                    if successful_results
                    else 0
                ),
                "skr03_stats": self.skr03_cache.get_stats() if self.skr03_cache else {},
                "embedding_stats": (
                    self.embedding_cache.get_stats() if self.embedding_cache else {}
                ),
            },
        }

    async def _process_pdf_with_cache(self, pdf_path: Path) -> dict[str, Any]:
        """
        Verarbeitet PDF mit Cache-Integration.

        Args:
            pdf_path: Pfad zur PDF-Datei

        Returns:
            Verarbeitungsergebnis mit Cache-Integration
        """
        try:
            # Basis-Verarbeitung durch AsyncUnifiedProcessor
            result = await self.process_pdf_async(pdf_path)

            # Convert ProcessingResult to dict for type compatibility
            result_dict: dict[str, Any]
            if isinstance(result, ProcessingResult):
                result_dict = result.model_dump()
            else:
                # Handle dict case - this branch is reachable for error cases
                result_dict = dict(result) if result else {}  # type: ignore[unreachable]

            # Cache-Integration für jede extrahierte Rechnung
            if "invoices" in result_dict:
                cached_invoices = []
                for invoice_data in result_dict["invoices"]:
                    cached_result = await self._apply_cache_to_invoice(invoice_data)
                    cached_invoices.append(cached_result)
                result_dict["invoices"] = cached_invoices

            return result_dict

        except (OSError, ValueError) as e:
            logger.error("Fehler bei PDF-Verarbeitung %s: %s", pdf_path, e)
            return {"pdf_path": str(pdf_path), "success": False, "error": str(e)}

    async def _apply_cache_to_invoice(
        self, invoice_data: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Wendet Cache-Logik auf Rechnungsdaten an.

        Args:
            invoice_data: Rechnungsdaten-Dictionary

        Returns:
            Erweiterte Rechnungsdaten mit Cache-Informationen
        """
        result = invoice_data.copy()
        result["cache_stats"] = {}

        # SKR03-Cache anwenden
        if (
            self.skr03_cache
            and "lieferant" in invoice_data
            and "beschreibung" in invoice_data
        ):
            cache_key = self.skr03_cache.generate_key(
                beschreibung=invoice_data["beschreibung"],
                lieferant=invoice_data["lieferant"],
            )

            cached_classification = self.skr03_cache.get_classification(cache_key)
            if cached_classification:
                result["skr03_classification"] = cached_classification
                result["cache_stats"]["skr03_cache_hit"] = True
            else:
                # Hier würde normale Klassifizierung stattfinden
                # Dann im Cache speichern
                result["cache_stats"]["skr03_cache_hit"] = False

        return result

    async def get_cache_statistics_async(self) -> dict[str, Any]:
        """
        Holt Cache-Statistiken asynchron.

        Returns:
            Dictionary mit Cache-Statistiken
        """
        loop = asyncio.get_event_loop()

        stats = {}

        if self.skr03_cache:
            stats["skr03"] = await loop.run_in_executor(
                None, self.skr03_cache.get_stats
            )

        if self.embedding_cache:
            stats["embeddings"] = await loop.run_in_executor(
                None, self.embedding_cache.get_stats
            )

        return stats


def create_cached_processor(
    processor_type: str = "sync", cache_dir: Path | None = None, **kwargs: Any
) -> CachedUnifiedProcessor | CachedAsyncUnifiedProcessor:
    """
    Factory-Funktion für gecachte Prozessoren.

    Args:
        processor_type: "sync" oder "async"
        cache_dir: Cache-Verzeichnis
        **kwargs: Zusätzliche Processor-Parameter

    Returns:
        Gecachter Processor der gewünschten Art
    """
    if processor_type == "async":
        return CachedAsyncUnifiedProcessor(cache_dir=cache_dir, **kwargs)
    else:
        return CachedUnifiedProcessor(cache_dir=cache_dir, **kwargs)
