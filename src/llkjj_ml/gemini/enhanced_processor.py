"""
Enhanced Gemini Processor - Erweiterte Gemini-Verarbeitung mit Retry und Validation

Erweitert den GeminiDirectProcessor um:
- Retry-Mechanismen mit exponential backoff
- Validierung der Extraktion auf Vollständigkeit
- Caching für verbesserte Performance
- Enhanced Error-Handling

Author: LLKJJ Team
Version: 1.0.0
Date: 2025-09-15
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from llkjj_ml.gemini.direct_processor import (
    GeminiDirectProcessor,
    GeminiDirectResult,
)

logger = logging.getLogger(__name__)


class RetryConfig(BaseModel):
    """Konfiguration für Retry-Mechanismen."""

    max_retries: int = 3
    initial_delay: float = 1.0  # Sekunden
    max_delay: float = 30.0  # Sekunden
    exponential_base: float = 2.0
    retry_on_errors: list[str] = [
        "quota_exceeded",
        "rate_limit",
        "timeout",
        "connection_error",
        "internal_error",
    ]


class ValidationConfig(BaseModel):
    """Konfiguration für Extraktion-Validierung."""

    required_fields: list[str] = [
        "supplier_name",
        "invoice_date",
        "invoice_number",
        "total_amount",
    ]
    minimum_confidence: float = 0.7
    require_invoice_items: bool = True
    minimum_item_count: int = 1


class ProcessingMetrics(BaseModel):
    """Metriken für Performance-Monitoring."""

    total_attempts: int = 0
    successful_attempts: int = 0
    failed_attempts: int = 0
    retry_attempts: int = 0
    avg_processing_time: float = 0.0
    last_processing_time: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0


class EnhancedGeminiProcessor(GeminiDirectProcessor):
    """
    Erweiterte Gemini-Verarbeitung mit Retry und Caching.

    Baut auf dem GeminiDirectProcessor auf und fügt hinzu:
    - Retry-Mechanismen für robuste Verarbeitung
    - Validierung der Extraktions-Vollständigkeit
    - Performance-Monitoring
    - Intelligent Caching
    """

    def __init__(
        self,
        config: Any = None,
        retry_config: RetryConfig | None = None,
        validation_config: ValidationConfig | None = None,
    ):
        """
        Initialisiert den Enhanced Processor.

        Args:
            config: Basis-Konfiguration für Gemini
            retry_config: Retry-Mechanismus Konfiguration
            validation_config: Validierungs-Konfiguration
        """
        super().__init__(config)

        self.retry_config = retry_config or RetryConfig()
        self.validation_config = validation_config or ValidationConfig()
        self.metrics = ProcessingMetrics()

        # Simple In-Memory Cache für häufige Lieferanten
        self._supplier_cache: dict[str, dict[str, Any]] = {}
        self._processing_cache: dict[str, GeminiDirectResult] = {}

        logger.info("✅ EnhancedGeminiProcessor initialisiert mit Retry & Validation")

    async def process_with_retry(
        self, pdf_path: Path, use_cache: bool = True
    ) -> GeminiDirectResult:
        """
        Verarbeitung mit automatischen Wiederholungen bei Fehlern.

        Args:
            pdf_path: Pfad zur PDF-Datei
            use_cache: Ob Cache verwendet werden soll

        Returns:
            GeminiDirectResult mit enhanced validation

        Raises:
            Exception: Nach erschöpften Retry-Versuchen
        """
        self.metrics.total_attempts += 1

        # Cache-Check
        if use_cache:
            cached_result = await self._check_cache(pdf_path)
            if cached_result:
                self.metrics.cache_hits += 1
                return cached_result

        self.metrics.cache_misses += 1

        # Retry-Loop
        last_exception = None
        for attempt in range(self.retry_config.max_retries + 1):
            try:
                start_time = time.time()

                # Basis-Verarbeitung
                result = await super().process_pdf_direct(pdf_path)

                # Enhanced Validation
                validation_result = self.validate_extraction_completeness(result)

                processing_time = time.time() - start_time
                self.metrics.last_processing_time = processing_time
                self._update_avg_processing_time(processing_time)

                if validation_result["is_complete"]:
                    # Erfolg
                    result.success = True
                    self.metrics.successful_attempts += 1

                    # Cache speichern
                    if use_cache:
                        await self._store_in_cache(pdf_path, result)

                    logger.info(
                        f"Enhanced processing successful after {attempt + 1} attempts"
                    )
                    return result
                else:
                    # Validation failed
                    result.errors.extend(validation_result["missing_fields"])
                    logger.warning(
                        f"Validation failed: {validation_result['missing_fields']}"
                    )

                    # Bei letztem Versuch: Partial Result zurückgeben
                    if attempt == self.retry_config.max_retries:
                        result.success = False
                        return result

                    # Retry für Validation-Fehler
                    raise ValueError(
                        f"Validation failed: {validation_result['reason']}"
                    )

            except Exception as e:
                last_exception = e
                self.metrics.retry_attempts += 1

                logger.warning(f"Attempt {attempt + 1} failed: {e}")

                # Letzter Versuch - Exception werfen
                if attempt == self.retry_config.max_retries:
                    self.metrics.failed_attempts += 1
                    break

                # Exponential Backoff
                delay = min(
                    self.retry_config.initial_delay
                    * (self.retry_config.exponential_base**attempt),
                    self.retry_config.max_delay,
                )

                logger.info(f"Retrying in {delay:.1f} seconds...")
                await asyncio.sleep(delay)

        # Alle Versuche fehlgeschlagen
        self.metrics.failed_attempts += 1
        raise Exception(
            f"Enhanced processing failed after {self.retry_config.max_retries + 1} attempts. Last error: {last_exception}"
        )

    def validate_extraction_completeness(
        self, result: GeminiDirectResult
    ) -> dict[str, Any]:
        """
        Prüft ob alle kritischen Felder extrahiert wurden.

        Args:
            result: Gemini-Verarbeitungsergebnis

        Returns:
            Dict mit Validierungs-Informationen
        """
        validation_info: dict[str, Any] = {
            "is_complete": True,
            "missing_fields": [],
            "confidence_issues": [],
            "item_issues": [],
            "reason": "",
        }

        # 1. Required Fields prüfen
        for field in self.validation_config.required_fields:
            value = result.invoice_data.get(field)
            if not value or value in ("", "N/A"):
                validation_info["missing_fields"].append(field)
                validation_info["is_complete"] = False

        # 2. Minimum Confidence prüfen
        overall_confidence = result.invoice_data.get("confidence", 0)
        if overall_confidence < self.validation_config.minimum_confidence:
            validation_info["confidence_issues"].append(
                f"Overall confidence {overall_confidence} below minimum {self.validation_config.minimum_confidence}"
            )
            validation_info["is_complete"] = False

        # 3. Invoice Items prüfen
        if self.validation_config.require_invoice_items:
            if len(result.invoice_items) < self.validation_config.minimum_item_count:
                validation_info["item_issues"].append(
                    f"Only {len(result.invoice_items)} items found, minimum {self.validation_config.minimum_item_count} required"
                )
                validation_info["is_complete"] = False

        # 4. Item-spezifische Validierung
        for i, item in enumerate(result.invoice_items):
            if not item.description or item.description.strip() == "":
                validation_info["item_issues"].append(
                    f"Item {i+1}: Missing description"
                )
                validation_info["is_complete"] = False

            if not item.skr03_account:
                validation_info["item_issues"].append(
                    f"Item {i+1}: Missing SKR03 account"
                )
                validation_info["is_complete"] = False

        # 5. Zusammenfassung
        if not validation_info["is_complete"]:
            reasons = []
            if validation_info["missing_fields"]:
                reasons.append(f"Missing fields: {validation_info['missing_fields']}")
            if validation_info["confidence_issues"]:
                reasons.append(
                    f"Low confidence: {validation_info['confidence_issues']}"
                )
            if validation_info["item_issues"]:
                reasons.append(f"Item issues: {validation_info['item_issues']}")

            validation_info["reason"] = "; ".join(reasons)

        return validation_info

    async def enhance_with_supplier_templates(
        self, result: GeminiDirectResult
    ) -> GeminiDirectResult:
        """
        Verbessert Ergebnisse mit bekannten Lieferanten-Templates.

        Args:
            result: Basis-Ergebnis von Gemini

        Returns:
            Enhanced result mit Template-Daten
        """
        supplier_name = result.invoice_data.get("supplier_name", "").lower().strip()

        if supplier_name in self._supplier_cache:
            template = self._supplier_cache[supplier_name]

            # SKR03-Accounts aus Template ergänzen
            for item in result.invoice_items:
                if not item.skr03_account and template.get("default_skr03_account"):
                    item.skr03_account = template["default_skr03_account"]
                    item.classification_reasoning += (
                        f" (aus Lieferanten-Template für {supplier_name})"
                    )

            # Standard-Kategorien ergänzen
            if template.get("default_category"):
                for item in result.invoice_items:
                    if not item.skr03_category:
                        item.skr03_category = template["default_category"]

            logger.debug(f"Enhanced result with template for supplier: {supplier_name}")

        return result

    async def learn_from_validation(
        self,
        original_result: GeminiDirectResult,
        validated_data: dict[str, Any],
        user_corrections: dict[str, Any],
    ) -> None:
        """
        Lernt aus User-Validierungen für bessere zukünftige Ergebnisse.

        Args:
            original_result: Original Gemini-Ergebnis
            validated_data: Finale validierte Daten
            user_corrections: User-Korrekturen
        """
        supplier_name = validated_data.get("supplier_name", "").lower().strip()

        if supplier_name and supplier_name not in self._supplier_cache:
            # Neuen Supplier-Template erstellen
            self._supplier_cache[supplier_name] = {
                "default_skr03_account": validated_data.get("items", [{}])[0].get(
                    "skr03_account"
                ),
                "default_category": validated_data.get("items", [{}])[0].get(
                    "skr03_category"
                ),
                "typical_amounts": [validated_data.get("total_amount", 0)],
                "learning_count": 1,
                "last_updated": time.time(),
            }

            logger.info(f"Learned new supplier template: {supplier_name}")

        elif supplier_name in self._supplier_cache:
            # Bestehenden Template aktualisieren
            template = self._supplier_cache[supplier_name]
            template["learning_count"] += 1
            template["last_updated"] = time.time()

            # Typical amounts erweitern
            amount = validated_data.get("total_amount", 0)
            if amount > 0:
                template["typical_amounts"].append(amount)
                # Nur die letzten 10 Werte behalten
                template["typical_amounts"] = template["typical_amounts"][-10:]

            logger.debug(
                f"Updated supplier template: {supplier_name} ({template['learning_count']} times)"
            )

    def get_processing_metrics(self) -> dict[str, Any]:
        """
        Liefert Performance-Metriken für Monitoring.

        Returns:
            Dict mit aktuellen Metriken
        """
        success_rate = 0.0
        if self.metrics.total_attempts > 0:
            success_rate = (
                self.metrics.successful_attempts / self.metrics.total_attempts
            )

        return {
            "total_attempts": self.metrics.total_attempts,
            "successful_attempts": self.metrics.successful_attempts,
            "failed_attempts": self.metrics.failed_attempts,
            "retry_attempts": self.metrics.retry_attempts,
            "success_rate": success_rate,
            "avg_processing_time": self.metrics.avg_processing_time,
            "last_processing_time": self.metrics.last_processing_time,
            "cache_hits": self.metrics.cache_hits,
            "cache_misses": self.metrics.cache_misses,
            "cache_hit_rate": self.metrics.cache_hits
            / max(1, self.metrics.cache_hits + self.metrics.cache_misses),
            "supplier_templates": len(self._supplier_cache),
        }

    async def _check_cache(self, pdf_path: Path) -> GeminiDirectResult | None:
        """Prüft ob Ergebnis im Cache vorhanden ist."""
        cache_key = self._generate_cache_key(pdf_path)
        return self._processing_cache.get(cache_key)

    async def _store_in_cache(self, pdf_path: Path, result: GeminiDirectResult) -> None:
        """Speichert Ergebnis im Cache."""
        cache_key = self._generate_cache_key(pdf_path)

        # Cache-Größe begrenzen (LRU-ähnlich)
        if len(self._processing_cache) > 100:
            # Älteste Einträge entfernen
            oldest_key = next(iter(self._processing_cache))
            del self._processing_cache[oldest_key]

        self._processing_cache[cache_key] = result

    def _generate_cache_key(self, pdf_path: Path) -> str:
        """Generiert Cache-Key basierend auf Datei-Eigenschaften."""
        try:
            stat = pdf_path.stat()
            return f"{pdf_path.name}_{stat.st_size}_{stat.st_mtime}"
        except OSError:
            return f"{pdf_path.name}_{time.time()}"

    def _update_avg_processing_time(self, new_time: float) -> None:
        """Aktualisiert durchschnittliche Processing-Zeit."""
        if self.metrics.avg_processing_time == 0:
            self.metrics.avg_processing_time = new_time
        else:
            # Exponential moving average
            alpha = 0.1
            self.metrics.avg_processing_time = (
                alpha * new_time + (1 - alpha) * self.metrics.avg_processing_time
            )
