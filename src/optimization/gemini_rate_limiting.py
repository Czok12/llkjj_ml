#!/usr/bin/env python3
"""
LLKJJ ML Pipeline - Enhanced Gemini API Rate Limiting
====================================================

Dynamisches Rate-Limiting für optimale Gemini API Performance:
- Adaptive rate limiting basierend auf API-Response-Zeiten
- Cost-aware request throttling
- Intelligent retry logic mit exponential backoff
- Performance-optimierte batch processing

Autor: LLKJJ ML Pipeline Team
Version: 4.2.0 (Enhanced Rate Limiting)
Datum: 19. August 2025
"""

import asyncio
import logging
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class GeminiAPIMetrics:
    """Metriken für Gemini API Performance-Tracking."""

    request_count: int = 0
    total_response_time: float = 0.0
    successful_requests: int = 0
    failed_requests: int = 0
    rate_limited_requests: int = 0
    total_cost_estimated: float = 0.0

    @property
    def average_response_time(self) -> float:
        """Durchschnittliche Response-Zeit."""
        return self.total_response_time / max(1, self.request_count)

    @property
    def success_rate(self) -> float:
        """Erfolgsrate der API-Requests."""
        return self.successful_requests / max(1, self.request_count)

    @property
    def cost_per_request(self) -> float:
        """Durchschnittliche Kosten pro Request."""
        return self.total_cost_estimated / max(1, self.successful_requests)


class EnhancedGeminiRateLimiter:
    """
    Intelligenter Rate-Limiter für Gemini API mit adaptiver Anpassung.

    Features:
    - Dynamische Semaphore-Anpassung basierend auf Performance
    - Cost-aware throttling für Budget-Kontrolle
    - Response-Zeit-basierte Optimierung
    - Exponential backoff bei Rate-Limiting
    """

    def __init__(
        self,
        initial_concurrent_requests: int = 5,
        max_concurrent_requests: int = 10,
        min_concurrent_requests: int = 2,
        cost_threshold_per_hour: float = 10.0,  # USD pro Stunde
        adaptive_window_minutes: int = 5,
    ):
        self.current_limit = initial_concurrent_requests
        self.max_limit = max_concurrent_requests
        self.min_limit = min_concurrent_requests
        self.cost_threshold = cost_threshold_per_hour
        self.adaptive_window = timedelta(minutes=adaptive_window_minutes)

        # Dynamisches Semaphore
        self.semaphore = asyncio.Semaphore(self.current_limit)

        # Performance-Tracking
        self.metrics = GeminiAPIMetrics()
        self.response_times: deque[float] = deque(
            maxlen=100
        )  # Letzte 100 Response-Zeiten
        self.cost_tracking: deque[tuple[datetime, float]] = deque(maxlen=1000)

        # Adaptive Anpassung
        self.last_adaptation = datetime.now()
        self.performance_threshold_ms = 2000  # 2 Sekunden als "langsam" definiert

        logger.info(
            "🚀 EnhancedGeminiRateLimiter initialisiert: %d concurrent requests",
            self.current_limit,
        )

    async def acquire_with_metrics(self, estimated_cost: float = 0.1) -> bool:
        """
        Erwirbt Semaphore mit Cost-Tracking und Performance-Monitoring.

        Args:
            estimated_cost: Geschätzte Kosten für diesen Request in USD

        Returns:
            True wenn Request erlaubt, False wenn cost-throttled
        """
        # Cost-Throttling prüfen
        if not self._check_cost_threshold(estimated_cost):
            logger.warning(
                "💰 Cost-Throttling aktiv: Request abgelehnt (Kosten-Limit erreicht)"
            )
            return False

        # Semaphore erwerben
        await self.semaphore.acquire()

        # Kosten tracken
        self.cost_tracking.append((datetime.now(), estimated_cost))
        self.metrics.total_cost_estimated += estimated_cost

        return True

    def release_with_metrics(
        self, response_time_ms: float, success: bool, rate_limited: bool = False
    ) -> None:
        """
        Gibt Semaphore frei und aktualisiert Performance-Metriken.

        Args:
            response_time_ms: Response-Zeit in Millisekunden
            success: Ob der Request erfolgreich war
            rate_limited: Ob der Request rate-limited wurde
        """
        # Metriken aktualisieren
        self.metrics.request_count += 1
        self.metrics.total_response_time += response_time_ms

        if success:
            self.metrics.successful_requests += 1
        else:
            self.metrics.failed_requests += 1

        if rate_limited:
            self.metrics.rate_limited_requests += 1

        # Response-Zeit für adaptive Anpassung tracken
        self.response_times.append(response_time_ms)

        # Semaphore freigeben
        self.semaphore.release()

        # Adaptive Anpassung prüfen
        if self._should_adapt():
            asyncio.create_task(self._adapt_rate_limit())

    def _check_cost_threshold(self, estimated_cost: float) -> bool:
        """Prüft ob Request unter Cost-Threshold liegt."""
        current_hour_cost = self._calculate_hourly_cost()
        return (current_hour_cost + estimated_cost) <= self.cost_threshold

    def _calculate_hourly_cost(self) -> float:
        """Berechnet Kosten der letzten Stunde."""
        now = datetime.now()
        one_hour_ago = now - timedelta(hours=1)

        hourly_cost = sum(
            cost for timestamp, cost in self.cost_tracking if timestamp >= one_hour_ago
        )

        return hourly_cost

    def _should_adapt(self) -> bool:
        """Prüft ob adaptive Anpassung durchgeführt werden sollte."""
        return (
            datetime.now() - self.last_adaptation >= self.adaptive_window
            and len(self.response_times) >= 10  # Mindestens 10 Samples
        )

    async def _adapt_rate_limit(self) -> None:
        """Führt adaptive Anpassung des Rate-Limits durch."""
        self.last_adaptation = datetime.now()

        # Aktuelle Performance analysieren
        avg_response_time = sum(self.response_times) / len(self.response_times)
        recent_success_rate = self.metrics.success_rate

        old_limit = self.current_limit

        # Adaptive Logik
        if (
            avg_response_time > self.performance_threshold_ms
            and recent_success_rate < 0.9
        ):
            # Performance schlecht → Rate-Limit reduzieren
            new_limit = max(self.min_limit, self.current_limit - 1)
            reason = "Performance-Degradation"

        elif (
            avg_response_time < (self.performance_threshold_ms * 0.5)
            and recent_success_rate > 0.95
        ):
            # Performance gut → Rate-Limit erhöhen
            new_limit = min(self.max_limit, self.current_limit + 1)
            reason = "Performance-Optimierung"

        else:
            # Keine Änderung notwendig
            return

        # Rate-Limit anpassen
        if new_limit != self.current_limit:
            await self._update_semaphore_limit(new_limit)

            logger.info(
                "⚡ Rate-Limit angepasst: %d → %d (%s) | Ø Response: %.0fms, Success: %.1f%%",
                old_limit,
                new_limit,
                reason,
                avg_response_time,
                recent_success_rate * 100,
            )

    async def _update_semaphore_limit(self, new_limit: int) -> None:
        """Aktualisiert Semaphore-Limit dynamisch."""
        self.current_limit = new_limit
        self.semaphore = asyncio.Semaphore(new_limit)

        # Warte bis alle aktuellen Requests abgeschlossen sind
        await asyncio.sleep(0.1)  # Kurz warten für laufende Requests

    def get_performance_stats(self) -> dict[str, Any]:
        """Liefert umfassende Performance-Statistiken."""
        hourly_cost = self._calculate_hourly_cost()

        return {
            "rate_limiting": {
                "current_limit": self.current_limit,
                "max_limit": self.max_limit,
                "min_limit": self.min_limit,
                "utilization_percent": (self.current_limit / self.max_limit) * 100,
            },
            "performance": {
                "average_response_time_ms": self.metrics.average_response_time,
                "success_rate": self.metrics.success_rate,
                "total_requests": self.metrics.request_count,
                "failed_requests": self.metrics.failed_requests,
                "rate_limited_requests": self.metrics.rate_limited_requests,
            },
            "cost_tracking": {
                "hourly_cost_usd": hourly_cost,
                "cost_threshold_usd": self.cost_threshold,
                "cost_utilization_percent": (hourly_cost / self.cost_threshold) * 100,
                "average_cost_per_request": self.metrics.cost_per_request,
                "total_estimated_cost": self.metrics.total_cost_estimated,
            },
            "adaptive_behavior": {
                "last_adaptation": self.last_adaptation.isoformat(),
                "adaptation_window_minutes": self.adaptive_window.total_seconds() / 60,
                "performance_threshold_ms": self.performance_threshold_ms,
                "recent_response_times": list(self.response_times)[-10:],  # Letzte 10
            },
        }


class GeminiAPIOptimizer:
    """
    High-Level API-Optimizer für Gemini mit Best-Practice-Integration.

    Kombiniert Rate-Limiting, Retry-Logic und Performance-Monitoring.
    """

    def __init__(self, rate_limiter: EnhancedGeminiRateLimiter):
        self.rate_limiter = rate_limiter
        self.retry_delays = [1, 2, 4, 8, 16]  # Exponential backoff in Sekunden

    async def execute_with_optimization(
        self,
        api_call_func: Callable[..., Any],
        estimated_cost: float = 0.1,
        max_retries: int = 3,
        **api_kwargs,
    ) -> tuple[Any, dict[str, Any]]:
        """
        Führt Gemini API-Call mit vollständiger Optimierung durch.

        Args:
            api_call_func: Async function für Gemini API-Call
            estimated_cost: Geschätzte Kosten in USD
            max_retries: Maximale Anzahl Wiederholungen
            **api_kwargs: Argumente für API-Call

        Returns:
            Tuple: (API-Response, Performance-Metriken)
        """
        # Cost-Throttling prüfen
        if not await self.rate_limiter.acquire_with_metrics(estimated_cost):
            raise RuntimeError("API-Request abgelehnt: Cost-Threshold erreicht")

        start_time = time.time()
        last_error = None

        try:
            for attempt in range(max_retries + 1):
                try:
                    # API-Call ausführen
                    response = await api_call_func(**api_kwargs)

                    # Erfolgreiche Response
                    response_time_ms = (time.time() - start_time) * 1000
                    self.rate_limiter.release_with_metrics(
                        response_time_ms, success=True
                    )

                    return response, {
                        "success": True,
                        "attempts": attempt + 1,
                        "response_time_ms": response_time_ms,
                        "cost_estimate": estimated_cost,
                    }

                except Exception as e:
                    last_error = e

                    # Rate-Limiting erkannt
                    if "rate limit" in str(e).lower() or "quota" in str(e).lower():
                        response_time_ms = (time.time() - start_time) * 1000
                        self.rate_limiter.release_with_metrics(
                            response_time_ms, success=False, rate_limited=True
                        )

                        if attempt < max_retries:
                            delay = self.retry_delays[
                                min(attempt, len(self.retry_delays) - 1)
                            ]
                            logger.warning(
                                "⚠️ Rate-Limit erreicht (Versuch %d/%d), warte %ds...",
                                attempt + 1,
                                max_retries + 1,
                                delay,
                            )
                            await asyncio.sleep(delay)
                            continue

                    # Anderer Fehler
                    if attempt < max_retries:
                        delay = self.retry_delays[
                            min(attempt, len(self.retry_delays) - 1)
                        ]
                        logger.warning(
                            "⚠️ API-Fehler (Versuch %d/%d): %s, warte %ds...",
                            attempt + 1,
                            max_retries + 1,
                            str(e),
                            delay,
                        )
                        await asyncio.sleep(delay)
                        continue

                    # Alle Versuche fehlgeschlagen
                    break

            # Fehler nach allen Versuchen
            response_time_ms = (time.time() - start_time) * 1000
            self.rate_limiter.release_with_metrics(response_time_ms, success=False)

            raise RuntimeError(
                f"API-Call nach {max_retries + 1} Versuchen fehlgeschlagen: {last_error}"
            )

        except Exception:
            # Semaphore bei Exception freigeben
            response_time_ms = (time.time() - start_time) * 1000
            self.rate_limiter.release_with_metrics(response_time_ms, success=False)
            raise


# Convenience-Funktionen für Integration
def create_optimized_rate_limiter(**kwargs) -> EnhancedGeminiRateLimiter:
    """Factory function für optimierte Rate-Limiter-Erstellung."""
    return EnhancedGeminiRateLimiter(**kwargs)


def create_api_optimizer(
    rate_limiter: [EnhancedGeminiRateLimiter] = None,
) -> GeminiAPIOptimizer:
    """Factory function für API-Optimizer-Erstellung."""
    if rate_limiter is None:
        rate_limiter = create_optimized_rate_limiter()
    return GeminiAPIOptimizer(rate_limiter)
