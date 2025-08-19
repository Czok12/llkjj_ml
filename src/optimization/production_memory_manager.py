#!/usr/bin/env python3
"""
LLKJJ ML Pipeline - Production-Ready Memory Manager
=================================================

🧠 MEMORY-MANAGEMENT: Intelligente Resource-Optimierung

Production-grade Memory-Management für:
- ChromaDB Memory-Leak-Prevention
- PyTorch MPS Memory-Optimization
- Garbage Collection Coordination
- Resource Pool Management

Autor: LLKJJ Memory-Management Team
Version: 1.0.0 (Production Optimization)
Datum: 19. August 2025
"""

import gc
import logging
import threading
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import psutil
import torch

logger = logging.getLogger(__name__)


@dataclass
class MemoryMetrics:
    """
    📊 MEMORY METRICS für Monitoring und Alerting
    """

    total_memory_mb: float
    available_memory_mb: float
    used_memory_mb: float
    memory_percent: float
    python_memory_mb: float
    torch_memory_mb: float
    mps_memory_mb: float | None = None


class ProductionMemoryManager:
    """
    🧠 PRODUCTION MEMORY MANAGER

    Intelligentes Memory-Management für production-readiness:
    - Automatische Garbage Collection
    - PyTorch MPS Memory-Optimization
    - ChromaDB Resource-Cleanup
    - Memory-Leak-Detection
    """

    def __init__(
        self,
        max_memory_threshold: float = 0.85,
        cleanup_interval_seconds: int = 300,
        enable_automatic_cleanup: bool = True,
    ):
        self.max_memory_threshold = max_memory_threshold
        self.cleanup_interval_seconds = cleanup_interval_seconds
        self.enable_automatic_cleanup = enable_automatic_cleanup

        # Memory-Monitoring
        self._memory_history: list[MemoryMetrics] = []
        self._cleanup_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        # Resource-Tracking
        self._active_tensors: set[torch.Tensor] = set()
        self._chroma_clients: set[Any] = set()

        logger.info(
            f"🧠 ProductionMemoryManager initialisiert (Threshold: {max_memory_threshold:.1%})"
        )

        if enable_automatic_cleanup:
            self._start_background_cleanup()

    def get_current_metrics(self) -> MemoryMetrics:
        """
        📊 Aktuelle Memory-Metriken abrufen

        Returns:
            Detaillierte Memory-Metriken für Monitoring
        """

        # System-Memory
        memory = psutil.virtual_memory()
        process = psutil.Process()
        python_memory_mb = process.memory_info().rss / (1024 * 1024)

        # PyTorch-Memory
        torch_memory_mb = 0.0
        mps_memory_mb = None

        if torch.cuda.is_available():
            torch_memory_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            try:
                mps_memory_mb = torch.mps.current_allocated_memory() / (1024 * 1024)
                torch_memory_mb = mps_memory_mb
            except Exception as e:
                logger.debug(f"MPS memory query failed: {e}")

        metrics = MemoryMetrics(
            total_memory_mb=memory.total / (1024 * 1024),
            available_memory_mb=memory.available / (1024 * 1024),
            used_memory_mb=memory.used / (1024 * 1024),
            memory_percent=memory.percent / 100.0,
            python_memory_mb=python_memory_mb,
            torch_memory_mb=torch_memory_mb,
            mps_memory_mb=mps_memory_mb,
        )

        # Track metrics history
        self._memory_history.append(metrics)
        if len(self._memory_history) > 100:  # Keep last 100 measurements
            self._memory_history.pop(0)

        return metrics

    def is_memory_critical(self) -> bool:
        """
        🚨 Prüft ob Memory-Situation kritisch ist

        Returns:
            True wenn Memory-Cleanup erforderlich ist
        """

        metrics = self.get_current_metrics()
        return metrics.memory_percent > self.max_memory_threshold

    def force_cleanup(self, aggressive: bool = False) -> dict[str, Any]:
        """
        🧹 Erzwungene Memory-Bereinigung

        Args:
            aggressive: Ob aggressive Cleanup-Strategien verwendet werden sollen

        Returns:
            Cleanup-Result mit Details über freigegebenen Speicher
        """

        logger.info(
            f"🧹 Starting {'aggressive' if aggressive else 'standard'} memory cleanup"
        )

        initial_metrics = self.get_current_metrics()
        cleanup_actions: list[str] = []

        try:
            # 1. Standard Python Garbage Collection
            collected = gc.collect()
            cleanup_actions.append(f"Python GC: {collected} objects collected")

            # 2. PyTorch Memory Cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                cleanup_actions.append("PyTorch CUDA: Cache emptied")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                try:
                    torch.mps.empty_cache()
                    cleanup_actions.append("PyTorch MPS: Cache emptied")
                except Exception as e:
                    logger.debug(f"MPS cache cleanup failed: {e}")

            # 3. ChromaDB Cleanup (if aggressive)
            if aggressive:
                self._cleanup_chroma_resources()
                cleanup_actions.append("ChromaDB: Resources cleaned")

            # 4. Tensor Reference Cleanup
            if aggressive:
                self._cleanup_tensor_references()
                cleanup_actions.append(
                    f"Tensors: {len(self._active_tensors)} references cleaned"
                )

            # 5. Final GC pass
            if aggressive:
                for _ in range(3):  # Multiple GC passes for thorough cleanup
                    gc.collect()
                cleanup_actions.append("Aggressive GC: Multiple passes completed")

            final_metrics = self.get_current_metrics()
            memory_freed_mb = (
                initial_metrics.used_memory_mb - final_metrics.used_memory_mb
            )

            logger.info(f"🧹 Memory cleanup completed: {memory_freed_mb:.1f} MB freed")

            return {
                "success": True,
                "memory_freed_mb": memory_freed_mb,
                "initial_memory_percent": initial_metrics.memory_percent,
                "final_memory_percent": final_metrics.memory_percent,
                "cleanup_actions": cleanup_actions,
            }

        except Exception as e:
            logger.error(f"🚨 Memory cleanup failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "cleanup_actions": cleanup_actions,
            }

    @contextmanager
    def memory_tracked_context(
        self, operation_name: str
    ) -> Generator[None, None, None]:
        """
        📊 Context-Manager für Memory-Tracking von Operationen

        Args:
            operation_name: Name der Operation für Logging

        Yields:
            None (Context für Code-Ausführung)
        """

        start_metrics = self.get_current_metrics()
        start_time = time.time()

        logger.debug(f"🔍 Memory tracking started for: {operation_name}")

        try:
            yield

        finally:
            end_time = time.time()
            end_metrics = self.get_current_metrics()
            duration_ms = (end_time - start_time) * 1000

            memory_delta_mb = end_metrics.used_memory_mb - start_metrics.used_memory_mb

            logger.info(
                f"📊 {operation_name}: "
                f"{duration_ms:.0f}ms, "
                f"Memory delta: {memory_delta_mb:+.1f} MB"
            )

            # Automatischer Cleanup bei kritischem Memory-Verbrauch
            if self.is_memory_critical():
                logger.warning(
                    f"🚨 Memory critical after {operation_name}, triggering cleanup"
                )
                self.force_cleanup(aggressive=False)

    def register_tensor(self, tensor: torch.Tensor) -> None:
        """
        📝 Registriert Tensor für Tracking

        Args:
            tensor: PyTorch Tensor zum Tracking
        """

        self._active_tensors.add(tensor)
        logger.debug(f"📝 Tensor registered: {tensor.shape} ({tensor.dtype})")

    def unregister_tensor(self, tensor: torch.Tensor) -> None:
        """
        🗑️ Entfernt Tensor aus Tracking

        Args:
            tensor: PyTorch Tensor zum Entfernen
        """

        self._active_tensors.discard(tensor)
        logger.debug(f"🗑️ Tensor unregistered: {tensor.shape}")

    def register_chroma_client(self, client: Any) -> None:
        """
        📝 Registriert ChromaDB Client für Cleanup

        Args:
            client: ChromaDB Client-Instanz
        """

        self._chroma_clients.add(client)
        logger.debug("📝 ChromaDB client registered")

    def unregister_chroma_client(self, client: Any) -> None:
        """
        🗑️ Entfernt ChromaDB Client aus Tracking

        Args:
            client: ChromaDB Client zum Entfernen
        """

        self._chroma_clients.discard(client)
        logger.debug("🗑️ ChromaDB client unregistered")

    def get_memory_report(self) -> dict[str, Any]:
        """
        📋 Umfassender Memory-Report für Debugging

        Returns:
            Detaillierter Memory-Report mit Trends und Empfehlungen
        """

        current_metrics = self.get_current_metrics()

        # Trend-Analyse
        memory_trend = "stable"
        if len(self._memory_history) >= 5:
            recent_usage = [m.memory_percent for m in self._memory_history[-5:]]
            if recent_usage[-1] - recent_usage[0] > 0.1:  # >10% increase
                memory_trend = "increasing"
            elif recent_usage[0] - recent_usage[-1] > 0.1:  # >10% decrease
                memory_trend = "decreasing"

        # Empfehlungen
        recommendations: list[str] = []
        if current_metrics.memory_percent > 0.8:
            recommendations.append("Consider reducing batch size")
        if current_metrics.torch_memory_mb > 1000:
            recommendations.append(
                "PyTorch memory usage high - consider model optimization"
            )
        if len(self._active_tensors) > 50:
            recommendations.append(
                "Many active tensors - review tensor lifecycle management"
            )

        return {
            "current_metrics": current_metrics,
            "memory_trend": memory_trend,
            "tracked_tensors": len(self._active_tensors),
            "tracked_chroma_clients": len(self._chroma_clients),
            "cleanup_enabled": self.enable_automatic_cleanup,
            "recommendations": recommendations,
            "history_points": len(self._memory_history),
        }

    def _start_background_cleanup(self) -> None:
        """
        🔄 Startet Background-Thread für automatische Memory-Bereinigung
        """

        def cleanup_worker() -> None:
            while not self._stop_event.wait(self.cleanup_interval_seconds):
                try:
                    if self.is_memory_critical():
                        logger.info("🚨 Automatic memory cleanup triggered")
                        self.force_cleanup(aggressive=False)
                except Exception as e:
                    logger.error(f"🚨 Background cleanup error: {e}")

        self._cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        self._cleanup_thread.start()
        logger.info(
            f"🔄 Background cleanup thread started (interval: {self.cleanup_interval_seconds}s)"
        )

    def _cleanup_chroma_resources(self) -> None:
        """
        🧹 Bereinigt ChromaDB-Ressourcen
        """

        for client in list(self._chroma_clients):
            try:
                # ChromaDB-spezifische Cleanup-Operationen
                if hasattr(client, "reset"):
                    client.reset()
                elif hasattr(client, "_client") and hasattr(client._client, "reset"):
                    client._client.reset()
                logger.debug("🧹 ChromaDB client cleaned")
            except Exception as e:
                logger.debug(f"ChromaDB cleanup error: {e}")

    def _cleanup_tensor_references(self) -> None:
        """
        🧹 Bereinigt inaktive Tensor-Referenzen
        """

        active_tensors: set[Any] = set()
        for tensor in self._active_tensors:
            if tensor.numel() > 0:  # Tensor still valid
                active_tensors.add(tensor)

        cleaned_count = len(self._active_tensors) - len(active_tensors)
        self._active_tensors = active_tensors

        if cleaned_count > 0:
            logger.debug(f"🧹 {cleaned_count} stale tensor references cleaned")

    def shutdown(self) -> None:
        """
        🛑 Beendet Memory-Manager und bereinigt Ressourcen
        """

        logger.info("🛑 Shutting down ProductionMemoryManager")

        self._stop_event.set()

        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5.0)

        # Final cleanup
        self.force_cleanup(aggressive=True)

        logger.info("🛑 ProductionMemoryManager shutdown completed")


# 🧠 GLOBAL MEMORY MANAGER
_global_memory_manager: ProductionMemoryManager | None = None


def get_memory_manager() -> ProductionMemoryManager:
    """
    🧠 Singleton Memory Manager

    Returns:
        Globale ProductionMemoryManager-Instanz
    """
    global _global_memory_manager
    if _global_memory_manager is None:
        _global_memory_manager = ProductionMemoryManager()
    return _global_memory_manager


# 🚀 CONVENIENCE FUNCTIONS


def track_memory_operation(operation_name: str) -> Callable[..., Any]:
    """
    📊 Decorator für automatisches Memory-Tracking

    Args:
        operation_name: Name der Operation für Tracking
    """

    def decorator(func: Any) -> Any:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with get_memory_manager().memory_tracked_context(operation_name):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def ensure_memory_available(min_available_mb: float = 500) -> bool:
    """
    🔍 Stellt sicher dass genügend Memory verfügbar ist

    Args:
        min_available_mb: Mindest verfügbarer Speicher in MB

    Returns:
        True wenn genügend Memory verfügbar ist
    """
    manager = get_memory_manager()
    metrics = manager.get_current_metrics()

    if metrics.available_memory_mb < min_available_mb:
        logger.warning(
            f"🚨 Low memory: {metrics.available_memory_mb:.0f} MB < {min_available_mb:.0f} MB"
        )
        manager.force_cleanup(aggressive=True)

        # Re-check nach cleanup
        new_metrics = manager.get_current_metrics()
        return new_metrics.available_memory_mb >= min_available_mb

    return True
