#!/usr/bin/env python3
"""
LLKJJ ML Pipeline - Apple Silicon MPS Optimization
==================================================

🍎 APPLE SILICON OPTIMIZATION: PyTorch MPS-Backend-Optimierung

Performance-Verbesserungen für Apple M1/M2/M3 Chips:
- MPS (Metal Performance Shaders) Backend für PyTorch
- Optimierte Memory-Management für Neural Networks
- Fallback-Strategien für Inkompatibilitäten

Autor: LLKJJ Apple Silicon Team
Version: 1.0.0 (MPS Optimization)
Datum: 19. August 2025
"""

import logging
import warnings
from typing import Any

import torch

logger = logging.getLogger(__name__)


class AppleSiliconOptimizer:
    """
    🍎 APPLE SILICON MPS OPTIMIZATION

    Optimiert PyTorch für Apple Silicon Chips (M1/M2/M3):
    - Automatische MPS-Backend-Erkennung
    - Memory-optimierte Tensor-Operations
    - Fallback für inkompatible Operations
    """

    def __init__(self) -> None:
        self.device = self._detect_optimal_device()
        self.mps_available = self._check_mps_availability()
        self._optimize_pytorch_settings()

        logger.info(
            f"🍎 Apple Silicon Optimizer initialisiert (Device: {self.device}, MPS: {self.mps_available})"
        )

    def _detect_optimal_device(self) -> str:
        """
        Erkennt das optimale PyTorch-Device für Apple Silicon.

        Returns:
            Optimal device string: "mps", "cpu", oder "cuda"
        """
        try:
            # 1. Prüfe MPS-Verfügbarkeit (Apple Silicon)
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                logger.info("🍎 MPS (Metal Performance Shaders) verfügbar")
                return "mps"

            # 2. Fallback zu CUDA (falls verfügbar)
            elif torch.cuda.is_available():
                logger.info("🐍 CUDA verfügbar")
                return "cuda"

            # 3. CPU-Fallback
            else:
                logger.info("💻 CPU-Backend wird verwendet")
                return "cpu"

        except Exception as e:
            logger.warning(f"⚠️ Device-Erkennung fehlgeschlagen: {e}, verwende CPU")
            return "cpu"

    def _check_mps_availability(self) -> bool:
        """
        Überprüft detailliert die MPS-Verfügbarkeit und -Konfiguration.
        """
        if self.device != "mps":
            return False

        try:
            # Test-Tensor für MPS-Funktionalität
            test_tensor = torch.randn(10, 10, device="mps")
            result = torch.mm(test_tensor, test_tensor.t())
            del test_tensor, result  # Memory cleanup

            logger.info("✅ MPS-Funktionalität validiert")
            return True

        except Exception as e:
            logger.warning(f"⚠️ MPS-Test fehlgeschlagen: {e}")
            return False

    def _optimize_pytorch_settings(self) -> None:
        """
        Optimiert PyTorch-Settings für Apple Silicon.
        """
        try:
            # Deaktiviere problematische MPS-Warnings
            warnings.filterwarnings(
                "ignore", message=".*pin_memory.*not supported on MPS.*"
            )

            # Optimierte Memory-Settings für Apple Silicon
            if self.mps_available:
                # MPS-spezifische Optimierungen
                torch.mps.empty_cache()  # Initial cache cleanup

                # Set optimal number of threads (Apple Silicon hat viele efficiency cores)
                optimal_threads = min(
                    torch.get_num_threads(), 8
                )  # Nicht mehr als 8 für MPS
                torch.set_num_threads(optimal_threads)

                logger.info(
                    f"🍎 MPS-Optimierungen aktiviert (Threads: {optimal_threads})"
                )

            # Allgemeine PyTorch-Optimierungen
            torch.set_flush_denormal(True)  # Performance für kleine Zahlen

        except Exception as e:
            logger.warning(f"⚠️ PyTorch-Optimierung fehlgeschlagen: {e}")

    def get_device(self) -> torch.device:
        """
        Liefert das optimale PyTorch-Device.

        Returns:
            torch.device für optimale Performance
        """
        return torch.device(self.device)

    def optimize_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Optimiert ein PyTorch-Modell für Apple Silicon.

        Args:
            model: PyTorch-Modell zum Optimieren

        Returns:
            Optimiertes Modell
        """
        try:
            # Move model to optimal device
            optimized_model = model.to(self.get_device())

            # Apple Silicon spezifische Optimierungen
            if self.mps_available:
                # Set model to evaluation mode für MPS-Optimierung
                optimized_model.eval()

                # Disable gradient computation für Inference-Optimierung
                for param in optimized_model.parameters():
                    param.requires_grad = False

                logger.info("🍎 Modell für MPS optimiert")

            return optimized_model

        except Exception as e:
            logger.warning(
                f"⚠️ Modell-Optimierung fehlgeschlagen: {e}, verwende Original"
            )
            return model

    def optimize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Optimiert einen Tensor für Apple Silicon.

        Args:
            tensor: Input-Tensor

        Returns:
            Optimierter Tensor
        """
        try:
            if self.mps_available and tensor.device.type != "mps":
                # Move zu MPS mit error handling
                return tensor.to(self.get_device())
            return tensor

        except Exception as e:
            logger.warning(f"⚠️ Tensor-Optimierung fehlgeschlagen: {e}")
            return tensor

    def cleanup_memory(self) -> None:
        """
        Bereinigt Apple Silicon GPU-Memory.
        """
        try:
            if self.mps_available:
                torch.mps.empty_cache()
                logger.debug("🍎 MPS-Memory bereinigt")

            # Zusätzliche Memory-Bereinigung
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            logger.warning(f"⚠️ Memory-Cleanup fehlgeschlagen: {e}")

    def get_performance_info(self) -> dict[str, Any]:
        """
        Liefert Performance-Informationen für Apple Silicon.

        Returns:
            Dictionary mit Performance-Metriken
        """
        info: dict[str, Any] = {
            "device": self.device,
            "mps_available": self.mps_available,
            "pytorch_version": torch.__version__,
            "num_threads": torch.get_num_threads(),
        }

        if self.mps_available:
            try:
                # MPS-spezifische Informationen
                info.update(
                    {
                        "mps_built": torch.backends.mps.is_built(),
                        "mps_enabled": torch.backends.mps.is_available(),
                    }
                )
            except Exception as e:
                logger.warning(f"⚠️ MPS-Info-Abfrage fehlgeschlagen: {e}")

        return info


# 🍎 GLOBAL APPLE SILICON OPTIMIZER INSTANCE
_apple_silicon_optimizer: AppleSiliconOptimizer | None = None


def get_apple_silicon_optimizer() -> AppleSiliconOptimizer:
    """
    🍎 Singleton Apple Silicon Optimizer

    Returns:
        Globale AppleSiliconOptimizer-Instanz
    """
    global _apple_silicon_optimizer
    if _apple_silicon_optimizer is None:
        _apple_silicon_optimizer = AppleSiliconOptimizer()
    return _apple_silicon_optimizer


def optimize_for_apple_silicon() -> None:
    """
    🍎 Convenience-Funktion für Apple Silicon Optimization

    Initialisiert und konfiguriert automatisch alle Apple Silicon Optimierungen.
    """
    optimizer = get_apple_silicon_optimizer()
    logger.info(
        f"🍎 Apple Silicon Optimization aktiviert: {optimizer.get_performance_info()}"
    )


def suppress_mps_warnings() -> None:
    """
    🔇 Unterdrückt bekannte MPS-Warnungen für saubere Logs
    """
    # Suppress spezifische MPS-Warnungen
    warnings.filterwarnings("ignore", message=".*pin_memory.*not supported on MPS.*")
    warnings.filterwarnings("ignore", message=".*MPS.*fallback.*")
    warnings.filterwarnings("ignore", message=".*Metal.*")

    logger.info("🔇 MPS-Warnungen unterdrückt für saubere Logs")


# 🚀 AUTO-INITIALIZATION für Apple Silicon
def auto_configure_apple_silicon() -> dict[str, Any]:
    """
    🚀 Automatische Apple Silicon Konfiguration

    Wird beim Import automatisch ausgeführt um optimale Performance sicherzustellen.

    Returns:
        Configuration info für Debugging
    """
    try:
        # Suppress Warnings zuerst
        suppress_mps_warnings()

        # Initialisiere Optimizer
        optimizer = get_apple_silicon_optimizer()

        # Memory cleanup
        optimizer.cleanup_memory()

        config_info: dict[str, Any] = {
            "auto_configured": True,
            "optimizer_info": optimizer.get_performance_info(),
            "recommendations": [],
        }

        # Performance-Empfehlungen
        if optimizer.mps_available:
            config_info["recommendations"].append(
                "MPS-Backend aktiv - optimale Performance"
            )
        elif optimizer.device == "cpu":
            config_info["recommendations"].append(
                "CPU-Fallback - erwäge PyTorch MPS Update"
            )

        logger.info("🚀 Apple Silicon Auto-Konfiguration abgeschlossen")
        return config_info

    except Exception as e:
        logger.warning(f"⚠️ Apple Silicon Auto-Konfiguration fehlgeschlagen: {e}")
        return {"auto_configured": False, "error": str(e)}


# 🍎 AUTO-CONFIGURE beim Import
_auto_config_result = auto_configure_apple_silicon()
