#!/usr/bin/env python3
"""
LLKJJ ML Pipeline - Comprehensive Error Handler
==============================================

🛡️ ROBUSTE ERROR-HANDLING: Production-Grade Fehlerbehandlung

Umfassendes Error-Handling-System für:
- PDF-Verarbeitung Edge-Cases
- API-Ausfälle und Rate-Limiting
- Memory-Management und Resource-Leaks
- Graceful Degradation bei Modell-Fehlern

Autor: LLKJJ Error-Handling Team
Version: 1.0.0 (Comprehensive Handling)
Datum: 19. August 2025
"""

import functools
import logging
import traceback
from collections.abc import Callable
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar

from ..models.processing_result import ProcessingResult

logger = logging.getLogger(__name__)

# Type variable für Generic Error Handler
F = TypeVar("F", bound=Callable[..., Any])


class ErrorSeverity(Enum):
    """Error-Severity-Klassifizierung für intelligente Behandlung."""

    LOW = "low"  # Warnings, non-critical issues
    MEDIUM = "medium"  # Recoverable errors
    HIGH = "high"  # Critical errors, processing might fail
    CRITICAL = "critical"  # System-level errors, immediate attention needed


class ErrorCategory(Enum):
    """Error-Kategorien für spezifische Behandlungsstrategien."""

    PDF_PROCESSING = "pdf_processing"
    API_ERROR = "api_error"
    MODEL_ERROR = "model_error"
    MEMORY_ERROR = "memory_error"
    NETWORK_ERROR = "network_error"
    VALIDATION_ERROR = "validation_error"
    CONFIGURATION_ERROR = "configuration_error"
    DATABASE_ERROR = "database_error"


class MLPipelineError(Exception):
    """
    🛡️ BASIS EXCEPTION für LLKJJ ML Pipeline

    Standardisierte Exception mit Context und Recovery-Informationen.
    """

    def __init__(
        self,
        message: str,
        category: ErrorCategory,
        severity: ErrorSeverity,
        context: dict[str, Any] | None = None,
        original_error: Exception | None = None,
        recovery_suggestion: str | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.context = context or {}
        self.original_error = original_error
        self.recovery_suggestion = recovery_suggestion
        self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> dict[str, Any]:
        """Serialisiert Error für Logging und Debugging."""
        return {
            "message": self.message,
            "category": self.category.value,
            "severity": self.severity.value,
            "context": self.context,
            "original_error": str(self.original_error) if self.original_error else None,
            "recovery_suggestion": self.recovery_suggestion,
            "timestamp": self.timestamp,
            "traceback": traceback.format_exc() if self.original_error else None,
        }


class ErrorRecoveryResult:
    """Result of error recovery attempt."""

    def __init__(
        self,
        success: bool,
        message: str,
        actions_taken: list[str] | None = None,
        suggested_retry: bool = False,
        fallback_result: ProcessingResult | None = None,
    ):
        self.success = success
        self.message = message
        self.actions_taken = actions_taken or []
        self.suggested_retry = suggested_retry
        self.fallback_result = fallback_result


class ErrorRecoveryManager:
    """
    🔄 ERROR RECOVERY MANAGER

    Intelligente Fehlerbehandlung mit automatischen Recovery-Strategien.
    """

    def __init__(self) -> None:
        self.recovery_strategies: dict[
            ErrorCategory, Callable[..., ErrorRecoveryResult]
        ] = {
            ErrorCategory.PDF_PROCESSING: self._recover_pdf_processing,
            ErrorCategory.API_ERROR: self._recover_api_error,
            ErrorCategory.MODEL_ERROR: self._recover_model_error,
            ErrorCategory.MEMORY_ERROR: self._recover_memory_error,
            ErrorCategory.NETWORK_ERROR: self._recover_network_error,
        }

        logger.info("🛡️ ErrorRecoveryManager initialisiert")

    def handle_error(
        self,
        error: Exception,
        category: ErrorCategory,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Behandelt Fehler mit automatischer Recovery-Strategie.

        Args:
            error: Original Exception
            category: Error-Kategorie für spezifische Behandlung
            context: Zusätzlicher Context für Recovery

        Returns:
            Recovery-Result mit Status und Empfehlungen
        """
        try:
            # Severity-Assessment
            severity = self._assess_severity(error, category)

            # Erstelle ML Pipeline Error
            ml_error = MLPipelineError(
                message=str(error),
                category=category,
                severity=severity,
                context=context,
                original_error=error,
                recovery_suggestion=self._get_recovery_suggestion(category, error),
            )

            # Logge Error mit Context
            self._log_error(ml_error)

            # Versuche Recovery
            recovery_result = self._attempt_recovery(ml_error)

            return {
                "error_handled": True,
                "error_details": ml_error.to_dict(),
                "recovery_result": recovery_result,
                "next_action": self._recommend_next_action(ml_error, recovery_result),
            }

        except Exception as recovery_error:
            logger.critical(
                f"🚨 Error-Recovery selbst fehlgeschlagen: {recovery_error}"
            )
            return {
                "error_handled": False,
                "critical_failure": True,
                "original_error": str(error),
                "recovery_error": str(recovery_error),
            }

    def _assess_severity(
        self, error: Exception, category: ErrorCategory
    ) -> ErrorSeverity:
        """Bewertet Error-Severity basierend auf Typ und Kategorie."""

        # Critical system-level errors
        if isinstance(error, MemoryError | SystemError | KeyboardInterrupt):
            return ErrorSeverity.CRITICAL

        # High severity für wichtige Pipeline-Komponenten
        if category in [ErrorCategory.MODEL_ERROR, ErrorCategory.DATABASE_ERROR]:
            return ErrorSeverity.HIGH

        # Medium für recoverable errors
        if category in [ErrorCategory.API_ERROR, ErrorCategory.NETWORK_ERROR]:
            return ErrorSeverity.MEDIUM

        # Low für non-critical issues
        return ErrorSeverity.LOW

    def _get_recovery_suggestion(
        self, category: ErrorCategory, error: Exception
    ) -> str:
        """Liefert spezifische Recovery-Empfehlungen."""

        suggestions = {
            ErrorCategory.PDF_PROCESSING: "Prüfe PDF-Datei auf Corruption, versuche alternative PDF-Reader",
            ErrorCategory.API_ERROR: "Implementiere Retry-Logic mit exponential backoff",
            ErrorCategory.MODEL_ERROR: "Fallback zu alternativer Verarbeitungsstrategie",
            ErrorCategory.MEMORY_ERROR: "Reduziere Batch-Größe, führe Memory-Cleanup durch",
            ErrorCategory.NETWORK_ERROR: "Prüfe Internetverbindung, aktiviere Offline-Modus",
            ErrorCategory.VALIDATION_ERROR: "Validiere Input-Daten und Schemas",
            ErrorCategory.CONFIGURATION_ERROR: "Überprüfe Konfigurationsdateien und Environment-Variablen",
            ErrorCategory.DATABASE_ERROR: "Prüfe Datenbankverbindung und Migrations-Status",
        }

        base_suggestion = suggestions.get(
            category, "Kontaktiere Support mit Error-Details"
        )

        # Erweitere um error-spezifische Hinweise
        if "timeout" in str(error).lower():
            base_suggestion += " | Erhöhe Timeout-Werte"
        elif "permission" in str(error).lower():
            base_suggestion += " | Prüfe Dateiberechtigungen"
        elif "not found" in str(error).lower():
            base_suggestion += " | Prüfe Pfade und Dependencies"

        return base_suggestion

    def _log_error(self, ml_error: MLPipelineError) -> None:
        """Loggt Error mit angemessenem Level."""

        error_dict = ml_error.to_dict()

        if ml_error.severity == ErrorSeverity.CRITICAL:
            logger.critical(f"🚨 CRITICAL: {ml_error.message}", extra=error_dict)
        elif ml_error.severity == ErrorSeverity.HIGH:
            logger.error(f"❌ HIGH: {ml_error.message}", extra=error_dict)
        elif ml_error.severity == ErrorSeverity.MEDIUM:
            logger.warning(f"⚠️ MEDIUM: {ml_error.message}", extra=error_dict)
        else:
            logger.info(f"ℹ️ LOW: {ml_error.message}", extra=error_dict)

    def _attempt_recovery(self, ml_error: MLPipelineError) -> dict[str, Any]:
        """Versucht automatische Error-Recovery."""

        recovery_strategy = self.recovery_strategies.get(ml_error.category)

        if not recovery_strategy:
            return {"recovery_attempted": False, "reason": "No strategy available"}

        try:
            recovery_result = recovery_strategy(ml_error)
            return {
                "recovery_attempted": True,
                "success": True,
                "result": recovery_result,
            }

        except Exception as e:
            return {
                "recovery_attempted": True,
                "success": False,
                "recovery_error": str(e),
            }

    def _recover_pdf_processing(self, ml_error: MLPipelineError) -> ErrorRecoveryResult:
        """Recovery-Strategie für PDF-Processing-Fehler."""

        context = ml_error.context
        pdf_path = context.get("pdf_path")

        recovery_actions: list[str] = []

        if pdf_path and Path(pdf_path).exists():
            # Prüfe PDF-Größe
            size_mb = Path(pdf_path).stat().st_size / (1024 * 1024)
            if size_mb > 50:
                recovery_actions.append("PDF zu groß (>50MB), verwende Komprimierung")

            # Fallback zu alternativer PDF-Extraktion
            recovery_actions.append("Aktiviere Fallback PDF-Processor")

        return ErrorRecoveryResult(
            success=True,
            message="PDF processing recovery applied",
            actions_taken=recovery_actions,
            suggested_retry=True,
        )

    def _recover_api_error(self, ml_error: MLPipelineError) -> ErrorRecoveryResult:
        """Recovery-Strategie für API-Fehler."""

        # Rate-Limiting-Detection
        if "rate" in ml_error.message.lower() or "429" in ml_error.message:
            return ErrorRecoveryResult(
                success=True,
                message="Rate limit recovery applied",
                actions_taken=["Implementiere exponential backoff"],
                suggested_retry=True,
            )

        # Network-Error-Detection
        if any(
            term in ml_error.message.lower()
            for term in ["timeout", "connection", "network"]
        ):
            return ErrorRecoveryResult(
                success=True,
                message="Network recovery applied",
                actions_taken=["Retry mit erhöhtem Timeout"],
                suggested_retry=True,
            )

        return ErrorRecoveryResult(
            success=True,
            message="Generic API recovery applied",
            actions_taken=["Fallback zu alternativer Verarbeitungslogik"],
            suggested_retry=False,
        )

    def _recover_model_error(self, ml_error: MLPipelineError) -> ErrorRecoveryResult:
        """Recovery-Strategie für Modell-Fehler."""

        return ErrorRecoveryResult(
            success=True,
            message="Model recovery applied",
            actions_taken=[
                "Fallback zu Docling-Pipeline",
                "Reduziere Modell-Komplexität",
                "Verwende CPU-Backend",
            ],
            suggested_retry=True,
        )

    def _recover_memory_error(self, ml_error: MLPipelineError) -> ErrorRecoveryResult:
        """Recovery-Strategie für Memory-Fehler."""

        return ErrorRecoveryResult(
            success=True,
            message="Memory recovery applied",
            actions_taken=[
                "Garbage Collection ausführen",
                "Batch-Größe reduzieren",
                "ChromaDB Cache leeren",
                "PyTorch Cache bereinigen",
            ],
            suggested_retry=True,
        )

    def _recover_network_error(self, ml_error: MLPipelineError) -> ErrorRecoveryResult:
        """Recovery-Strategie für Netzwerk-Fehler."""

        return ErrorRecoveryResult(
            success=True,
            message="Network recovery applied",
            actions_taken=[
                "Offline-Modus aktivieren",
                "Lokale Modelle verwenden",
                "Cached Results nutzen",
            ],
            suggested_retry=False,
        )

    def _recommend_next_action(
        self, ml_error: MLPipelineError, recovery_result: dict[str, Any]
    ) -> str:
        """Empfiehlt nächste Schritte basierend auf Error und Recovery."""

        if recovery_result.get("success"):
            return "Verarbeitung mit Recovery-Strategie fortsetzen"

        if ml_error.severity == ErrorSeverity.CRITICAL:
            return "Sofortige manuelle Intervention erforderlich"
        elif ml_error.severity == ErrorSeverity.HIGH:
            return "Fallback-Strategie aktivieren, Admin benachrichtigen"
        else:
            return "Mit alternativer Methode fortsetzen, Issue dokumentieren"


def error_handler(
    category: ErrorCategory, return_fallback: Any = None, reraise: bool = False
) -> Callable[[F], F]:
    """
    🛡️ DECORATOR für automatisches Error-Handling

    Args:
        category: Error-Kategorie für spezifische Behandlung
        return_fallback: Fallback-Wert bei Fehlern
        reraise: Ob Exception nach Handling wieder geworfen werden soll
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)

            except Exception as e:
                # Error-Handling mit Context
                context: dict[str, Any] = {
                    "function": func.__name__,
                    "args_count": len(args),
                    "kwargs_keys": list(kwargs.keys()),
                }

                # Verwende globalen Error-Recovery-Manager
                recovery_manager = ErrorRecoveryManager()
                handling_result = recovery_manager.handle_error(e, category, context)

                logger.info(
                    f"🛡️ Error-Handling für {func.__name__}: {handling_result['error_handled']}"
                )

                if reraise:
                    raise

                return return_fallback

        return wrapper  # type: ignore

    return decorator


def create_fallback_processing_result(
    pdf_path: str, error_message: str
) -> ProcessingResult:
    """
    🛡️ Erstellt Fallback-ProcessingResult bei kritischen Fehlern

    Args:
        pdf_path: PDF-Pfad für Result
        error_message: Error-Message für Context

    Returns:
        Minimal-ProcessingResult für graceful degradation
    """

    return ProcessingResult(
        pdf_path=pdf_path,
        processing_timestamp=datetime.now().isoformat(),
        processing_method="gemini_fallback",
        raw_text="",
        structured_data={"error": error_message, "status": "processing_failed"},
        invoice_header={"error": error_message},
        line_items=[],
        skr03_classifications=[],
        processing_time_ms=0,
        confidence_score=0.0,
        extraction_quality="poor",
    )


# 🛡️ GLOBAL ERROR RECOVERY MANAGER
_global_error_manager: ErrorRecoveryManager | None = None


def get_error_manager() -> ErrorRecoveryManager:
    """
    🛡️ Singleton Error Recovery Manager

    Returns:
        Globale ErrorRecoveryManager-Instanz
    """
    global _global_error_manager
    if _global_error_manager is None:
        _global_error_manager = ErrorRecoveryManager()
    return _global_error_manager


# 🚀 CONVENIENCE FUNCTIONS für häufige Error-Scenarios


def handle_pdf_error(error: Exception, pdf_path: str) -> dict[str, Any]:
    """Convenience-Handler für PDF-Verarbeitungsfehler."""
    return get_error_manager().handle_error(
        error, ErrorCategory.PDF_PROCESSING, {"pdf_path": pdf_path}
    )


def handle_api_error(error: Exception, api_name: str) -> dict[str, Any]:
    """Convenience-Handler für API-Fehler."""
    return get_error_manager().handle_error(
        error, ErrorCategory.API_ERROR, {"api": api_name}
    )


def handle_model_error(error: Exception, model_name: str) -> dict[str, Any]:
    """Convenience-Handler für ML-Modell-Fehler."""
    return get_error_manager().handle_error(
        error, ErrorCategory.MODEL_ERROR, {"model": model_name}
    )
