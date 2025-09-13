"""
Deutsche Fehlermeldungen für LLKJJ ML-Pipeline.

Benutzerfreundliche deutsche Meldungen für häufige Fehler.
Pragmatische Lösung für 2 interne Nutzer.
"""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class GermanErrorMessages:
    """Sammlung deutscher Fehlermeldungen für häufige ML-Pipeline Fehler."""

    @staticmethod
    def pdf_not_found(pdf_path: Path) -> str:
        """PDF-Datei nicht gefunden."""
        return f"📄 PDF-Datei wurde nicht gefunden: {pdf_path.name}"

    @staticmethod
    def pdf_empty(pdf_path: Path) -> str:
        """PDF-Datei ist leer."""
        return f"📄 PDF-Datei ist leer oder beschädigt: {pdf_path.name}"

    @staticmethod
    def pdf_too_large(file_size_mb: float, limit_mb: float) -> str:
        """PDF-Datei ist zu groß."""
        return (
            f"📄 PDF-Datei ist zu groß ({file_size_mb:.1f} MB). Maximum: {limit_mb} MB"
        )

    @staticmethod
    def pdf_invalid_format(pdf_path: Path) -> str:
        """Datei ist kein PDF."""
        return f"📄 Datei ist kein gültiges PDF: {pdf_path.name}"

    @staticmethod
    def gemini_api_failed(original_error: str = "") -> str:
        """Gemini API nicht erreichbar."""
        base_msg = "🤖 Gemini AI ist momentan nicht erreichbar"
        if "quota" in original_error.lower() or "limit" in original_error.lower():
            return f"{base_msg} - API-Limit erreicht"
        elif (
            "network" in original_error.lower()
            or "connection" in original_error.lower()
        ):
            return f"{base_msg} - Netzwerkproblem"
        else:
            return f"{base_msg} - Bitte später erneut versuchen"

    @staticmethod
    def memory_insufficient(required_mb: float) -> str:
        """Nicht genügend Speicher."""
        return f"💾 Nicht genügend Arbeitsspeicher für diese PDF-Datei ({required_mb:.0f} MB erforderlich)"

    @staticmethod
    def processing_failed(pdf_name: str) -> str:
        """Verarbeitung komplett fehlgeschlagen."""
        return f"⚠️ Verarbeitung von {pdf_name} ist fehlgeschlagen - PDF möglicherweise beschädigt"

    @staticmethod
    def skr03_classification_failed() -> str:
        """SKR03-Klassifizierung fehlgeschlagen."""
        return "📊 SKR03-Kontenzuordnung konnte nicht automatisch bestimmt werden"


def wrap_pdf_errors(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator für deutsche PDF-Fehlermeldungen.

    Wandelt technische Exceptions in benutzerfreundliche deutsche Meldungen um.
    """

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except FileNotFoundError as e:
            # PDF path aus Args extrahieren
            pdf_path = None
            for arg in args:
                if isinstance(arg, Path) and str(arg).endswith(".pdf"):
                    pdf_path = arg
                    break

            if pdf_path:
                german_msg = GermanErrorMessages.pdf_not_found(pdf_path)
                logger.error(f"PDF-Fehler: {german_msg}")
                raise FileNotFoundError(german_msg) from e
            else:
                raise e

        except ValueError as e:
            error_str = str(e).lower()

            if "pdf file not found" in error_str or "nicht gefunden" in error_str:
                # PDF path aus Fehlermeldung extrahieren
                if ":" in str(e):
                    path_part = str(e).split(":")[-1].strip()
                    pdf_path = Path(path_part)
                    german_msg = GermanErrorMessages.pdf_not_found(pdf_path)
                else:
                    german_msg = "📄 PDF-Datei wurde nicht gefunden"

                logger.error(f"PDF-Fehler: {german_msg}")
                raise ValueError(german_msg) from e

            elif "too large" in error_str or "zu groß" in error_str:
                # File size aus Fehlermeldung extrahieren
                if "MB" in str(e):
                    try:
                        parts = str(e).split()
                        size_idx = next(
                            i for i, part in enumerate(parts) if "MB" in part
                        )
                        # Extract the size from the part containing "MB"
                        size_str = parts[size_idx].replace("MB", "")
                        file_size = float(size_str)
                        german_msg = GermanErrorMessages.pdf_too_large(file_size, 100)
                    except (ValueError, StopIteration):
                        german_msg = "📄 PDF-Datei ist zu groß für die Verarbeitung"
                else:
                    german_msg = "📄 PDF-Datei ist zu groß für die Verarbeitung"

                logger.error(f"PDF-Fehler: {german_msg}")
                raise ValueError(german_msg) from e

            elif "empty" in error_str or "leer" in error_str:
                german_msg = "📄 PDF-Datei ist leer oder beschädigt"
                logger.error(f"PDF-Fehler: {german_msg}")
                raise ValueError(german_msg) from e

            elif "not a pdf" in error_str or "kein pdf" in error_str:
                german_msg = "📄 Datei ist kein gültiges PDF-Format"
                logger.error(f"PDF-Fehler: {german_msg}")
                raise ValueError(german_msg) from e

            else:
                # Unbekannter ValueError - Original beibehalten
                raise e

        except MemoryError as e:
            german_msg = GermanErrorMessages.memory_insufficient(0)
            logger.error(f"Speicher-Fehler: {german_msg}")
            raise MemoryError(german_msg) from e

        except RuntimeError as e:
            error_str = str(e).lower()

            if "gemini" in error_str:
                german_msg = GermanErrorMessages.gemini_api_failed(str(e))
                logger.error(f"Gemini-Fehler: {german_msg}")
                raise RuntimeError(german_msg) from e

            elif "all processing strategies failed" in error_str:
                german_msg = GermanErrorMessages.processing_failed("PDF-Datei")
                logger.error(f"Verarbeitungs-Fehler: {german_msg}")
                raise RuntimeError(german_msg) from e

            else:
                # Unbekannter RuntimeError - Original beibehalten
                raise e

    return wrapper


# Convenience Functions für direkten Import
def german_pdf_error(pdf_path: Path, error_type: str) -> str:
    """Erstelle deutsche PDF-Fehlermeldung basierend auf Typ."""
    error_mapping = {
        "not_found": GermanErrorMessages.pdf_not_found,
        "empty": GermanErrorMessages.pdf_empty,
        "invalid": GermanErrorMessages.pdf_invalid_format,
    }

    error_func = error_mapping.get(error_type)
    if error_func:
        return error_func(pdf_path)
    else:
        return f"📄 Unbekannter PDF-Fehler: {pdf_path.name}"


def german_gemini_error(original_error: str = "") -> str:
    """Erstelle deutsche Gemini-Fehlermeldung."""
    return GermanErrorMessages.gemini_api_failed(original_error)


def german_memory_error(required_mb: float = 0) -> str:
    """Erstelle deutsche Speicher-Fehlermeldung."""
    return GermanErrorMessages.memory_insufficient(required_mb)
