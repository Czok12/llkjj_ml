#!/usr/bin/env python3
"""
LLKJJ ML Pipeline - Pydantic Error Handling Extensions
======================================================

Enhanced error handling for Pydantic validation errors in German invoice processing.
Provides German-language error messages and business-specific validation helpers.

Author: LLKJJ ML Pipeline
Version: 1.0.0 (Pydantic v2 Migration)
"""

import logging
from typing import Any

from pydantic import ValidationError

logger = logging.getLogger(__name__)


class GermanValidationError(Exception):
    """Deutsche Validierungs-Fehler für bessere Benutzerfreundlichkeit."""

    def __init__(self, validation_error: ValidationError, context: str = ""):
        self.validation_error = validation_error
        self.context = context
        self.german_message = self._translate_to_german()
        super().__init__(self.german_message)

    def _translate_to_german(self) -> str:
        """Übersetze Pydantic-Validierungsfehler ins Deutsche."""
        errors = []

        for error in self.validation_error.errors():
            field_path = " → ".join(str(loc) for loc in error["loc"])
            error_type = error["type"]
            input_value = error.get("input", "")

            # Deutsche Übersetzungen für häufige Validierungsfehler
            german_messages = {
                "missing": f"Pflichtfeld '{field_path}' fehlt",
                "string_too_short": f"'{field_path}' ist zu kurz (Eingabe: '{input_value}')",
                "string_too_long": f"'{field_path}' ist zu lang (Eingabe: '{input_value}')",
                "string_pattern_mismatch": f"'{field_path}' entspricht nicht dem erwarteten Format (Eingabe: '{input_value}')",
                "greater_than_equal": f"'{field_path}' muss größer oder gleich {error.get('ge', 0)} sein (Eingabe: {input_value})",
                "less_than_equal": f"'{field_path}' muss kleiner oder gleich {error.get('le', 0)} sein (Eingabe: {input_value})",
                "greater_than": f"'{field_path}' muss größer als {error.get('gt', 0)} sein (Eingabe: {input_value})",
                "decimal_places": f"'{field_path}' darf maximal {error.get('decimal_places', 2)} Nachkommastellen haben",
                "value_error": f"'{field_path}': {error.get('msg', 'Ungültiger Wert')}",
                "type_error": f"'{field_path}' hat den falschen Datentyp (Eingabe: '{input_value}')",
                "literal_error": f"'{field_path}' muss einer der erlaubten Werte sein (Eingabe: '{input_value}')",
            }

            german_msg = german_messages.get(
                error_type,
                f"'{field_path}': {error.get('msg', 'Unbekannter Validierungsfehler')}",
            )
            errors.append(german_msg)

        context_prefix = f"[{self.context}] " if self.context else ""
        return f"{context_prefix}Validierungsfehler: {'; '.join(errors)}"

    def get_field_errors(self) -> dict[str, list[str]]:
        """Gruppiere Fehler nach Feldern für strukturierte Rückgabe."""
        field_errors: dict[str, list[str]] = {}

        for error in self.validation_error.errors():
            field_path = " → ".join(str(loc) for loc in error["loc"])
            error_msg = error.get("msg", "Unbekannter Fehler")

            if field_path not in field_errors:
                field_errors[field_path] = []
            field_errors[field_path].append(error_msg)

        return field_errors


def validate_and_handle_errors(
    model_class: type, data: dict[str, Any], context: str = ""
) -> Any:
    """
    Validiere Daten mit Pydantic-Modell und behandle Fehler benutzerfreundlich.

    Args:
        model_class: Pydantic BaseModel-Klasse
        data: Zu validierende Daten
        context: Kontext für Fehlermeldungen

    Returns:
        Validierte Modellinstanz

    Raises:
        GermanValidationError: Bei Validierungsfehlern
    """
    try:
        return model_class(**data)
    except ValidationError as e:
        logger.error(f"Validation failed for {model_class.__name__}: {e}")
        raise GermanValidationError(e, context) from e


def safe_parse_invoice_data(raw_data: dict[str, Any]) -> tuple[Any | None, list[str]]:
    """
    Sichere Parsing von Rechnungsdaten mit detailliertem Error-Reporting.

    Args:
        raw_data: Rohe Extraktionsdaten

    Returns:
        Tuple von (Invoice-Model oder None, Liste der Fehlermeldungen)
    """
    from src.models.invoice import Invoice

    errors = []
    invoice = None

    try:
        invoice = validate_and_handle_errors(Invoice, raw_data, "Rechnungsverarbeitung")
        logger.info("✅ Rechnung erfolgreich validiert")
    except GermanValidationError as e:
        errors.append(e.german_message)

        # Versuche Teilvalidierung für bessere Fehleranalyse
        try:
            # Prüfe Header separat
            if "header" in raw_data:
                from src.models.invoice import InvoiceHeader

                validate_and_handle_errors(
                    InvoiceHeader, raw_data["header"], "Rechnungskopf"
                )
                logger.info("✅ Rechnungskopf validiert")
        except GermanValidationError as header_error:
            errors.append(f"Rechnungskopf: {header_error.german_message}")

        try:
            # Prüfe Line Items separat
            if "line_items" in raw_data and isinstance(raw_data["line_items"], list):
                from src.models.invoice import LineItem

                for i, item_data in enumerate(raw_data["line_items"]):
                    validate_and_handle_errors(LineItem, item_data, f"Position {i+1}")
                logger.info("✅ Rechnungspositionen validiert")
        except GermanValidationError as item_error:
            errors.append(f"Rechnungspositionen: {item_error.german_message}")

        logger.warning(
            f"Rechnungsvalidierung fehlgeschlagen: {len(errors)} Fehler gefunden"
        )

    return invoice, errors


def safe_parse_skr03_classification(
    raw_data: dict[str, Any],
) -> tuple[Any | None, list[str]]:
    """
    Sichere Parsing von SKR03-Klassifizierungsdaten.

    Args:
        raw_data: Rohe Klassifizierungsdaten

    Returns:
        Tuple von (SKR03Classification-Model oder None, Liste der Fehlermeldungen)
    """
    from src.models.skr03 import SKR03Classification

    errors = []
    classification = None

    try:
        classification = validate_and_handle_errors(
            SKR03Classification, raw_data, "SKR03-Klassifizierung"
        )
        logger.info("✅ SKR03-Klassifizierung erfolgreich validiert")
    except GermanValidationError as e:
        errors.append(e.german_message)
        logger.warning(f"SKR03-Klassifizierung fehlgeschlagen: {e.german_message}")

    return classification, errors


def validate_processing_result(
    result_data: dict[str, Any],
) -> tuple[Any | None, list[str]]:
    """
    Validiere ProcessingResult mit umfassendem Error-Handling.

    Args:
        result_data: ProcessingResult-Daten

    Returns:
        Tuple von (ProcessingResult oder None, Liste der Fehlermeldungen)
    """
    from src.pipeline.processor import ProcessingResult

    errors = []
    result = None

    try:
        result = validate_and_handle_errors(
            ProcessingResult, result_data, "Verarbeitungsergebnis"
        )
        logger.info("✅ Verarbeitungsergebnis erfolgreich validiert")

        # Zusätzliche Business-Logic-Validierung
        if result.confidence_score < 0.5:
            errors.append(
                "Warnung: Niedrige Konfidenz-Score (< 0.5) - manuelle Überprüfung empfohlen"
            )

        if result.processing_time_ms > 30000:  # 30 Sekunden
            errors.append(
                "Warnung: Lange Verarbeitungszeit (> 30s) - Performance prüfen"
            )

    except GermanValidationError as e:
        errors.append(e.german_message)
        logger.error(
            f"Verarbeitungsergebnis-Validierung fehlgeschlagen: {e.german_message}"
        )

    return result, errors


# Hilfs-Funktionen für häufige Validierungsszenarien
def is_valid_german_vat_rate(rate: int) -> bool:
    """Prüfe ob MwSt-Satz für Deutschland gültig ist."""
    return rate in [0, 7, 19]


def is_valid_skr03_account(account: str) -> bool:
    """Prüfe ob SKR03-Kontonummer gültig ist."""
    if not account.isdigit() or len(account) != 4:
        return False

    account_int = int(account)
    valid_ranges = [
        (1000, 1999),
        (2000, 2999),
        (3000, 3999),
        (4000, 4999),
        (5000, 5999),
        (6000, 6999),
        (7000, 7999),
        (8000, 8999),
        (9000, 9999),
    ]

    return any(start <= account_int <= end for start, end in valid_ranges)


def format_currency_german(amount: float | int) -> str:
    """Formatiere Betrag nach deutschen Standards."""
    return f"{amount:,.2f} EUR".replace(",", "X").replace(".", ",").replace("X", ".")


# Export der wichtigsten Funktionen
__all__ = [
    "GermanValidationError",
    "validate_and_handle_errors",
    "safe_parse_invoice_data",
    "safe_parse_skr03_classification",
    "validate_processing_result",
    "is_valid_german_vat_rate",
    "is_valid_skr03_account",
    "format_currency_german",
]
