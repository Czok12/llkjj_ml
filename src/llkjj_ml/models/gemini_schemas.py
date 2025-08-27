#!/usr/bin/env python3
"""
LLKJJ ML Pipeline - Gemini Response Validation Schemas
====================================================

Pydantic v2 Modelle für die Validierung von Gemini AI Responses.
Stellt sicher, dass jede Gemini-Antwort die korrekte Struktur hat
und verhindert "Garbage In" in der Trainingsdatensammlung.

Strategische Bedeutung:
- Qualitätssicherung für Phase 1 (Gemini-produktiv)
- Saubere Trainingsdaten für Phase 2 (lokale Autonomie)
- Frühe Fehlererkennung und Debugging

Autor: LLKJJ ML Pipeline Team
Version: 1.0.0 (Pydantic v2)
Datum: 18. August 2025
"""

import logging
from datetime import date
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

logger = logging.getLogger(__name__)


class GeminiInvoiceHeader(BaseModel):
    """
    Pydantic-Schema für Gemini Invoice Header Extraktion.

    Validiert die Rechnungskopf-Daten aus Gemini AI Response.
    """

    lieferant: str = Field(
        ..., min_length=2, max_length=200, description="Firmenname des Lieferanten"
    )
    kundennummer: str | None = Field(None, max_length=50, description="Kundennummer")
    rechnungsnummer: str = Field(
        ..., min_length=1, max_length=100, description="Rechnungsnummer"
    )
    rechnungsdatum: str = Field(..., description="Rechnungsdatum im Format YYYY-MM-DD")
    lieferdatum: str | None = Field(
        None, description="Lieferdatum im Format YYYY-MM-DD"
    )
    zahlungsziel: str | None = Field(
        None, max_length=50, description="Zahlungsziel in Tagen oder Text"
    )

    @field_validator("rechnungsdatum", "lieferdatum")
    @classmethod
    def validate_date_format(cls, v: str | None) -> str | None:
        """Validiere Datumsformat YYYY-MM-DD"""
        if v is None:
            return v
        try:
            # Versuche Parsing als ISO-Datum
            if len(v) == 10:
                year, month, day = map(int, v.split("-"))
                date(year, month, day)  # Validation durch date-Constructor
                return v
            else:
                raise ValueError("Datum muss im Format YYYY-MM-DD sein")
        except (ValueError, AttributeError) as exc:
            raise ValueError(
                f"Ungültiges Datumsformat: {v}. Erwartet: YYYY-MM-DD"
            ) from exc


class GeminiLineItem(BaseModel):
    """
    Pydantic-Schema für Gemini Line Item Extraktion.

    Validiert einzelne Rechnungspositionen aus Gemini AI Response.
    """

    position: int = Field(..., ge=1, le=1000, description="Positionsnummer (1-1000)")
    artikelnummer: str | None = Field(
        None, max_length=100, description="Artikelnummer/SKU"
    )
    beschreibung: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Vollständige Artikelbeschreibung",
    )
    marke: str | None = Field(
        None, max_length=100, description="Herstellermarke (GIRA, Hager, etc.)"
    )
    menge: float = Field(
        ...,
        ge=-99999,
        le=99999,
        description="Menge (-99999 bis 99999, negativ bei Retouren/Gutschriften)",
    )

    @field_validator("menge")
    @classmethod
    def validate_quantity(cls, v: float) -> float:
        """Handle zero quantities by converting to minimal value"""
        if v == 0:
            logger.warning(f"⚠️ Zero quantity converted to 1.0: {v} -> 1.0")
            return 1.0
        return v

    einheit: str = Field(
        ..., max_length=20, description="Mengeneinheit (Stk, m, kg, etc.)"
    )
    einzelpreis: float = Field(
        ...,
        ge=-999999,
        le=999999,
        description="Einzelpreis in EUR (-999999 bis 999999, negativ bei Gutschriften)",
    )
    gesamtpreis: float = Field(
        ...,
        ge=-9999999,
        le=9999999,
        description="Gesamtpreis in EUR (-9999999 bis 9999999, negativ bei Gutschriften)",
    )
    elektro_kategorie: str | None = Field(
        None, max_length=100, description="Elektrotechnik-Kategorie"
    )
    skr03_vorschlag: str | None = Field(
        None, max_length=10, description="SKR03-Konto-Vorschlag"
    )
    skr03_beschreibung: str | None = Field(
        None, max_length=200, description="SKR03-Konto-Beschreibung"
    )


class GeminiTotals(BaseModel):
    """
    Pydantic-Schema für Gemini Totals Extraktion.

    Validiert Rechnungssummen aus Gemini AI Response.
    """

    nettosumme: float = Field(
        ...,
        ge=-9999999,
        le=9999999,
        description="Nettosumme in EUR (negativ bei Gutschriften)",
    )
    mwst_betrag: float = Field(
        ...,
        ge=-9999999,
        le=9999999,
        description="Mehrwertsteuerbetrag in EUR (negativ bei Gutschriften)",
    )
    mwst_satz: float = Field(
        ..., ge=0, le=100, description="Mehrwertsteuersatz in Prozent"
    )
    bruttosumme: float = Field(
        ...,
        ge=-9999999,
        le=9999999,
        description="Bruttosumme in EUR (negativ bei Gutschriften)",
    )


class GeminiZusatzinfos(BaseModel):
    """
    Pydantic-Schema für Gemini Zusatzinformationen.

    Validiert zusätzliche Rechnungsinformationen.
    """

    zahlungsart: str | None = Field(None, max_length=100, description="Zahlungsart")
    lieferadresse: str | None = Field(None, max_length=500, description="Lieferadresse")
    bemerkungen: str | None = Field(
        None, max_length=1000, description="Besondere Bemerkungen"
    )
    bestellnummer: str | None = Field(None, max_length=100, description="Bestellnummer")


class GeminiExtractionResult(BaseModel):
    """
    Vollständiges Pydantic-Schema für Gemini AI Response.

    Top-Level-Container für alle extrahierten Rechnungsdaten.
    Stellt sicher, dass jede Gemini-Response vollständig und korrekt ist.
    """

    model_config = ConfigDict(
        extra="allow",  # Erlaube zusätzliche Felder für Flexibilität
        str_strip_whitespace=True,  # Automatisches Trimming
        validate_assignment=True,  # Validierung bei Assignment
    )

    invoice_header: GeminiInvoiceHeader = Field(..., description="Rechnungskopf-Daten")
    line_items: list[GeminiLineItem] = Field(
        ..., min_length=1, max_length=1000, description="Rechnungspositionen"
    )
    totals: GeminiTotals | None = Field(None, description="Rechnungssummen")
    zusatzinfos: GeminiZusatzinfos | None = Field(
        None, description="Zusätzliche Informationen"
    )

    # Metadaten für Debugging und Tracking
    extraction_timestamp: str | None = Field(
        None, description="Zeitstempel der Extraktion"
    )
    gemini_model: str | None = Field(None, description="Verwendetes Gemini-Modell")
    confidence_indicators: dict[str, Any] | None = Field(
        None, description="Konfidenz-Indikatoren"
    )

    @field_validator("line_items")
    @classmethod
    def validate_line_items_not_empty(
        cls, v: list[GeminiLineItem]
    ) -> list[GeminiLineItem]:
        """Stelle sicher, dass mindestens eine Position vorhanden ist"""
        if not v or len(v) == 0:
            raise ValueError("Mindestens eine Rechnungsposition muss vorhanden sein")
        return v


# Utility-Funktionen für Validierung


def validate_gemini_response(
    response_data: dict[str, Any],
) -> tuple[GeminiExtractionResult | None, list[str]]:
    """
    Validiere Gemini-Response mit deutscher Fehlerbehandlung.

    Args:
        response_data: Rohe Gemini-Response als Dictionary

    Returns:
        Tuple von (GeminiExtractionResult oder None, Liste der deutschen Fehlermeldungen)
    """
    errors = []
    result = None

    try:
        result = GeminiExtractionResult(**response_data)
        return result, []

    except ValidationError as e:
        # Deutsche Übersetzung der Pydantic-Fehler
        for error in e.errors():
            field_path = " → ".join(str(loc) for loc in error["loc"])
            error_type = error["type"]
            input_value = error.get("input", "")

            # Deutsche Fehlermeldungen
            german_messages = {
                "missing": f"Pflichtfeld '{field_path}' fehlt",
                "value_error": f"Ungültiger Wert für '{field_path}': {error.get('msg', 'Validierungsfehler')}",
                "type_error": f"Falscher Datentyp für '{field_path}' (Eingabe: '{input_value}')",
                "greater_than": f"'{field_path}' muss größer als {error.get('ctx', {}).get('limit_value', 0)} sein",
                "less_than": f"'{field_path}' muss kleiner als {error.get('ctx', {}).get('limit_value', 0)} sein",
                "string_too_short": f"'{field_path}' ist zu kurz (min. {error.get('ctx', {}).get('limit_value', 0)} Zeichen)",
                "string_too_long": f"'{field_path}' ist zu lang (max. {error.get('ctx', {}).get('limit_value', 0)} Zeichen)",
            }

            german_msg = german_messages.get(
                error_type,
                f"Validierungsfehler in '{field_path}': {error.get('msg', 'Unbekannter Fehler')}",
            )
            errors.append(german_msg)

    except (TypeError, ValueError) as e:
        errors.append(f"Datenstruktur-Fehler: {str(e)}")

    return result, errors


def create_validation_report(errors: list[str]) -> str:
    """
    Erstelle einen deutschen Validierungsbericht.

    Args:
        errors: Liste der Validierungsfehler

    Returns:
        Formatierter deutscher Fehlerbericht
    """
    if not errors:
        return "✅ Gemini-Response erfolgreich validiert - alle Daten korrekt!"

    report = f"❌ Gemini-Response-Validierung fehlgeschlagen ({len(errors)} Fehler):\n"
    for i, error in enumerate(errors, 1):
        report += f"  {i}. {error}\n"

    report += "\n💡 Hinweis: Prüfe Gemini-Prompt und Antwortformat."
    return report


# Schema-Example für Testing und Dokumentation
EXAMPLE_GEMINI_RESPONSE = {
    "invoice_header": {
        "lieferant": "Sonepar Deutschland AG",
        "kundennummer": "123456",
        "rechnungsnummer": "2024021489",
        "rechnungsdatum": "2024-12-10",
        "lieferdatum": "2024-12-09",
        "zahlungsziel": "30",
    },
    "line_items": [
        {
            "position": 1,
            "artikelnummer": "6189404",
            "beschreibung": "GIRA Rahmen 1-fach reinweiß glänzend",
            "marke": "GIRA",
            "menge": 10.0,
            "einheit": "Stk",
            "einzelpreis": 3.45,
            "gesamtpreis": 34.50,
            "elektro_kategorie": "Schalterprogramm",
            "skr03_vorschlag": "3400",
            "skr03_beschreibung": "Wareneingang Elektromaterial",
        }
    ],
    "totals": {
        "nettosumme": 1234.56,
        "mwst_betrag": 234.56,
        "mwst_satz": 19.0,
        "bruttosumme": 1469.12,
    },
    "gemini_model": "gemini-2.0-flash-exp",
    "extraction_timestamp": "2025-08-18T14:30:00",
}
