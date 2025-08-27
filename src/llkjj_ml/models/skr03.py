#!/usr/bin/env python3
"""
LLKJJ ML Pipeline - SKR03 Classification Models
===============================================

Pydantic BaseModel definitions for German SKR03 chart of accounts classification.
Handles validation of account numbers, business logic, and confidence scoring.

Author: LLKJJ ML Pipeline
Version: 1.0.0 (Pydantic v2 Migration)
"""

from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator


class SKR03Account(BaseModel):
    """SKR03 chart of accounts entry with validation."""

    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True)

    konto: str = Field(
        ..., pattern=r"^\d{4}$", description="SKR03 Kontonummer (4-stellig)"
    )
    bezeichnung: str = Field(
        ..., min_length=3, max_length=200, description="Kontobezeichnung"
    )
    kontoart: Literal[
        "Aktiva", "Passiva", "Aufwendungen", "Erträge", "Abschluss", "Vorträge"
    ] = Field(..., description="SKR03 Kontoart")
    kontoklasse: int = Field(..., ge=0, le=9, description="SKR03 Kontoklasse (0-9)")

    @field_validator("konto")
    @classmethod
    def validate_skr03_account(cls, v: str) -> str:
        """Validiere SKR03-Kontonummer format und range"""
        if not v.isdigit() or len(v) != 4:
            raise ValueError(f"SKR03-Konto muss 4-stellig sein: {v}")

        account_int = int(v)
        # SKR03 valid ranges (simplified)
        valid_ranges = [
            (1000, 1999),  # Anlagevermögen
            (2000, 2999),  # Umlaufvermögen
            (3000, 3999),  # Eigenkapital/Fremdkapital
            (4000, 4999),  # Betriebliche Erträge
            (5000, 5999),  # Materialaufwand
            (6000, 6999),  # Personalaufwand
            (7000, 7999),  # Sonstige Aufwendungen
            (8000, 8999),  # Ergebnisrechnung
            (9000, 9999),  # Abschlusskonten
        ]

        if not any(start <= account_int <= end for start, end in valid_ranges):
            raise ValueError(f"Ungültiger SKR03-Kontonummer-Bereich: {v}")

        return v

    @field_validator("kontoklasse")
    @classmethod
    def validate_kontoklasse_consistency(cls, v: int, info: ValidationInfo) -> int:
        """Validiere Konsistenz zwischen Kontonummer und Kontoklasse"""
        if info.data.get("konto"):
            expected_class = int(info.data["konto"][0])
            if v != expected_class:
                raise ValueError(
                    f"Kontoklasse {v} stimmt nicht mit Kontonummer überein (erwartet: {expected_class})"
                )
        return v


class SKR03Classification(BaseModel):
    """Single SKR03 classification result with confidence scoring."""

    model_config = ConfigDict(validate_assignment=True)

    konto: str = Field(..., pattern=r"^\d{4}$", description="SKR03 Kontonummer")
    bezeichnung: str = Field(..., min_length=3, description="Kontobezeichnung")
    betrag: Decimal = Field(..., decimal_places=2, description="Buchungsbetrag in EUR")
    soll_haben: Literal["Soll", "Haben"] = Field(
        ..., description="Soll- oder Haben-Buchung"
    )

    # Confidence and quality metrics
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Klassifizierungs-Konfidenz (0.0-1.0)"
    )
    rule_based: bool = Field(default=False, description="Regelbasierte Klassifizierung")
    ml_prediction: bool = Field(default=False, description="ML-basierte Vorhersage")
    manual_review: bool = Field(
        default=False, description="Manuelle Überprüfung erforderlich"
    )

    # Source information
    source_text: str = Field(default="", description="Quelltext für Klassifizierung")
    matching_keywords: list[str] = Field(
        default_factory=list, description="Gefundene Schlüsselwörter"
    )

    @field_validator("betrag")
    @classmethod
    def validate_amount(cls, v: Decimal) -> Decimal:
        """Validiere Buchungsbetrag (muss != 0 sein)"""
        if v == 0:
            raise ValueError("Buchungsbetrag darf nicht 0 sein")
        return v

    @field_validator("confidence")
    @classmethod
    def validate_confidence_thresholds(cls, v: float, info: ValidationInfo) -> float:
        """Validiere Konfidenz und setze Review-Flags"""
        # Low confidence requires manual review
        if v < 0.6 and not info.data.get("manual_review", False):
            # Note: We can't modify other fields in validator, but we can warn
            pass
        return v

    def get_classification_summary(self) -> str:
        """Erstelle Zusammenfassung der Klassifizierung"""
        source = (
            "Regel"
            if self.rule_based
            else ("ML" if self.ml_prediction else "Unbekannt")
        )
        review = " [Review]" if self.manual_review else ""
        return (
            f"Konto {self.konto} ({self.bezeichnung}) | "
            f"{self.soll_haben}: {self.betrag:.2f} EUR | "
            f"Konfidenz: {self.confidence:.3f} ({source}){review}"
        )


class SKR03MultipleClassification(BaseModel):
    """Multiple SKR03 classifications for complex transactions."""

    model_config = ConfigDict(validate_assignment=True)

    classifications: list[SKR03Classification] = Field(
        ..., min_length=1, description="Liste der Klassifizierungen"
    )
    total_amount: Decimal = Field(..., decimal_places=2, description="Gesamtbetrag")
    balanced: bool = Field(default=False, description="Soll/Haben-Balance geprüft")

    # Overall quality metrics
    average_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Durchschnittliche Konfidenz"
    )
    requires_review: bool = Field(default=False, description="Überprüfung erforderlich")

    @field_validator("classifications")
    @classmethod
    def validate_unique_accounts(
        cls, v: list[SKR03Classification]
    ) -> list[SKR03Classification]:
        """Validiere eindeutige Kontonummern pro Soll/Haben-Seite"""
        soll_accounts = []
        haben_accounts = []

        for classification in v:
            if classification.soll_haben == "Soll":
                soll_accounts.append(classification.konto)
            else:
                haben_accounts.append(classification.konto)

        # Check for duplicates within each side
        if len(soll_accounts) != len(set(soll_accounts)):
            raise ValueError("Doppelte Soll-Konten in Klassifizierung")
        if len(haben_accounts) != len(set(haben_accounts)):
            raise ValueError("Doppelte Haben-Konten in Klassifizierung")

        return v

    def calculate_balance(self) -> Decimal:
        """Berechne Soll/Haben-Balance"""
        soll_total = Decimal(
            str(sum(c.betrag for c in self.classifications if c.soll_haben == "Soll"))
        )
        haben_total = Decimal(
            str(sum(c.betrag for c in self.classifications if c.soll_haben == "Haben"))
        )
        return soll_total - haben_total

    def validate_balance(self, tolerance: Decimal = Decimal("0.01")) -> bool:
        """Validiere dass Soll = Haben (mit Toleranz)"""
        balance = self.calculate_balance()
        is_balanced = abs(balance) <= tolerance
        self.balanced = is_balanced
        return is_balanced

    def calculate_metrics(self) -> None:
        """Berechne Gesamtmetriken automatisch"""
        if not self.classifications:
            return

        # Average confidence
        self.average_confidence = sum(c.confidence for c in self.classifications) / len(
            self.classifications
        )

        # Requires review if any single classification needs it or low overall confidence
        self.requires_review = (
            any(c.manual_review for c in self.classifications)
            or self.average_confidence < 0.7
            or not self.balanced
        )

    def get_summary(self) -> str:
        """Erstelle Zusammenfassung der Multi-Klassifizierung"""
        balance = self.calculate_balance()
        balance_status = "✓" if self.balanced else "⚠"
        review_status = "[Review]" if self.requires_review else "[OK]"

        return (
            f"{len(self.classifications)} Buchungen | "
            f"Summe: {self.total_amount:.2f} EUR | "
            f"Balance: {balance:.2f} {balance_status} | "
            f"Konfidenz: {self.average_confidence:.3f} {review_status}"
        )


class ElectricalContractorMapping(BaseModel):
    """Elektrohandwerk-spezifische SKR03-Mappings und Regeln."""

    model_config = ConfigDict(validate_assignment=True)

    # Common electrical contractor accounts
    material_accounts: list[str] = Field(
        default_factory=lambda: ["5400", "5410", "5420"],  # Rohstoffe/Material
        description="Konten für Elektromaterial",
    )
    revenue_accounts: list[str] = Field(
        default_factory=lambda: ["4100", "4110", "4120"],  # Umsatzerlöse
        description="Konten für Elektroinstallation/Service",
    )
    expense_accounts: list[str] = Field(
        default_factory=lambda: ["6200", "6210", "6300"],  # Fremdleistungen/Fahrzeug
        description="Konten für Betriebsausgaben",
    )

    # VAT accounts
    vat_19_input: str = Field(default="1576", description="Vorsteuer 19%")
    vat_19_output: str = Field(default="3806", description="Umsatzsteuer 19%")
    vat_7_input: str = Field(default="1571", description="Vorsteuer 7%")
    vat_7_output: str = Field(default="3801", description="Umsatzsteuer 7%")

    # Common supplier patterns
    supplier_keywords: dict[str, str] = Field(
        default_factory=lambda: {
            "sonepar": "5400",  # Elektrogroßhandel
            "rexel": "5400",
            "elektro": "5400",
            "kabel": "5400",
            "schalter": "5400",
            "tankstelle": "6530",  # Fahrzeugkosten
            "werkzeug": "6600",  # Geringwertige Wirtschaftsgüter
        },
        description="Lieferanten-Keywords zu SKR03-Konten",
    )

    def classify_by_supplier(
        self, supplier_name: str, amount: Decimal
    ) -> SKR03Classification | None:
        """Klassifiziere basierend auf Lieferanten-Namen"""
        supplier_lower = supplier_name.lower()

        for keyword, account in self.supplier_keywords.items():
            if keyword in supplier_lower:
                # Get account description (simplified)
                descriptions = {
                    "5400": "Rohstoffe und Material",
                    "6530": "Fahrzeugkosten",
                    "6600": "Geringwertige Wirtschaftsgüter",
                }

                return SKR03Classification(
                    konto=account,
                    bezeichnung=descriptions.get(account, "Sonstige Aufwendungen"),
                    betrag=amount,
                    soll_haben="Soll",
                    confidence=0.8,
                    rule_based=True,
                    source_text=supplier_name,
                    matching_keywords=[keyword],
                )

        return None

    def get_vat_classification(
        self, vat_rate: int, amount: Decimal, is_input: bool = True
    ) -> SKR03Classification:
        """Erstelle MwSt-Klassifizierung basierend auf Satz und Richtung"""
        if vat_rate == 19:
            account = self.vat_19_input if is_input else self.vat_19_output
            description = "Vorsteuer 19%" if is_input else "Umsatzsteuer 19%"
        elif vat_rate == 7:
            account = self.vat_7_input if is_input else self.vat_7_output
            description = "Vorsteuer 7%" if is_input else "Umsatzsteuer 7%"
        else:
            raise ValueError(f"Ungültiger MwSt-Satz: {vat_rate}%")

        return SKR03Classification(
            konto=account,
            bezeichnung=description,
            betrag=amount,
            soll_haben="Soll" if is_input else "Haben",
            confidence=1.0,  # VAT classification is always certain
            rule_based=True,
            source_text=f"MwSt {vat_rate}%",
        )


# Export specialized electrical contractor instance
ELECTRICAL_CONTRACTOR_MAPPING = ElectricalContractorMapping()
