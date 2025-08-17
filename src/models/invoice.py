#!/usr/bin/env python3
"""
LLKJJ ML Pipeline - German Invoice Data Models
==============================================

Pydantic BaseModel definitions for German electrical contractor invoices.
Optimized for German business processes, VAT rates, and accounting standards.

Author: LLKJJ ML Pipeline
Version: 1.0.0 (Pydantic v2 Migration)
"""

from datetime import date
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator


class Address(BaseModel):
    """German address with validation for postal codes and formatting."""

    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True)

    strasse: str = Field(
        ..., min_length=5, max_length=100, description="Straße und Hausnummer"
    )
    plz: str = Field(
        ..., pattern=r"^\d{5}$", description="Deutsche Postleitzahl (5-stellig)"
    )
    ort: str = Field(..., min_length=2, max_length=100, description="Ortsname")
    land: str = Field(default="Deutschland", description="Land")

    @field_validator("plz")
    @classmethod
    def validate_german_plz(cls, v: str) -> str:
        """Validiere deutsche Postleitzahl"""
        if not v.isdigit() or len(v) != 5:
            raise ValueError(f"Deutsche PLZ muss 5-stellig sein: {v}")
        # Basic range check for German postal codes
        plz_int = int(v)
        if plz_int < 1000 or plz_int > 99999:
            raise ValueError(f"Ungültige deutsche PLZ: {v}")
        return v

    def get_full_address(self) -> str:
        """Formatierte Adresse für deutsche Standards"""
        return f"{self.strasse}, {self.plz} {self.ort}"


class Contact(BaseModel):
    """Contact information with German phone number validation."""

    model_config = ConfigDict(str_strip_whitespace=True)

    name: str = Field(..., min_length=2, max_length=100, description="Kontaktperson")
    telefon: str | None = Field(None, description="Telefonnummer")
    email: str | None = Field(None, description="E-Mail-Adresse")

    @field_validator("telefon")
    @classmethod
    def validate_phone(cls, v: str | None) -> str | None:
        """Validiere deutsche Telefonnummer (einfache Prüfung)"""
        if v is None:
            return v
        # Simple German phone validation
        cleaned = v.replace(" ", "").replace("-", "").replace("/", "")
        if not cleaned.startswith(("+49", "0")) or len(cleaned) < 10:
            raise ValueError(f"Ungültige deutsche Telefonnummer: {v}")
        return v


class Company(BaseModel):
    """German company information with business validation."""

    model_config = ConfigDict(str_strip_whitespace=True)

    name: str = Field(..., min_length=2, max_length=200, description="Firmenname")
    adresse: Address = Field(..., description="Firmenadresse")
    ustid: str | None = Field(None, description="Umsatzsteuer-ID")
    contact: Contact | None = Field(None, description="Ansprechpartner")

    @field_validator("ustid")
    @classmethod
    def validate_ustid(cls, v: str | None) -> str | None:
        """Validiere deutsche USt-ID"""
        if v is None:
            return v
        # German VAT ID format: DE + 9 digits
        if not v.startswith("DE") or len(v) != 11 or not v[2:].isdigit():
            raise ValueError(f"Ungültige deutsche USt-ID: {v}")
        return v.upper()


class LineItem(BaseModel):
    """Invoice line item with German VAT and pricing validation."""

    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True)

    position: int = Field(..., ge=1, description="Positionsnummer")
    artikel_nr: str | None = Field(None, max_length=50, description="Artikelnummer")
    beschreibung: str = Field(
        ..., min_length=3, max_length=500, description="Artikelbeschreibung"
    )
    menge: Decimal = Field(..., gt=0, decimal_places=3, description="Menge")
    einheit: str = Field(default="Stk", max_length=10, description="Mengeneinheit")
    einzelpreis: Decimal = Field(
        ..., ge=0, decimal_places=2, description="Einzelpreis netto in EUR"
    )
    gesamt_netto: Decimal = Field(
        ..., ge=0, decimal_places=2, description="Gesamtpreis netto in EUR"
    )
    mwst_satz: Literal[0, 7, 19] = Field(19, description="MwSt-Satz in % (0, 7, 19)")
    mwst_betrag: Decimal = Field(
        ..., ge=0, decimal_places=2, description="MwSt-Betrag in EUR"
    )
    gesamt_brutto: Decimal = Field(
        ..., ge=0, decimal_places=2, description="Gesamtpreis brutto in EUR"
    )

    @field_validator("gesamt_netto")
    @classmethod
    def validate_line_total(cls, v: Decimal, info: ValidationInfo) -> Decimal:
        """Validiere dass Gesamtpreis = Menge × Einzelpreis"""
        if info.data.get("menge") and info.data.get("einzelpreis"):
            expected = info.data["menge"] * info.data["einzelpreis"]
            # Allow small rounding differences
            if abs(v - expected) > Decimal("0.01"):
                raise ValueError(
                    f"Gesamtpreis ({v}) stimmt nicht überein mit Menge × Einzelpreis ({expected})"
                )
        return v

    @field_validator("mwst_betrag")
    @classmethod
    def validate_vat_amount(cls, v: Decimal, info: ValidationInfo) -> Decimal:
        """Validiere MwSt-Betrag basierend auf Nettobetrag und Satz"""
        if info.data.get("gesamt_netto") and info.data.get("mwst_satz"):
            expected = (
                info.data["gesamt_netto"] * Decimal(str(info.data["mwst_satz"])) / 100
            )
            if abs(v - expected) > Decimal("0.01"):
                raise ValueError(
                    f"MwSt-Betrag ({v}) stimmt nicht mit Berechnung überein ({expected})"
                )
        return v

    def calculate_totals(self) -> None:
        """Berechne automatisch alle Totals (für Helper-Funktionen)"""
        self.gesamt_netto = self.menge * self.einzelpreis
        self.mwst_betrag = self.gesamt_netto * Decimal(str(self.mwst_satz)) / 100
        self.gesamt_brutto = self.gesamt_netto + self.mwst_betrag


class InvoiceHeader(BaseModel):
    """German invoice header with business validation."""

    model_config = ConfigDict(
        str_strip_whitespace=True, validate_assignment=True, use_enum_values=True
    )

    # Invoice identification
    rechnung_nummer: str = Field(
        ..., pattern=r"^[A-Z0-9-]{3,20}$", description="Rechnungsnummer"
    )
    rechnungs_datum: date = Field(..., description="Rechnungsdatum")
    leistungs_datum: date | None = Field(None, description="Leistungsdatum")

    # Company information
    lieferant: Company = Field(..., description="Lieferant/Verkäufer")
    kunde: Company = Field(..., description="Kunde/Käufer")

    # Payment terms
    zahlungsziel: int = Field(
        default=30, ge=1, le=365, description="Zahlungsziel in Tagen"
    )
    skonto_tage: int | None = Field(None, ge=1, le=30, description="Skonto-Tage")
    skonto_prozent: Decimal | None = Field(
        None, ge=0, le=10, decimal_places=2, description="Skonto in %"
    )

    # Reference
    kunden_nummer: str | None = Field(None, max_length=50, description="Kundennummer")
    bestell_nummer: str | None = Field(None, max_length=50, description="Bestellnummer")

    @field_validator("leistungs_datum")
    @classmethod
    def validate_leistungs_datum(
        cls, v: date | None, info: ValidationInfo
    ) -> date | None:
        """Validiere dass Leistungsdatum <= Rechnungsdatum"""
        if v and info.data.get("rechnungs_datum"):
            if v > info.data["rechnungs_datum"]:
                raise ValueError("Leistungsdatum kann nicht nach Rechnungsdatum liegen")
        return v


class InvoiceFooter(BaseModel):
    """German invoice totals and VAT summary."""

    model_config = ConfigDict(validate_assignment=True)

    # Netto totals by VAT rate
    netto_0_prozent: Decimal = Field(default=Decimal("0.00"), ge=0, decimal_places=2)
    netto_7_prozent: Decimal = Field(default=Decimal("0.00"), ge=0, decimal_places=2)
    netto_19_prozent: Decimal = Field(default=Decimal("0.00"), ge=0, decimal_places=2)

    # VAT amounts by rate
    mwst_0_prozent: Decimal = Field(default=Decimal("0.00"), ge=0, decimal_places=2)
    mwst_7_prozent: Decimal = Field(default=Decimal("0.00"), ge=0, decimal_places=2)
    mwst_19_prozent: Decimal = Field(default=Decimal("0.00"), ge=0, decimal_places=2)

    # Totals
    summe_netto: Decimal = Field(
        ..., ge=0, decimal_places=2, description="Gesamtsumme netto"
    )
    summe_mwst: Decimal = Field(..., ge=0, decimal_places=2, description="Gesamte MwSt")
    summe_brutto: Decimal = Field(
        ..., ge=0, decimal_places=2, description="Gesamtsumme brutto"
    )

    @field_validator("summe_netto")
    @classmethod
    def validate_netto_total(cls, v: Decimal, info: ValidationInfo) -> Decimal:
        """Validiere dass Netto-Gesamtsumme korrekt ist"""
        expected = (
            info.data.get("netto_0_prozent", Decimal("0"))
            + info.data.get("netto_7_prozent", Decimal("0"))
            + info.data.get("netto_19_prozent", Decimal("0"))
        )
        if abs(v - expected) > Decimal("0.01"):
            raise ValueError(
                f"Netto-Gesamtsumme ({v}) stimmt nicht mit Einzelsummen überein ({expected})"
            )
        return v

    @field_validator("summe_mwst")
    @classmethod
    def validate_mwst_total(cls, v: Decimal, info: ValidationInfo) -> Decimal:
        """Validiere dass MwSt-Gesamtsumme korrekt ist"""
        expected = (
            info.data.get("mwst_0_prozent", Decimal("0"))
            + info.data.get("mwst_7_prozent", Decimal("0"))
            + info.data.get("mwst_19_prozent", Decimal("0"))
        )
        if abs(v - expected) > Decimal("0.01"):
            raise ValueError(
                f"MwSt-Gesamtsumme ({v}) stimmt nicht mit Einzelsummen überein ({expected})"
            )
        return v

    def calculate_from_line_items(self, line_items: list[LineItem]) -> None:
        """Berechne Footer-Totals aus Line Items"""
        # Reset totals
        self.netto_0_prozent = Decimal("0.00")
        self.netto_7_prozent = Decimal("0.00")
        self.netto_19_prozent = Decimal("0.00")
        self.mwst_0_prozent = Decimal("0.00")
        self.mwst_7_prozent = Decimal("0.00")
        self.mwst_19_prozent = Decimal("0.00")

        # Aggregate by VAT rate
        for item in line_items:
            if item.mwst_satz == 0:
                self.netto_0_prozent += item.gesamt_netto
                self.mwst_0_prozent += item.mwst_betrag
            elif item.mwst_satz == 7:
                self.netto_7_prozent += item.gesamt_netto
                self.mwst_7_prozent += item.mwst_betrag
            elif item.mwst_satz == 19:
                self.netto_19_prozent += item.gesamt_netto
                self.mwst_19_prozent += item.mwst_betrag

        # Calculate totals
        self.summe_netto = (
            self.netto_0_prozent + self.netto_7_prozent + self.netto_19_prozent
        )
        self.summe_mwst = (
            self.mwst_0_prozent + self.mwst_7_prozent + self.mwst_19_prozent
        )
        self.summe_brutto = self.summe_netto + self.summe_mwst


class Invoice(BaseModel):
    """Complete German invoice with header, line items, and footer."""

    model_config = ConfigDict(validate_assignment=True)

    header: InvoiceHeader = Field(..., description="Rechnungskopf")
    line_items: list[LineItem] = Field(
        ..., min_length=1, description="Rechnungspositionen"
    )
    footer: InvoiceFooter = Field(..., description="Rechnungssummen")

    # Metadata
    confidence_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Extraktions-Konfidenz"
    )
    extraction_source: str = Field(
        default="unknown", description="Quelle der Extraktion"
    )

    @field_validator("line_items")
    @classmethod
    def validate_unique_positions(cls, v: list[LineItem]) -> list[LineItem]:
        """Validiere eindeutige Positionsnummern"""
        positions = [item.position for item in v]
        if len(positions) != len(set(positions)):
            raise ValueError("Positionsnummern müssen eindeutig sein")
        return v

    def validate_totals(self) -> bool:
        """Validiere dass Header/Footer Totals mit Line Items übereinstimmen"""
        # Calculate expected totals from line items
        expected_footer = InvoiceFooter(
            summe_netto=Decimal("0"), summe_mwst=Decimal("0"), summe_brutto=Decimal("0")
        )
        expected_footer.calculate_from_line_items(self.line_items)

        # Compare with actual footer (allow small rounding differences)
        tolerance = Decimal("0.01")
        return (
            abs(self.footer.summe_netto - expected_footer.summe_netto) <= tolerance
            and abs(self.footer.summe_mwst - expected_footer.summe_mwst) <= tolerance
            and abs(self.footer.summe_brutto - expected_footer.summe_brutto)
            <= tolerance
        )

    def get_summary(self) -> str:
        """Erstelle eine Zusammenfassung der Rechnung"""
        return (
            f"Rechnung {self.header.rechnung_nummer} | "
            f"Lieferant: {self.header.lieferant.name} | "
            f"Datum: {self.header.rechnungs_datum} | "
            f"Summe: {self.footer.summe_brutto:.2f} EUR | "
            f"{len(self.line_items)} Positionen"
        )
