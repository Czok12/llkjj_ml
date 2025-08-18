#!/usr/bin/env python3
"""
LLKJJ ML Pipeline - Deutsche Elektrotechnik NER-Spezialisierung
===============================================================

🇩🇪 Domain-spezifische Named Entity Recognition für deutsches Elektrohandwerk

Spezialisierte NER-Funktionalitäten:
- GIRA, Hager, Siemens → Hersteller-Entitäten
- Artikelnummer-Pattern-Learning: Elektriker-spezifische SKU-Formate
- Mengen-Unit-Normalization: "5 St" vs "5 Stück" vs "5x" → einheitliche Erkennung
- Price-Extraction-Robustness: Euro-Zeichen, Komma vs. Punkt, Netto/Brutto

Autor: LLKJJ ML Pipeline Team
Version: 1.0.0 (Deutsche Elektrotechnik-Spezialisierung)
Datum: 18. August 2025
"""

import logging
import re
from typing import Any

import spacy
from pydantic import BaseModel, Field
from spacy.tokens import Doc

from src.config import Config

logger = logging.getLogger(__name__)


class ElektroEntität(BaseModel):
    """Elektrotechnik-spezifische Entität."""

    text: str = Field(..., description="Entitäts-Text")
    label: str = Field(..., description="Entitäts-Label (HERSTELLER, ARTIKEL, etc.)")
    start: int = Field(..., description="Start-Position im Text")
    end: int = Field(..., description="End-Position im Text")
    confidence: float = Field(default=1.0, description="Erkennungs-Konfidenz")
    normalized: str = Field(default="", description="Normalisierte Form")
    category: str = Field(default="", description="Elektro-Kategorie")


class ElektroNERResult(BaseModel):
    """Ergebnis der Elektrotechnik NER-Analyse."""

    text: str = Field(..., description="Ursprünglicher Text")
    entities: list[ElektroEntität] = Field(..., description="Erkannte Entitäten")
    hersteller: list[str] = Field(
        default_factory=list, description="Erkannte Hersteller"
    )
    artikel_nummern: list[str] = Field(
        default_factory=list, description="Artikel-Nummern"
    )
    mengen: list[dict[str, Any]] = Field(
        default_factory=list, description="Normalisierte Mengen"
    )
    preise: list[dict[str, Any]] = Field(
        default_factory=list, description="Extrahierte Preise"
    )


class DeutscheElektrotechnikNER:
    """
    🔌 Deutsche Elektrotechnik Named Entity Recognition

    Spezialisiert auf deutsches Elektrohandwerk mit branchenspezifischen
    Entitäten, Mustern und Normalisierungen.
    """

    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.nlp = self._load_spacy_model()
        self._setup_patterns()
        logger.info("🔌 Deutsche Elektrotechnik NER initialisiert")

    def _load_spacy_model(self) -> Any:
        """Lädt und konfiguriert das deutsche spaCy-Modell."""
        try:
            nlp = spacy.load(self.config.spacy_model_name)

            # Custom pipeline components hinzufügen
            if "elektro_ner" not in nlp.pipe_names:
                nlp.add_pipe("elektro_ner", after="ner")

            return nlp

        except OSError:
            logger.warning("⚠️ Deutsches spaCy-Modell nicht gefunden, lade Basis-Modell")
            return spacy.load("de_core_web_sm")

    def _setup_patterns(self) -> None:
        """Setup für Elektrotechnik-spezifische Pattern."""

        # Deutsche Elektro-Hersteller (erweiterte Liste)
        self.elektro_hersteller = {
            # Premium-Marken
            "GIRA",
            "Hager",
            "Siemens",
            "ABB",
            "Schneider Electric",
            "Legrand",
            "Busch-Jaeger",
            "Merten",
            "Jung",
            "Berker",
            # Installations-Material
            "Wago",
            "Phoenix Contact",
            "Weidmüller",
            "Wieland",
            "Hensel",
            "Spelsberg",
            "OBO Bettermann",
            "Lapp Kabel",
            # Beleuchtung & LED
            "Osram",
            "Philips",
            "Ledvance",
            "Paulmann",
            "SLV",
            "Trilux",
            # Mess & Prüftechnik
            "Fluke",
            "Benning",
            "Gossen Metrawatt",
            "Chauvin Arnoux",
            # Installation & Werkzeug
            "Rennsteig",
            "Knipex",
            "Wiha",
            "Wera",
            "Klauke",
            # Größere Lieferanten
            "Sonepar",
            "Rexel",
            "Conrad",
            "ELV",
            "Reichelt",
        }

        # Artikelnummer-Pattern für deutsches Elektrohandwerk
        self.artikel_patterns = [
            r"\b\d{6,12}\b",  # 6-12 stellige Nummern (Standard)
            r"\b[A-Z]{2,4}\d{4,8}\b",  # Hersteller-Prefix + Nummer
            r"\b\d{3}[-\.]\d{3}[-\.]\d{3}\b",  # xxx-xxx-xxx Format
            r"\bEAN\s*:?\s*\d{13}\b",  # EAN-Codes
            r"\b[A-Z]\d{6,8}[A-Z]?\b",  # Letter-Number-Letter Format
        ]

        # Mengen-Unit-Pattern (deutsch)
        self.mengen_patterns = [
            r"(\d+(?:,\d+)?)\s*(St\.?|Stück|Stk\.?|x|mal|×)",  # Stück
            r"(\d+(?:,\d+)?)\s*(m|Meter|mm|cm)",  # Länge
            r"(\d+(?:,\d+)?)\s*(kg|g|Gramm|Kilogramm)",  # Gewicht
            r"(\d+(?:,\d+)?)\s*(l|Liter|ml)",  # Volumen
            r"(\d+(?:,\d+)?)\s*(m²|qm|Quadratmeter)",  # Fläche
            r"(\d+(?:,\d+)?)\s*(Paar|paar|Pck\.?|Pack|Packung)",  # Verpackung
        ]

        # Preis-Pattern (deutsche Formate)
        self.preis_patterns = [
            r"(\d{1,3}(?:\.\d{3})*,\d{2})\s*€",  # 1.234,56 €
            r"€\s*(\d{1,3}(?:\.\d{3})*,\d{2})",  # € 1.234,56
            r"(\d+,\d{2})\s*EUR",  # 123,45 EUR
            r"(\d+\.\d{2})\s*€",  # Englisches Format 123.45 €
            r"(\d+(?:,\d{2})?)\s*€(?:/[A-Za-z]+)?",  # 123€ oder 123€/Stk
        ]

        # Kompilierte Regex-Pattern
        self.compiled_patterns = {
            "artikel": [re.compile(pattern) for pattern in self.artikel_patterns],
            "mengen": [
                re.compile(pattern, re.IGNORECASE) for pattern in self.mengen_patterns
            ],
            "preise": [re.compile(pattern) for pattern in self.preis_patterns],
        }

    # @spacy.Language.component("elektro_ner")  # Deaktiviert für Kompatibilität
    def elektro_ner_component(self, doc: Doc) -> Doc:
        """Custom spaCy-Pipeline-Component für Elektrotechnik-NER."""
        spans = []

        # Hersteller-Erkennung
        for hersteller in self.elektro_hersteller:
            pattern = re.compile(rf"\b{re.escape(hersteller)}\b", re.IGNORECASE)
            for match in pattern.finditer(doc.text):
                start_char = match.start()
                end_char = match.end()
                span = doc.char_span(start_char, end_char, label="HERSTELLER")
                if span:
                    spans.append(span)

        # Artikel-Nummern
        for pattern in self.compiled_patterns["artikel"]:
            for match in pattern.finditer(doc.text):
                start_char = match.start()
                end_char = match.end()
                span = doc.char_span(start_char, end_char, label="ARTIKEL_NR")
                if span:
                    spans.append(span)

        # Spans zu doc hinzufügen
        doc.spans["elektro_entities"] = spans
        return doc

    def extract_entities(self, text: str) -> ElektroNERResult:
        """
        🔍 Hauptfunktion für Entitäts-Extraktion

        Args:
            text: Zu analysierender Text

        Returns:
            ElektroNERResult: Alle erkannten Elektrotechnik-Entitäten
        """
        try:
            # spaCy-Processing
            doc = self.nlp(text)

            entities = []
            hersteller = []
            artikel_nummern = []

            # Standard spaCy-Entitäten
            for ent in doc.ents:
                elektro_ent = ElektroEntität(
                    text=ent.text,
                    label=ent.label_,
                    start=ent.start_char,
                    end=ent.end_char,
                    confidence=0.9,  # spaCy-Confidence
                )
                entities.append(elektro_ent)

            # Custom Elektro-Entitäten
            if "elektro_entities" in doc.spans:
                for span in doc.spans["elektro_entities"]:
                    elektro_ent = ElektroEntität(
                        text=span.text,
                        label=span.label_,
                        start=span.start_char,
                        end=span.end_char,
                        confidence=1.0,  # Custom Pattern = hohe Konfidenz
                    )
                    entities.append(elektro_ent)

                    # Kategorie-spezifische Listen
                    if span.label_ == "HERSTELLER":
                        hersteller.append(span.text)
                    elif span.label_ == "ARTIKEL_NR":
                        artikel_nummern.append(span.text)

            # Mengen-Extraktion
            mengen = self._extract_mengen(text)

            # Preis-Extraktion
            preise = self._extract_preise(text)

            return ElektroNERResult(
                text=text,
                entities=entities,
                hersteller=list(set(hersteller)),  # Duplikate entfernen
                artikel_nummern=list(set(artikel_nummern)),
                mengen=mengen,
                preise=preise,
            )

        except Exception as e:
            logger.error("❌ Elektrotechnik NER-Extraktion fehlgeschlagen: %s", e)
            return ElektroNERResult(
                text=text,
                entities=[],
                hersteller=[],
                artikel_nummern=[],
                mengen=[],
                preise=[],
            )

    def _extract_mengen(self, text: str) -> list[dict[str, Any]]:
        """
        📏 Extrahiert und normalisiert Mengen-Angaben

        Args:
            text: Text zur Analyse

        Returns:
            list: Normalisierte Mengen mit Einheit
        """
        mengen = []

        for pattern in self.compiled_patterns["mengen"]:
            for match in pattern.finditer(text):
                value_str = match.group(1).replace(
                    ",", "."
                )  # Deutsche → Englische Dezimal
                unit = match.group(2)

                try:
                    value = float(value_str)

                    # Unit-Normalisierung
                    normalized_unit = self._normalize_unit(unit)

                    mengen.append(
                        {
                            "original_text": match.group(0),
                            "value": value,
                            "original_unit": unit,
                            "normalized_unit": normalized_unit,
                            "start": match.start(),
                            "end": match.end(),
                        }
                    )

                except ValueError:
                    logger.warning("⚠️ Konnte Menge nicht parsen: %s", match.group(0))

        return mengen

    def _extract_preise(self, text: str) -> list[dict[str, Any]]:
        """
        💰 Extrahiert und normalisiert Preis-Angaben

        Args:
            text: Text zur Analyse

        Returns:
            list: Normalisierte Preise in Euro
        """
        preise = []

        for pattern in self.compiled_patterns["preise"]:
            for match in pattern.finditer(text):
                price_str = match.group(1)

                try:
                    # Deutsche Dezimal-Notation → Float
                    if "," in price_str and "." in price_str:
                        # Format: 1.234,56
                        price_str = price_str.replace(".", "").replace(",", ".")
                    elif "," in price_str:
                        # Format: 123,45
                        price_str = price_str.replace(",", ".")

                    price = float(price_str)

                    preise.append(
                        {
                            "original_text": match.group(0),
                            "value": price,
                            "currency": "EUR",
                            "start": match.start(),
                            "end": match.end(),
                        }
                    )

                except ValueError:
                    logger.warning("⚠️ Konnte Preis nicht parsen: %s", match.group(0))

        return preise

    def _normalize_unit(self, unit: str) -> str:
        """
        🔄 Normalisiert Einheiten zu Standardformen

        Args:
            unit: Original-Einheit

        Returns:
            str: Normalisierte Einheit
        """
        unit_mapping = {
            # Stück-Varianten
            "St.": "Stück",
            "St": "Stück",
            "Stk.": "Stück",
            "Stk": "Stück",
            "x": "Stück",
            "mal": "Stück",
            "×": "Stück",
            # Längen-Varianten
            "m": "Meter",
            "mm": "Millimeter",
            "cm": "Zentimeter",
            # Gewicht-Varianten
            "kg": "Kilogramm",
            "g": "Gramm",
            # Volumen-Varianten
            "l": "Liter",
            "ml": "Milliliter",
            # Flächen-Varianten
            "m²": "Quadratmeter",
            "qm": "Quadratmeter",
            # Verpackung-Varianten
            "Pck.": "Packung",
            "Pck": "Packung",
            "Pack": "Packung",
        }

        return unit_mapping.get(unit, unit)

    def get_hersteller_context(self, text: str) -> dict[str, Any]:
        """
        🏭 Extrahiert Hersteller-Kontext für SKR03-Klassifizierung

        Args:
            text: Text zur Analyse

        Returns:
            dict: Hersteller-Kontext mit Kategorisierung
        """
        result = self.extract_entities(text)

        # Hersteller-Kategorisierung für SKR03
        hersteller_kategorien = {
            "installation": [
                "GIRA",
                "Hager",
                "Busch-Jaeger",
                "Merten",
                "Jung",
                "Berker",
            ],
            "kabel_verbindung": ["Wago", "Phoenix Contact", "Lapp Kabel", "Hensel"],
            "beleuchtung": ["Osram", "Philips", "Ledvance", "Paulmann", "SLV"],
            "messtechnik": ["Fluke", "Benning", "Gossen Metrawatt"],
            "werkzeug": ["Knipex", "Wiha", "Wera", "Klauke"],
        }

        erkannte_kategorien = []
        for hersteller in result.hersteller:
            for kategorie, hersteller_liste in hersteller_kategorien.items():
                if hersteller.upper() in [h.upper() for h in hersteller_liste]:
                    erkannte_kategorien.append(kategorie)

        return {
            "hersteller": result.hersteller,
            "kategorien": list(set(erkannte_kategorien)),
            "artikel_count": len(result.artikel_nummern),
            "durchschnittspreis": (
                sum(p["value"] for p in result.preise) / len(result.preise)
                if result.preise
                else 0.0
            ),
            "empfohlene_skr03_kategorien": self._get_skr03_empfehlungen(
                erkannte_kategorien
            ),
        }

    def _get_skr03_empfehlungen(self, kategorien: list[str]) -> list[str]:
        """SKR03-Empfehlungen basierend auf Hersteller-Kategorien."""
        kategorie_mapping = {
            "installation": ["3400"],  # Elektro-Installation
            "kabel_verbindung": ["3410"],  # Kabel und Verbindungen
            "beleuchtung": ["3420"],  # Beleuchtung
            "messtechnik": ["4985"],  # Mess- und Prüfgeräte
            "werkzeug": ["4985"],  # Werkzeuge
        }

        empfehlungen = []
        for kategorie in kategorien:
            empfehlungen.extend(kategorie_mapping.get(kategorie, []))

        return list(set(empfehlungen)) if empfehlungen else ["3400"]  # Default: Elektro


# Convenience Functions
def create_elektro_ner(config: Config | None = None) -> DeutscheElektrotechnikNER:
    """Factory-Function für DeutscheElektrotechnikNER."""
    return DeutscheElektrotechnikNER(config)


def analyze_elektro_text(text: str) -> ElektroNERResult:
    """
    Schnelle Elektrotechnik-Analyse ohne Instanz-Erstellung.

    Args:
        text: Zu analysierender Text

    Returns:
        ElektroNERResult: Analyseergebnis
    """
    ner = create_elektro_ner()
    return ner.extract_entities(text)


def get_hersteller_from_text(text: str) -> list[str]:
    """
    Extrahiert nur Hersteller aus Text.

    Args:
        text: Text zur Analyse

    Returns:
        list[str]: Erkannte Hersteller
    """
    result = analyze_elektro_text(text)
    return result.hersteller
