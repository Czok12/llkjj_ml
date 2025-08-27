"""
Vereinfachte Elektrotechnik NER-Implementierung mit vollständiger Type-Safety
Spezialisiert auf deutschen Elektrohandel mit mypy --strict Compliance
"""

import logging
import re
from dataclasses import dataclass
from re import Pattern
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ElektroEntität:
    """Type-safe Elektro-Entität für Named Entity Recognition"""

    text: str
    label: str
    start: int
    end: int
    confidence: float
    zusatz_info: dict[str, Any]


@dataclass
class ElektroNERResult:
    """Type-safe Ergebnis der Elektro-NER Analyse"""

    original_text: str
    entities: list[ElektroEntität]
    hersteller: list[str]
    artikel_nummern: list[str]
    mengen: list[dict[str, Any]]
    preise: list[dict[str, Any]]
    elektro_begriffe: list[str]
    confidence_score: float
    verarbeitungszeit: float
    metadaten: dict[str, Any]


class VereinfachteElektrotechnikNER:
    """
    Vereinfachte Elektrotechnik Named Entity Recognition mit Type-Safety
    Optimiert für deutschen Elektrohandel und mypy --strict Compliance
    """

    def __init__(self) -> None:
        """Initialisierung mit kompilierten Regex-Patterns"""
        self.patterns = self._compile_patterns()
        self.hersteller_mapping = self._build_hersteller_mapping()

    def _compile_patterns(self) -> dict[str, Pattern[str]]:
        """Kompiliert alle Regex-Patterns mit Type-Safety"""
        patterns: dict[str, Pattern[str]] = {}

        # Hersteller Pattern (einfach)
        hersteller_pattern = (
            r"\b(SIEMENS|ABB|SCHNEIDER|LEGRAND|GIRA|JUNG|WAGO|PHOENIX|EATON|RITTAL)\b"
        )
        patterns["hersteller"] = re.compile(hersteller_pattern, re.IGNORECASE)

        # Artikel-Nummern (kombiniert)
        artikel_pattern = (
            r"\b(?:[A-Z]{2,6}[-\s]?\d{4,8}|[A-Z]\d{6,10}|\d{6,12}[A-Z]{0,3})\b"
        )
        patterns["artikel_nummern"] = re.compile(artikel_pattern)

        # Elektro-Begriffe (kombiniert)
        elektro_begriffe = [
            "Kabel",
            "Leitung",
            "Schalter",
            "Steckdose",
            "Sicherung",
            "Automat",
            "Verteiler",
            "Schrank",
            "Klemme",
            "Kontakt",
            "Relais",
            "Schütz",
            "Motor",
            "Trafo",
            "Netzteil",
            "LED",
            "Lampe",
            "Leuchte",
        ]
        elektro_pattern = r"\b(" + "|".join(elektro_begriffe) + r")\b"
        patterns["elektro_begriffe"] = re.compile(elektro_pattern, re.IGNORECASE)

        # Mengen-Pattern
        mengen_pattern = (
            r"(\d+(?:[.,]\d+)?)\s*(St(?:ück)?|m|Meter|km|mm|cm|kg|g|Liter?|l)\b"
        )
        patterns["mengen"] = re.compile(mengen_pattern, re.IGNORECASE)

        # Preis-Pattern
        preis_pattern = r"(\d+(?:[.,]\d{2})?)[\s]*(?:€|EUR|Euro)\b"
        patterns["preise"] = re.compile(preis_pattern, re.IGNORECASE)

        return patterns

    def _build_hersteller_mapping(self) -> dict[str, dict[str, Any]]:
        """Erstellt Hersteller-Metadaten Mapping"""
        return {
            "SIEMENS": {"land": "Deutschland", "kategorie": "Automation"},
            "ABB": {"land": "Schweiz", "kategorie": "Energietechnik"},
            "SCHNEIDER": {"land": "Frankreich", "kategorie": "Automation"},
            "LEGRAND": {"land": "Frankreich", "kategorie": "Elektroinstallation"},
            "GIRA": {"land": "Deutschland", "kategorie": "Gebäudetechnik"},
            "JUNG": {"land": "Deutschland", "kategorie": "Schalter"},
            "WAGO": {"land": "Deutschland", "kategorie": "Klemmen"},
            "PHOENIX": {"land": "Deutschland", "kategorie": "Verbindungstechnik"},
            "EATON": {"land": "USA", "kategorie": "Energiemanagement"},
            "RITTAL": {"land": "Deutschland", "kategorie": "Schaltschränke"},
        }

    def analyze_text(self, text: str) -> ElektroNERResult:
        """
        Analysiert Text und extrahiert Elektro-Entitäten

        Args:
            text: Zu analysierender Text

        Returns:
            ElektroNERResult mit allen extrahierten Entitäten
        """
        import time

        start_time = time.time()

        entities: list[ElektroEntität] = []
        hersteller: list[str] = []
        artikel_nummern: list[str] = []
        mengen: list[dict[str, Any]] = []
        preise: list[dict[str, Any]] = []
        elektro_begriffe: list[str] = []

        # Hersteller extrahieren
        for match in self.patterns["hersteller"].finditer(text):
            hersteller_name = match.group(1).upper()
            hersteller.append(hersteller_name)

            zusatz_info = self.hersteller_mapping.get(hersteller_name, {})

            entities.append(
                ElektroEntität(
                    text=match.group(1),
                    label="HERSTELLER",
                    start=match.start(),
                    end=match.end(),
                    confidence=0.95,
                    zusatz_info=zusatz_info,
                )
            )

        # Artikel-Nummern extrahieren
        for match in self.patterns["artikel_nummern"].finditer(text):
            artikel_nr = match.group(0)
            artikel_nummern.append(artikel_nr)

            entities.append(
                ElektroEntität(
                    text=artikel_nr,
                    label="ARTIKEL_NUMMER",
                    start=match.start(),
                    end=match.end(),
                    confidence=0.85,
                    zusatz_info={"pattern_type": "elektro_artikelnummer"},
                )
            )

        # Elektro-Begriffe extrahieren
        for match in self.patterns["elektro_begriffe"].finditer(text):
            begriff = match.group(0)
            elektro_begriffe.append(begriff)

            entities.append(
                ElektroEntität(
                    text=begriff,
                    label="ELEKTRO_BEGRIFF",
                    start=match.start(),
                    end=match.end(),
                    confidence=0.80,
                    zusatz_info={"kategorie": "elektro_terminologie"},
                )
            )

        # Mengen extrahieren
        for match in self.patterns["mengen"].finditer(text):
            value_str = match.group(1).replace(",", ".")
            unit = match.group(2)

            try:
                value = float(value_str)
                normalized_unit = self._normalize_unit(unit)

                menge_data: dict[str, Any] = {
                    "wert": value,
                    "einheit": unit,
                    "normalisierte_einheit": normalized_unit,
                    "text": match.group(0),
                }
                mengen.append(menge_data)

                entities.append(
                    ElektroEntität(
                        text=match.group(0),
                        label="MENGE",
                        start=match.start(),
                        end=match.end(),
                        confidence=0.90,
                        zusatz_info=menge_data,
                    )
                )
            except ValueError:
                logger.warning("Konnte Menge nicht parsen: %s", value_str)

        # Preise extrahieren
        for match in self.patterns["preise"].finditer(text):
            preis_str = match.group(1).replace(",", ".")

            try:
                preis_value = float(preis_str)

                preis_data: dict[str, Any] = {
                    "betrag": preis_value,
                    "währung": "EUR",
                    "text": match.group(0),
                }
                preise.append(preis_data)

                entities.append(
                    ElektroEntität(
                        text=match.group(0),
                        label="PREIS",
                        start=match.start(),
                        end=match.end(),
                        confidence=0.88,
                        zusatz_info=preis_data,
                    )
                )
            except ValueError:
                logger.warning("Konnte Preis nicht parsen: %s", preis_str)

        verarbeitungszeit = time.time() - start_time
        confidence_score = self._calculate_confidence(entities)

        return ElektroNERResult(
            original_text=text,
            entities=entities,
            hersteller=hersteller,
            artikel_nummern=artikel_nummern,
            mengen=mengen,
            preise=preise,
            elektro_begriffe=elektro_begriffe,
            confidence_score=confidence_score,
            verarbeitungszeit=verarbeitungszeit,
            metadaten={
                "pattern_count": len(self.patterns),
                "entity_count": len(entities),
                "text_length": len(text),
            },
        )

    def _normalize_unit(self, unit: str) -> str:
        """Normalisiert Einheiten auf Standardform"""
        unit_mapping = {
            "st": "Stück",
            "stück": "Stück",
            "stk": "Stück",
            "m": "Meter",
            "meter": "Meter",
            "mm": "Millimeter",
            "cm": "Zentimeter",
            "km": "Kilometer",
            "kg": "Kilogramm",
            "g": "Gramm",
            "l": "Liter",
            "liter": "Liter",
        }
        return unit_mapping.get(unit.lower(), unit)

    def _calculate_confidence(self, entities: list[ElektroEntität]) -> float:
        """Berechnet Gesamt-Confidence Score"""
        if not entities:
            return 0.0

        total_confidence = sum(entity.confidence for entity in entities)
        return total_confidence / len(entities)

    def get_statistics(self, results: list[ElektroNERResult]) -> dict[str, Any]:
        """Erstellt Statistiken über mehrere Analyse-Ergebnisse"""
        if not results:
            return {"error": "Keine Ergebnisse verfügbar"}

        entity_counts: dict[str, int] = {}
        total_entities = 0
        total_confidence = 0.0
        total_time = 0.0

        for result in results:
            total_entities += len(result.entities)
            total_confidence += result.confidence_score
            total_time += result.verarbeitungszeit

            for entity in result.entities:
                entity_counts[entity.label] = entity_counts.get(entity.label, 0) + 1

        avg_confidence = total_confidence / len(results) if results else 0.0
        avg_time = total_time / len(results) if results else 0.0

        return {
            "total_documents": len(results),
            "total_entities": total_entities,
            "avg_entities_per_doc": total_entities / len(results),
            "entity_type_distribution": entity_counts,
            "avg_confidence": avg_confidence,
            "avg_processing_time": avg_time,
            "performance_metrics": {
                "entities_per_second": (
                    total_entities / total_time if total_time > 0 else 0
                )
            },
        }
