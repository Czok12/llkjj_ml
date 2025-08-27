#!/usr/bin/env python3
"""
LLKJJ ML Pipeline - Text Feature Extractors
==========================================

Text-basierte Feature-Extraktion für deutsche Elektrohandwerk-Rechnungen.
Implementiert NLP-Features, N-Grams, und elektro-spezifische Begriffserkennung.

Author: LLKJJ ML Pipeline
Version: 1.0.0
"""

import logging
import re
from collections import Counter
from typing import Any

from . import FeatureExtractionResult, FeatureExtractor, FeatureMetadata

logger = logging.getLogger(__name__)


class TextFeatureExtractor(FeatureExtractor):
    """
    Extrahiert text-basierte Features aus Rechnungsdaten.

    Features:
    - N-Gram-Analyse (1-3 Grams)
    - Elektro-spezifische Begriffe
    - Text-Statistiken
    - Sprachcharakteristika
    """

    def __init__(self, name: str = "text_features", enabled: bool = True) -> None:
        super().__init__(name, enabled)

        # Deutsche Elektro-Begriffe
        self.elektro_keywords = {
            "material": [
                "kabel",
                "leitung",
                "draht",
                "schalter",
                "steckdose",
                "lampe",
                "led",
                "halogen",
                "trafo",
                "transformator",
                "sicherung",
                "automat",
                "fi",
                "rcd",
                "verteiler",
                "schaltschrank",
                "installation",
                "verkabelung",
                "elektromaterial",
                "elektroartikel",
                "elektrik",
                "elektro",
            ],
            "arbeiten": [
                "installation",
                "montage",
                "reparatur",
                "wartung",
                "prüfung",
                "inbetriebnahme",
                "anschluss",
                "verkabelung",
                "verdrahtung",
                "elektroinstallation",
                "elektroarbeiten",
                "elektromontage",
            ],
            "geräte": [
                "motor",
                "pumpe",
                "ventilator",
                "heizung",
                "klimaanlage",
                "ofen",
                "backofen",
                "herd",
                "spülmaschine",
                "waschmaschine",
                "trockner",
                "durchlauferhitzer",
                "boiler",
                "warmwasserspeicher",
            ],
            "lieferanten": [
                "sonepar",
                "rexel",
                "elektro",
                "großhandel",
                "fachhandel",
                "elektrogroßhandel",
                "eltefa",
                "voltimum",
                "conrad",
                "reichelt",
            ],
            "services": [
                "stunden",
                "arbeitszeit",
                "fahrzeit",
                "anfahrt",
                "pauschale",
                "service",
                "notdienst",
                "bereitschaft",
                "überstunden",
            ],
        }

        # Flache Liste aller Begriffe
        self.all_elektro_terms = []
        for category_terms in self.elektro_keywords.values():
            self.all_elektro_terms.extend(category_terms)

        # Häufige deutsche Stopwords
        self.stopwords = {
            "der",
            "die",
            "das",
            "und",
            "oder",
            "aber",
            "mit",
            "für",
            "von",
            "zu",
            "auf",
            "in",
            "an",
            "bei",
            "nach",
            "vor",
            "über",
            "unter",
            "zwischen",
            "durch",
            "gegen",
            "ohne",
            "um",
            "bis",
            "seit",
            "während",
            "wegen",
            "trotz",
            "statt",
            "anstatt",
            "außer",
            "innerhalb",
            "außerhalb",
        }

    def extract_features(
        self, invoice_data: dict[str, Any], **kwargs: Any
    ) -> FeatureExtractionResult:
        """Extrahiere alle Text-Features"""

        features: dict[str, float | int | str | bool | list[Any]] = {}

        # Text sammeln
        text_content = self._collect_text_content(invoice_data)

        if not text_content:
            return FeatureExtractionResult(
                features={},
                metadata=FeatureMetadata(
                    name=self.name,
                    extractor_type=self.__class__.__name__,
                    success=False,
                    error_message="No text content found",
                ),
            )

        # Basic text statistics
        features.update(self._extract_basic_stats(text_content))

        # Elektro-specific features
        features.update(self._extract_elektro_features(text_content))

        # N-gram features
        features.update(self._extract_ngram_features(text_content))

        # Language characteristics
        features.update(self._extract_language_features(text_content))

        # Price and number patterns
        features.update(self._extract_numeric_patterns(text_content))

        return FeatureExtractionResult(
            features=features,
            metadata=FeatureMetadata(
                name=self.name,
                extractor_type=self.__class__.__name__,
                success=True,
                feature_count=len(features),
            ),
            raw_data={"text_content": text_content},
        )

    def _collect_text_content(self, invoice_data: dict[str, Any]) -> str:
        """Sammle allen verfügbaren Text aus Rechnungsdaten"""
        text_parts = []

        # Verschiedene Textquellen
        text_fields = [
            "raw_text",
            "text",
            "content",
            "description",
            "beschreibung",
            "lieferant",
            "vendor",
            "supplier",
            "kunde",
            "customer",
        ]

        for field in text_fields:
            if field in invoice_data and invoice_data[field]:
                text_parts.append(str(invoice_data[field]))

        # Line items
        if "line_items" in invoice_data:
            for item in invoice_data["line_items"]:
                if isinstance(item, dict):
                    if "beschreibung" in item:
                        text_parts.append(str(item["beschreibung"]))
                    if "description" in item:
                        text_parts.append(str(item["description"]))

        # Tables
        if "tables" in invoice_data:
            for table in invoice_data["tables"]:
                if isinstance(table, dict) and "content" in table:
                    text_parts.append(str(table["content"]))

        return " ".join(text_parts).lower()

    def _extract_basic_stats(
        self, text: str
    ) -> dict[str, float | int | str | bool | list[Any]]:
        """Grundlegende Textstatistiken"""
        words = text.split()

        return {
            "text_length": float(len(text)),
            "word_count": float(len(words)),
            "avg_word_length": (
                float(sum(len(word) for word in words) / len(words)) if words else 0.0
            ),
            "sentence_count": float(len(re.findall(r"[.!?]+", text))),
            "digit_ratio": (
                float(len(re.findall(r"\d", text)) / len(text)) if text else 0.0
            ),
            "uppercase_ratio": (
                float(len(re.findall(r"[A-Z]", text)) / len(text)) if text else 0.0
            ),
            "punctuation_ratio": (
                float(len(re.findall(r"[^\w\s]", text)) / len(text)) if text else 0.0
            ),
        }

    def _extract_elektro_features(
        self, text: str
    ) -> dict[str, float | int | str | bool | list[Any]]:
        """Elektro-spezifische Features"""
        features: dict[str, float | int | str | bool | list[Any]] = {}

        # Kategorie-spezifische Zählungen
        for category, keywords in self.elektro_keywords.items():
            count = sum(1 for keyword in keywords if keyword in text)
            features[f"elektro_{category}_count"] = float(count)
            features[f"elektro_{category}_density"] = (
                float(count / len(text.split())) if text.split() else 0.0
            )

        # Gesamte Elektro-Dichte
        total_elektro_terms = sum(1 for term in self.all_elektro_terms if term in text)
        features["elektro_total_count"] = float(total_elektro_terms)
        features["elektro_density"] = (
            float(total_elektro_terms / len(text.split())) if text.split() else 0.0
        )

        # Spezifische Muster
        features["has_voltage_pattern"] = float(
            bool(re.search(r"\d+\s*[vV](?:olt)?", text))
        )
        features["has_ampere_pattern"] = float(
            bool(re.search(r"\d+\s*[aA](?:mp|mpere)?", text))
        )
        features["has_watt_pattern"] = float(
            bool(re.search(r"\d+\s*[wW](?:att)?", text))
        )
        features["has_phase_pattern"] = float(
            bool(re.search(r"3\s*[pP](?:hase|hasen)", text))
        )

        # Installation-spezifische Begriffe
        installation_patterns = [
            r"installation",
            r"montage",
            r"anschluss",
            r"verkabelung",
            r"elektroinstallation",
            r"elektroarbeiten",
        ]
        features["installation_mentions"] = float(
            sum(1 for pattern in installation_patterns if re.search(pattern, text))
        )

        return features

    def _extract_ngram_features(
        self, text: str
    ) -> dict[str, float | int | str | bool | list[Any]]:
        """N-Gram Features für häufige Begriffskombinationen"""
        words = [
            word
            for word in text.split()
            if word not in self.stopwords and len(word) > 2
        ]
        features: dict[str, float | int | str | bool | list[Any]] = {}

        # Unigrams (Top-20 häufigste relevante Begriffe)
        unigram_counts = Counter(words)
        top_unigrams = unigram_counts.most_common(20)

        for i, (word, count) in enumerate(top_unigrams):
            features[f"top_unigram_{i+1}_{word}"] = float(count)

        # Bigrams (Top-10 häufigste Wortpaare)
        bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words) - 1)]
        bigram_counts = Counter(bigrams)
        top_bigrams = bigram_counts.most_common(10)

        for i, (bigram, count) in enumerate(top_bigrams):
            features[f"top_bigram_{i+1}_{bigram}"] = float(count)

        # Trigrams (Top-5 häufigste Wortgruppen)
        trigrams = [
            f"{words[i]}_{words[i+1]}_{words[i+2]}" for i in range(len(words) - 2)
        ]
        trigram_counts = Counter(trigrams)
        top_trigrams = trigram_counts.most_common(5)

        for i, (trigram, count) in enumerate(top_trigrams):
            features[f"top_trigram_{i+1}_{trigram}"] = float(count)

        return features

    def _extract_language_features(
        self, text: str
    ) -> dict[str, float | int | str | bool | list[Any]]:
        """Deutsche Sprachcharakteristika"""
        features: dict[str, float | int | str | bool | list[Any]] = {}

        # Deutsche Sonderzeichen
        umlaut_count = float(len(re.findall(r"[äöüÄÖÜß]", text)))
        features["umlaut_count"] = umlaut_count
        features["umlaut_density"] = umlaut_count / len(text) if text else 0.0

        # Zusammengesetzte Wörter (typisch für Deutsch)
        long_words = [word for word in text.split() if len(word) > 12]
        features["long_word_count"] = float(len(long_words))
        features["long_word_ratio"] = (
            float(len(long_words) / len(text.split())) if text.split() else 0.0
        )

        # Häufige deutsche Endungen
        german_endings = ["-ung", "-heit", "-keit", "-schaft", "-tum", "-nis"]
        for ending in german_endings:
            count = len(re.findall(f"\\w+{ending}\\b", text))
            features[f"ending{ending}_count"] = float(count)

        return features

    def _extract_numeric_patterns(
        self, text: str
    ) -> dict[str, float | int | str | bool | list[Any]]:
        """Numerische Muster in Rechnungen"""
        features: dict[str, float | int | str | bool | list[Any]] = {}

        # Preismuster
        price_patterns = [
            r"\d+,\d{2}\s*€",  # Deutsche Preisformate
            r"\d+\.\d{3},\d{2}\s*€",
            r"€\s*\d+,\d{2}",
            r"eur\s*\d+,\d{2}",
        ]

        total_price_matches = 0
        for pattern in price_patterns:
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            total_price_matches += matches

        features["price_pattern_count"] = float(total_price_matches)

        # Artikelnummern und Codes
        features["article_number_count"] = float(len(re.findall(r"\b\d{6,}\b", text)))
        features["product_code_count"] = float(
            len(re.findall(r"\b[A-Z]{2,}\d{2,}\b", text))
        )

        # Mengenangaben
        quantity_patterns = [
            r"\d+\s*stk",
            r"\d+\s*stück",
            r"\d+\s*m\b",
            r"\d+\s*meter",
            r"\d+\s*kg",
            r"\d+\s*g\b",
            r"\d+\s*l\b",
            r"\d+\s*liter",
        ]

        features["quantity_pattern_count"] = float(
            sum(
                len(re.findall(pattern, text, re.IGNORECASE))
                for pattern in quantity_patterns
            )
        )

        return features

    def get_feature_names(self) -> list[str]:
        """Gebe alle möglichen Feature-Namen zurück"""
        base_features = [
            # Basic stats
            "text_length",
            "word_count",
            "avg_word_length",
            "sentence_count",
            "digit_ratio",
            "uppercase_ratio",
            "punctuation_ratio",
            # Elektro features
            "elektro_total_count",
            "elektro_density",
            "has_voltage_pattern",
            "has_ampere_pattern",
            "has_watt_pattern",
            "has_phase_pattern",
            "installation_mentions",
            # Language features
            "umlaut_count",
            "umlaut_density",
            "long_word_count",
            "long_word_ratio",
            # Numeric patterns
            "price_pattern_count",
            "article_number_count",
            "product_code_count",
            "quantity_pattern_count",
        ]

        # Elektro-Kategorien
        for category in self.elektro_keywords:
            base_features.extend(
                [f"elektro_{category}_count", f"elektro_{category}_density"]
            )

        # Deutsche Endungen
        german_endings = ["-ung", "-heit", "-keit", "-schaft", "-tum", "-nis"]
        for ending in german_endings:
            base_features.append(f"ending{ending}_count")

        # N-Grams (dynamische Namen, hier Platzhalter)
        base_features.extend([f"top_unigram_{i}" for i in range(1, 21)])
        base_features.extend([f"top_bigram_{i}" for i in range(1, 11)])
        base_features.extend([f"top_trigram_{i}" for i in range(1, 6)])

        return base_features


class GermanNERExtractor(FeatureExtractor):
    """
    Named Entity Recognition für deutsche Elektro-Rechnungen.

    Erkennt spezifische Entitäten wie:
    - Firmennamen
    - Adressen
    - Produktbezeichnungen
    - Technische Spezifikationen
    """

    def __init__(self, name: str = "german_ner", enabled: bool = True) -> None:
        super().__init__(name, enabled)

        # Patterns für deutsche Entitäten
        self.entity_patterns = {
            "company": [
                r"\b[A-ZÄÖÜ][a-zäöüß]+\s+(?:GmbH|AG|e\.K\.|UG|OHG|KG)\b",
                r"\b[A-ZÄÖÜ][a-zäöüß]+\s+&\s+[A-ZÄÖÜ][a-zäöüß]+\b",
                r"\b[A-ZÄÖÜ][a-zäöüß]+\s+Elektro\b",
            ],
            "address": [
                r"\b\d{5}\s+[A-ZÄÖÜ][a-zäöüß]+\b",  # PLZ + Ort
                r"\b[A-ZÄÖÜ][a-zäöüß]+(?:straße|str\.|weg|platz|gasse)\s+\d+\b",
            ],
            "phone": [r"\b(?:\+49|0)\s*\d{1,5}\s*[-/]?\s*\d+\b"],
            "email": [r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b"],
            "technical_spec": [
                r"\b\d+\s*[VAW]\b",  # Volt/Ampere/Watt
                r"\b\d+\s*mm²?\b",  # Querschnitt
                r"\b\d+\s*m\b",  # Meter
                r"\bIP\d{2}\b",  # Schutzart
                r"\bCE\b",  # CE-Kennzeichnung
            ],
        }

    def extract_features(
        self, invoice_data: dict[str, Any], **kwargs: Any
    ) -> FeatureExtractionResult:
        """Extrahiere NER-Features"""

        features: dict[str, float | int | str | bool | list[Any]] = {}
        text_content = self._collect_text_content(invoice_data)

        if not text_content:
            return FeatureExtractionResult(
                features={},
                metadata=FeatureMetadata(
                    name=self.name,
                    extractor_type=self.__class__.__name__,
                    success=False,
                    error_message="No text content found",
                ),
            )

        entities_found = {}

        # Extrahiere Entitäten nach Typ
        for entity_type, patterns in self.entity_patterns.items():
            entities = []
            for pattern in patterns:
                matches = re.findall(pattern, text_content)
                entities.extend(matches)

            entities_found[entity_type] = entities

            # Features für jeden Entitätstyp
            features[f"{entity_type}_count"] = float(len(entities))
            features[f"{entity_type}_unique_count"] = float(len(set(entities)))

        # Spezielle Features
        features.update(
            self._extract_company_features(entities_found.get("company", []))
        )
        features.update(
            self._extract_technical_features(entities_found.get("technical_spec", []))
        )

        return FeatureExtractionResult(
            features=features,
            metadata=FeatureMetadata(
                name=self.name,
                extractor_type=self.__class__.__name__,
                success=True,
                feature_count=len(features),
            ),
            raw_data={"entities": entities_found},
        )

    def _collect_text_content(self, invoice_data: dict[str, Any]) -> str:
        """Sammle Text für NER"""
        text_parts = []

        fields = ["raw_text", "text", "content", "lieferant", "kunde"]
        for field in fields:
            if field in invoice_data and invoice_data[field]:
                text_parts.append(str(invoice_data[field]))

        return " ".join(text_parts)

    def _extract_company_features(
        self, companies: list[str]
    ) -> dict[str, float | int | str | bool | list[Any]]:
        """Firmen-spezifische Features"""
        features: dict[str, float | int | str | bool | list[Any]] = {}

        # Bekannte Elektro-Großhändler
        major_suppliers = [
            "sonepar",
            "rexel",
            "eltefa",
            "voltimum",
            "elektro",
            "conrad",
            "reichelt",
            "wago",
            "phoenix",
            "siemens",
        ]

        is_major_supplier = any(
            supplier.lower() in " ".join(companies).lower()
            for supplier in major_suppliers
        )

        features["is_major_elektro_supplier"] = float(is_major_supplier)
        features["company_name_length_avg"] = (
            float(sum(len(company) for company in companies) / len(companies))
            if companies
            else 0.0
        )

        return features

    def _extract_technical_features(
        self, tech_specs: list[str]
    ) -> dict[str, float | int | str | bool | list[Any]]:
        """Technische Spezifikations-Features"""
        features: dict[str, float | int | str | bool | list[Any]] = {}

        # Zähle verschiedene technische Angaben
        voltage_specs = [spec for spec in tech_specs if re.search(r"\d+\s*V", spec)]
        current_specs = [spec for spec in tech_specs if re.search(r"\d+\s*A", spec)]
        power_specs = [spec for spec in tech_specs if re.search(r"\d+\s*W", spec)]

        features["voltage_spec_count"] = float(len(voltage_specs))
        features["current_spec_count"] = float(len(current_specs))
        features["power_spec_count"] = float(len(power_specs))
        features["has_ce_marking"] = float(any("CE" in spec for spec in tech_specs))
        features["has_ip_rating"] = float(
            any(re.search(r"IP\d{2}", spec) for spec in tech_specs)
        )

        return features

    def get_feature_names(self) -> list[str]:
        """Gebe alle NER Feature-Namen zurück"""
        base_features = []

        # Basis-Zählungen für jeden Entitätstyp
        for entity_type in self.entity_patterns:
            base_features.extend(
                [f"{entity_type}_count", f"{entity_type}_unique_count"]
            )

        # Spezielle Features
        base_features.extend(
            [
                "is_major_elektro_supplier",
                "company_name_length_avg",
                "voltage_spec_count",
                "current_spec_count",
                "power_spec_count",
                "has_ce_marking",
                "has_ip_rating",
            ]
        )

        return base_features
