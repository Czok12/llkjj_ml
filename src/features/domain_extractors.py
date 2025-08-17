#!/usr/bin/env python3
"""
LLKJJ ML Pipeline - Domain-Specific Feature Extractors
=====================================================

Elektro-spezifische und geschäftliche Feature-Extraktion für deutsche
Handwerksfirmen. Implementiert domänen-spezifisches Wissen und Muster.

Author: LLKJJ ML Pipeline
Version: 1.0.0
"""

import logging
import re
from datetime import date, datetime
from typing import Any

from src.features import FeatureExtractionResult, FeatureExtractor, FeatureMetadata

logger = logging.getLogger(__name__)


class ElectricalContractorExtractor(FeatureExtractor):
    """
    Elektrohandwerk-spezifische Feature-Extraktion.

    Features:
    - Lieferanten-Klassifizierung
    - Produkt-Kategorisierung
    - Preisanalyse
    - Steuer-Muster
    """

    def __init__(self, name: str = "electrical_features", enabled: bool = True) -> None:
        super().__init__(name, enabled)

        # Elektro-Großhändler und ihre Charakteristika
        self.major_suppliers = {
            "sonepar": {
                "type": "großhandel",
                "specialization": "elektromaterial",
                "typical_accounts": ["5400", "5410"],
                "keywords": ["sonepar", "belectric", "hagemeyer"],
            },
            "rexel": {
                "type": "großhandel",
                "specialization": "elektromaterial",
                "typical_accounts": ["5400", "5410"],
                "keywords": ["rexel", "eltefa", "voltimum"],
            },
            "elektro_einzelhandel": {
                "type": "einzelhandel",
                "specialization": "kleinteile",
                "typical_accounts": ["5400", "6600"],
                "keywords": ["conrad", "reichelt", "elektro", "fachhandel"],
            },
            "hersteller": {
                "type": "hersteller",
                "specialization": "direktbezug",
                "typical_accounts": ["5400"],
                "keywords": ["siemens", "abb", "schneider", "wago", "phoenix"],
            },
        }

        # Produktkategorien mit SKR03-Zuordnung
        self.product_categories = {
            "installationsmaterial": {
                "keywords": [
                    "kabel",
                    "leitung",
                    "draht",
                    "rohr",
                    "kanal",
                    "dose",
                    "schalter",
                    "steckdose",
                ],
                "skr03_accounts": ["5400", "5410"],
                "typical_vat": 19,
                "price_range": (0.1, 1000.0),
            },
            "beleuchtung": {
                "keywords": [
                    "lampe",
                    "led",
                    "halogen",
                    "leuchte",
                    "birne",
                    "spot",
                    "strahler",
                ],
                "skr03_accounts": ["5400", "5420"],
                "typical_vat": 19,
                "price_range": (5.0, 500.0),
            },
            "schaltgeraete": {
                "keywords": [
                    "schütz",
                    "relais",
                    "sicherung",
                    "automat",
                    "fi",
                    "rcd",
                    "leitungsschutz",
                ],
                "skr03_accounts": ["5400"],
                "typical_vat": 19,
                "price_range": (10.0, 2000.0),
            },
            "messgeraete": {
                "keywords": [
                    "messgerät",
                    "multimeter",
                    "spannungsprüfer",
                    "duspol",
                    "benning",
                ],
                "skr03_accounts": ["6600", "0840"],  # GWG oder Anlagevermögen
                "typical_vat": 19,
                "price_range": (20.0, 5000.0),
            },
            "werkzeug": {
                "keywords": [
                    "zange",
                    "schraubendreher",
                    "bohrer",
                    "säge",
                    "werkzeug",
                    "knipex",
                ],
                "skr03_accounts": ["6600"],  # Geringwertige Wirtschaftsgüter
                "typical_vat": 19,
                "price_range": (5.0, 800.0),
            },
            "dienstleistung": {
                "keywords": [
                    "montage",
                    "installation",
                    "reparatur",
                    "wartung",
                    "prüfung",
                    "stunden",
                ],
                "skr03_accounts": ["4100", "4110", "4120"],  # Umsatzerlöse
                "typical_vat": 19,
                "price_range": (50.0, 10000.0),
            },
        }

        # Deutsche Steuerstrukturen
        self.vat_patterns = {
            19: ["19%", "19,00%", "19.00%", "neunzehn prozent"],
            7: ["7%", "7,00%", "7.00%", "sieben prozent"],
            0: ["0%", "0,00%", "0.00%", "steuerfrei", "umsatzsteuerfrei"],
        }

    def extract_features(
        self, invoice_data: dict[str, Any], **kwargs: Any
    ) -> FeatureExtractionResult:
        """Extrahiere Elektrohandwerk-Features"""

        features: dict[str, float | int | str | bool | list[Any]] = {}

        # Lieferanten-Analyse
        features.update(self._extract_supplier_features(invoice_data))

        # Produkt-Analyse
        features.update(self._extract_product_features(invoice_data))

        # Preisstruktur-Analyse
        features.update(self._extract_pricing_features(invoice_data))

        # Steuer-Analyse
        features.update(self._extract_tax_features(invoice_data))

        # Geschäfts-Kontext
        features.update(self._extract_business_context(invoice_data))

        return FeatureExtractionResult(
            features=features,
            metadata=FeatureMetadata(
                name=self.name,
                extractor_type=self.__class__.__name__,
                success=True,
                feature_count=len(features),
            ),
        )

    def _extract_supplier_features(
        self, invoice_data: dict[str, Any]
    ) -> dict[str, float | int | str | bool | list[Any]]:
        """Lieferanten-spezifische Features"""
        features: dict[str, float | int | str | bool | list[Any]] = {}

        # Sammle Lieferanten-Informationen
        supplier_text = ""
        for field in ["lieferant", "vendor", "supplier", "company"]:
            if field in invoice_data and invoice_data[field]:
                supplier_text += str(invoice_data[field]).lower() + " "

        # Haupt-Lieferanten-Klassifizierung
        detected_supplier = None
        for supplier_type, info in self.major_suppliers.items():
            for keyword in info["keywords"]:
                if keyword in supplier_text:
                    detected_supplier = supplier_type
                    break
            if detected_supplier:
                break

        # Supplier Features
        features["supplier_type"] = detected_supplier or "unknown"
        features["is_major_elektro_supplier"] = float(detected_supplier is not None)

        is_grosshandel = (
            detected_supplier is not None
            and self.major_suppliers[detected_supplier]["type"] == "großhandel"
        )
        features["is_grosshandel"] = float(is_grosshandel)

        is_hersteller = (
            detected_supplier is not None
            and self.major_suppliers[detected_supplier]["type"] == "hersteller"
        )
        features["is_hersteller"] = float(is_hersteller)

        # Lieferanten-Vertrauen (basierend auf Bekanntheit)
        if detected_supplier is not None:
            features["supplier_trust_score"] = 1.0
            features["supplier_specialization"] = str(
                self.major_suppliers[detected_supplier]["specialization"]
            )
        else:
            features["supplier_trust_score"] = 0.5  # Unbekannter Lieferant
            features["supplier_specialization"] = "unknown"

        # Rechnungsfrequenz-Indikatoren
        features["supplier_name_length"] = float(len(supplier_text.strip()))
        features["has_gmbh_suffix"] = float("gmbh" in supplier_text)
        features["has_elektro_in_name"] = float("elektro" in supplier_text)

        return features

    def _extract_product_features(
        self, invoice_data: dict[str, Any]
    ) -> dict[str, float | int | str | bool | list[Any]]:
        """Produkt-Kategorisierung und -Analyse"""
        features: dict[str, float | int | str | bool | list[Any]] = {}

        # Sammle Produktbeschreibungen
        product_text = ""
        line_items = invoice_data.get("line_items", [])

        for item in line_items:
            if isinstance(item, dict):
                for field in ["beschreibung", "description", "artikel", "product"]:
                    if field in item and item[field]:
                        product_text += str(item[field]).lower() + " "

        # Zusätzlich aus Haupttext
        raw_text = invoice_data.get("raw_text", "").lower()
        product_text += raw_text

        # Kategorien-Erkennung
        category_scores = {}
        dominant_category = None
        max_score = 0

        for category, info in self.product_categories.items():
            keywords_list = info.get("keywords", [])
            if isinstance(keywords_list, list):
                # Explizite Type-Safe Lösung für sum() mit Generator
                match_count = 0
                for keyword in keywords_list:
                    if keyword in product_text:
                        match_count += 1
                score = match_count
            else:
                score = 0
            category_scores[category] = score
            features[f"category_{category}_score"] = float(score)

            if score > max_score:
                max_score = score
                dominant_category = category

        features["dominant_product_category"] = dominant_category or "unknown"
        features["category_diversity"] = float(
            len([s for s in category_scores.values() if s > 0])
        )
        features["max_category_score"] = float(max_score)

        # Spezifische Produkt-Features
        features["has_cable_products"] = float(
            "kabel" in product_text or "leitung" in product_text
        )
        features["has_lighting_products"] = float(
            "led" in product_text or "lampe" in product_text
        )
        features["has_safety_products"] = float(
            "fi" in product_text or "rcd" in product_text
        )
        features["has_measurement_tools"] = float(
            "messgerät" in product_text or "multimeter" in product_text
        )

        # Item-Anzahl-Analyse
        features["item_count"] = float(len(line_items))

        if line_items:
            # Durchschnittliche Beschreibungslänge
            description_lengths = []
            for item in line_items:
                if isinstance(item, dict):
                    desc = item.get("beschreibung", item.get("description", ""))
                    description_lengths.append(len(str(desc)))

            if description_lengths:
                features["avg_item_description_length"] = float(
                    sum(description_lengths) / len(description_lengths)
                )
            else:
                features["avg_item_description_length"] = 0.0
        else:
            features["avg_item_description_length"] = 0.0

        return features

    def _extract_pricing_features(
        self, invoice_data: dict[str, Any]
    ) -> dict[str, float | int | str | bool | list[Any]]:
        """Preisstruktur und -muster"""
        features: dict[str, float | int | str | bool | list[Any]] = {}

        # Gesamtbetrag-Analyse
        total_amount = 0.0
        for field in ["gesamt_betrag", "total_amount", "betrag", "summe"]:
            if field in invoice_data and invoice_data[field]:
                try:
                    total_amount = float(invoice_data[field])
                    break
                except (ValueError, TypeError):
                    continue

        features["total_amount"] = total_amount

        # Betragskategorisierung für Elektrohandwerk
        if total_amount == 0:
            features["amount_category"] = "unknown"
        elif total_amount < 50:
            features["amount_category"] = "kleinteile"
        elif total_amount < 500:
            features["amount_category"] = "normal"
        elif total_amount < 2000:
            features["amount_category"] = "grossprojekt"
        else:
            features["amount_category"] = "industrie"

        features["is_small_order"] = float(total_amount < 100)
        features["is_large_order"] = float(total_amount > 1000)
        features["is_industrial_order"] = float(total_amount > 5000)

        # Line-Item-Preisanalyse
        line_items = invoice_data.get("line_items", [])
        if line_items:
            item_prices = []
            for item in line_items:
                if isinstance(item, dict):
                    for price_field in ["einzelpreis", "preis", "price", "unit_price"]:
                        if price_field in item:
                            try:
                                price = float(item[price_field])
                                item_prices.append(price)
                                break
                            except (ValueError, TypeError):
                                continue

            if item_prices:
                avg_price = sum(item_prices) / len(item_prices)
                max_price = max(item_prices)
                min_price = min(item_prices)

                features["avg_item_price"] = float(avg_price)
                features["max_item_price"] = float(max_price)
                features["min_item_price"] = float(min_price)
                features["price_range"] = float(max_price - min_price)

                # Preisverteilung
                cheap_items = [p for p in item_prices if p < 10]
                expensive_items = [p for p in item_prices if p > 100]

                features["cheap_item_ratio"] = float(
                    len(cheap_items) / len(item_prices)
                )
                features["expensive_item_ratio"] = float(
                    len(expensive_items) / len(item_prices)
                )

            else:
                features.update(
                    {
                        "avg_item_price": 0.0,
                        "max_item_price": 0.0,
                        "min_item_price": 0.0,
                        "price_range": 0.0,
                        "cheap_item_ratio": 0.0,
                        "expensive_item_ratio": 0.0,
                    }
                )
        else:
            features.update(
                {
                    "avg_item_price": 0.0,
                    "max_item_price": 0.0,
                    "min_item_price": 0.0,
                    "price_range": 0.0,
                    "cheap_item_ratio": 0.0,
                    "expensive_item_ratio": 0.0,
                }
            )

        return features

    def _extract_tax_features(
        self, invoice_data: dict[str, Any]
    ) -> dict[str, float | int | str | bool | list[Any]]:
        """Steuer-spezifische Features"""
        features: dict[str, float | int | str | bool | list[Any]] = {}

        raw_text = invoice_data.get("raw_text", "").lower()

        # VAT-Satz-Erkennung
        detected_vat_rates = []
        for vat_rate, patterns in self.vat_patterns.items():
            for pattern in patterns:
                if pattern in raw_text:
                    detected_vat_rates.append(vat_rate)
                    break

        # VAT Features
        features["has_19_percent_vat"] = float(19 in detected_vat_rates)
        features["has_7_percent_vat"] = float(7 in detected_vat_rates)
        features["has_zero_vat"] = float(0 in detected_vat_rates)
        features["vat_complexity"] = float(len(set(detected_vat_rates)))

        # Typische Elektrohandwerk-Steuerstruktur
        if 19 in detected_vat_rates and len(detected_vat_rates) == 1:
            features["standard_vat_structure"] = 1.0
        elif 19 in detected_vat_rates and 7 in detected_vat_rates:
            features["mixed_vat_structure"] = 1.0
        else:
            features["unusual_vat_structure"] = 1.0

        # Steuerbeträge-Analyse
        vat_amount = 0.0
        for field in ["mwst_betrag", "vat_amount", "steuer", "tax"]:
            if field in invoice_data and invoice_data[field]:
                try:
                    vat_amount = float(invoice_data[field])
                    break
                except (ValueError, TypeError):
                    continue

        features["vat_amount"] = vat_amount

        # Netto/Brutto-Verhältnis
        total_amount = features.get("total_amount", 0.0)
        if (
            isinstance(total_amount, int | float)
            and total_amount > 0
            and vat_amount > 0
        ):
            vat_ratio = vat_amount / total_amount
            features["vat_ratio"] = vat_ratio

            # Plausibilitätsprüfung
            if 0.15 <= vat_ratio <= 0.20:  # ~19% VAT
                features["plausible_vat_ratio"] = 1.0
            elif 0.06 <= vat_ratio <= 0.08:  # ~7% VAT
                features["plausible_vat_ratio"] = 1.0
            else:
                features["plausible_vat_ratio"] = 0.0
        else:
            features["vat_ratio"] = 0.0
            features["plausible_vat_ratio"] = 0.0

        return features

    def _extract_business_context(
        self, invoice_data: dict[str, Any]
    ) -> dict[str, float | int | str | bool | list[Any]]:
        """Geschäftlicher Kontext"""
        features: dict[str, float | int | str | bool | list[Any]] = {}

        raw_text = invoice_data.get("raw_text", "").lower()

        # Rechnungstyp-Erkennung
        is_material = float(
            any(
                keyword in raw_text
                for keyword in ["material", "artikel", "ware", "lieferung"]
            )
        )
        is_service = float(
            any(
                keyword in raw_text
                for keyword in ["arbeitszeit", "montage", "installation", "service"]
            )
        )

        features["is_material_invoice"] = is_material
        features["is_service_invoice"] = is_service
        features["is_mixed_invoice"] = float(bool(is_material) and bool(is_service))

        # Projektgröße-Indikatoren
        features["mentions_project"] = float("projekt" in raw_text)
        features["mentions_baustelle"] = float("baustelle" in raw_text)
        features["mentions_neubau"] = float("neubau" in raw_text)
        features["mentions_sanierung"] = float("sanierung" in raw_text)

        # Dringlichkeits-Indikatoren
        features["mentions_eilig"] = float(
            any(
                keyword in raw_text
                for keyword in ["eilig", "dringend", "sofort", "notfall"]
            )
        )
        features["mentions_regulaer"] = float(
            any(keyword in raw_text for keyword in ["regulär", "standard", "normal"])
        )

        # Zahlungskonditionen
        features["mentions_skonto"] = float("skonto" in raw_text)
        features["mentions_vorkasse"] = float("vorkasse" in raw_text)
        features["mentions_nachnahme"] = float("nachnahme" in raw_text)

        # Qualitäts-/Compliance-Indikatoren
        features["mentions_ce"] = float("ce" in raw_text)
        features["mentions_din"] = float("din" in raw_text)
        features["mentions_vde"] = float("vde" in raw_text)
        features["mentions_tuv"] = float("tüv" in raw_text or "tuv" in raw_text)

        return features

    def get_feature_names(self) -> list[str]:
        """Gebe alle Elektrohandwerk Feature-Namen zurück"""
        base_features = [
            # Supplier features
            "supplier_type",
            "is_major_elektro_supplier",
            "is_grosshandel",
            "is_hersteller",
            "supplier_trust_score",
            "supplier_specialization",
            "supplier_name_length",
            "has_gmbh_suffix",
            "has_elektro_in_name",
            # Product features
            "dominant_product_category",
            "category_diversity",
            "max_category_score",
            "has_cable_products",
            "has_lighting_products",
            "has_safety_products",
            "has_measurement_tools",
            "item_count",
            "avg_item_description_length",
            # Pricing features
            "total_amount",
            "amount_category",
            "is_small_order",
            "is_large_order",
            "is_industrial_order",
            "avg_item_price",
            "max_item_price",
            "min_item_price",
            "price_range",
            "cheap_item_ratio",
            "expensive_item_ratio",
            # Tax features
            "has_19_percent_vat",
            "has_7_percent_vat",
            "has_zero_vat",
            "vat_complexity",
            "standard_vat_structure",
            "mixed_vat_structure",
            "unusual_vat_structure",
            "vat_amount",
            "vat_ratio",
            "plausible_vat_ratio",
            # Business context
            "is_material_invoice",
            "is_service_invoice",
            "is_mixed_invoice",
            "mentions_project",
            "mentions_baustelle",
            "mentions_neubau",
            "mentions_sanierung",
            "mentions_eilig",
            "mentions_regulaer",
            "mentions_skonto",
            "mentions_vorkasse",
            "mentions_nachnahme",
            "mentions_ce",
            "mentions_din",
            "mentions_vde",
            "mentions_tuv",
        ]

        # Produkt-Kategorien
        for category in self.product_categories.keys():
            base_features.append(f"category_{category}_score")

        return base_features


class TemporalFeatureExtractor(FeatureExtractor):
    """
    Zeitliche und saisonale Feature-Extraktion.

    Features:
    - Saisonalitäts-Muster
    - Wochentag/Monat-Effekte
    - Zeitabstände
    - Fälligkeiten
    """

    def __init__(self, name: str = "temporal_features", enabled: bool = True) -> None:
        super().__init__(name, enabled)

    def extract_features(
        self, invoice_data: dict[str, Any], **kwargs: Any
    ) -> FeatureExtractionResult:
        """Extrahiere zeitliche Features"""

        features: dict[str, float | int | str | bool | list[Any]] = {}

        # Datums-Extraktion
        invoice_date = self._extract_invoice_date(invoice_data)

        if invoice_date:
            # Basis-Zeitfeatures
            features.update(self._extract_date_features(invoice_date))

            # Saisonale Features
            features.update(self._extract_seasonal_features(invoice_date))

            # Geschäfts-Zeitraum Features
            features.update(self._extract_business_period_features(invoice_date))

        else:
            # Default-Werte wenn kein Datum verfügbar
            features.update(self._get_default_temporal_features())

        # Zeitabstände und Fristen
        features.update(self._extract_deadline_features(invoice_data))

        return FeatureExtractionResult(
            features=features,
            metadata=FeatureMetadata(
                name=self.name,
                extractor_type=self.__class__.__name__,
                success=True,
                feature_count=len(features),
            ),
        )

    def _extract_invoice_date(self, invoice_data: dict[str, Any]) -> datetime | None:
        """Extrahiere Rechnungsdatum"""

        # Verschiedene Datumsfelder probieren
        date_fields = ["rechnungs_datum", "invoice_date", "datum", "date"]

        for field in date_fields:
            if field in invoice_data and invoice_data[field]:
                date_value = invoice_data[field]

                # Verschiedene Datumsformate handhaben
                if isinstance(date_value, datetime):
                    return date_value
                elif isinstance(date_value, date):
                    return datetime.combine(date_value, datetime.min.time())
                elif isinstance(date_value, str):
                    # Deutsche Datumsformate
                    date_patterns = [
                        "%d.%m.%Y",
                        "%d/%m/%Y",
                        "%Y-%m-%d",
                        "%d.%m.%y",
                        "%d/%m/%y",
                    ]

                    for pattern in date_patterns:
                        try:
                            return datetime.strptime(date_value, pattern)
                        except ValueError:
                            continue

        # Fallback: Datum aus Text extrahieren
        raw_text = invoice_data.get("raw_text", "")
        date_pattern = r"\b(\d{1,2}[./]\d{1,2}[./]\d{2,4})\b"
        matches = re.findall(date_pattern, raw_text)

        for match in matches:
            for pattern in ["%d.%m.%Y", "%d/%m/%Y", "%d.%m.%y", "%d/%m/%y"]:
                try:
                    return datetime.strptime(match, pattern)
                except ValueError:
                    continue

        return None

    def _extract_date_features(
        self, invoice_date: datetime
    ) -> dict[str, float | int | str | bool | list[Any]]:
        """Basis-Datums-Features"""
        features: dict[str, float | int | str | bool | list[Any]] = {}

        features["invoice_year"] = float(invoice_date.year)
        features["invoice_month"] = float(invoice_date.month)
        features["invoice_day"] = float(invoice_date.day)
        features["invoice_weekday"] = float(invoice_date.weekday())  # 0=Monday
        features["invoice_week"] = float(invoice_date.isocalendar()[1])

        # Wochentag-Kategorien
        features["is_monday"] = float(invoice_date.weekday() == 0)
        features["is_friday"] = float(invoice_date.weekday() == 4)
        features["is_weekend"] = float(invoice_date.weekday() >= 5)
        features["is_weekday"] = float(invoice_date.weekday() < 5)

        # Monatsposition
        features["is_month_start"] = float(invoice_date.day <= 7)
        features["is_month_middle"] = float(8 <= invoice_date.day <= 22)
        features["is_month_end"] = float(invoice_date.day >= 23)

        return features

    def _extract_seasonal_features(
        self, invoice_date: datetime
    ) -> dict[str, float | int | str | bool | list[Any]]:
        """Saisonale Features"""
        features: dict[str, float | int | str | bool | list[Any]] = {}

        month = invoice_date.month

        # Jahreszeiten
        if month in [12, 1, 2]:
            features["season"] = "winter"
        elif month in [3, 4, 5]:
            features["season"] = "spring"
        elif month in [6, 7, 8]:
            features["season"] = "summer"
        else:
            features["season"] = "autumn"

        features["is_winter"] = float(features["season"] == "winter")
        features["is_spring"] = float(features["season"] == "spring")
        features["is_summer"] = float(features["season"] == "summer")
        features["is_autumn"] = float(features["season"] == "autumn")

        # Elektrohandwerk-spezifische Saisonalität
        # Winter: Heizung, Beleuchtung (Weihnachten)
        features["high_heating_season"] = float(month in [11, 12, 1, 2])
        features["christmas_lighting_season"] = float(month in [11, 12])

        # Frühling: Außenarbeiten, Renovierung
        features["renovation_season"] = float(month in [3, 4, 5])
        features["outdoor_work_season"] = float(month in [4, 5, 6, 7, 8, 9])

        # Sommer: Klimaanlagen, Urlaub (weniger Arbeit)
        features["vacation_season"] = float(month in [7, 8])
        features["air_conditioning_season"] = float(month in [5, 6, 7, 8, 9])

        return features

    def _extract_business_period_features(
        self, invoice_date: datetime
    ) -> dict[str, float | int | str | bool | list[Any]]:
        """Geschäftszeitraum-Features"""
        features: dict[str, float | int | str | bool | list[Any]] = {}

        # Geschäftsjahr (kann vom Kalenderjahr abweichen)
        features["calendar_year"] = float(invoice_date.year)

        # Quartal
        quarter = (invoice_date.month - 1) // 3 + 1
        features["quarter"] = float(quarter)
        features["is_q1"] = float(quarter == 1)
        features["is_q2"] = float(quarter == 2)
        features["is_q3"] = float(quarter == 3)
        features["is_q4"] = float(quarter == 4)

        # Jahresende-Effekte (Steuer, Abschreibungen)
        features["is_year_end"] = float(invoice_date.month == 12)
        features["is_year_start"] = float(invoice_date.month == 1)

        # Deutsche Feiertage (vereinfacht)
        # Implementierung von Weihnachten und Neujahr als Basis-Feiertage
        features["near_christmas"] = float(
            invoice_date.month == 12 and invoice_date.day >= 20
        )
        features["near_new_year"] = float(
            (invoice_date.month == 12 and invoice_date.day >= 28)
            or (invoice_date.month == 1 and invoice_date.day <= 7)
        )

        return features

    def _extract_deadline_features(
        self, invoice_data: dict[str, Any]
    ) -> dict[str, float | int | str | bool | list[Any]]:
        """Zahlungsfristen und Deadlines"""
        features: dict[str, float | int | str | bool | list[Any]] = {}

        # Zahlungsziel extrahieren
        payment_terms = 30  # Standard
        raw_text = invoice_data.get("raw_text", "").lower()

        # Zahlungsziel-Muster
        payment_patterns = [
            r"zahlungsziel:?\s*(\d+)\s*tag",
            r"zahlung\s+binnen\s+(\d+)\s*tag",
            r"(\d+)\s*tag[en]?\s+netto",
            r"netto\s+(\d+)",
            r"payment\s+terms:?\s*(\d+)\s*day",
        ]

        for pattern in payment_patterns:
            match = re.search(pattern, raw_text)
            if match:
                try:
                    payment_terms = int(match.group(1))
                    break
                except (ValueError, IndexError):
                    continue

        features["payment_terms_days"] = float(payment_terms)

        # Kategorisierung der Zahlungsbedingungen
        if payment_terms <= 14:
            features["payment_terms_category"] = "short"
        elif payment_terms <= 30:
            features["payment_terms_category"] = "standard"
        else:
            features["payment_terms_category"] = "long"

        features["has_short_payment_terms"] = float(payment_terms <= 14)
        features["has_standard_payment_terms"] = float(14 < payment_terms <= 30)
        features["has_long_payment_terms"] = float(payment_terms > 30)

        # Skonto-Erkennung
        skonto_days = 0
        skonto_percent = 0.0

        skonto_patterns = [
            r"(\d+)%\s*skonto\s+bei\s+zahlung\s+binnen\s+(\d+)\s*tag",
            r"skonto\s+(\d+)%\s+bei\s+(\d+)\s*tag",
            r"(\d+)\s*tag[en]?\s+(\d+)%\s*skonto",
        ]

        for pattern in skonto_patterns:
            match = re.search(pattern, raw_text)
            if match:
                try:
                    if "skonto" in pattern and "prozent" not in pattern:
                        skonto_percent = float(match.group(1))
                        skonto_days = int(match.group(2))
                    break
                except (ValueError, IndexError):
                    continue

        features["skonto_days"] = float(skonto_days)
        features["skonto_percent"] = skonto_percent
        features["has_skonto"] = float(skonto_days > 0)

        return features

    def _get_default_temporal_features(
        self,
    ) -> dict[str, float | int | str | bool | list[Any]]:
        """Default-Werte wenn kein Datum verfügbar"""
        return {
            "invoice_year": 0.0,
            "invoice_month": 0.0,
            "invoice_day": 0.0,
            "invoice_weekday": 0.0,
            "invoice_week": 0.0,
            "is_monday": False,
            "is_friday": False,
            "is_weekend": False,
            "is_weekday": False,
            "is_month_start": False,
            "is_month_middle": False,
            "is_month_end": False,
            "season": "unknown",
            "is_winter": False,
            "is_spring": False,
            "is_summer": False,
            "is_autumn": False,
            "high_heating_season": False,
            "christmas_lighting_season": False,
            "renovation_season": False,
            "outdoor_work_season": False,
            "vacation_season": False,
            "air_conditioning_season": False,
            "calendar_year": 0.0,
            "quarter": 0.0,
            "is_q1": False,
            "is_q2": False,
            "is_q3": False,
            "is_q4": False,
            "is_year_end": False,
            "is_year_start": False,
            "near_christmas": False,
            "near_new_year": False,
            "payment_terms_days": 30.0,
            "payment_terms_category": "standard",
            "has_short_payment_terms": False,
            "has_standard_payment_terms": True,
            "has_long_payment_terms": False,
            "skonto_days": 0.0,
            "skonto_percent": 0.0,
            "has_skonto": False,
        }

    def get_feature_names(self) -> list[str]:
        """Gebe alle zeitlichen Feature-Namen zurück"""
        return [
            # Date features
            "invoice_year",
            "invoice_month",
            "invoice_day",
            "invoice_weekday",
            "invoice_week",
            "is_monday",
            "is_friday",
            "is_weekend",
            "is_weekday",
            "is_month_start",
            "is_month_middle",
            "is_month_end",
            # Seasonal features
            "season",
            "is_winter",
            "is_spring",
            "is_summer",
            "is_autumn",
            "high_heating_season",
            "christmas_lighting_season",
            "renovation_season",
            "outdoor_work_season",
            "vacation_season",
            "air_conditioning_season",
            # Business period features
            "calendar_year",
            "quarter",
            "is_q1",
            "is_q2",
            "is_q3",
            "is_q4",
            "is_year_end",
            "is_year_start",
            "near_christmas",
            "near_new_year",
            # Payment terms features
            "payment_terms_days",
            "payment_terms_category",
            "has_short_payment_terms",
            "has_standard_payment_terms",
            "has_long_payment_terms",
            "skonto_days",
            "skonto_percent",
            "has_skonto",
        ]
