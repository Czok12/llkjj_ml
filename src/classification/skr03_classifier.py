"""
SKR03 Kontierungs-Engine für LLKJJ ML Pipeline

Automatische Zuordnung von Rechnungspositionen zu SKR03-Konten
für deutsches Elektrotechnik-Handwerk nach GoBD-Standard.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SKR03Account:
    """SKR03 Konto Definition"""

    account_number: str
    account_name: str
    account_type: str  # aktiv, passiv, aufwand, ertrag
    parent_account: str | None = None
    is_elektro_relevant: bool = False
    description: str = ""


@dataclass
class ClassificationResult:
    """Ergebnis der SKR03-Klassifikation"""

    suggested_account: str
    confidence_score: float
    reasoning: str
    alternative_accounts: list[tuple[str, float]] | None = (
        None  # [(account, confidence), ...]
    )
    gkg_classification: str | None = None  # "anlagegut" oder "betriebsausgabe"

    def __post_init__(self) -> None:
        if self.alternative_accounts is None:
            self.alternative_accounts = []


class SKR03Classifier:
    """Intelligente SKR03-Klassifikation für Elektrotechnik-Handwerk"""

    def __init__(self, config_path: Path | None = None):
        """
        Initialisiert SKR03 Classifier

        Args:
            config_path: Pfad zur SKR03-Konfiguration
        """
        self.config_path = config_path
        self.accounts = self._load_skr03_accounts()
        self.classification_rules = self._load_classification_rules()
        self.elektro_keywords = self._load_elektro_keywords()

        logger.info(f"SKR03Classifier initialisiert mit {len(self.accounts)} Konten")

    def _load_skr03_accounts(self) -> dict[str, SKR03Account]:
        """Lädt SKR03-Kontenplan für Elektrohandwerk"""

        # Elektrotechnik-spezifische SKR03-Konten
        accounts_data = {
            # ANLAGEVERMÖGEN
            "0130": SKR03Account(
                "0130",
                "Maschinen und maschinelle Anlagen",
                "aktiv",
                is_elektro_relevant=True,
                description="Elektrische Maschinen, größere Werkzeuge über 800€",
            ),
            "0490": SKR03Account(
                "0490",
                "Geringwertige Wirtschaftsgüter",
                "aktiv",
                is_elektro_relevant=True,
                description="Werkzeuge, Messgeräte 150-800€",
            ),
            "0480": SKR03Account(
                "0480",
                "Geschäfts- und Betriebsausstattung",
                "aktiv",
                is_elektro_relevant=True,
                description="Büroausstattung, Firmenfahrzeuge",
            ),
            # MATERIALKONTEN (Aufwand)
            "4830": SKR03Account(
                "4830",
                "Roh-, Hilfs- und Betriebsstoffe",
                "aufwand",
                is_elektro_relevant=True,
                description="Elektromaterial: Kabel, Schalter, Steckdosen, Sicherungen",
            ),
            "4600": SKR03Account(
                "4600",
                "Fremdleistungen",
                "aufwand",
                is_elektro_relevant=True,
                description="Subunternehmer, externe Montage, Fremdarbeiten",
            ),
            "4400": SKR03Account(
                "4400",
                "Bezugskosten",
                "aufwand",
                is_elektro_relevant=True,
                description="Transport, Verpackung, Zoll für Elektromaterial",
            ),
            # BÜRO UND VERWALTUNG
            "4935": SKR03Account(
                "4935",
                "Büromaterial",
                "aufwand",
                description="Papier, Stifte, Büroartikel",
            ),
            "4930": SKR03Account(
                "4930",
                "Sonstige betriebliche Aufwendungen",
                "aufwand",
                description="Software, Lizenzen, kleinere Ausgaben",
            ),
            # FAHRZEUGKOSTEN
            "4520": SKR03Account(
                "4520",
                "Kraftfahrzeugkosten (nicht abziehbare Vorsteuer)",
                "aufwand",
                description="Kraftstoff, Parkgebühren",
            ),
            "4530": SKR03Account(
                "4530",
                "Kraftfahrzeugkosten (abziehbare Vorsteuer)",
                "aufwand",
                description="Reparaturen, Wartung, TÜV",
            ),
            # SONSTIGE BETRIEBSAUSGABEN
            "4970": SKR03Account(
                "4970",
                "Fortbildungskosten",
                "aufwand",
                description="Schulungen, Seminare, Weiterbildung",
            ),
            "4920": SKR03Account(
                "4920",
                "Rechts- und Beratungskosten",
                "aufwand",
                description="Steuerberater, Anwalt, Unternehmensberatung",
            ),
            # SPEZIELLE ELEKTRO-KONTEN
            "4840": SKR03Account(
                "4840",
                "Verpackungsmaterial",
                "aufwand",
                is_elektro_relevant=True,
                description="Installationsrohre, Kabelkanäle, Schutzverkleidungen",
            ),
        }

        return accounts_data

    def _load_classification_rules(self) -> dict[str, dict[str, Any]]:
        """Lädt Klassifikationsregeln für Elektrotechnik"""

        rules = {
            # ELEKTROMATERIAL-REGELN
            "elektromaterial": {
                "keywords": [
                    "kabel",
                    "leitung",
                    "nym",
                    "nyy",
                    "installationsleitung",
                    "steckdose",
                    "schuko",
                    "schalter",
                    "taster",
                    "dimmer",
                    "sicherung",
                    "automat",
                    "fi-schalter",
                    "leitungsschutzschalter",
                    "lampe",
                    "leuchte",
                    "led",
                    "halogen",
                    "glühbirne",
                    "verteilerdose",
                    "abzweigdose",
                    "klemme",
                    "lüsterklemme",
                    "rohr",
                    "kabelkanal",
                    "installationsrohr",
                ],
                "default_account": "4830",
                "confidence_base": 0.9,
            },
            # WERKZEUG-REGELN
            "werkzeug": {
                "keywords": [
                    "bohrmaschine",
                    "bohrhammer",
                    "säge",
                    "multimeter",
                    "spannungsprüfer",
                    "isolationstester",
                    "erdungsmessgerät",
                    "crimpzange",
                    "abisolierzange",
                    "seitenschneider",
                    "schraubendreher",
                    "werkzeugkoffer",
                    "leiter",
                ],
                "price_rules": {
                    "over_800": "0130",  # Anlagevermögen
                    "150_to_800": "0490",  # GWG
                    "under_150": "4830",  # Sofort abziehbar
                },
                "confidence_base": 0.85,
            },
            # BÜROMATERIAL-REGELN
            "bueroartikel": {
                "keywords": [
                    "papier",
                    "stifte",
                    "marker",
                    "ordner",
                    "hefter",
                    "toner",
                    "druckerpatrone",
                    "briefumschlag",
                    "software",
                    "lizenz",
                    "office",
                    "windows",
                ],
                "default_account": "4935",
                "confidence_base": 0.8,
            },
            # FREMDLEISTUNG-REGELN
            "fremdleistung": {
                "keywords": [
                    "montage",
                    "installation",
                    "reparatur",
                    "wartung",
                    "prüfung",
                    "abnahme",
                    "inbetriebnahme",
                    "service",
                ],
                "default_account": "4600",
                "confidence_base": 0.9,
            },
        }

        return rules

    def _load_elektro_keywords(self) -> dict[str, list[str]]:
        """Lädt Elektrotechnik-spezifische Keywords"""

        return {
            "lieferanten": [
                "conrad",
                "rexel",
                "elv",
                "reichelt",
                "wago",
                "phoenix",
                "siemens",
                "abb",
                "schneider",
                "legrand",
                "hager",
                "gira",
                "jung",
                "busch-jaeger",
                "berker",
                "merten",
            ],
            "elektro_begriffe": [
                "elektro",
                "electrical",
                "installation",
                "elektrik",
                "strom",
                "spannung",
                "ampere",
                "volt",
                "watt",
                "phase",
                "nullleiter",
                "schutzleiter",
                "erde",
            ],
            "ausschlussbegriffe": [
                "lebensmittel",
                "kleidung",
                "möbel",
                "kosmetik",
                "spielzeug",
                "sport",
                "garten",
                "haushalt",
            ],
        }

    def classify_item(
        self,
        item_description: str,
        item_price: float = 0.0,
        supplier: str = "",
        additional_context: dict[str, Any] | None = None,
    ) -> ClassificationResult:
        """
        Klassifiziert eine Rechnungsposition nach SKR03

        Args:
            item_description: Artikelbezeichnung
            item_price: Preis der Position (für GWG-Prüfung)
            supplier: Lieferant (für Kontext)
            additional_context: Zusätzliche Informationen

        Returns:
            ClassificationResult: Klassifikationsergebnis
        """
        if additional_context is None:
            additional_context = {}

        # Normalisiere Text für Analyse
        description_lower = item_description.lower()
        supplier_lower = supplier.lower()

        # Score für verschiedene Kategorien berechnen
        scores = {}

        # Elektromaterial-Score
        elektro_score = self._calculate_elektro_score(description_lower, supplier_lower)
        if elektro_score > 0:
            scores["elektromaterial"] = elektro_score

        # Werkzeug-Score
        werkzeug_score = self._calculate_werkzeug_score(description_lower, item_price)
        if werkzeug_score > 0:
            scores["werkzeug"] = werkzeug_score

        # Büroartikel-Score
        buero_score = self._calculate_buero_score(description_lower)
        if buero_score > 0:
            scores["bueroartikel"] = buero_score

        # Fremdleistung-Score
        service_score = self._calculate_service_score(description_lower)
        if service_score > 0:
            scores["fremdleistung"] = service_score

        # Beste Klassifikation ermitteln
        if not scores:
            # Fallback: Elektromaterial (häufigste Kategorie)
            return ClassificationResult(
                suggested_account="4830",
                confidence_score=0.3,
                reasoning="Keine eindeutige Klassifikation möglich - Elektromaterial als Standard",
                gkg_classification=self._classify_gwg(item_price),
            )

        # Höchster Score gewinnt
        best_category = max(scores.keys(), key=lambda k: scores[k])
        best_score = scores[best_category]

        # Konto basierend auf Kategorie und Preis bestimmen
        suggested_account = self._determine_account(best_category, item_price)

        # Alternative Konten sammeln
        alternatives = []
        for category, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)[
            1:3
        ]:
            alt_account = self._determine_account(category, item_price)
            if alt_account != suggested_account:
                alternatives.append((alt_account, score))

        return ClassificationResult(
            suggested_account=suggested_account,
            confidence_score=min(best_score, 0.95),
            reasoning=f"Klassifiziert als {best_category} basierend auf Textanalyse",
            alternative_accounts=alternatives,
            gkg_classification=self._classify_gwg(item_price),
        )

    def _calculate_elektro_score(self, description: str, supplier: str) -> float:
        """Berechnet Score für Elektromaterial"""
        score = 0.0

        elektro_keywords = self.classification_rules["elektromaterial"]["keywords"]

        # Keyword-Matching
        for keyword in elektro_keywords:
            if keyword in description:
                score += 0.3

        # Lieferanten-Bonus
        elektro_lieferanten = self.elektro_keywords["lieferanten"]
        for lieferant in elektro_lieferanten:
            if lieferant in supplier:
                score += 0.4
                break

        # Elektro-Begriff-Bonus
        for begriff in self.elektro_keywords["elektro_begriffe"]:
            if begriff in description:
                score += 0.2

        return min(score, 1.0)

    def _calculate_werkzeug_score(self, description: str, price: float) -> float:
        """Berechnet Score für Werkzeuge"""
        score = 0.0

        werkzeug_keywords = self.classification_rules["werkzeug"]["keywords"]

        # Keyword-Matching
        for keyword in werkzeug_keywords:
            if keyword in description:
                score += 0.4

        # Preis-Indikator (teure Artikel sind oft Werkzeuge)
        if price > 50:
            score += 0.2
        if price > 200:
            score += 0.3

        return min(score, 1.0)

    def _calculate_buero_score(self, description: str) -> float:
        """Berechnet Score für Büroartikel"""
        score = 0.0

        buero_keywords = self.classification_rules["bueroartikel"]["keywords"]

        # Keyword-Matching
        for keyword in buero_keywords:
            if keyword in description:
                score += 0.5

        return min(score, 1.0)

    def _calculate_service_score(self, description: str) -> float:
        """Berechnet Score für Fremdleistungen"""
        score = 0.0

        service_keywords = self.classification_rules["fremdleistung"]["keywords"]

        # Keyword-Matching
        for keyword in service_keywords:
            if keyword in description:
                score += 0.6

        return min(score, 1.0)

    def _determine_account(self, category: str, price: float) -> str:
        """Bestimmt SKR03-Konto basierend auf Kategorie und Preis"""

        if category == "werkzeug":
            # GWG-Logik für Werkzeuge
            if price >= 800:
                return "0130"  # Anlagevermögen
            elif price >= 150:
                return "0490"  # GWG
            else:
                return "4830"  # Betriebsausgabe

        elif category == "elektromaterial":
            return "4830"  # Elektromaterial

        elif category == "bueroartikel":
            return "4935"  # Büromaterial

        elif category == "fremdleistung":
            return "4600"  # Fremdleistungen

        else:
            return "4830"  # Standard: Elektromaterial

    def _classify_gwg(self, price: float) -> str:
        """Klassifiziert nach GWG-Regeln (Geringwertige Wirtschaftsgüter)"""

        if price >= 800:
            return "anlagegut"  # Muss aktiviert werden
        elif price >= 150:
            return "gwg"  # Geringwertiges Wirtschaftsgut
        else:
            return "betriebsausgabe"  # Sofort abziehbar

    def batch_classify(self, items: list[dict[str, Any]]) -> list[ClassificationResult]:
        """
        Klassifiziert mehrere Positionen

        Args:
            items: Liste von Positionen mit description, price, supplier

        Returns:
            List[ClassificationResult]: Klassifikationsergebnisse
        """
        logger.info(f"Starte Batch-Klassifikation für {len(items)} Positionen")

        results = []
        for i, item in enumerate(items):
            try:
                result = self.classify_item(
                    item_description=item.get("description", ""),
                    item_price=item.get("price", 0.0),
                    supplier=item.get("supplier", ""),
                )
                results.append(result)

            except Exception as e:
                logger.error(f"Fehler bei Position {i+1}: {e}")
                # Fallback-Klassifikation
                fallback = ClassificationResult(
                    suggested_account="4830",
                    confidence_score=0.1,
                    reasoning=f"Fehler bei Klassifikation: {e}",
                )
                results.append(fallback)

        logger.info(f"Batch-Klassifikation abgeschlossen: {len(results)} Ergebnisse")
        return results

    def get_account_info(self, account_number: str) -> SKR03Account | None:
        """Gibt Informationen zu einem SKR03-Konto zurück"""
        return self.accounts.get(account_number)

    def export_classification_report(
        self, results: list[ClassificationResult], output_path: Path
    ) -> None:
        """
        Exportiert Klassifikationsbericht

        Args:
            results: Klassifikationsergebnisse
            output_path: Ausgabedatei
        """
        report_data: dict[str, object] = {
            "summary": {
                "total_items": len(results),
                "high_confidence": len(
                    [r for r in results if r.confidence_score > 0.8]
                ),
                "medium_confidence": len(
                    [r for r in results if 0.5 < r.confidence_score <= 0.8]
                ),
                "low_confidence": len(
                    [r for r in results if r.confidence_score <= 0.5]
                ),
            },
            "account_distribution": {},
            "confidence_distribution": {},
            "classifications": [],
        }

        # Account-Verteilung
        for result in results:
            account = result.suggested_account
            account_distribution = report_data["account_distribution"]
            if isinstance(account_distribution, dict):
                if account in account_distribution:
                    current_count = account_distribution[account]
                    if isinstance(current_count, int | float):
                        account_distribution[account] = current_count + 1
                else:
                    account_distribution[account] = 1

        # Detaillierte Klassifikationen
        for i, result in enumerate(results):
            classification_entry = {
                "item_number": i + 1,
                "suggested_account": result.suggested_account,
                "confidence_score": result.confidence_score,
                "reasoning": result.reasoning,
                "alternatives": result.alternative_accounts,
                "gwg_classification": result.gkg_classification,
            }
            classifications = report_data["classifications"]
            if isinstance(classifications, list):
                classifications.append(classification_entry)

        # Speichere Bericht
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Klassifikationsbericht exportiert: {output_path}")


if __name__ == "__main__":
    # Test der SKR03-Klassifikation
    classifier = SKR03Classifier()

    test_items = [
        {
            "description": "Installationsleitung NYM-J 3x1,5 mm²",
            "price": 125.00,
            "supplier": "Conrad Electronic",
        },
        {
            "description": "Multimeter Digital Fluke 87V",
            "price": 456.78,
            "supplier": "Conrad Electronic",
        },
        {
            "description": "Büropapier DIN A4 80g/m²",
            "price": 12.50,
            "supplier": "Office Depot",
        },
        {
            "description": "Montage und Installation Schaltschrank",
            "price": 850.00,
            "supplier": "Elektro Service GmbH",
        },
    ]

    print("SKR03-Klassifikationsergebnisse:")
    print("=" * 50)

    for i, item in enumerate(test_items):
        description = str(item["description"]) if item["description"] else ""
        price_val = item["price"]
        price = float(price_val) if isinstance(price_val, int | float | str) else 0.0
        supplier = str(item["supplier"]) if item["supplier"] else ""
        result = classifier.classify_item(description, price, supplier)

        print(f"\nPosition {i+1}: {item['description']}")
        print(f"Preis: {item['price']:.2f}€")
        print(f"SKR03-Konto: {result.suggested_account}")
        print(f"Confidence: {result.confidence_score:.2f}")
        print(f"GWG-Status: {result.gkg_classification}")
        print(f"Begründung: {result.reasoning}")

        if result.alternative_accounts:
            print("Alternativen:", result.alternative_accounts)
