#!/usr/bin/env python3
"""
SKR03Manager - Produktive Einbindung des SKR03-Regelwerks
========================================================

Zentraler Manager für SKR03-Klassifizierung und Kontenverwaltung.
Lädt und verwaltet:
- skr03_regeln.yaml v2.0 (29 Kategorien für Elektrotechnik)
- Kontenplan.csv (300+ SKR03 Konten)
- Intelligente Keyword-Matching-Logik

Author: LLKJJ ML Pipeline
Version: 2.0.0 - Produktive Integration
"""

import csv
import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class KontenplanParser:
    """Parser für den vollständigen SKR03-Kontenplan aus CSV"""

    def __init__(self, csv_pfad: Path):
        self.csv_pfad = csv_pfad
        self.konten: dict[str, dict[str, Any]] = {}
        self.kategorien: dict[str, list[str]] = {}
        self._lade_kontenplan()

    def _lade_kontenplan(self) -> None:
        """Lädt den Kontenplan aus CSV-Datei"""
        try:
            if not self.csv_pfad.exists():
                logger.warning("Kontenplan nicht gefunden: %s", self.csv_pfad)
                return

            with open(self.csv_pfad, encoding="utf-8-sig") as f:  # BOM-sicher
                reader = csv.DictReader(f, delimiter=";")  # Deutsche CSV mit Semicolon
                for row in reader:
                    konto_nr = row.get("Kontonummer", "").strip()
                    if konto_nr:
                        self.konten[konto_nr] = {
                            "bezeichnung": row.get("Kontenbezeichnung", ""),
                            "kategorie": row.get("Kontenkategorie", ""),
                            "unterart": row.get("Kontenunterart", ""),
                            "ust_position": row.get("USt_Position", ""),
                            "soll_haben": row.get("Soll_Haben", ""),
                            "steuerart": row.get("Steuerart", ""),
                        }

                        # Kategorien sammeln
                        kat = row.get("Kontenkategorie", "")
                        if kat:
                            if kat not in self.kategorien:
                                self.kategorien[kat] = []
                            self.kategorien[kat].append(konto_nr)

            logger.info(
                "✅ Kontenplan geladen: %d Konten, %d Kategorien",
                len(self.konten),
                len(self.kategorien),
            )

        except (OSError, csv.Error) as e:
            logger.error("❌ Fehler beim Laden des Kontenplans: %s", e)

    def ist_gueltig(self, konto_nr: str) -> bool:
        """Prüft ob Kontonummer im SKR03-Plan existiert"""
        return konto_nr in self.konten

    def get_konto_info(self, konto_nr: str) -> dict[str, Any] | None:
        """Gibt Informationen zu einem Konto zurück"""
        return self.konten.get(konto_nr)

    def get_konten_by_kategorie(self, kategorie: str) -> list[str]:
        """Gibt alle Konten einer Kategorie zurück"""
        return self.kategorien.get(kategorie, [])


class SKR03Manager:
    """
    Zentraler Manager für SKR03-Regelwerk und Klassifizierung

    Verwaltet:
    - YAML-Klassifizierungsregeln (config/skr03_regeln.yaml)
    - CSV-Kontenplan (Kontenplan.csv)
    - Intelligente Keyword-Matching-Logik
    """

    def __init__(self, regeln_pfad: Path, kontenplan_pfad: Path):
        self.regeln_pfad = regeln_pfad
        self.kontenplan_pfad = kontenplan_pfad

        # Kernkomponenten
        self.klassifizierungsregeln: dict[str, Any] = {}
        self.kontenplan_parser: KontenplanParser | None = None

        # Status-Tracking
        self._regeln_geladen = False
        self._kontenplan_geladen = False

        # Initialisierung
        self._lade_regelwerk()
        self._lade_kontenplan()

    def _lade_regelwerk(self) -> None:
        """Lädt SKR03-Klassifizierungsregeln aus YAML"""
        try:
            if not self.regeln_pfad.exists():
                logger.error(f"❌ SKR03-Regeln nicht gefunden: {self.regeln_pfad}")
                return

            with open(self.regeln_pfad, encoding="utf-8") as f:
                regeln_daten = yaml.safe_load(f)

            self.klassifizierungsregeln = regeln_daten.get("kategorien", {})
            self._regeln_geladen = len(self.klassifizierungsregeln) > 0

            if self._regeln_geladen:
                logger.info(
                    f"✅ SKR03-Regeln geladen: {len(self.klassifizierungsregeln)} Kategorien"
                )

                # Statistiken loggen
                total_keywords = sum(
                    len(kategorie.get("schlüsselwörter", []))
                    for kategorie in self.klassifizierungsregeln.values()
                )
                logger.info(f"   📊 Gesamt Keywords: {total_keywords}")

                # Beispiele der Kategorien
                kategorien_liste = list(self.klassifizierungsregeln.keys())[:5]
                logger.debug(f"   📂 Kategorien (Beispiele): {kategorien_liste}")
            else:
                logger.warning("⚠️ Keine Kategorien in SKR03-Regeln gefunden")

        except Exception as e:
            logger.error(f"❌ Fehler beim Laden der SKR03-Regeln: {e}")
            self._regeln_geladen = False

    def _lade_kontenplan(self) -> None:
        """Lädt vollständigen Kontenplan für Validierung"""
        try:
            self.kontenplan_parser = KontenplanParser(self.kontenplan_pfad)
            self._kontenplan_geladen = len(self.kontenplan_parser.konten) > 0

            if self._kontenplan_geladen:
                logger.info("✅ Kontenplan verfügbar für Validierung")
            else:
                logger.warning("⚠️ Kontenplan konnte nicht geladen werden")

        except Exception as e:
            logger.error(f"❌ Fehler beim Laden des Kontenplans: {e}")
            self._kontenplan_geladen = False

    def ist_bereit(self) -> bool:
        """Prüft ob Manager einsatzbereit ist"""
        return self._regeln_geladen

    def get_statistiken(self) -> dict[str, Any]:
        """Gibt Status-Statistiken zurück"""
        regeln_count = len(self.klassifizierungsregeln)
        konten_count = (
            len(self.kontenplan_parser.konten) if self.kontenplan_parser else 0
        )

        return {
            "regeln_geladen": regeln_count,
            "konten_geladen": konten_count,
            "ist_bereit": self.ist_bereit(),
            "regeln_verfügbar": self._regeln_geladen,
            "kontenplan_verfügbar": self._kontenplan_geladen,
        }

    def klassifiziere_artikel(
        self, beschreibung: str, lieferant: str = ""
    ) -> tuple[str, str, float, list[str]]:
        """
        Klassifiziert einen Artikel basierend auf Beschreibung und Lieferant

        Returns:
            Tuple[kategorie, konto, konfidenz, matched_keywords]
        """
        if not self._regeln_geladen:
            logger.warning("SKR03-Manager nicht bereit - verwende Fallback")
            return "wareneingang_elektro_allgemein", "3400", 0.3, []

        beschreibung_lower = beschreibung.lower()
        lieferant_lower = lieferant.lower()
        kombinierter_text = f"{beschreibung_lower} {lieferant_lower}".strip()

        best_score = 0.0
        best_category = "wareneingang_elektro_allgemein"
        matched_keywords = []

        # Durchsuche alle Kategorien
        for kategorie, regeln in self.klassifizierungsregeln.items():
            score = 0.0
            gefundene_keywords = []

            schluesselwoerter = regeln.get("schlüsselwörter", [])

            for keyword in schluesselwoerter:
                keyword_lower = keyword.lower()

                # Keyword-Matching mit verschiedenen Gewichtungen
                if keyword_lower in kombinierter_text:
                    # Basis-Score
                    base_score = 1.0

                    # Längen-Bonus (längere Keywords sind spezifischer)
                    laenge_bonus = len(keyword) * 0.1

                    # Lieferanten-Bonus (Keywords im Lieferant sind wichtiger)
                    lieferant_bonus = 2.0 if keyword_lower in lieferant_lower else 0.0

                    # Vollwort-Bonus (ganze Wörter sind wichtiger als Teilwörter)
                    vollwort_bonus = (
                        0.5 if f" {keyword_lower} " in f" {kombinierter_text} " else 0.0
                    )

                    keyword_score = (
                        base_score + laenge_bonus + lieferant_bonus + vollwort_bonus
                    )
                    score += keyword_score
                    gefundene_keywords.append(keyword)

            # Beste Kategorie aktualisieren
            if score > best_score:
                best_score = score
                best_category = kategorie
                matched_keywords = gefundene_keywords

        # Konfidenz berechnen
        if best_score == 0:
            konfidenz = 0.2  # Minimale Konfidenz für Fallback
        else:
            # Konfidenz basierend auf Score und Anzahl Matches
            base_confidence = min(0.8, 0.3 + (best_score * 0.05))
            keyword_bonus = min(0.15, len(matched_keywords) * 0.03)
            konfidenz = base_confidence + keyword_bonus

        # Standard-Konto aus Regeln abrufen
        category_info = self.klassifizierungsregeln.get(best_category, {})
        standard_konto = category_info.get("standard_konto", "3400")

        # Konto-Validierung (falls Kontenplan verfügbar)
        if self.kontenplan_parser and not self.kontenplan_parser.ist_gueltig(
            standard_konto
        ):
            logger.warning(
                f"⚠️ Ungültiges Konto {standard_konto} für Kategorie {best_category}"
            )
            standard_konto = "3400"  # Fallback auf sicheres Elektromaterial-Konto

        logger.debug(
            f"SKR03-Klassifizierung: '{beschreibung}' → {best_category} "
            f"(Konto: {standard_konto}, Konfidenz: {konfidenz:.3f}, "
            f"Keywords: {matched_keywords[:3]})"
        )

        return best_category, standard_konto, konfidenz, matched_keywords

    def get_kategorie_info(self, kategorie: str) -> dict[str, Any] | None:
        """Gibt detaillierte Informationen zu einer Kategorie zurück"""
        return self.klassifizierungsregeln.get(kategorie)

    def get_verfuegbare_kategorien(self) -> list[str]:
        """Gibt alle verfügbaren Kategorien zurück"""
        return list(self.klassifizierungsregeln.keys())

    def validiere_konto(self, konto_nr: str) -> bool:
        """Validiert eine Kontonummer gegen den SKR03-Plan"""
        if not self.kontenplan_parser:
            return True  # Keine Validierung möglich
        return self.kontenplan_parser.ist_gueltig(konto_nr)


# Convenience-Funktion für einfache Nutzung
def lade_skr03_manager(
    regeln_pfad: str = "config/skr03_regeln.yaml",
    kontenplan_pfad: str = "Kontenplan.csv",
) -> SKR03Manager:
    """Lädt SKR03Manager mit Standard-Pfaden"""
    # Base path ist jetzt das Projekt-Root (eine Ebene über src/)
    base_path = Path(__file__).parent.parent

    regeln_path = base_path / regeln_pfad
    konten_path = base_path / kontenplan_pfad

    return SKR03Manager(regeln_path, konten_path)


if __name__ == "__main__":
    # Test des SKR03Managers
    import sys

    logging.basicConfig(level=logging.DEBUG)

    # Test-Initialisierung
    manager = lade_skr03_manager()

    if manager.ist_bereit():
        stats = manager.get_statistiken()
        print("✅ SKR03Manager funktioniert!")
        print(f"   📊 Statistiken: {stats}")

        # Test-Klassifizierungen
        test_artikel = [
            ("GIRA Lichtschalter E2", "FAMO GmbH"),
            ("Sicherungsautomat 16A", "Sonepar"),
            ("Bürostuhl ergonomisch", "IKEA"),
            ("Elektrokabel NYM-J 3x1,5", "Elektrogroßhandel"),
        ]

        print("\n🧪 Test-Klassifizierungen:")
        for beschreibung, lieferant in test_artikel:
            kategorie, konto, konfidenz, keywords = manager.klassifiziere_artikel(
                beschreibung, lieferant
            )
            print(f"   {beschreibung} → {kategorie} (Konto: {konto}, {konfidenz:.2f})")
            if keywords:
                print(f"      Keywords: {keywords[:3]}")
    else:
        print("❌ SKR03Manager konnte nicht initialisiert werden")
        sys.exit(1)
