#!/usr/bin/env python3
"""
Docling Extraktionsqualität-Analyse
Detaillierte Bewertung der Datenextraktion von Docling
"""

import json
from pathlib import Path


def analyze_docling_extraction_quality() -> None:
    """Analysiert die Qualität der Docling-Datenextraktion"""

    print("🔍 DOCLING EXTRAKTIONSQUALITÄT-ANALYSE")
    print("=" * 50)

    # Lade Docling-Output
    docling_file = Path("demo_output/Sonepar_test3_docling_best_practices.json")
    with open(docling_file, encoding="utf-8") as f:
        data = json.load(f)

    # Grundlegende Statistiken
    stats = data["extraction_stats"]
    print("\n📊 GRUNDLEGENDE STATISTIKEN:")
    print(f"  Verarbeitungszeit: {stats['processing_time']:.2f} Sekunden")
    print(f"  Seiten: {stats['total_pages']}")
    print(f"  Tabellen gefunden: {stats['tables_found']}")
    print(f"  Textblöcke: {stats['text_blocks_found']}")
    print(f"  OCR verwendet: {stats['ocr_used']}")
    print(f"  TableFormer Modus: {stats['table_former_mode']}")

    # Tabellen-Analyse
    tables = data.get("tables", [])
    print("\n📋 TABELLEN-EXTRAKTIONSQUALITÄT:")
    print("-" * 35)

    # Tabelle 1 - Hauptpositionen
    if len(tables) >= 1:
        table1 = tables[0]
        print("TABELLE 1 (Hauptpositionen):")
        print(f"  📊 {table1['rows']} Zeilen × {table1['columns']} Spalten")

        # Bewerte Datenqualität
        correct_positions = 0
        total_positions = len(table1["data"])

        print("  📝 POSITIONSANALYSE:")
        for i, pos in enumerate(table1["data"], 1):
            artikel_nr = pos.get("Pos Artikel-Nr.", "")
            menge = pos.get("Menge", "")
            nettopreis = pos.get("Nettopreis", "")
            gesamt = pos.get("Gesamt", "")
            bezeichnung = pos.get("Artikelbezeichnung", "")

            # Prüfe Datenqualität
            has_artikel = artikel_nr.isdigit() and len(artikel_nr) >= 8
            has_menge = menge.isdigit()
            has_preis = nettopreis.isdigit()
            has_gesamt = gesamt.isdigit()
            has_bezeichnung = len(bezeichnung) > 10

            quality_score = sum(
                [has_artikel, has_menge, has_preis, has_gesamt, has_bezeichnung]
            )

            if quality_score >= 4:
                correct_positions += 1
                status = "✅"
            else:
                status = "❌"

            print(
                f"    Position {i}: {status} {artikel_nr[:10]}... (Score: {quality_score}/5)"
            )

        print(
            f"  🎯 Qualitätsscore: {correct_positions}/{total_positions} ({correct_positions/total_positions*100:.1f}%)"
        )

    # Tabelle 2 - Fortsetzung + Summen
    if len(tables) >= 2:
        table2 = tables[1]
        print("\nTABELLE 2 (Fortsetzung + Summen):")
        print(f"  📊 {table2['rows']} Zeilen × {table2['columns']} Spalten")

        # Finde wichtige Summen
        wichtige_summen = {
            "Warenwert": "276,32",
            "MwSt 19%": "52,50",
            "Endbetrag": "328,82",
            "Skonto-Fälligkeit": "05.02.25",
            "Netto-Fälligkeit": "05.02.25",
        }

        gefundene_summen = 0
        print("  💰 SUMMEN-EXTRAKTION:")

        for pos in table2["data"]:
            pos_text = " ".join(str(v) for v in pos.values())

            for summe_name, erwarteter_wert in wichtige_summen.items():
                if erwarteter_wert in pos_text:
                    print(f"    ✅ {summe_name}: {erwarteter_wert} gefunden")
                    gefundene_summen += 1
                    break

        print(
            f"  🎯 Summen-Score: {gefundene_summen}/{len(wichtige_summen)} ({gefundene_summen/len(wichtige_summen)*100:.1f}%)"
        )

    # Fibu-relevante Datenextraktion
    print("\n💼 FIBU-RELEVANTE DATENEXTRAKTION:")
    print("-" * 35)

    full_text = data["content"]["full_text"]

    fibu_fields = {
        "Kunden-Nr.": "031938",
        "Rechnungs-Nr.": "56981199",
        "Rechnungsdatum": "22.01.25",
        "Lieferant": "Sonepar",
        "Kunde": "Elektro Czok UG",
        "Warenwert": "276,32",
        "MwSt-Satz": "19,0%",
        "MwSt-Betrag": "52,50",
        "Endbetrag": "328,82",
        "Zahlungsziel": "05.02.25",
        "Lieferanschrift": "Garteler Weiden 9",
        "PLZ/Ort": "27711 Osterholz-Scharmbeck",
    }

    extracted_count = 0
    for field, expected_value in fibu_fields.items():
        found = expected_value in full_text
        status = "✅" if found else "❌"
        print(f"  {status} {field}: {expected_value}")
        if found:
            extracted_count += 1

    print(
        f"  🎯 Fibu-Score: {extracted_count}/{len(fibu_fields)} ({extracted_count/len(fibu_fields)*100:.1f}%)"
    )

    # SKR03-Kontierungsrelevanz
    print("\n🏦 SKR03-KONTIERUNGSRELEVANZ:")
    print("-" * 30)

    # Analysiere Produktkategorien für automatische Kontierung
    if len(tables) >= 1:
        produktkategorien: dict[str, list[dict[str, str]]] = {}

        for pos in tables[0]["data"]:
            bezeichnung = pos.get("Artikelbezeichnung", "").lower()
            nettopreis = pos.get("Nettopreis", "0")

            # Kategorisierung für SKR03
            if "gira" in bezeichnung and (
                "rahmen" in bezeichnung or "abdeckung" in bezeichnung
            ):
                kategorie = "Elektroinstallation (4400)"
            elif "siemens" in bezeichnung and "abdeckstreifen" in bezeichnung:
                kategorie = "Elektroinstallation (4400)"
            elif "adapterrahmen" in bezeichnung:
                kategorie = "Elektroinstallation (4400)"
            elif "blindabdeckung" in bezeichnung:
                kategorie = "Elektroinstallation (4400)"
            elif "datentech" in bezeichnung:
                kategorie = "IT-Technik (4405)"
            else:
                kategorie = "Sonstiges Material (4499)"

            if kategorie not in produktkategorien:
                produktkategorien[kategorie] = []

            produktkategorien[kategorie].append(
                {"artikel": pos.get("Pos Artikel-Nr.", ""), "preis": nettopreis}
            )

        print("  📂 AUTOMATISCHE KATEGORISIERUNG:")
        for kategorie, artikel in produktkategorien.items():
            gesamt_wert = sum(
                float(a["preis"]) for a in artikel if a["preis"].isdigit()
            )
            print(f"    {kategorie}: {len(artikel)} Artikel, {gesamt_wert:.2f}€")

    # Datenstruktur-Qualität
    print("\n🏗️ DATENSTRUKTUR-QUALITÄT:")
    print("-" * 25)

    # JSON-Struktur bewerten
    has_metadata = "metadata" in data
    has_tables = len(tables) > 0
    has_structured_data = has_tables and len(tables[0].get("data", [])) > 0
    has_text = len(full_text) > 1000
    has_html = has_tables and "html" in tables[0]
    has_markdown = has_tables and "markdown" in tables[0]

    struktur_score = sum(
        [
            has_metadata,
            has_tables,
            has_structured_data,
            has_text,
            has_html,
            has_markdown,
        ]
    )

    print(f"  ✅ Metadaten vorhanden: {has_metadata}")
    print(f"  ✅ Tabellen strukturiert: {has_structured_data}")
    print(f"  ✅ Volltext extrahiert: {has_text}")
    print(f"  ✅ HTML-Format: {has_html}")
    print(f"  ✅ Markdown-Format: {has_markdown}")
    print(f"  🎯 Struktur-Score: {struktur_score}/6 ({struktur_score/6*100:.1f}%)")

    # Gesamtbewertung
    print("\n🏆 GESAMTBEWERTUNG DOCLING:")
    print("=" * 30)

    # Berechne Gesamtscore
    position_score = (
        correct_positions / total_positions if "correct_positions" in locals() else 0
    )
    summen_score = (
        gefundene_summen / len(wichtige_summen) if "gefundene_summen" in locals() else 0
    )
    fibu_score = extracted_count / len(fibu_fields)
    struktur_score_pct = struktur_score / 6

    gesamtscore = (position_score + summen_score + fibu_score + struktur_score_pct) / 4

    print(f"  📊 Positionsextraktion: {position_score*100:.1f}%")
    print(f"  💰 Summenextraktion: {summen_score*100:.1f}%")
    print(f"  💼 Fibu-Daten: {fibu_score*100:.1f}%")
    print(f"  🏗️ Datenstruktur: {struktur_score_pct*100:.1f}%")
    print(f"\n  🎯 GESAMTSCORE: {gesamtscore*100:.1f}%")

    # Bewertung
    if gesamtscore >= 0.9:
        bewertung = "🏆 AUSGEZEICHNET - Produktionsreif"
    elif gesamtscore >= 0.8:
        bewertung = "🥇 SEHR GUT - Minimale Nachbearbeitung"
    elif gesamtscore >= 0.7:
        bewertung = "🥈 GUT - Leichte Nachbearbeitung nötig"
    elif gesamtscore >= 0.6:
        bewertung = "🥉 BEFRIEDIGEND - Deutliche Nachbearbeitung"
    else:
        bewertung = "❌ UNGENÜGEND - Starke Überarbeitung nötig"

    print(f"\n  {bewertung}")

    # Empfehlungen
    print("\n💡 EMPFEHLUNGEN:")
    print("-" * 15)
    if gesamtscore >= 0.85:
        print("✅ Docling ist BEREIT für Gemini-Training")
        print("✅ Datenqualität ausreichend für spaCy-Training")
        print("✅ Direkte Verwendung für Fibu-Automatisierung möglich")
    elif gesamtscore >= 0.7:
        print("⚠️ Docling braucht leichte Nachbearbeitung")
        print("💡 Post-Processing für kritische Felder empfohlen")
        print("✅ Grundsätzlich tauglich für Training")
    else:
        print("❌ Docling allein nicht ausreichend")
        print("💡 Hybrid-Ansatz mit OCR empfohlen")
        print("⚠️ Zusätzliche Validierung erforderlich")


if __name__ == "__main__":
    analyze_docling_extraction_quality()
