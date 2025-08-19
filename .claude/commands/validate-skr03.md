# SKR03 Klassifizierung Validieren

Validiere SKR03-Kontierung für deutsche Elektrohandwerk-Rechnungen.

## Schritte:

1. Lese aktuelle SKR03-Regeln aus `data/config/skr03_regeln.yaml`
2. Teste gegen neue/bestehende Rechnungsdaten
3. Berechne Konfidenz-Scores für Klassifizierungen
4. Identifiziere Verbesserungsmöglichkeiten
5. Prüfe deutsche Elektrotechnik-Terminologie
6. Validiere gegen SKR03-Standard für UG (haftungsbeschränkt)
7. Erstelle Validierungsbericht mit Empfehlungen

## Verwendung:

```
/validate-skr03 neue_rechnung.pdf
/validate-skr03 --all-test-pdfs
/validate-skr03 --accuracy-report
```

## Parameter:

$ARGUMENTS - PDF-Datei, Option oder Konto-Nummer

## Validierungskriterien:

- Klassifizierungsgenauigkeit > 92%
- Deutsche Elektrohandwerk-Begriffe korrekt erkannt
- SKR03-Konten korrekt zugeordnet
- Konfidenz-Scores dokumentiert

**IMPORTANT**: Fokus auf deutsche Elektrotechnik-Spezifika und SKR03-Konformität.
