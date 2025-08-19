# Test PDF Processing Pipeline

Führe einen vollständigen Test der PDF-Verarbeitungs-Pipeline durch.

## Schritte:

1. Validiere Eingabe-PDF existiert
2. Aktiviere Debug-Modus für detaillierte Logs
3. Führe GeminiDirectProcessor aus
4. Validiere SKR03-Klassifizierung
5. Prüfe ChromaDB-Integration und Konfidenz-Scores
6. Erstelle detaillierten Qualitätsbericht
7. Vergleiche mit Baseline-Performance

## Verwendung:

```
/test-pdf-pipeline test_pdfs/Sonepar_test3.pdf
```

## Parameter:

$ARGUMENTS - Pfad zur Test-PDF-Datei

## Erwartetes Verhalten:

- Verarbeitungszeit < 30 Sekunden
- SKR03-Klassifizierung Konfidenz > 0.85
- Erfolgreiche ChromaDB-Speicherung
- Vollständige Rechnungsdaten-Extraktion

**IMPORTANT**: Verwende Poetry für alle Python-Befehle und dokumentiere Performance-Metriken.
