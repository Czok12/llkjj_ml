---
applyTo: "**"
---

# LLKJJ ML Plugin - KI-Assistent Anweisungen (2025 Best Practices)

## Projektübersicht

**Zweck**: Deutsches Elektrohandwerk-Buchhaltungs-Plugin für intelligente Rechnungsverarbeitung mit KI/ML-Pipeline

**WICHTIG: Nutze immer `poetry run ...` für alle Terminal-Befehle!**

**Kontext**: Dies ist ein Plugin-Modul für die LLKJJ Buchhaltungssoftware - dem deutschen buchhaltungsbutler.de-Ersatz für Elektrotechnik-Handwerksfirmen mit Rechtsform UG (haftungsbeschränkt), die zur doppelten Buchführung verpflichtet sind.

**Domäne**: Dokumenten-KI, OCR, NLP, SKR03-Kontierung, Machine Learning
**Sprache**: Deutsche Optimierung mit Elektrotechnik-Spezialisierung

## Kern-Technologien & Stack

### Primäre Technologien

- **Python 3.10+**: Moderne Type Hints, Pattern Matching, async/await
- **Poetry**: Dependency Management und CLI-Werkzeuge (`poetry run ...` ist Pflicht!)
- **Docling 2.44.0**: IBMs PDF-Verarbeitung mit TableFormer KI
- **spaCy 3.7+**: Deutsche NLP und Entitätserkennung
- **Gemini 2.5 Pro**: KI-Verbesserung für Extraktionsqualität

### KI/ML Frameworks

- **Docling**: PDF → strukturierte Datenextraktion
- **Google Gemini**: LLM-basierte Content-Verbesserung
- **spaCy**: Named Entity Recognition für deutsche Texte
- **ChromaDB**: Vektordatenbank für intelligente Klassifizierung
- **Transformers**: Für zukünftige Modell-Integration

### Entwicklungstools

- **Pre-commit**: Code-Qualitäts-Automatisierung
- **Ruff**: Schnelles Python-Linting
- **Black**: Code-Formatierung
- **mypy**: Statische Typ-Überprüfung

## Plugin-Architektur (Nach Konsolidierung)

### Kern-Services

1. **unified_processor.py** (642 LOC)

   - PDF → Docling → Gemini → SKR03 Pipeline
   - Einzige Verantwortung: Dokumentverarbeitung

2. **src/trainer.py** (ehemals training_service.py)

   - spaCy-Modelltraining und Datenexport
   - Einzige Verantwortung: ML-Training

3. **main.py** (337 LOC)
   - Einheitliche CLI-Schnittstelle
   - Befehle: process, export, train, pipeline, analyze

### Entfernte Redundanzen (KISS-Prinzip angewandt)

- ❌ Redundante Dateien → Konsolidiert in Kern-Services
- ❌ Überkomplexe src/ Struktur → Vereinfacht zu 2 Services
- ❌ Mehrfache Entry Points → 1 einheitliche CLI

## LLKJJ Plugin-Kontext

### Geschäftszweck

- **Zielgruppe**: Elektrotechnik-Handwerksfirmen (UG) in Deutschland
- **Buchhaltungsart**: Doppelte Buchführung nach SKR03
- **Ersetzt**: buchhaltungsbutler.de und manuelle Rechnungseingabe
- **Automatisiert**: PDF-Upload → OCR → KI-Analyse → SKR03-Kontierung → DATEV-Export

## KI-Assistent Verhaltensrichtlinien

### 1. Code-Qualitätsstandards (2025)

**Type Safety First**

- Nutze moderne Python 3.10+ Type Hints: `dict[str, Any]` nicht `Dict[str, Any]`
- Union-Syntax: `str | int` nicht `Union[str, int]`
- Immer Return-Type-Annotationen bereitstellen
- Dataclasses für strukturierte Daten verwenden

**Fehlerbehandlung**

- Spezifische Exception-Typen, vermeide blanke `except:`
- Umfassende Fehlerprotokollierung mit Kontext
- Graceful Degradation für nicht-kritische Ausfälle

**Dokumentationsstandards**

- Klare Docstrings im Google-Stil
- Type-Annotationen dienen als Dokumentation
- Code-Kommentare nur für Geschäftslogik
- Offensichtliche Kommentare vermeiden

### 2. KI/ML Best Practices (2025)

**Modell-Dokumentation (CLeAR Framework)**

- **Comparable** (Vergleichbar): Standardisierte Metriken und Evaluierung
- **Legible** (Lesbar): Klare Erklärungen des Modellverhaltens
- **Actionable** (Umsetzbar): Spezifische Anleitungen für Nutzer und Maintainer
- **Robust**: Edge Cases und Fehlermodi behandeln

**Data Pipeline Prinzipien**

- Unveränderliche Datenverarbeitung wo möglich
- Klare Datenherkunft und Versionierung
- Umfassende Validierung bei jedem Schritt
- Performance-Monitoring und Logging

**Deutsche Sprachoptimierung**

- spaCy Deutsche Modelle: `de_core_news_sm`
- Elektrotechnik-Domänen-Entitäten und Terminologie
- SKR03-Buchhaltungsklassifizierungskontext
- Deutsche Sonderzeichen und Encoding handhaben

### 3. Dokumentation für KI-Reader (2025)

**Struktur für LLM-Verbrauch**

- Klare, beschreibende Überschriften verwenden
- Kurze, fokussierte Absätze (3-5 Zeilen)
- Semantische Formatierung: Code-Blöcke, Tabellen, Listen
- Konsistente Terminologie durchgehend

**Code-Dokumentation**

- Selbstdokumentierender Code mit aussagekräftigen Namen
- Type Hints als primäre Dokumentation
- Docstrings nur für öffentliche APIs
- Beispiele in Docstrings für komplexe Funktionen

### 4. Entwicklungsworkflow

**Git-Praktiken**

- Atomare Commits mit klaren Nachrichten
- Feature-Branches für bedeutende Änderungen
- Aussagekräftige Commit-Nachrichten, die "warum" erklären
- Historie sauber und linear halten

**Test-Strategie**

- Unit-Tests für Kernfunktionalität
- Integrationstests für Pipeline-Workflows
- Performance-Benchmarks für Verarbeitungsgeschwindigkeit
- Qualitätsmetriken für Extraktionsgenauigkeit

**Performance-Optimierung**

- Async/await für I/O-Operationen
- Batch-Verarbeitung für mehrere Dokumente
- Speichereffiziente Datenstrukturen
- Caching für teure Operationen

## Domänen-spezifischer Kontext

### Deutsche Elektrobranche

- **SKR03**: Standard-Kontenrahmen für deutsche Unternehmen
- **Elektrotechnik**: Elektroingenieurwesen/Elektrohandwerk
- **Häufige Begriffe**: Rechnung, Artikel, Menge, Einzelpreis, Gesamt
- **Regulatorisches**: Deutsches Steuerrecht, Rechnungsanforderungen

### Dokumentverarbeitung

- **PDF-Typen**: Gescannte Rechnungen, digitale Rechnungen, gemischter Inhalt
- **Qualitätsstufen**: Hoch (digital), Mittel (sauber gescannt), Niedrig (schlechter Scan)
- **Ausgabeformat**: Strukturiertes JSON mit Konfidenz-Scores

### ML-Pipeline

- **Input**: Deutsche Elektrohandwerk-PDFs
- **Verarbeitung**: OCR → Verbesserung → Klassifizierung → Extraktion
- **Output**: Strukturierte Rechnungsdaten mit SKR03-Klassifizierungen

## Interaktionsrichtlinien

### Bei Code-Hilfe

1. **Kontext verstehen**: Nach dem spezifischen Anwendungsfall fragen
2. **Architektur befolgen**: Die KISS-Konsolidierung respektieren
3. **Type Safety**: Immer ordentliche Type-Annotationen einschließen
4. **Performance**: Deutsche Textverarbeitungsspezifika berücksichtigen
5. **Testing**: Angemessene Testabdeckung vorschlagen

### Beim Debugging

1. **Logs überprüfen**: Verarbeitungslogs für Kontext prüfen
2. **Daten validieren**: Deutsche Kodierung korrekt sicherstellen
3. **Performance**: Speicherverbrauch bei großen PDFs überwachen
4. **Qualität**: Extraktions-Konfidenz-Scores überprüfen

### Bei Feature-Hinzufügungen

1. **KISS-Prinzip**: Ergänzungen einfach und fokussiert halten
2. **Single Responsibility**: Belange nicht vermischen
3. **Deutsche Optimierung**: Sprachspezifische Bedürfnisse berücksichtigen
4. **Rückwärtskompatibilität**: Bestehende Workflows beibehalten

## WICHTIG: Poetry-Kommandos

**ALLE Terminal-Befehle MÜSSEN mit `poetry run` beginnen:**

```bash
# Korrekt:
poetry run python main.py process document.pdf
poetry run mypy .
poetry run ruff check .
poetry run black .

# FALSCH (nicht verwenden):
python main.py process document.pdf
mypy .
ruff check .
black .
```

## Projektziele & Erfolgsmetriken

### Primäre Ziele

- **Genauigkeit**: >90% SKR03-Klassifizierungsgenauigkeit
- **Geschwindigkeit**: <30 Sekunden pro Dokumentverarbeitung
- **Zuverlässigkeit**: <1% Ausfallrate in der Produktion
- **Wartbarkeit**: Ein Entwickler kann die gesamte Codebase verstehen

### Qualitätsindikatoren

- **Type Safety**: 100% mypy-Compliance
- **Code Coverage**: >80% Testabdeckung
- **Dokumentation**: Alle öffentlichen APIs dokumentiert
- **Performance**: Sub-lineare Skalierung mit Dokumentgröße

## Zukünftige Überlegungen

### Potenzielle Verbesserungen

- **Erweiterte Modelle**: Integration neuerer Transformer-Modelle
- **Batch-Optimierung**: Parallele Verarbeitung für große Dokumentensätze
- **API-Integration**: REST-API für externe Service-Integration
- **Qualitätsverbesserungen**: Verbesserte Konfidenz-Bewertung

### Architektur-Evolution

- KISS-Prinzipien beibehalten
- Feature Creep vermeiden
- Konsolidierungsvorteile behalten
- Jede Komplexitätserweiterung dokumentieren

---

_Zuletzt aktualisiert: 16. August 2025_
_Version: 2.1.0 (Post-Konsolidierung, Deutsch)_
_Framework: Basierend auf 2025 KI/ML-Dokumentations-Best-Practices_
