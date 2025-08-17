# Systemarchitektur: llkjj-ml

Dieses Dokument beschreibt die technische Architektur der `llkjj-ml` Pipeline, einer Anwendung zur automatisierten Verarbeitung und buchhalterischen Klassifizierung von PDF-Rechnungen im Elektrotechnik-Handwerk.

## 1. Übersicht

Das System ist als eine modulare, kommandozeilen-gesteuerte Pipeline konzipiert. Der Haupteinstiegspunkt ist `main.py`, welches verschiedene Workflows und Aufgaben über Sub-Befehle steuert. Die Architektur folgt dem **Single Responsibility Principle**, indem spezifische Aufgaben in dedizierte Module ausgelagert sind.

Der Kernprozess umfasst das Einlesen einer PDF-Rechnung, die Extraktion von Text und strukturierten Daten, die Anreicherung und Klassifizierung mittels KI (Gemini) und regelbasierten Systemen sowie die Speicherung der Ergebnisse.

## 2. Architektur-Stil

Die Anwendung verwendet eine **modulare Pipeline-Architektur**. Jede Verarbeitungsstufe ist eine eigenständige Komponente, die von einem zentralen Orchestrator (`UnifiedProcessor` oder `ModularProcessor`) koordiniert wird. Dies ermöglicht eine hohe Flexibilität, Testbarkeit und Wartbarkeit.

- **Konfigurationsgetrieben:** Das gesamte Verhalten wird zentral über die `src/config.py` gesteuert, die Pydantic zur Validierung und zum Laden von Umgebungsvariablen (`.env`) nutzt.
- **Modulare Komponenten:** Klare Trennung von Datenextraktion, Klassifizierung, Datenhaltung und Qualitätsbewertung.
- **Datenmodelle:** Strenge Typisierung und Validierung der Datenstrukturen durch Pydantic-Modelle in `src/models/`.

## 3. Kernkomponenten

Die Logik ist primär im `src/` Verzeichnis untergebracht und in folgende Module aufgeteilt:

### `main.py`
- **Zweck:** Der Haupt-Einstiegspunkt der Anwendung.
- **Funktionalität:** Definiert die CLI-Schnittstelle mittels `argparse`. Nimmt Befehle wie `process`, `train`, `database` oder `workflow1..4` entgegen und ruft die entsprechenden Service-Klassen auf. Initialisiert die Konfiguration und das Logging.

### `src/config.py`
- **Zweck:** Zentrales Konfigurationsmanagement.
- **Funktionalität:** Nutzt `pydantic-settings` zur Definition aller Konfigurationsparameter (API-Schlüssel, Pfade, Modellnamen, Geschäftsregeln). Lädt sensible Daten aus einer `.env`-Datei.

### `src/pipeline`
- **Zweck:** Orchestrierung der Verarbeitungspipeline.
- **Klassen:**
    - `UnifiedProcessor`: Der Haupt-Orchestrator, der die verschiedenen Module (Extractor, Classifier, Assessor) zu einer einzigen, kohärenten Pipeline verbindet.
    - `ProcessingResult`: Ein Pydantic-Modell, das die standardisierte Ausgabe der Verarbeitung kapselt.

### `src/processing`
- **Zweck:** Enthält die Kernlogik für die Geschäftsregeln und die Datenverarbeitung.
- **Klassen:**
    - `ModularProcessor`: Bietet alternative, isolierte Workflows (z.B. nur Text-Extraktion, nur Gemini-Verarbeitung), ideal für Debugging und Tests.
    - `DataClassifier`: Verantwortlich für die buchhalterische Klassifizierung. Nutzt den `SKR03Manager` und ein RAG-System (Retrieval-Augmented Generation) über die Vektordatenbank, um Rechnungspositionen SKR03-Konten zuzuordnen.
    - `QualityAssessor`: Bewertet die Qualität der Extraktion und berechnet einen Konfidenz-Score.

### `src/extraction`
- **Zweck:** Extraktion von Daten aus den Eingangs-PDFs.
- **Klassen:**
    - `DataExtractor`: Die zentrale Klasse für die Datenextraktion. Sie nutzt `docling` für die OCR und Text-Extraktion. Zusätzlich kann sie das `Gemini`-Modell ansteuern, um die extrahierten Daten zu bereinigen, zu strukturieren und anzureichern.

### `src/database`
- **Zweck:** Verwaltung der Vektordatenbank (`ChromaDB`).
- **Klassen:**
    - `OptimizedDatabaseManager`: Stellt eine optimierte Schnittstelle zur Datenbank bereit, inklusive Connection Pooling, Batch-Verarbeitung und Migrations-Tools.
    - Die Datenbank wird für das RAG-System genutzt, um semantisch ähnliche, bereits validierte Buchungsbeispiele zu finden und so die Klassifizierungsgenauigkeit zu verbessern.

### `src/models`
- **Zweck:** Definition der zentralen Datenstrukturen.
- **Funktionalität:** Enthält Pydantic-Modelle wie `Invoice` und `SKR03Classification`, die eine strikte Validierung und Typsicherheit für alle verarbeiteten Daten gewährleisten.

## 4. Datenfluss (Workflow 4: Komplette Pipeline)

Ein typischer Durchlauf einer Rechnung durch das System sieht wie folgt aus:

1.  **Initiierung:** Ein Benutzer ruft `python main.py workflow4 <pdf_pfad>` auf.
2.  **Extraktion:** Der `ModularProcessor` ruft den `DataExtractor` auf.
    - `docling` wandelt das PDF in Text und Tabellen um.
    - Der extrahierte Text wird an das `Gemini`-Modell gesendet, welches strukturierte Daten (Rechnungsnummer, Datum, Beträge, Positionen) im JSON-Format zurückgibt.
3.  **Klassifizierung:** Der `ModularProcessor` übergibt die extrahierten Rechnungspositionen an den `DataClassifier`.
    - Für jede Position sucht der `DataClassifier` mittels des `SKR03Manager` und der Vektordatenbank nach passenden SKR03-Konten.
    - Er kombiniert regelbasierte Treffer (Keyword-Matching) mit semantischer Ähnlichkeit (RAG) zu einer finalen Klassifizierung mit Konfidenz-Score.
4.  **Speicherung:** Das finale Ergebnis, welches die extrahierten Daten und die SKR03-Klassifizierungen enthält, wird als JSON-Datei im `data/output`-Verzeichnis gespeichert.

## 5. Konfigurationsmanagement

- Die Konfiguration wird über die `Config`-Klasse in `src/config.py` geladen.
- Standardwerte sind im Code definiert.
- Sensible Daten wie der `GOOGLE_API_KEY` werden aus einer `.env`-Datei im Projekt-Stammverzeichnis geladen, die nicht im Git-Repository eingecheckt wird.
