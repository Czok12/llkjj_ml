
### 2. `Dokumentation.md`

Dieses Dokument ist für Entwickler gedacht. Es erklärt die Funktionsweise, die wichtigsten Workflows und wie man mit dem System arbeitet.

```markdown
# LLKJJ ML-Pipeline - Entwicklerdokumentation

**Version: 4.0 (Gemini-First)**

Dieses Dokument beschreibt die Funktionsweise und den Aufbau der LLKJJ ML-Pipeline. Es richtet sich an Entwickler, die das System warten, erweitern oder in das `llkjj_backend` integrieren.

## 1. Funktionsübersicht

Die Pipeline ist ein eigenständiges Python-Paket (`llkjj_ml`), das darauf spezialisiert ist, PDF-Eingangsrechnungen aus dem deutschen Elektrohandwerk zu verarbeiten.

**Kernfunktionen:**
-   **Direkte PDF-Analyse:** Extrahiert Rechnungsdaten (Kopf, Positionen, Summen) direkt aus PDF-Dateien.
-   **SKR03-Klassifizierung:** Ordnet jeder Rechnungsposition automatisch ein passendes Konto aus dem deutschen Standardkontenrahmen SKR03 zu.
-   **Kontinuierliches Lernen:** Jede verarbeitete Rechnung erzeugt Trainingsdaten, um zukünftige, vollständig lokale KI-Modelle (spaCy, RAG) zu trainieren.
-   **Qualitätsbewertung:** Jedes Ergebnis wird mit einem Konfidenz-Score und einer Qualitätsstufe versehen.

## 2. Haupt-Workflow (Gemini-First)

Der Standardprozess wird durch die Klasse `GeminiDirectProcessor` gesteuert und läuft wie folgt ab:

1.  **Eingabe:** Eine PDF-Datei wird an die Methode `process_pdf_gemini_first` übergeben.
2.  **Validierung:** Die PDF wird auf Gültigkeit und Größe geprüft.
3.  **Gemini-Analyse:**
    -   Die PDF wird an die Google Gemini API gesendet.
    -   Ein speziell entwickelter Prompt (`_get_german_elektro_analysis_prompt`) instruiert das Modell, alle relevanten Daten in einem strukturierten JSON-Format zu extrahieren. Dies beinhaltet Header-Daten, alle Rechnungspositionen und Vorschläge für spaCy-Annotationen.
4.  **Schema-Validierung:** Die JSON-Antwort von Gemini wird gegen ein strenges Pydantic-Schema (`GeminiExtractionResult`) validiert, um die Datenintegrität sicherzustellen.
5.  **RAG-Anreicherung:**
    -   Die von Gemini vorgeschlagenen SKR03-Konten für jede Position werden durch das lokale RAG-System (ChromaDB) verfeinert.
    -   Das System sucht nach ähnlichen, bereits (vom Benutzer) validierten Buchungen und passt den Vorschlag und die Konfidenz bei Bedarf an.
6.  **Qualitätsbewertung:** Der `QualityAssessor` berechnet einen finalen Konfidenz-Score basierend auf der Vollständigkeit der Extraktion und der Sicherheit der Klassifizierung.
7.  **Trainingsdaten-Generierung:**
    -   Der `SpacyAnnotationCorrector` korrigiert die von Gemini vorgeschlagenen Annotationen, um exakte Zeichen-Offsets für das spaCy-Training zu gewährleisten.
    -   Die validierten Klassifizierungen werden in der ChromaDB gespeichert, um das RAG-System für die Zukunft zu verbessern.
8.  **Ausgabe:** Ein Pydantic-validiertes `ProcessingResult`-Objekt wird zurückgegeben. Es enthält alle extrahierten Daten, Klassifizierungen, Metriken und Trainingsannotationen.

**Fehlerbehandlung:** Wenn ein Schritt in der Gemini-Pipeline fehlschlägt (z.B. API-Fehler, Validierungsfehler), wird der Prozess mit einer `RuntimeError` abgebrochen. Es gibt **keinen automatischen Fallback** auf die Docling-Methode.

## 3. Alternative Methode (Docling)

Für Test- und Vergleichszwecke existiert eine alternative Pipeline, die `Docling` für die OCR- und Tabellenextraktion verwendet.

-   **Aufruf:** Muss explizit über die CLI (`process-docling`) oder programmatisch aufgerufen werden.
-   **Prozessor:** `DoclingAlternativeProcessor`.
-   **Funktionsweise:** Nutzt den bewährten, mehrstufigen Prozess aus Extraktion (`DataExtractor`), Klassifizierung (`DataClassifier`) und Qualitätsbewertung.

## 4. Installation & Nutzung

Das Projekt wird mit Poetry verwaltet.

**Installation:**
```bash
poetry install
```

**Nutzung als CLI-Tool (für Entwickler):**

```bash
# Standard-Workflow (Gemini-First)
poetry run python main.py process /pfad/zur/rechnung.pdf

# Alternative (Docling)
poetry run python main.py process-docling /pfad/zur/rechnung.pdf
```

**Nutzung als Plugin (im `llkjj_backend`):**

```python
from llkjj_ml import GeminiDirectProcessor, ProcessingResult
from llkjj_ml.config import MLSettings

# Eigene Konfiguration erstellen (optional)
settings = MLSettings(google_api_key="...")

# Prozessor instanziieren
processor = GeminiDirectProcessor(settings)

try:
    # PDF verarbeiten
    result: ProcessingResult = processor.process_pdf_gemini_first("rechnung.pdf")

    # Mit den Ergebnissen arbeiten
    print(result.get_summary())

except RuntimeError as e:
    print(f"Verarbeitung fehlgeschlagen: {e}")
```

## 5. Konfiguration

Die Pipeline wird über Umgebungsvariablen konfiguriert, die in einer `.env`-Datei im Projekt-Root gespeichert werden können. Die wichtigste Variable ist `GOOGLE_API_KEY`. Alle verfügbaren Variablen sind in `src/config.py` in der `Config`-Klasse dokumentiert.

```

---

### 3. `Architektur.md`

Dieses Dokument beschreibt die übergeordnete Architektur und die Interaktion der Komponenten. Es ist ideal für das Onboarding neuer Entwickler oder um strategische Entscheidungen zu treffen.

```markdown
# LLKJJ ML-Pipeline - Architekturdokument

**Version: 4.0 (Gemini-First)**

## 1. Architektonische Vision

Das LLKJJ ML-System ist als **hochgradig modulares, eigenständiges Plugin** konzipiert, das von einem zentralen `llkjj_backend` konsumiert wird. Die Architektur folgt strengen Prinzipien der **Trennung von Verantwortlichkeiten (SoC)** und **losen Kopplung**, um maximale Wartbarkeit, Testbarkeit und zukünftige Erweiterbarkeit zu gewährleisten.

Die Kernphilosophie ist "Intelligenz als austauschbarer Service". Aktuell ist **Google Gemini** die primäre Intelligenz-Engine, die Architektur ist jedoch darauf ausgelegt, diese nahtlos durch eine zukünftige, **selbst trainierte spaCy/RAG-Lösung** zu ersetzen.

## 2. Systemübersicht & Komponenten-Interaktion

Das System ist in klar definierte Schichten und Komponenten unterteilt.

```mermaid
graph TD
    subgraph llkjj_backend [API Backend (FastAPI)]
        A[API Endpunkte </br> z.B. /documents/upload]
    end

    subgraph llkjj_ml [ML Plugin (Dieses Projekt)]
        B(Standard-Prozessor </br> GeminiDirectProcessor)
        C(Alternativ-Prozessor </br> DoclingAlternativeProcessor)

        subgraph Core Services
            D[SKR03Manager </br> Regelwerk & Kontenplan]
            E[QualityAssessor </br> Konfidenz & Qualität]
            F[ResourceManager </br> Singleton für Modelle]
        end

        subgraph KI Engines
            G[Gemini API Client]
            H[RAG-System </br> (ChromaDB + Embedding Model)]
            I[Docling Engine]
        end

        subgraph Datenmodelle [Datenmodelle (Pydantic)]
            J(ProcessingResult </br> Stabiler Datenvertrag)
            K(Gemini Schemas)
            L(Interne Modelle)
        end
    end

    A -->|ruft auf| B
    B --> G
    B --> D
    B --> H
    B --> E
    B --> J

    A -.->|ruft optional auf| C
    C --> I
    C --> D
    C --> H
    C --> E
    C --> J

    B -.-> F
    H -.-> F
    I -.-> F
```

## 3. Detaillierte Beschreibung der Komponenten

### 3.1. Prozessoren (Die Orchestratoren)

- **`GeminiDirectProcessor` (Standard):**

  - **Verantwortung:** Steuert den gesamten "Gemini-First"-Workflow.
  - **Interaktion:** Kommuniziert direkt mit der Gemini-API, dem `SKR03Manager` zur Verfeinerung, dem `RAG-System` zur Anreicherung und dem `QualityAssessor`.
  - **Output:** Erzeugt ein `ProcessingResult`-Objekt.
- **`DoclingAlternativeProcessor` (Optional):**

  - **Verantwortung:** Steuert den alternativen, OCR-basierten Workflow.
  - **Interaktion:** Nutzt intern den alten `UnifiedProcessor`, der auf `Docling` aufbaut.

### 3.2. Core Services (Die Helfer)

- **`ResourceManager`:**

  - **Pattern:** Singleton.
  - **Verantwortung:** Verhindert das mehrfache Laden von ressourcenintensiven Modellen wie `SentenceTransformer` oder `DoclingProcessor`. Bietet explizite `cleanup`-Methoden für ein sauberes Speichermanagement.
- **`SKR03Manager`:**

  - **Verantwortung:** Kapselt die gesamte Logik der deutschen SKR03-Buchhaltung. Lädt und interpretiert das YAML-Regelwerk und den CSV-Kontenplan. Bietet eine einfache Methode `klassifiziere_artikel`.
- **`QualityAssessor`:**

  - **Verantwortung:** Berechnet einen objektiven, gewichteten Konfidenz-Score für jedes Verarbeitungsergebnis. Entkoppelt die Qualitätsbewertung von der eigentlichen Extraktionslogik.

### 3.3. KI-Engines (Die "Gehirne")

- **Gemini API Client:**

  - **Verantwortung:** Schnittstelle zur Google Cloud. Sendet die PDF und den Prompt, empfängt das strukturierte JSON.
- **RAG-System (ChromaDB + Embedding Model):**

  - **Verantwortung:** Das "Langzeitgedächtnis" der Pipeline.
  - **Ingestion:** Speichert Vektoren von validierten Rechnungspositionen.
  - **Retrieval:** Findet semantisch ähnliche Einträge, um die KI-Vorschläge zu validieren und zu verfeinern.
  - **Feedback-Loop:** Wird durch Benutzerkorrekturen kontinuierlich verbessert.

### 3.4. Datenmodelle (Das Fundament)

- **`ProcessingResult`:**
  - **Rolle:** **Der öffentliche Datenvertrag.** Dies ist die einzige Datenstruktur, die das Plugin nach außen gibt. Ihre Stabilität ist garantiert.
- **Gemini-Schemas:**
  - **Rolle:** **Der interne Datenvertrag mit der KI.** Pydantic-Modelle, die die erwartete JSON-Struktur von Gemini validieren und so die Robustheit der Pipeline sicherstellen.

## 4. Design-Entscheidungen & Begründungen

- **Warum "Gemini-First"?**

  - **Effizienz:** Ein einziger API-Aufruf ersetzt einen mehrstufigen Prozess (OCR -> Tabellenerkennung -> NLP -> Klassifizierung), was die Latenz und Komplexität drastisch reduziert.
  - **Qualität:** Moderne multimodale LLMs übertreffen oft traditionelle OCR/Template-Methoden bei unstrukturierten Dokumenten.
  - **Strategie:** Ermöglicht die schnelle Erfassung hochwertiger, annotierter Daten, um die zukünftigen lokalen Modelle zu trainieren.
- **Warum kein Fallback?**

  - **Robustheit und Wartbarkeit:** Ein automatischer Fallback kann Fehler in der primären Pipeline (Gemini) verschleiern. Indem wir bei einem Fehler explizit scheitern, erzwingen wir die Analyse und Behebung des Kernproblems (z.B. ein fehlerhafter Prompt, ein API-Problem), anstatt einen stillen, potenziell minderwertigen Fallback-Pfad zu gehen.
- **Warum ein `ResourceManager`?**

  - **Speicher-Effizienz:** KI-Modelle belegen oft mehrere Gigabyte im RAM. Ein Singleton stellt sicher, dass jedes Modell nur einmal geladen wird, was für den Betrieb in ressourcenbeschränkten Umgebungen (wie Docker-Containern) unerlässlich ist.
- **Warum ein `ProcessingResult`-Modell?**

  - **Stabile API:** Es entkoppelt die internen Prozesse von der externen Schnittstelle. Wir können die gesamte interne Funktionsweise ändern, solange das finale `ProcessingResult`-Objekt konsistent bleibt. Dies ist der Kern einer guten Plugin-Architektur.

```

```
