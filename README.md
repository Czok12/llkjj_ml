
# Umfassender Projektkontext & Entwicklungsrichtlinien für KI-Agenten: Das LLKJJ Projekt

**Version: Master Context 1.0**

Dieses Dokument ist die **einzige und maßgebliche Quelle der Wahrheit (Single Source of Truth)** für die KI-gestützte Entwicklung am LLKJJ-Projekt. Es fasst die Vision, Architektur, technische Blaupause und strategische Ausrichtung zusammen. Halte dich strikt an die hier dargelegten Prinzipien, Architekturen und Anweisungen.

## 1. Vision & Manifest (Das "Warum")

Das Ziel des LLKJJ-Projekts ist die Schaffung einer persönlichen, hochautomatisierten und intelligenten Buchhaltungs- und Dokumentenmanagement-Engine. Es ist als private, überlegene Alternative zu Standardlösungen konzipiert und exakt auf die Bedürfnisse eines deutschen Elektrohandwerk-Betriebs zugeschnitten.

Es geht nicht darum, eine weitere App zu bauen, sondern eine **maßgeschneiderte, hochintegrierte und intelligente Engine**, die den administrativen Aufwand auf ein absolutes Minimum reduziert.

## 2. Leitprinzipien & Strategische Roadmap

### 2.1. Leitprinzipien

1. **Autonomie:** Das System soll lernen, Kontexte verstehen und eigenständig korrekte Entscheidungen treffen, um manuelle Eingriffe zu minimieren.
2. **Präzision:** Maximale Genauigkeit bei der Datenextraktion und SKR03-Klassifizierung ist das oberste Ziel (>92% Genauigkeit).
3. **Sicherheit:** Da es um sensible Finanzdaten geht, hat die Sicherheit auf allen Ebenen – von der Code-Analyse bis zur Datenspeicherung – höchste Priorität.
4. **Wartbarkeit:** Eine saubere, entkoppelte Architektur (KISS-Prinzip) ist entscheidend, um das System langfristig pflegen und erweitern zu können.

### 2.2. Strategische Roadmap (KI-Entwicklung)

Die Entwicklung der KI-Fähigkeiten folgt einer zweistufigen Strategie:

1. **Phase 1 (Aktuell): "Gemini-First"-Pipeline**

   * **Ansatz:** Wir nutzen Google's Gemini-Modell (`gemini-2.5d-flash` oder besser) als primäres "Gehirn" für die direkte PDF-Analyse und Datenerfassung. Ein einziger API-Aufruf ersetzt einen komplexen, mehrstufigen Prozess.
   * **Ziel:** Schnelle, qualitativ hochwertige Ergebnisse erzielen und gleichzeitig hochwertige Trainingsdaten (validierte Extraktionen, SKR03-Kontierungen, spaCy-Annotationen) für die nächste Phase sammeln.
2. **Phase 2 (Zukunft): Lokale, autonome KI-Lösung**

   * **Ansatz:** Nahtloser Übergang zu einer von externen APIs unabhängigen Lösung.
   * **Ziel:** Einsatz von selbst trainierten **spaCy-Modellen (NER, TextCat)** und einem lokalen **RAG-System (ChromaDB)**, das mit den in Phase 1 gesammelten Daten aufgebaut wird. Deine Arbeit heute bereitet diesen Übergang vor.

## 3. Gesamtarchitektur (Das "Wie" – High Level)

Die Systemarchitektur basiert auf einer strikten Trennung zwischen einem zentralen Backend-Kern und spezialisierten, austauschbaren Plugins.

```
+---------------------------------------------------------------------------------+
|                                 LLKJJ-System                                    |
|                                                                                 |
|  +---------------------------------------------------------------------------+  |
|  |                Workspace 1: Das Backend (Core-System)                     |  |
|  |---------------------------------------------------------------------------|  |
|  | [FastAPI] -> API Routers -> [Service Schicht] -> [Repository] -> PostgreSQL |  |
|  |      ^                (Business Logic)      (SQLAlchemy)      (Struktur)  |  |
|  |      |                                                                    |  |
|  |      +------------------(Definierte Schnittstellen)-----------------------+  |
|  +------------------------------------|----------------------------------------+  |
|                                       |                                         |
|      +--------------------------------+--------------------------------+        |
|      |                                |                                |        |
|      v                                v                                v        |
| +-------------------------+  +-------------------------+  +-------------------------+ |
| | Workspace 2:            |  | Workspace 3:            |  | Workspace 4:            | |
| | ML-Pipeline             |  | E-Invoice-System        |  | Export-System           | |
| | (`llkjj_ml_plugin`)     |  | (`llkjj_efaktura`)      |  | (`llkjj_export`)        | |
| |-------------------------|  |-------------------------|  |-------------------------| |
| | - OCR (Docling)         |  | - XRechnung / UBL XML   |  | - DATEV CSV Export      | |
| | - Gemini-First (aktiv)  |  | - PDF/A-3 Generierung   |  | - JSON / Standard CSV   | |
| | - RAG (ChromaDB)        |  | - KoSIT-Validierung     |  | - Universelle Schemas   | |
| +-------------------------+  +-------------------------+  +-------------------------+ |
+-------------------------------------------------------------------------------------+
```

- **Workspace 1: Das Backend (Core-System):** Der zentrale Orchestrator. Stellt die REST-API bereit, verwaltet Benutzer und Daten in PostgreSQL und kommuniziert mit den Plugins über definierte Schnittstellen.
- **Workspace 2: Die ML-Pipeline (`llkjj_ml_plugin`):** Das "Gehirn". Verarbeitet eine PDF-Datei und gibt ein strukturiertes, klassifiziertes Ergebnis zurück. **Dies ist dein primärer Fokus.**
- **Workspace 3: Das E-Invoice-System (`llkjj_efaktura`):** Spezialist für die Erstellung gesetzeskonformer E-Rechnungen (XRechnung, Factur-X).
- **Workspace 4: Das Export-System (`llkjj_export`):** Stellt Daten in verschiedenen Formaten (z.B. DATEV-CSV) für externe Systeme bereit.

## 4. Detailarchitektur & Workflow des ML-Plugins (`llkjj_ml_plugin`)

### 4.1. Interne Komponenten & Interaktion

Das ML-Plugin ist als hochgradig modulares, eigenständiges Paket konzipiert.

```mermaid
graph TD
    subgraph llkjj_backend [API Backend]
        A[API Endpunkt]
    end

    subgraph llkjj_ml [ML Plugin (Dein Fokus)]
        B(Standard-Prozessor </br> GeminiDirectProcessor)
        C(Alternativ-Prozessor </br> DoclingAlternativeProcessor)

        subgraph Core Services
            D[SKR03Manager]
            E[QualityAssessor]
            F[ResourceManager </br> Singleton für Modelle]
        end

        subgraph KI Engines
            G[Gemini API Client]
            H[RAG-System </br> (ChromaDB + Embedding Model)]
            I[Docling Engine]
        end

        subgraph Datenmodelle [Pydantic]
            J(ProcessingResult </br> **Öffentlicher Datenvertrag**)
            K(Gemini Schemas </br> Interner KI-Vertrag)
        end
    end

    A -->|ruft auf| B
    B --> G
    B --> D
    B --> H
    B --> E
    B --> J

    B -.-> F
    H -.-> F
    I -.-> F
```

### 4.2. Haupt-Workflow (Gemini-First)

Der Standardprozess wird durch `GeminiDirectProcessor` gesteuert:

1. **Eingabe & Validierung:** Eine PDF-Datei wird an `process_pdf_gemini_first` übergeben und geprüft.
2. **Gemini-Analyse:** Die PDF wird an die Google Gemini API gesendet. Ein spezifischer Prompt instruiert das Modell, alle Daten in einem strukturierten JSON zu extrahieren.
3. **Schema-Validierung:** Die JSON-Antwort von Gemini wird strikt gegen das Pydantic-Schema `GeminiExtractionResult` validiert. Dies ist ein kritischer Qualitätssicherungsschritt.
4. **RAG-Anreicherung:** Die von Gemini vorgeschlagenen SKR03-Konten werden durch das lokale RAG-System (ChromaDB) verfeinert und validiert, indem ähnliche, bereits bestätigte Buchungen gesucht werden.
5. **Qualitätsbewertung:** Der `QualityAssessor` berechnet einen finalen Konfidenz-Score.
6. **Trainingsdaten-Generierung:** Es werden korrigierte spaCy-Annotationen erzeugt und validierte Daten in ChromaDB gespeichert, um den Feedback-Loop zu schließen und das System kontinuierlich zu verbessern.
7. **Ausgabe:** Ein Pydantic-validiertes **`ProcessingResult`-Objekt** wird zurückgegeben. Dies ist die einzige garantierte Schnittstelle nach außen.

### 4.3. Wichtige technische Details & Design-Entscheidungen

- **Datenvertrag:** Das `ProcessingResult`-Modell ist der **stabile, öffentliche Datenvertrag** des ML-Plugins. Interne Änderungen dürfen dieses Schema nicht brechen.
- **Fehlerbehandlung:** Wenn ein Schritt in der Gemini-Pipeline fehlschlägt (API-Fehler, Validierungsfehler), wird der Prozess mit einer `RuntimeError` abgebrochen. **Es gibt keinen automatischen Fallback auf die Docling-Methode.** Dies ist eine bewusste Entscheidung, um Fehler in der primären Pipeline sichtbar zu machen und zu beheben, anstatt sie zu verschleiern.
- **Ressourcen-Management:** Ein `ResourceManager` (Singleton-Pattern) stellt sicher, dass ressourcenintensive Modelle (z.B. `SentenceTransformer`) nur einmal geladen werden, um den Speicherverbrauch zu minimieren.
- **Datenbanken:**
  - **PostgreSQL (im Backend):** Speichert alle strukturierten Geschäftsdaten. Der Zugriff erfolgt via SQLAlchemy 2.0 (async).
  - **ChromaDB (im ML-Plugin):** Dient als Vektordatenbank für das RAG-System und das "Langzeitgedächtnis".
- **Sicherheit:**
  - **Authentifizierung:** Basiert auf JWT mit Argon2-gehashten Passwörtern.
  - **Auditing:** Eine `SecurityAuditor`-Klasse (`Bandit`, `Safety`) ist Teil der CI-Pipeline.
  - **Secret Management:** Sensible Schlüssel werden "at rest" mit AES-256 verschlüsselt.

## 5. Anweisungen für die KI-gestützte Entwicklung (Deine Mission)

### 5.1. Generelle Anweisungen ("Golden Rules")

1. **Sprache:** Der gesamte Code, alle Kommentare und die Dokumentation sind auf **Deutsch**, um die Domänensprache (deutsche Buchhaltung) konsistent abzubilden.
2. **Architektur respektieren:** Halte dich strikt an die Trennung von Backend-Kern und Plugins. Keine direkten Abhängigkeiten zwischen den Workspaces, außer über die definierten Schnittstellen (`ProcessingResult` etc.).
3. **Sicherheit zuerst:** Jeder neue Code muss sicher sein. Vermeide hartcodierte Secrets, validiere alle Eingaben und nutze die bereitgestellten Sicherheits-Utilities.
4. **Typsicherheit:** Der gesamte Code muss `MyPy --strict` konform sein.
5. **Testen:** Für neue Funktionen müssen entsprechende `pytest`-Tests geschrieben werden.
6. **Fokus:** Deine Hauptaufgabe ist es, die "Gemini-First"-Pipeline zu verbessern und gleichzeitig den Übergang zu Phase 2 (lokale Modelle) vorzubereiten. Die Qualität und Konsistenz der gesammelten Trainingsdaten ist wichtiger als kurzfristige Performance-Hacks.

### 5.2. Anweisungen für Deinen Workspace: Die ML-Pipeline (`llkjj_ml_plugin`)

- **Dein Kernziel:** Eine PDF-Datei so präzise wie möglich in ein strukturiertes, SKR03-klassifiziertes `ProcessingResult`-Objekt zu verwandeln.
- **Dein Kontext:** Du bist ein Experte für Machine Learning, NLP (spaCy, Transformers), Computer Vision, Vektordatenbanken (ChromaDB) und die Integration von LLMs wie Gemini.
- **Deine Hauptaufgaben:** Verbesserung der Extraktions-Pipeline (`GeminiDirectProcessor`), Optimierung der Klassifizierungsmodelle, Implementierung von Feature Engineering und Stärkung des RAG-Systems.
- **Deine Interaktionen und Grenzen:**
  - Deine **einzige Schnittstelle** zum Backend ist das `ProcessingResult`-Schema.
  - Du darfst **keine direkten Abhängigkeiten** zum Backend-Code (`Workspace 1`) oder dessen PostgreSQL-Datenbank haben. Du agierst als eigenständige Blackbox.
  - Deine eigene Persistenz (z.B. für `ChromaDB`) verwaltest du in deinem eigenen Verzeichnis (z.B. `data/vectors`).

---
