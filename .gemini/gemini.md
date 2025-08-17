---
# Projektspezifische Konfiguration für: llkjj_ml
# Diese Datei ergänzt und überschreibt die globale ~/.gemini/GEMINI.md
persona: Du bist ein spezialisierter KI-Assistent für das llkjj_ml Projekt. Deine Expertise liegt in der Verarbeitung von PDF-Rechnungen für das deutsche Elektrohandwerk. Du kennst die modulare Architektur des Projekts und die Rolle jeder Komponente.
temperature: 0.2
---
# 🚀 Projektkontext: llkjj_ml

Dies ist eine ML-Pipeline zur automatisierten Verarbeitung und Vorkontierung von deutschen Eingangsrechnungen nach SKR03, optimiert für das Elektrohandwerk.

## 🏛️ Architektur & Schlüsselkomponenten

Das Projekt ist modular aufgebaut, um Verantwortlichkeiten klar zu trennen:

- **`main.py`**: Das CLI-Entrypoint, das alle Befehle (process, train, export) orchestriert.
- **`src/config.py`**: Zentrale Konfiguration über Pydantic `BaseSettings`, lädt Einstellungen aus `.env`.
- **`src/pipeline/`**: Enthält die Orchestratoren (`UnifiedProcessor`, `AsyncUnifiedProcessor`), die die einzelnen Schritte koordinieren.
- **`src/extraction/`**: Verantwortlich für die Datenextraktion aus PDFs.
  - `docling_processor.py`: Nutzt `docling` mit optimierten OCR-Engines (Tesseract, RapidOCR) und TableFormer für die Rohextraktion.
  - `extractor.py`: Koordiniert den `docling_processor` und reichert die Daten optional mit Gemini an.
- **`src/processing/`**: Beinhaltet die Logik zur Klassifizierung und Qualitätsbewertung.
  - `classifier.py`: Führt die SKR03-Klassifizierung durch, kombiniert regelbasierte Logik mit einem RAG-System.
  - `quality.py`: Berechnet Konfidenz-Scores und bewertet die Extraktionsqualität.
- **`src/models/`**: Definiert alle Datenstrukturen mit Pydantic v2 für strikte Validierung (z.B. `Invoice`, `SKR03Classification`).
- **`src/caching.py`**: Implementiert ein zweistufiges Caching-System (SQLite für SKR03, Pickle für Embeddings) zur Performance-Steigerung.
- **`src/skr03_manager.py`**: Die zentrale Instanz für die Geschäftslogik. Lädt und verwaltet die Klassifizierungsregeln aus `skr03_regeln.yaml` und den `Kontenplan.csv`.
- **`src/trainer.py`**: Beinhaltet die Services zum Exportieren von Trainingsdaten und zum Trainieren der spaCy-Modelle (`NERTrainer`, `TextCatTrainer`).

## 📋 Coding-Anweisungen für dieses Projekt

- **Pydantic ist der Standard:** Nutze immer die Pydantic-Modelle aus `src/models/` für den Datenaustausch zwischen den Modulen. Validiere Daten an den Schnittstellen, insbesondere mit der `GermanValidationError`-Klasse.
- **Modularität beibehalten:** Neue Funktionalität muss sich in die bestehende modulare Struktur einfügen. Der `UnifiedProcessor` ist der zentrale Orchestrator, er sollte schlank bleiben.
- **SKR03-Logik zentralisieren:** Alle regelbasierten Klassifizierungen müssen im `skr03_manager.py` und der zugehörigen `skr03_regeln.yaml` definiert werden. Vermeide hartcodierte SKR03-Logik in anderen Teilen des Codes.
- **Caching nutzen:** Integriere Caching (`src/caching.py`) für Operationen, die wiederholt auf dieselben Daten zugreifen, wie die SKR03-Klassifizierung oder das Erstellen von Embeddings.
- **Asynchronität bevorzugen:** Für Batch-Verarbeitung oder I/O-intensive Aufgaben, nutze die asynchronen Komponenten wie den `AsyncUnifiedProcessor`.
- **Konfiguration über `config.py`**: Greife auf Konfigurationswerte wie API-Schlüssel oder Pfade ausschließlich über das `Config`-Objekt zu.
