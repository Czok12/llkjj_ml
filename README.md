# LLKJJ ML Pipeline

## 1. Projekt-Übersicht

Dieses Projekt ist eine ML-gestützte Pipeline zur automatisierten Verarbeitung von PDF-Rechnungen für das Elektrotechnik-Handwerk. Der Fokus liegt auf der Extraktion von Rechnungsdaten und der anschließenden Klassifizierung der Positionen gemäß dem deutschen SKR03-Kontenrahmen.

Die Architektur ist modular und kommandozeilen-gesteuert. Die Logik ist in spezialisierte Komponenten für Extraktion, Klassifizierung und Datenhaltung aufgeteilt.

## 2. Kerntechnologien

- **Sprache:** Python 3.10+
- **Abhängigkeitsmanagement:** Poetry
- **Datenvalidierung:** Pydantic v2
- **CLI:** `argparse`
- **PDF-Extraktion:** `docling`, `PyMuPDF`
- **KI-Extraktion/Anreicherung:** Google Gemini
- **Vektordatenbank (RAG):** ChromaDB
- **Embedding-Modelle:** `sentence-transformers`
- **ML-Training (optional):** spaCy
- **Code-Qualität:** Ruff (Linting/Formatierung), Mypy (Typ-Prüfung)
- **Testing:** Pytest

## 3. Setup und Installation

1.  **Poetry installieren:** Falls noch nicht geschehen, [Poetry installieren](https://python-poetry.org/docs/#installation).
2.  **Abhängigkeiten installieren:**
    ```bash
    poetry install
    ```
3.  **Umgebungsvariablen einrichten:**
    - Kopieren Sie die Vorlagedatei: `cp .env.example .env`
    - Tragen Sie Ihren `GOOGLE_API_KEY` in die `.env`-Datei ein.

## 4. Wichtige CLI-Befehle

Alle Befehle werden mit `poetry run python main.py ...` ausgeführt.

### 4.1. Modulare Workflows

- **Workflow 1: Nur Text-Extraktion (PDF -> TXT)**
  ```bash
  poetry run python main.py workflow1 PFAD_ZUR_RECHNUNG.pdf
  ```
- **Workflow 2: Direkte Gemini-Verarbeitung (PDF -> JSON)**
  ```bash
  poetry run python main.py workflow2 PFAD_ZUR_RECHNUNG.pdf
  ```
- **Workflow 3: Text-zu-Gemini Verarbeitung (TXT -> JSON)**
  ```bash
  poetry run python main.py workflow3 PFAD_ZUR_TEXTDATEI.txt
  ```
- **Workflow 4: Komplette Pipeline (PDF -> Klassifiziertes JSON)**
  ```bash
  poetry run python main.py workflow4 PFAD_ZUR_RECHNUNG.pdf
  ```

### 4.2. Datenbank-Management

- **Datenbank initialisieren:**
  ```bash
  poetry run python main.py database init --path data/vectors
  ```
- **Statistiken anzeigen:**
  ```bash
  poetry run python main.py database stats --path data/vectors
  ```
- **Backup erstellen:**
  ```bash
  poetry run python main.py database backup --name mein_backup
  ```

### 4.3. Training (Beispiel)

- **Trainingsdaten exportieren:**
  ```bash
  poetry run python main.py export data/processed/ --output data/training/training_data.jsonl
  ```
- **Modell trainieren:**
  ```bash
  poetry run python main.py train data/training/training_data.jsonl --output output_model/
  ```

## 5. Architektur

Eine detaillierte Beschreibung der Systemarchitektur, der Komponenten und der Datenflüsse finden Sie in [docs/architecture.md](./docs/architecture.md).

## 6. Entwicklungs-Tasks

- **Linter und Formatter ausführen:**
  ```bash
  poetry run ruff check . --fix
  poetry run ruff format .
  ```
- **Typ-Prüfung ausführen:**
  ```bash
  poetry run mypy .
  ```
- **Tests ausführen:**
  ```bash
  poetry run pytest
  ```
