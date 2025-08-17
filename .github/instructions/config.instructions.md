---
applyTo: "**/*.{yaml,yml,json}"
---

# Konfigurationsdateien-Anweisungen für GitHub Copilot

## Kontext

Diese Dateien enthalten Konfigurationen für das **LLKJJ ML-Plugin**. Alle Konfigurationen müssen für deutsche Elektrohandwerks-Szenarien und Poetry-basierte Umgebungen optimiert sein.

## Konfigurationsstandards

### 1. YAML-Dateien

```yaml
# Korrekt: Deutsche Konfiguration mit Kommentaren
# LLKJJ ML Pipeline Konfiguration
pipeline:
  name: "LLKJJ Dokumentenverarbeitung"
  language: "de" # Deutsche Optimierung

  # OCR-Engine-Konfiguration
  ocr:
    primary_engine: "docling"
    fallback_engines: ["tesseract", "paddleocr"]
    language_codes: ["deu", "de"]

  # SKR03-Klassifizierung
  classification:
    confidence_threshold: 0.8
    skr03_ruleset: "data/config/skr03_regeln.yaml"
    fallback_account: "4960" # Sonstige betriebliche Aufwendungen

  # Performance-Einstellungen
  performance:
    max_processing_time_seconds: 30
    max_memory_usage_gb: 2.0
    batch_size: 5
```

### 2. JSON-Konfigurationen

```json
{
  "// Kommentar": "LLKJJ ML Pipeline Einstellungen",
  "projekt": {
    "name": "LLKJJ ML Plugin",
    "version": "2.1.0",
    "beschreibung": "KI-Pipeline für deutsche Elektrohandwerk-Buchhaltung"
  },
  "deutsche_validierung": {
    "plz_format": "^\\d{5}$",
    "ustid_format": "^DE\\d{9}$",
    "datum_format": "dd.mm.yyyy"
  },
  "skr03_konten": {
    "elektromaterial": ["4360", "4370"],
    "werkzeuge": ["4370", "4380"],
    "fahrtkosten": ["4510", "4520"]
  }
}
```

### 3. Poetry pyproject.toml

Bei Änderungen an `pyproject.toml` beachten:

```toml
[tool.poetry]
name = "llkjj-ml"
version = "2.1.0"
description = "ML-Pipeline für deutsche Elektrohandwerk-Buchhaltung"
authors = ["LLKJJ Team <team@llkjj.de>"]

[tool.poetry.dependencies]
python = "^3.10"
# KI/ML Core
docling = "^2.44.0"
spacy = "^3.7.0"
chromadb = "^0.4.0"
sentence-transformers = "^2.2.0"

# Deutsche NLP
de-core-news-sm = {url = "https://github.com/explosion/spacy-models/releases/download/de_core_news_sm-3.7.0/de_core_news_sm-3.7.0-py3-none-any.whl"}

[tool.poetry.scripts]
llkjj-process = "main:main"
llkjj-train = "src.trainer:main"

[tool.mypy]
strict = true
python_version = "3.10"
warn_return_any = true
warn_unused_configs = true

[tool.ruff]
line-length = 88
target-version = "py310"
select = ["E", "F", "W", "C90", "I"]

[tool.black]
line-length = 88
target-version = ['py310']
```

### 4. Docker/Container-Konfigurationen

```yaml
# docker-compose.yml für Entwicklung
version: "3.8"
services:
  llkjj-ml:
    build: .
    environment:
      - POETRY_VENV_IN_PROJECT=1
      - PYTHONPATH=/app
      - LANG=de_DE.UTF-8
      - LC_ALL=de_DE.UTF-8
    volumes:
      - ./data:/app/data
      - ./test_pdfs:/app/test_pdfs
    command: poetry run python main.py
```

### 5. GitHub Actions/Workflows

```yaml
# .github/workflows/test.yml
name: LLKJJ ML Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11"]

    steps:
      - uses: actions/checkout@v4
      - name: Setup Python ${{ matrix.python-version }}
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install Poetry
        uses: snok/install-poetry@v1
        with:
          virtualenvs-create: true
          virtualenvs-in-project: true

      - name: Load cached venv
        uses: actions/cache@v3
        with:
          path: .venv
          key: venv-${{ runner.os }}-${{ matrix.python-version }}-${{ hashFiles('poetry.lock') }}

      - name: Install dependencies
        run: poetry install --no-interaction --no-ansi

      - name: Install German spaCy model
        run: poetry run python -m spacy download de_core_news_sm

      - name: Run tests
        run: poetry run pytest tests/ -v --cov=src

      - name: Run type checking
        run: poetry run mypy .

      - name: Run linting
        run: poetry run ruff check .
```

## Logging-Konfiguration

```yaml
# logging_config.yaml
version: 1
disable_existing_loggers: false

formatters:
  standard:
    format: "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    datefmt: "%Y-%m-%d %H:%M:%S"

  german:
    format: "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    datefmt: "%d.%m.%Y %H:%M:%S" # Deutsches Datumsformat

handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    formatter: german
    stream: ext://sys.stdout

  file:
    class: logging.handlers.RotatingFileHandler
    level: DEBUG
    formatter: standard
    filename: logs/llkjj_ml.log
    maxBytes: 10485760 # 10MB
    backupCount: 5
    encoding: utf-8

loggers:
  src:
    level: DEBUG
    handlers: [console, file]
    propagate: false

  chromadb:
    level: WARNING
    handlers: [console, file]
    propagate: false

root:
  level: INFO
  handlers: [console, file]
```

## Environment-Konfiguration

```yaml
# config/development.yaml
environment: development
debug: true

database:
  chromadb:
    path: "data/vectors/chroma.sqlite3"
    collection_name: "llkjj_embeddings"

ml_models:
  spacy_model: "de_core_news_sm"
  sentence_transformer: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

api:
  gemini:
    model: "gemini-2.5-pro"
    temperature: 0.1
    max_tokens: 2048

processing:
  max_pdf_size_mb: 50
  supported_languages: ["de", "deu"]
  confidence_threshold: 0.8

# config/production.yaml
environment: production
debug: false

processing:
  max_processing_time: 25  # Strenger in Produktion
  confidence_threshold: 0.9  # Höhere Qualitätsanforderungen
```

## Wichtige Hinweise

### Sicherheit

- **Keine API-Keys in Konfigurationsdateien** - nutze Environment-Variablen
- **Deutsche Datenschutz-Compliance** beachten
- **Sensible Pfade** nicht in Version Control

### Performance

- **Memory-Limits** für große PDF-Verarbeitung setzen
- **Timeout-Werte** für alle externen API-Calls
- **Batch-Größen** für ChromaDB-Operationen optimieren

### Wartbarkeit

- **Versionierung** aller Konfigurationsänderungen
- **Umgebungsspezifische Werte** in separaten Dateien
- **Validierung** aller Konfigurationswerte beim Laden

---

**Tipp**: Nutze `poetry run python -c "from src.config import get_config; print(get_config())"` um Konfiguration zu validieren.
