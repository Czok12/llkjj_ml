# LLKJJ ML Service Package 🚀

Eigenständiges ML-Service-Package für deutsche Elektrohandwerk-Rechnungen mit KI-gestützter PDF-Verarbeitung und SKR03-Klassifizierung.

## 📋 Übersicht

Das **LLKJJ ML Service Package** ist eine vollständig eigenständige Python-Bibliothek, die aus dem größeren LLKJJ-Buchhaltungssystem extrahiert wurde. Es bietet hochperformante PDF-Verarbeitung mit modernster KI-Technologie, speziell optimiert für deutsche Elektrohandwerk-Rechnungen.

### 🎯 Hauptfunktionen

- **PDF-Extraktion**: IBMs Docling 2.44.0 mit TableFormer KI
- **Deutsche NLP**: spaCy-optimierte Textverarbeitung
- **KI-Enhancement**: Google Gemini 2.5 Pro Integration
- **SKR03-Klassifizierung**: Automatische Buchhaltungskonten-Zuordnung
- **Vektordatenbank**: ChromaDB für intelligente Klassifizierung
- **Performance**: Caching und GPU-Beschleunigung

### ✨ Besonderheiten

- 🇩🇪 **Deutsche Optimierung**: Speziell für deutsche Elektrobranche entwickelt
- 🔌 **Eigenständig**: Null externe Abhängigkeiten zu anderen Systemkomponenten
- ⚡ **High Performance**: GPU-beschleunigte Verarbeitung mit MPS/CUDA
- 🧪 **Getestet**: Umfassende Test-Suite mit 21+ Tests
- 🔧 **Konfigurierbar**: Vollständig über Umgebungsvariablen konfigurierbar

## 🚀 Quick Start

### Installation

```bash
# Repository klonen
git clone <repository-url>
cd llkjj_ml

# Dependencies installieren
poetry install
```

### Erste Schritte

```python
from ml_service import MLProcessor, process_pdf_simple

# Einfache Verwendung
result = process_pdf_simple("rechnung.pdf")
print(f"Extrahiert: {len(result.skr03_classifications)} Klassifizierungen")

# Erweiterte Verwendung
from ml_service import MLSettings

settings = MLSettings(cache_enabled=True)
with MLProcessor(settings) as processor:
    result = processor.process_pdf("rechnung.pdf")
    print(result.get_summary())
```

### CLI-Interface

```bash
# PDF verarbeiten
poetry run python -m ml_service process rechnung.pdf

# Konfiguration prüfen
poetry run python -m ml_service validate-config

# Capabilities testen
poetry run python -m ml_service test-capabilities

# Performance-Benchmark
poetry run python -m ml_service benchmark rechnung.pdf --runs 5
```

## 📁 Paket-Struktur

```
ml_service/
├── __init__.py         # Public API (MLProcessor, MLSettings, ProcessingResult)
├── __main__.py         # CLI Entry Point
├── cli.py              # Kommandozeilen-Interface
├── config.py           # MLSettings (Eigenständige Konfiguration)
├── processor.py        # MLProcessor (Kern-Verarbeitung)
└── tests/              # Test-Suite
    ├── conftest.py     # Pytest-Konfiguration
    └── test_ml_service.py  # Umfassende Tests (21 Tests)
```

## ⚙️ Konfiguration

Das Package ist vollständig über Umgebungsvariablen konfigurierbar:

```bash
# Gemini AI Configuration
export ML_GEMINI_API_KEY="your-api-key"
export ML_GEMINI_MODEL="gemini-2.5-flash"

# Cache Configuration
export ML_CACHE_ENABLED="true"
export ML_DATA_PATH="./data"
export ML_VECTOR_DB_PATH="./data/vectors"

# Performance Settings
export ML_USE_GPU="true"
export ML_GERMAN_OPTIMIZED="true"
```

### MLSettings Klasse

```python
from ml_service import MLSettings

# Standardkonfiguration
settings = MLSettings()

# Eigene Konfiguration
settings = MLSettings(
    gemini_model="gemini-2.5-pro",
    cache_enabled=False,
    use_gpu=True
)

# Validierung
validation = settings.validate_configuration()
print(f"Konfiguration gültig: {validation['valid']}")
```

## 🔧 API-Referenz

### MLProcessor

```python
from ml_service import MLProcessor, MLSettings

# Initialisierung
processor = MLProcessor(settings=MLSettings())

# PDF verarbeiten
result = processor.process_pdf("path/to/invoice.pdf")

# Capabilities abfragen
capabilities = processor.get_capabilities()

# Context Manager (empfohlen)
with MLProcessor() as processor:
    result = processor.process_pdf("invoice.pdf")
```

### ProcessingResult

```python
# Zugriff auf Ergebnisse
print(f"PDF: {result.pdf_path}")
print(f"Qualität: {result.extraction_quality}")
print(f"Konfidenz: {result.confidence_score}")
print(f"Zeit: {result.processing_time_ms}ms")

# Rechnungsdaten
invoice_data = result.invoice_data
print(f"Rechnungsnummer: {invoice_data.get('invoice_number')}")

# SKR03-Klassifizierungen
for classification in result.skr03_classifications:
    print(f"SKR03 {classification['skr03_code']}: {classification['confidence']}%")

# Zusammenfassung
summary = result.get_summary()
print(summary)
```

### Convenience Functions

```python
from ml_service import process_pdf_simple

# Einfachste Verwendung
result = process_pdf_simple("rechnung.pdf")
```

## 🧪 Testing

```bash
# Alle Tests ausführen
poetry run pytest ml_service/tests/ -v

# Spezifische Test-Kategorien
poetry run pytest ml_service/tests/test_ml_service.py::TestMLSettings -v
poetry run pytest ml_service/tests/test_ml_service.py::TestMLProcessor -v

# Test-Coverage
poetry run pytest ml_service/tests/ --cov=ml_service --cov-report=html
```

### Test-Kategorien

- **TestMLSettings** (4 Tests): Konfigurationsvalidierung
- **TestMLProcessor** (11 Tests): Kern-Verarbeitungsfunktionalität
- **TestProcessingResult** (2 Tests): Ergebnis-Datenstrukturen
- **TestConvenienceFunctions** (2 Tests): Einfache API-Funktionen
- **TestIntegration** (2 Tests): End-to-End-Integration

## 📊 Performance

### Benchmark-Ergebnisse

```bash
poetry run python -m ml_service benchmark test_pdfs/Sonepar_test3.pdf --runs 3
```

Typische Performance (MacBook Pro M2):
- **PDF-Verarbeitung**: ~30-35 Sekunden
- **OCR-Extraktion**: ~9-12 Sekunden
- **KI-Enhancement**: ~20-25 Sekunden
- **SKR03-Klassifizierung**: ~1-2 Sekunden
- **Qualitäts-Score**: 60-80% (abhängig von PDF-Qualität)

### Optimierungen

- ⚡ **GPU-Beschleunigung**: MPS (macOS) / CUDA (Linux)
- 💾 **Intelligentes Caching**: Wiederverwendung von Modell-Outputs
- 🧠 **Singleton Pattern**: Einmalige Modell-Initialisierung
- 🔄 **Batch-Processing**: Parallele Verarbeitung mehrerer PDFs

## 🛠️ Entwicklung

### Development Setup

```bash
# Dev-Dependencies installieren
poetry install --with dev

# Pre-commit hooks
pre-commit install

# Code Quality
poetry run ruff check .
poetry run black .
poetry run mypy .
```

### Package als Bibliothek verwenden

```python
# In anderem Projekt
from ml_service import MLProcessor, MLSettings, process_pdf_simple

# Eigenständige Konfiguration
settings = MLSettings(
    gemini_api_key="your-key",
    data_path="./custom-data"
)

processor = MLProcessor(settings)
result = processor.process_pdf("document.pdf")
```

## 🔄 Integration

### In bestehende Projekte

```python
# Einfache Integration
import ml_service

# PDF verarbeiten
result = ml_service.process_pdf_simple("invoice.pdf")

# Strukturierte Daten extrahieren
invoice_data = result.invoice_data
classifications = result.skr03_classifications
```

### Als Microservice

```python
# FastAPI Integration (Beispiel)
from fastapi import FastAPI, UploadFile
from ml_service import process_pdf_simple

app = FastAPI()

@app.post("/process-pdf")
async def process_pdf_endpoint(file: UploadFile):
    # Temporäre Datei speichern
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    # ML Service verwenden
    result = process_pdf_simple(temp_path)

    return {
        "pdf_path": result.pdf_path,
        "quality": result.extraction_quality,
        "confidence": result.confidence_score,
        "invoice_data": result.invoice_data,
        "classifications": result.skr03_classifications
    }
```

## 📈 Qualitätsmetriken

### Extraktion-Qualität

- **Header-Vollständigkeit**: Rechnungskopf-Erkennung
- **Artikel-Vollständigkeit**: Positionserkennung
- **Betrags-Konsistenz**: Preisvalidierung
- **Elektrotechnik-Optimierung**: Branchen-spezifische Patterns
- **Lieferanten-Erkennung**: Supplier-Identifikation

### SKR03-Klassifizierung

- **Konfidenz-Score**: 0-100% pro Klassifizierung
- **RAG-Nutzung**: Vector-Database-basierte Klassifizierung
- **Fallback-Strategien**: Rule-based und Keyword-basierte Klassifizierung

## 🚨 Troubleshooting

### Häufige Probleme

**Import-Fehler bei Legacy-Komponenten**
```python
# Falls src/ Module nicht verfügbar
# Package funktioniert mit vereinfachten Fallbacks
settings = MLSettings(use_legacy_fallback=False)
```

**GPU-Probleme**
```bash
# GPU-Support prüfen
poetry run python -c "import torch; print(f'MPS: {torch.backends.mps.is_available()}')"

# CPU-only Modus
export ML_USE_GPU="false"
```

**Gemini API-Fehler**
```bash
# API-Key validieren
poetry run python -m ml_service validate-config

# Alternative: Ohne KI-Enhancement
export ML_GEMINI_API_KEY=""
```

### Debug-Modus

```bash
# Verbose Logging
poetry run python -m ml_service process --verbose document.pdf

# Test-Capabilities
poetry run python -m ml_service test-capabilities
```

## 📝 Changelog

### Version 3.0.0 (Eigenständige Implementierung)
- ✅ Vollständige Unabhängigkeit von externen Systemkomponenten
- ✅ MLSettings-basierte Konfiguration
- ✅ Umfassende Test-Suite (21+ Tests)
- ✅ CLI-Interface mit 4 Kommandos
- ✅ ProcessingResult-Datenmodell
- ✅ Performance-Optimierungen und Caching

### Version 2.1.0 (Post-Konsolidierung)
- ✅ KISS-Prinzip angewandt: Konsolidierung auf 2 Kern-Services
- ✅ Blackbox-Transformation mit öffentlicher API
- ✅ Legacy-Kompatibilität mit Fallback-Implementierung

## 🤝 Beitragen

1. Fork das Repository
2. Feature-Branch erstellen: `git checkout -b feature/amazing-feature`
3. Tests hinzufügen: `poetry run pytest`
4. Code committen: `git commit -m 'Add amazing feature'`
5. Branch pushen: `git push origin feature/amazing-feature`
6. Pull Request erstellen

## 📄 Lizenz

Dieses Projekt ist unter der MIT-Lizenz lizenziert. Siehe `LICENSE` für Details.

## 🙏 Danksagungen

- **IBM Docling**: Hochperformante PDF-Verarbeitung
- **Google Gemini**: KI-Enhanced Content Extraction
- **spaCy**: Deutsche NLP-Optimierung
- **ChromaDB**: Vector Database für intelligente Klassifizierung

---

**LLKJJ ML Service** - Deutscher Elektrohandwerk trifft moderne KI-Technologie 🇩🇪⚡🤖
