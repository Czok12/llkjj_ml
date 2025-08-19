# LLKJJ ML Plugin - Claude Code Dokumentation

## Projektübersicht

Das LLKJJ ML Plugin ist eine spezialisierte Machine Learning Pipeline für die automatisierte Verarbeitung von PDF-Rechnungen im deutschen Elektrohandwerk. Es transformiert unstrukturierte PDF-Dokumente in strukturierte, SKR03-klassifizierte Buchhaltungsdaten.

## Architektur

### Kern-Komponenten
- **Plugin Interface** (`llkjj_ml_plugin.py`): Öffentliche API mit Blackbox-Pattern
- **Processing Pipeline** (`src/pipeline/`): Modulare Verarbeitungsstrategie
- **Data Models** (`src/models/`): Pydantic-basierte Datenstrukturen
- **Intelligence Layer** (`src/intelligence/`): KI-Klassifizierung und RAG-System

### Verarbeitungsstrategien
1. **Gemini-First Pipeline**: Primäre Strategie mit Google Gemini 2.5 API
2. **Docling Alternative**: Fallback mit lokaler OCR-Verarbeitung
3. **Unified Processor**: Orchestriert beide Ansätze

## Technologie-Stack

### Core Dependencies
```toml
docling = ">=2.44.0"          # PDF-Extraktion
pydantic = ">=2.9.2"          # Datenvalidierung
google-genai = ">=1.30.0"     # Gemini API
chromadb = ">=0.5.5"          # Vektordatenbank
sentence-transformers = ">=3.0.1"  # Embeddings
spacy = ">=3.7.6"             # NLP-Pipeline
torch = ">=2.4.1"             # ML Framework
```

### Development Tools
```toml
mypy = "^1.11.2"              # Typisierung
ruff = "^0.6.4"               # Linting
black = "^24.8.0"             # Formatierung
pytest = "^8.3.3"            # Testing
bandit = "^1.8.6"             # Sicherheitsanalyse
```

## Verwendung

### Basic Usage
```python
from llkjj_ml_plugin import MLPlugin

# Plugin initialisieren
plugin = MLPlugin()

# PDF verarbeiten
result = plugin.process_pdf("rechnung.pdf")

# Ergebnisse verwenden
for item in result.skr03_classifications:
    print(f"{item['description']} -> {item['skr03_account']}")
```

### Convenience Function
```python
from llkjj_ml_plugin import process_pdf_simple

result = process_pdf_simple("rechnung.pdf")
print(f"Konfidenz: {result.confidence_score:.1%}")
```

## Entwicklungscommands

### Umgebung Setup
```bash
poetry install                    # Dependencies installieren
poetry install --with dev        # Mit dev-dependencies
poetry install --with gpu        # GPU-Support aktivieren
```

### Code Quality
```bash
poetry run ruff check src        # Linting
poetry run mypy src              # Type checking
poetry run black src             # Code formatting
poetry run bandit -r src         # Security scan
```

### Testing
```bash
poetry run pytest               # Alle Tests
poetry run pytest -v            # Verbose output
poetry run pytest tests/test_integration_*.py  # Integration tests
poetry run pytest --cov=src     # Mit Coverage
```

### Ausführung
```bash
poetry run python main.py                    # Haupt-CLI
poetry run python -m src.cli                 # CLI-Modul
poetry run python scripts/populate_rag_system.py  # RAG-Setup
```

## Konfiguration

### Environment Variables
```env
GOOGLE_API_KEY=your_gemini_api_key
CHROMADB_HOST=localhost
CHROMADB_PORT=8000
LOG_LEVEL=INFO
```

### Config-Dateien
- `src/config.py`: Hauptkonfiguration
- `deployment/config/`: Umgebungsspezifische Configs
- `src/config/skr03_regeln.yaml`: SKR03-Klassifizierungsregeln

## Datenstrukturen

### ProcessingResult
Das einheitliche Output-Schema für alle Verarbeitungsstrategien:
```python
class ProcessingResult(BaseModel):
    pdf_path: str
    processing_timestamp: datetime
    raw_text: str
    structured_data: dict[str, Any]
    invoice_data: dict[str, Any]
    skr03_classifications: list[dict[str, Any]]
    processing_time_ms: float
    confidence_score: float
    extraction_quality: Literal["high", "medium", "low"]
```

## Sicherheit

### Implementierte Maßnahmen
- **Input Validation**: Pydantic-Schemas für alle Datenstrukturen
- **Security Scanning**: Bandit für statische Codeanalyse
- **Dependency Scanning**: Safety für bekannte Vulnerabilities
- **Type Safety**: MyPy strict mode für Typsicherheit
- **Secret Management**: Environment-basierte Konfiguration

### Security Commands
```bash
poetry run bandit -r src -f json > security_report.json
poetry run safety check
poetry run mypy src --strict
```

## Performance

### Optimierungen
- **Resource Manager**: Singleton für ML-Modell-Caching
- **Apple Silicon**: Native M1/M2 Optimierungen
- **Batch Processing**: Effiziente Massenverarbeitung
- **Memory Management**: Proaktive Speicherverwaltung

### Benchmarking
```bash
poetry run python deployment/performance_benchmark.py
poetry run python -m src.monitoring.performance_benchmarking
```

## Monitoring & Debugging

### Logging
- **Structured Logging**: JSON-Format für Production
- **Log Levels**: DEBUG, INFO, WARNING, ERROR, CRITICAL
- **Rotating Logs**: Automatische Rotation und Archivierung

### Debug Tools
```bash
# Performance Profiling
poetry run python -m cProfile main.py

# Memory Profiling
poetry run python -m memory_profiler scripts/test_memory.py

# Error Analytics
poetry run python src/monitoring/error_analytics_dashboard.py
```

## Deployment

### Production Setup
```bash
# Production Dependencies
poetry install --only main

# Production Config
cp deployment/config/production.yaml config.yaml

# Health Check
poetry run python deployment/health_service.py

# Production Manager
poetry run python deployment/production_manager.py
```

### Docker (falls verfügbar)
```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip install poetry && poetry install --only main
CMD ["poetry", "run", "python", "main.py"]
```

## Häufige Probleme

### Gemini API Errors
```bash
# API Key prüfen
echo $GOOGLE_API_KEY

# Rate Limiting testen
poetry run python src/optimization/gemini_rate_limiting.py

# Error Logs analysieren
tail -f logs/gemini_errors/gemini_error_*.log
```

### Memory Issues
```bash
# Memory Monitoring
poetry run python src/monitoring/enhanced_batch_memory_manager.py

# Resource Cleanup
poetry run python -c "from src.utils.resource_manager import ResourceManager; ResourceManager().cleanup_all()"
```

### ChromaDB Issues
```bash
# ChromaDB Reset
poetry run python scripts/reset_chromadb.py

# ChromaDB Migration
poetry run python src/database/migrations.py

# Production ChromaDB
poetry run python src/database/production_chromadb.py
```

## Erweiterungen

### Neue Features entwickeln
1. Feature in `src/features/` implementieren
2. Tests in `tests/` erstellen
3. Pipeline-Integration in `src/pipeline/`
4. Dokumentation aktualisieren

### Custom Extractors
```python
from src.features.domain_extractors import BaseDomainExtractor

class CustomExtractor(BaseDomainExtractor):
    def extract(self, text: str) -> dict:
        # Custom extraction logic
        return {"custom_field": "value"}
```

### Plugin Erweiterung
```python
class ExtendedMLPlugin(MLPlugin):
    def custom_processing(self, pdf_path: str) -> CustomResult:
        # Extended functionality
        pass
```

## Migration & Updates

### Version Updates
```bash
# Dependencies aktualisieren
poetry update

# Breaking Changes prüfen
poetry run python scripts/compatibility_check.py

# Migration ausführen
poetry run python src/database/migrations.py
```

### Datenformat-Migration
```python
# Legacy Format zu neuer ProcessingResult
from src.models.processing_result import migrate_legacy_result
new_result = migrate_legacy_result(old_result)
```

---

**Entwicklungsstatus**: Production Ready
**Letzte Aktualisierung**: August 2025
**Team**: LLKJJ ML Pipeline Team
