---
applyTo: "tests/**/*.py"
---

# Test-spezifische Anweisungen für GitHub Copilot

## Kontext

Diese Dateien sind Teil der Test-Suite für das **LLKJJ ML-Plugin**. Alle Tests müssen die deutsche Elektrohandwerk-Domäne und SKR03-Klassifizierung berücksichtigen.

## Test-Framework & Tools

- **pytest**: Primäres Test-Framework
- **pytest-asyncio**: Für asynchrone ML-Pipeline-Tests
- **unittest.mock**: Für ML-Model-Mocking
- **Poetry**: ALLE Befehle mit `poetry run pytest`

## Test-Standards

### 1. Naming Conventions (Deutsch)

```python
# Korrekt: Deutsche Test-Namen
def test_pdf_extraktion_mit_tabellen():
    """Test der PDF-Extraktion mit Tabellenerkennung."""

def test_skr03_klassifizierung_elektromaterial():
    """Test der SKR03-Klassifizierung für Elektromaterial."""

async def test_async_verarbeitung_grosses_dokument():
    """Test der asynchronen Verarbeitung großer PDF-Dateien."""
```

### 2. Test-Struktur

```python
# Template für ML-Pipeline-Tests
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from src.processor import UnifiedProcessor
from src.models.invoice import ProcessingResult

class TestMLPipeline:
    """Test-Suite für ML-Pipeline-Komponenten."""

    @pytest.fixture
    def sample_pdf(self) -> Path:
        """Beispiel-PDF für Tests."""
        return Path("test_pdfs/Sonepar_test3.pdf")

    @pytest.fixture
    def expected_skr03_mapping(self) -> dict[str, str]:
        """Erwartete SKR03-Kontenzuordnung."""
        return {
            "Kabel NYM-J": "4360",  # Büroausstattung
            "Steckdosen": "4360",   # Büroausstattung
        }

    async def test_dokumentverarbeitung_end_to_end(
        self,
        sample_pdf: Path,
        expected_skr03_mapping: dict[str, str]
    ):
        """End-to-End Test der kompletten Pipeline."""
        # Arrange
        processor = UnifiedProcessor()

        # Act
        result = await processor.process_document_async(sample_pdf)

        # Assert
        assert isinstance(result, ProcessingResult)
        assert result.konfidenz_score >= 0.8
        assert any(
            konto in result.klassifizierung
            for konto in expected_skr03_mapping.values()
        )
```

### 3. Performance-Tests

```python
import time
import pytest

@pytest.mark.performance
async def test_verarbeitungsgeschwindigkeit_unter_30_sekunden():
    """Sicherstellen dass Verarbeitung unter 30 Sekunden dauert."""
    start_time = time.time()

    # Verarbeitung...

    verarbeitungszeit = time.time() - start_time
    assert verarbeitungszeit < 30.0, f"Verarbeitung dauerte {verarbeitungszeit:.2f}s"

@pytest.mark.performance
def test_memory_usage_unter_2gb():
    """Memory-Usage bei großen PDFs überwachen."""
    import psutil
    process = psutil.Process()

    # Vor Verarbeitung
    memory_before = process.memory_info().rss / 1024 / 1024 / 1024  # GB

    # Verarbeitung großer Datei...

    memory_after = process.memory_info().rss / 1024 / 1024 / 1024
    memory_increase = memory_after - memory_before

    assert memory_increase < 2.0, f"Memory-Anstieg: {memory_increase:.2f}GB"
```

### 4. ML-spezifische Tests

```python
@pytest.mark.ml
async def test_konfidenz_score_berechnung():
    """Test der ML-Konfidenz-Score-Berechnung."""
    from src.pipeline.quality_assessor import QualityAssessor

    assessor = QualityAssessor()

    # Hochqualitative Extraktion simulieren
    high_quality_data = {
        "rechnungsnummer": "RE-2024-001",
        "datum": "2024-01-15",
        "betrag_gesamt": 1234.56,
        "mehrwertsteuer": 234.56
    }

    score = assessor.calculate_confidence(high_quality_data)
    assert score >= 0.9, f"Erwartete hohe Konfidenz, bekommen: {score}"

@pytest.mark.ml
def test_skr03_regeln_vollstaendigkeit():
    """Test dass alle wichtigen Elektro-Kategorien abgedeckt sind."""
    from src.skr03_manager import SKR03Manager

    manager = SKR03Manager()

    # Wichtige Elektro-Begriffe testen
    test_begriffe = [
        "Kabel",
        "Steckdose",
        "Schalter",
        "Sicherung",
        "Verteilerschrank",
        "LED-Lampe"
    ]

    for begriff in test_begriffe:
        klassifizierung = manager.klassifiziere_keyword(begriff)
        assert klassifizierung is not None, f"Keine Klassifizierung für: {begriff}"
        assert klassifizierung.startswith("4"), f"Ungültiges SKR03-Konto: {klassifizierung}"
```

### 5. Integration-Tests

```python
@pytest.mark.integration
async def test_chromadb_integration():
    """Test der ChromaDB-Vektordatenbank-Integration."""
    from src.database.chroma_client import ChromaDBClient

    client = ChromaDBClient()

    # Test-Embedding hinzufügen
    test_text = "Installationskabel NYM-J 3x1,5 mm²"
    embedding_id = await client.add_embedding(test_text, {"skr03": "4360"})

    # Ähnlichkeitssuche
    similar = await client.search_similar(test_text, limit=5)

    assert len(similar) > 0
    assert any(item["skr03"] == "4360" for item in similar)

@pytest.mark.integration
def test_docling_processor_verfuegbarkeit():
    """Sicherstellen dass Docling-Processor verfügbar ist."""
    from src.extraction.docling_processor import AdvancedDoclingProcessor

    processor = AdvancedDoclingProcessor()
    assert processor.is_available(), "Docling-Processor nicht verfügbar"
```

### 6. Fehlerbehandlung-Tests

```python
@pytest.mark.error_handling
async def test_korrupte_pdf_behandlung():
    """Test der Behandlung korrupter PDF-Dateien."""
    from src.processor import UnifiedProcessor

    processor = UnifiedProcessor()

    # Korrupte PDF simulieren
    with pytest.raises(PDFProcessingError) as exc_info:
        await processor.process_document_async(Path("non_existent.pdf"))

    assert "PDF-Verarbeitung fehlgeschlagen" in str(exc_info.value)

@pytest.mark.error_handling
def test_netzwerk_fehler_resilience():
    """Test der Netzwerk-Fehler-Behandlung bei Gemini API."""
    from src.enhancement.gemini_enhancer import GeminiEnhancer

    with patch('src.enhancement.gemini_enhancer.requests.post') as mock_post:
        mock_post.side_effect = ConnectionError("Netzwerk nicht verfügbar")

        enhancer = GeminiEnhancer()
        result = enhancer.enhance_with_fallback("test text")

        # Sollte auf lokale Verarbeitung zurückfallen
        assert result is not None
        assert "fallback" in result.metadata
```

## Test-Daten & Fixtures

### Deutsche Test-PDFs verwenden

```python
@pytest.fixture(scope="session")
def deutsche_test_rechnungen() -> list[Path]:
    """Deutsche Elektrohandwerk-Test-Rechnungen."""
    return [
        Path("test_pdfs/Sonepar_test3.pdf"),
        Path("test_pdfs/Elektro_Grosskunde_Rechnung.pdf"),
        Path("test_pdfs/Kabel_Lieferung_2024.pdf")
    ]

@pytest.fixture
def skr03_test_mappings() -> dict[str, dict[str, Any]]:
    """SKR03-Test-Zuordnungen für Elektromaterial."""
    return {
        "4360": {  # Büroausstattung
            "keywords": ["Kabel", "Leitungen", "Installationsmaterial"],
            "confidence_threshold": 0.8
        },
        "4370": {  # Werkzeuge
            "keywords": ["Zange", "Schraubendreher", "Messgerät"],
            "confidence_threshold": 0.9
        }
    }
```

## Build & Test-Kommandos

```bash
# Alle Tests
poetry run pytest tests/ -v

# Nur Unit-Tests
poetry run pytest tests/ -m "not integration and not performance"

# Performance-Tests
poetry run pytest tests/ -m performance --tb=short

# ML-spezifische Tests
poetry run pytest tests/ -m ml -v

# Coverage-Report
poetry run pytest tests/ --cov=src --cov-report=html

# Spezifische Test-Datei
poetry run pytest tests/test_pipeline_e2e.py -v
```

## Assertions & Validierung

### Deutsche Datenvalidierung

```python
def assert_valid_german_invoice_data(data: dict[str, Any]):
    """Validiert deutsche Rechnungsdaten."""
    # Datum im deutschen Format
    assert re.match(r'\d{1,2}\.\d{1,2}\.\d{4}', data.get('datum', ''))

    # USt-ID Format
    if 'ustid' in data:
        assert re.match(r'DE\d{9}', data['ustid'])

    # PLZ Format
    if 'plz' in data:
        assert re.match(r'\d{5}', data['plz'])

    # Beträge als Decimal
    assert isinstance(data.get('betrag_gesamt'), (int, float, Decimal))

def assert_valid_skr03_klassifizierung(klassifizierung: dict[str, Any]):
    """Validiert SKR03-Klassifizierung."""
    for konto, konfidenz in klassifizierung.items():
        assert konto.startswith('4'), f"Ungültiges SKR03-Konto: {konto}"
        assert 0.0 <= konfidenz <= 1.0, f"Ungültige Konfidenz: {konfidenz}"
```

## Debugging & Logging in Tests

```python
import logging

# Test-spezifisches Logging
@pytest.fixture(autouse=True)
def configure_test_logging():
    """Konfiguriert Logging für Tests."""
    logging.getLogger('src').setLevel(logging.DEBUG)
    logging.getLogger('chromadb').setLevel(logging.WARNING)

def test_with_debug_logging():
    """Beispiel für Test mit Debug-Ausgabe."""
    logger = logging.getLogger(__name__)
    logger.debug("Starte ML-Pipeline-Test")

    # Test-Code...

    logger.debug("Test erfolgreich abgeschlossen")
```

---

**Wichtig**: Alle Tests müssen auf deutsche Elektrohandwerk-Szenarien ausgelegt sein und realistische SKR03-Klassifizierungen verwenden. Performance-Tests sollen die Produktionsanforderungen (<30s pro Dokument) validieren.
