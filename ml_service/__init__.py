#!/usr/bin/env python3
"""
LLKJJ ML Service - Eigenständiges Plugin-Paket
==============================================

Dieses Paket stellt das ML-Plugin als völlig eigenständiges,
wiederverwendbares Modul zur Verfügung.

**EIGENSTÄNDIGKEIT:**
- Keine Abhängigkeiten zu externen Systemen oder DMS-Plugins
- Zentrale Konfigurationsverwaltung über llkjj_backend.core.settings
- Vollständig über Umgebungsvariablen konfigurierbar
- Installation als separates Python-Paket möglich

**PUBLIC API:**
```python
from ml_service import MLProcessor, settings

# Mit zentraler Konfiguration
processor = MLProcessor()
result = processor.process_pdf("invoice.pdf")

# Konfiguration erfolgt über llkjj_backend.core.settings
print(f"Verwendetes Modell: {settings.ml.gemini_model}")
processor = MLProcessor()
result = processor.process_pdf("invoice.pdf")
```

Author: LLKJJ ML Pipeline Team
Version: 3.0.0 (Eigenständiges Plugin)
"""

from llkjj_api.core.settings import settings

from .processor import MLProcessor, ProcessingResult, process_pdf_simple

__version__ = "3.0.0"
__author__ = "LLKJJ ML Pipeline Team"
__description__ = "Eigenständiges ML-Plugin für deutsche Rechnungsverarbeitung"

# Public API exports
__all__ = [
    "MLProcessor",
    "ProcessingResult",
    "process_pdf_simple",
    "settings",
    "__version__",
]

# Plugin-Metadaten für externe Systeme
PLUGIN_INFO = {
    "name": "LLKJJ ML Service",
    "version": __version__,
    "description": __description__,
    "author": __author__,
    "capabilities": [
        "pdf_extraction",
        "german_nlp",
        "skr03_classification",
        "quality_assessment",
        "vector_search",
        "cache_optimization",
    ],
    "supported_formats": ["PDF"],
    "languages": ["de", "en"],
    "dependencies": [
        "docling>=2.44.0",
        "spacy>=3.7.0",
        "transformers",
        "chromadb",
        "pydantic>=2.0.0",
    ],
    "system_requirements": {
        "python": ">=3.10",
        "memory_gb": 4,
        "gpu_optional": True,
        "disk_space_gb": 2,
    },
}
