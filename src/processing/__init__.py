"""
Processing Module für LLKJJ ML Pipeline

Dieses Modul enthält spezialisierte Klassen für die modulare Verarbeitung
von Rechnungsdaten mit Fokus auf deutsche Elektrotechnik-Buchhaltung.

Module:
- extractor: DataExtractor für PDF-Extraktion und Datengewinnung
- classifier: DataClassifier für SKR03-Klassifizierung und RAG-System
- quality: QualityAssessor und QualityReporter für Qualitätsbewertung

Das processing-Modul folgt dem Single Responsibility Principle und ermöglicht
bessere Wartbarkeit und Testbarkeit des Codes.

Autor: LLKJJ ML Pipeline Team
Version: 2.1.0 (Post-Konsolidierung)
"""

from .classifier import DataClassifier
from .extractor import DataExtractor
from .quality import QualityAssessor, QualityReporter

__all__ = ["DataClassifier", "DataExtractor", "QualityAssessor", "QualityReporter"]

__version__ = "2.1.0"
