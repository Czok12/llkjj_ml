#!/usr/bin/env python3
"""
LLKJJ ML Plugin - Eigenständige Konfiguration
============================================

Eigenständige Settings-Klasse für das ML-Plugin, die komplett unabhängig
von externen Systemen funktioniert. Liest Konfiguration aus Umgebungsvariablen
oder .env-Datei und macht das Plugin als eigenständiges Paket installierbar.

**EIGENSTÄNDIGKEIT:**
- Keine Abhängigkeiten zu core/settings.py oder anderen externen Modulen
- Vollständig konfigurierbar über Umgebungsvariablen
- Optimierte Defaults für deutsche Elektrohandwerk-Rechnungen
- Wiederverwendbar in jedem Python-Projekt

Author: LLKJJ ML Pipeline Team
Version: 3.0.0 (Eigenständige Konfiguration)
"""

from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Try to load .env file if available (optional)
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    # dotenv is optional - plugin works without it
    pass


class MLSettings(BaseSettings):
    """
    Eigenständige ML-Plugin Konfiguration

    Diese Klasse stellt alle notwendigen Einstellungen für das ML-Plugin bereit,
    ohne externe Abhängigkeiten zu anderen Systemkomponenten.

    **UMGEBUNGSVARIABLEN:**
    Alle Einstellungen können über Umgebungsvariablen überschrieben werden:
    - ML_GOOGLE_API_KEY: Google Gemini API-Schlüssel
    - ML_GEMINI_MODEL: Gemini-Modell Name
    - ML_DATA_PATH: Basispfad für Datenverzeichnisse
    - ML_VECTOR_DB_PATH: Pfad für ChromaDB Vector Store
    - ML_SPACY_MODEL: spaCy-Modell für deutsche Texte
    - ML_EMBEDDING_MODEL: SentenceTransformer-Modell
    - ML_OCR_ENGINE: OCR-Engine (rapid, tesseract, easyocr)
    - ML_TABLE_MODE: Tabellenextraktion (accurate, fast)
    - ML_CACHE_ENABLED: Cache aktivieren/deaktivieren
    - ML_LOG_LEVEL: Logging-Level (DEBUG, INFO, WARNING, ERROR)
    """

    model_config = SettingsConfigDict(
        env_prefix="ML_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # === CORE API CONFIGURATION ===
    google_api_key: str | None = Field(
        default=None, description="Google Gemini API Key für KI-basierte Verbesserungen"
    )

    gemini_model: str = Field(
        default="gemini-2.5-flash",
        description="Gemini-Modell für Extraktionsverbesserung",
    )

    # === PFAD-KONFIGURATION ===
    # Plugin-Root automatisch ermitteln
    plugin_root: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent,
        description="Root-Verzeichnis des ML-Plugins",
    )

    data_path: Path = Field(
        default_factory=lambda: Path("data"),
        description="Basis-Datenverzeichnis (relativ zu plugin_root)",
    )

    vector_db_path: Path = Field(
        default_factory=lambda: Path("data/vectors"),
        description="ChromaDB Vector Store Pfad",
    )

    models_cache_path: Path = Field(
        default_factory=lambda: Path("data/models_cache"),
        description="Cache für ML-Modelle",
    )

    logs_path: Path = Field(
        default_factory=lambda: Path("logs"), description="Log-Verzeichnis"
    )

    # === ML-MODELL KONFIGURATION ===
    spacy_model: str = Field(
        default="de_core_news_sm",
        description="spaCy-Modell für deutsche NLP-Verarbeitung",
    )

    embedding_model: str = Field(
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        description="SentenceTransformer-Modell für Embeddings",
    )

    # Alternative kompakte Version für bessere Performance
    embedding_model_compact: str = Field(
        default="all-MiniLM-L12-v2",
        description="Kompaktes Embedding-Modell für schnellere Verarbeitung",
    )

    # === VERARBEITUNGS-KONFIGURATION ===
    ocr_engine: str = Field(
        default="rapid", description="OCR-Engine: rapid, tesseract, easyocr"
    )

    table_mode: str = Field(
        default="accurate", description="Tabellenextraktion: accurate, fast"
    )

    use_gpu: bool = Field(
        default=True, description="GPU-Beschleunigung aktivieren wenn verfügbar"
    )

    german_optimized: bool = Field(
        default=True, description="Deutsche Sprachoptimierungen aktivieren"
    )

    # === CACHE-KONFIGURATION ===
    cache_enabled: bool = Field(default=True, description="Caching-System aktivieren")

    cache_ttl_hours: int = Field(default=24, description="Cache TTL in Stunden")

    memory_cache_size: int = Field(
        default=1000, description="Maximale Anzahl Einträge im Memory-Cache"
    )

    # === QUALITÄTS-KONFIGURATION ===
    min_confidence_threshold: float = Field(
        default=0.5, description="Minimale Konfidenz für Klassifizierung"
    )

    high_quality_threshold: float = Field(
        default=0.8, description="Schwellwert für 'high quality' Bewertung"
    )

    medium_quality_threshold: float = Field(
        default=0.6, description="Schwellwert für 'medium quality' Bewertung"
    )

    # === ELEKTROHANDWERK-SPEZIFISCHE KONFIGURATION ===
    skr03_rules_file: str = Field(
        default="src/config/skr03_regeln.yaml",
        description="SKR03-Regelwerk für deutsche Buchhaltung",
    )

    elektro_suppliers_file: str = Field(
        default="src/config/elektro_lieferanten.txt",
        description="Bekannte Elektrohandwerk-Lieferanten",
    )

    gemini_prompt_file: str = Field(
        default="src/config/gemini_extraction_prompt.txt",
        description="Gemini-Prompt für Extraktion",
    )

    # === LOGGING-KONFIGURATION ===
    log_level: str = Field(
        default="INFO", description="Logging-Level: DEBUG, INFO, WARNING, ERROR"
    )

    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log-Format-String",
    )

    # === PERFORMANCE-KONFIGURATION ===
    max_pdf_size_mb: int = Field(default=50, description="Maximale PDF-Größe in MB")

    max_processing_time_seconds: int = Field(
        default=300, description="Maximale Verarbeitungszeit in Sekunden"
    )

    batch_size: int = Field(default=32, description="Batch-Größe für ML-Verarbeitung")

    def get_absolute_path(self, relative_path: Path | str) -> Path:
        """
        Konvertiert relativen Pfad zu absolutem Pfad basierend auf plugin_root.

        Args:
            relative_path: Relativer Pfad zum Plugin-Root

        Returns:
            Absoluter Pfad
        """
        if isinstance(relative_path, str):
            relative_path = Path(relative_path)

        if relative_path.is_absolute():
            return relative_path

        return self.plugin_root / relative_path

    def ensure_directories(self) -> None:
        """
        Erstellt alle notwendigen Verzeichnisse falls sie nicht existieren.
        """
        directories = [
            self.get_absolute_path(self.data_path),
            self.get_absolute_path(self.vector_db_path),
            self.get_absolute_path(self.models_cache_path),
            self.get_absolute_path(self.logs_path),
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def get_config_file_path(self, config_file: str) -> Path:
        """
        Ermittelt absoluten Pfad zu Konfigurationsdateien.

        Args:
            config_file: Relativer Pfad zur Konfigurationsdatei

        Returns:
            Absoluter Pfad zur Konfigurationsdatei
        """
        return self.get_absolute_path(config_file)

    def validate_configuration(self) -> dict[str, Any]:
        """
        Validiert Konfiguration und gibt Zusammenfassung zurück.

        Returns:
            Dict mit Validierungsergebnissen und Konfigurationszusammenfassung
        """
        validation_results: dict[str, Any] = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "summary": {},
        }

        # Google API Key Check
        if not self.google_api_key:
            validation_results["warnings"].append(
                "Kein Google API Key konfiguriert - Gemini-Features deaktiviert"
            )

        # Pfad-Validierung
        try:
            self.ensure_directories()
            validation_results["summary"]["directories_created"] = True
        except Exception as e:
            validation_results["errors"].append(
                f"Fehler beim Erstellen der Verzeichnisse: {e}"
            )
            validation_results["valid"] = False

        # Konfigurationsdateien prüfen
        config_files = [
            self.skr03_rules_file,
            self.elektro_suppliers_file,
            self.gemini_prompt_file,
        ]

        missing_files = []
        for config_file in config_files:
            file_path = self.get_config_file_path(config_file)
            if not file_path.exists():
                missing_files.append(str(file_path))

        if missing_files:
            validation_results["warnings"].append(
                f"Konfigurationsdateien nicht gefunden: {missing_files}"
            )

        # Zusammenfassung erstellen
        validation_results["summary"].update(
            {
                "plugin_root": str(self.plugin_root),
                "data_path": str(self.get_absolute_path(self.data_path)),
                "vector_db_path": str(self.get_absolute_path(self.vector_db_path)),
                "cache_enabled": self.cache_enabled,
                "gpu_enabled": self.use_gpu,
                "german_optimized": self.german_optimized,
                "gemini_available": bool(self.google_api_key),
            }
        )

        return validation_results

    @classmethod
    def create_default(cls) -> "MLSettings":
        """
        Erstellt eine Standardkonfiguration für das ML-Plugin.

        Returns:
            MLSettings-Instanz mit optimierten Defaults
        """
        return cls()

    def __str__(self) -> str:
        """String-Repräsentation der Konfiguration."""
        return (
            f"MLSettings(plugin_root={self.plugin_root}, "
            f"gemini_model={self.gemini_model}, "
            f"cache_enabled={self.cache_enabled}, "
            f"german_optimized={self.german_optimized})"
        )


# Globale Standardkonfiguration für einfache Verwendung
default_settings = MLSettings.create_default()
