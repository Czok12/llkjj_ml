#!/usr/bin/env python3
"""
LLKJJ ML Pipeline - Hauptkonfiguration (REFACTORED)

Zentrale Konfiguration für die automatisierte PDF-Rechnungsverarbeitung
für Elektrotechnik-Handwerk UG.

ONLY configuration values, NO business logic or prompts!
"""

from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

# Load environment variables
load_dotenv()


class Config(BaseSettings):
    """Hauptkonfiguration für LLKJJ ML Pipeline - NUR Konfigurationswerte"""

    # API Configuration
    google_api_key: str | None = Field(default=None)
    gemini_model: str = Field(default="gemini-2.5-pro")

    # Paths
    project_root: Path = Path(__file__).parent.parent
    data_raw_path: Path = Field(default_factory=lambda: Path("data/raw"))
    data_processed_path: Path = Field(default_factory=lambda: Path("data/processed"))
    training_data_path: Path = Field(default_factory=lambda: Path("data/training"))
    models_path: Path = Field(default_factory=lambda: Path("models"))
    vector_db_path: Path = Field(default_factory=lambda: Path("data/vectors"))

    # Configuration file paths (moved out of code!)
    skr03_mapping_file: str = "config/skr03_elektro_mapping.yaml"
    gemini_prompt_file: str = "config/gemini_extraction_prompt.txt"
    spacy_entities_file: str = "config/spacy_entities.yaml"

    # ML Configuration
    spacy_model_name: str = "de_core_news_sm"
    sentence_transformer_model: str = (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    training_iterations: int = 30
    batch_size: int = 8

    # Vector Database
    chroma_collection_name: str = "llkjj_invoices"
    embedding_dimension: int = 384

    # Processing Configuration
    max_pdf_pages: int = 50
    ocr_language: str = "deu"

    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/llkjj_ml.log"

    # Business Rules für Elektrotechnik
    umsatzsteuer_regulaer: float = 0.19
    umsatzsteuer_ermaessigt: float = 0.07
    gwg_grenze: float = 800.0  # GWG-Grenze für Anlagegüter

    # Bekannte Elektro-Lieferanten
    elektro_lieferanten: list[str] = [
        "Rexel",
        "Conrad",
        "ELV",
        "Wago",
        "Phoenix Contact",
        "Siemens",
        "ABB",
        "Schneider Electric",
        "Legrand",
        "Hager",
        "Gira",
        "Jung",
        "Busch-Jaeger",
        "Berker",
    ]

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"

    def __post_init__(self) -> None:
        """Erstelle notwendige Verzeichnisse"""
        for path in [
            self.data_raw_path,
            self.data_processed_path,
            self.training_data_path,
            self.models_path,
            self.vector_db_path,
            Path("logs"),
            Path("config"),  # Config directory
        ]:
            path.mkdir(parents=True, exist_ok=True)


def load_config() -> Config:
    """Lade die Config-Instanz"""
    return Config()
