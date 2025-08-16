"""
LLKJJ ML Pipeline - Hauptkonfiguration

Zentrale Konfiguration für die automatisierte PDF-Rechnungsverarbeitung
für Elektrotechnik-Handwerk UG mit Gemini 2.5 Pro Integration.
"""

from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

# Load environment variables
load_dotenv()


class Config(BaseSettings):
    """Hauptkonfiguration für LLKJJ ML Pipeline"""

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

    # SKR03 Elektrohandwerk Configuration
    skr03_regeln_file: str = "src/config/skr03_regeln.yaml"
    gemini_prompt_file: str = "src/config/gemini_extraction_prompt.txt"

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
        extra = "ignore"  # Ignoriere unbekannte Felder aus .env

    def __post_init__(self) -> None:
        """Erstelle notwendige Verzeichnisse"""
        for path in [
            self.data_raw_path,
            self.data_processed_path,
            self.training_data_path,
            self.models_path,
            self.vector_db_path,
            Path("logs"),
        ]:
            path.mkdir(parents=True, exist_ok=True)


# Default SKR03 accounts from skr03_regeln.yaml
# These are fallback values - actual mapping is in skr03_regeln.yaml
DEFAULT_ACCOUNTS = {
    "elektromaterial": "3400",  # Wareneingang 19% Vorsteuer
    "office": "4935",  # Büromaterial
    "tools": "4985",  # Werkzeuge und Kleingeräte
    "services": "4400",  # Fremdleistungen
}

SPACY_ENTITIES = [
    "INVOICE_NUMBER",  # Rechnungsnummer
    "INVOICE_DATE",  # Rechnungsdatum
    "SUPPLIER",  # Lieferant
    "NET_AMOUNT",  # Nettobetrag
    "VAT_AMOUNT",  # Umsatzsteuer
    "GROSS_AMOUNT",  # Bruttobetrag
    "ITEM_DESCRIPTION",  # Artikelbezeichnung
    "ITEM_NUMBER",  # Artikelnummer
    "QUANTITY",  # Menge
    "UNIT_PRICE",  # Einzelpreis
    "TOTAL_PRICE",  # Gesamtpreis
    "SKR03_ACCOUNT",  # SKR03 Konto
    "PRODUCT_CATEGORY",  # Produktkategorie
]
