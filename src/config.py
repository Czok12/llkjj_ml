"""
LLKJJ ML Pipeline - Hauptkonfiguration

Zentrale Konfiguration für die automatisierte PDF-Rechnungsverarbeitung
für Elektrotechnik-Handwerk UG mit Gemini 2.5 Pro Integration.
"""

from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load environment variables
load_dotenv()


class Config(BaseSettings):
    """Hauptkonfiguration für LLKJJ ML Pipeline"""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # API Configuration
    google_api_key: str | None = Field(default=None)
    gemini_model: str = Field(default="gemini-2.0-flash-exp")

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
    spacy_model_name: str = Field(default="de_core_news_sm")
    sentence_transformer_model: str = Field(
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    training_iterations: int = Field(default=100)
    batch_size: int = Field(default=8)

    # Vector Database
    chroma_collection_name: str = Field(default="llkjj_invoices")
    embedding_dimension: int = Field(default=384)

    # Processing Configuration
    max_pdf_pages: int = Field(default=50)
    max_pdf_size_mb: int = Field(default=50)
    ocr_language: str = Field(default="deu")

    # Logging
    log_level: str = Field(default="INFO")
    log_file: str = Field(default="logs/llkjj_ml.log")

    # Business Rules für Elektrotechnik (konfigurierbar via .env)
    umsatzsteuer_regulaer: float = Field(default=0.19)
    umsatzsteuer_ermaessigt: float = Field(default=0.07)
    gwg_grenze: float = Field(default=800.0)  # GWG-Grenze für Anlagegüter

    # Default SKR03 Accounts (konfigurierbar via .env)
    default_account_elektro: str = Field(default="4830")
    default_account_office: str = Field(default="4935")
    default_account_tools: str = Field(default="4600")
    default_account_assets: str = Field(default="0490")

    # External Data Files
    elektro_lieferanten_file: str = Field(default="src/config/elektro_lieferanten.txt")

    @property
    def elektro_lieferanten(self) -> list[str]:
        """Lade Elektro-Lieferanten aus externer Datei"""
        try:
            lieferanten_path = Path(self.elektro_lieferanten_file)
            if not lieferanten_path.exists():
                # Fallback zur hartcodierten Liste wenn Datei nicht existiert
                return [
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

            with open(lieferanten_path, encoding="utf-8") as f:
                lieferanten = []
                for line in f:
                    line = line.strip()
                    # Ignoriere Kommentare und leere Zeilen
                    if line and not line.startswith("#"):
                        lieferanten.append(line)
                return lieferanten
        except Exception:
            # Fallback bei Fehlern
            return ["Rexel", "Conrad", "ELV", "Wago", "Phoenix Contact"]

    @property
    def default_accounts(self) -> dict[str, str]:
        """SKR03 Standard-Konten basierend auf Konfiguration"""
        return {
            "elektromaterial": self.default_account_elektro,
            "office": self.default_account_office,
            "tools": self.default_account_tools,
            "assets": self.default_account_assets,
        }

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


# SpaCy Entitäten für NER-Training
# Diese sind Pipeline-spezifisch und können global bleiben
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
