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
    skr03_mapping_file: str = "config/skr03_elektro.json"
    default_account_elektro: str = "4830"  # Elektromaterial
    default_account_office: str = "4935"  # Büromaterial
    default_account_tools: str = "0490"  # Werkzeuge
    default_account_services: str = "4400"  # Fremdleistungen

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


# Funktion zum Laden der Config (ohne globale Instanz)
def load_config() -> Config:
    """Lade die Config-Instanz"""
    return Config()


# SKR03 Mapping für Elektrohandwerk (KORRIGIERTE KONTEN)
SKR03_ELEKTRO_MAPPING = {
    # Wareneingang (korrekte Konten)
    "elektromaterial": "3400",  # Wareneingang 19% Vorsteuer
    "kabel": "3400",
    "schalter": "3400",
    "steckdosen": "3400",
    "lampen": "3400",
    "leuchten": "3400",
    "sicherungen": "3400",
    "installation": "3400",
    "verbrauchsmaterial": "3400",
    # Werkzeuge und Geräte (korrekte Kontierung)
    "werkzeuge_klein": "4985",  # Werkzeuge und Kleingeräte (Sofortaufwand)
    "handwerkzeug": "4985",
    "messgeraete_klein": "4985",
    # Anlagegüter (über 800€)
    "maschinen": "0210",  # Maschinen
    "bohrmaschinen": "0210",
    "saegen": "0210",
    "werkzeuge_gross": "0440",  # Werkzeuge (aktivierungspflichtig)
    "anlagen": "0200",  # Technische Anlagen
    # Betriebs- und Geschäftsausstattung
    "bueroausstattung": "0400",  # Betriebsausstattung
    "computer": "0420",  # Büroeinrichtung
    "software": "0420",
    "fahrzeuge": "0350",  # Lkw
    "pkw": "0320",  # Pkw
    # Fremdleistungen
    "fremdleistung": "4400",  # Fremdleistungen
    "montage": "4400",
    "subunternehmer": "4400",
}

# Gemini Prompt Templates
GEMINI_EXTRACTION_PROMPT = """
Analysiere diese PDF-Rechnung für ein deutsches Elektrotechnik-Handwerksunternehmen und extrahiere folgende Informationen:

RECHNUNGSDATEN:
- Rechnungsnummer
- Rechnungsdatum
- Lieferant/Firma
- Lieferantennummer
- Auftragsnummer (falls vorhanden)

FINANZIELLE DATEN:
- Nettobetrag
- Umsatzsteuer (19% oder 7%)
- Bruttobetrag
- Währung

POSITIONEN (für jede Rechnungsposition):
- Positionsnummer (falls vorhanden)
- Artikelbezeichnung
- Artikelnummer/Waren-ID (falls vorhanden)
- Menge
- Einheit (Stück, m, kg, etc.)
- Einzelpreis (netto)
- Gesamtpreis (netto)
- Umsatzsteuersatz
- Produktkategorie (z.B. Elektromaterial, Werkzeug, Büromaterial)

SKR03 KONTIERUNG (KORRIGIERTE KONTEN):
Ordne jede Position einem SKR03-Konto zu:
- 3400: Wareneingang 19% Vorsteuer (Elektromaterial, Kabel, Schalter, etc.)
- 3410: Wareneingang 7% Vorsteuer (ermäßigter Steuersatz)
- 4985: Werkzeuge und Kleingeräte (Sofortaufwand, unter 800€)
- 0200: Technische Anlagen (über 800€)
- 0210: Maschinen (über 800€)
- 0400: Betriebsausstattung
- 0420: Büroeinrichtung
- 4400: Fremdleistungen/Montage

Gib das Ergebnis als strukturiertes JSON zurück.
"""

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
