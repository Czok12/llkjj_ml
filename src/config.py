"""
LLKJJ ML Pipeline - Hauptkonfiguration

Zentrale Konfiguration für die automatisierte PDF-Rechnungsverarbeitung
für Elektrotechnik-Handwerk UG mit Gemini 2.5 Pro Integration.
"""

import logging
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load environment variables
load_dotenv()


logger = logging.getLogger(__name__)


class Config(BaseSettings):
    """Hauptkonfiguration für LLKJJ ML Pipeline"""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # API Configuration
    google_api_key: str | None = Field(default=None)
    gemini_model: str = Field(default="gemini-2.5-flash")

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
    # Empfohlene, semantisch passende Defaults (werden beim Zugriff validiert)
    default_account_elektro: str = Field(default="3400")
    default_account_office: str = Field(default="4930")
    default_account_tools: str = Field(default="4985")
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
        except (OSError, UnicodeError) as e:
            # Fallback bei Dateizugriffs- oder Dekodierungsfehlern
            logger.warning("Fehler beim Lesen der Lieferanten-Datei: %s", e)
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

    def _normalize_konto(self, konto: str) -> str:
        """Normalisiert eine Kontonummer für Vergleich/Lookup.

        Entfernt Nicht-Ziffern und führende Nullen. Liefert leere Zeichenkette
        bei ungültigen Eingaben zurück.
        """
        if not konto:
            return ""
        digits = "".join(ch for ch in konto if ch.isdigit())
        # Entferne führende Nullen, falls der Kontenplan ohne führende Nullen vorliegt
        normalized = digits.lstrip("0")
        return normalized or digits

    def _load_kontenplan_parser(self) -> Any | None:
        """Versucht, den Kontenplan-Parser zu laden (lazy import).

        Gibt None zurück wenn der Parser oder die CSV nicht verfügbar ist.
        """
        try:
            # Lazy import, um zyklische Importe zu vermeiden
            from src.skr03_manager import KontenplanParser

            kontenplan_path = self.project_root / "src" / "config" / "Kontenplan.csv"
            if not kontenplan_path.exists():
                logger.debug("Kontenplan-Datei nicht gefunden: %s", kontenplan_path)
                return None
            return KontenplanParser(kontenplan_path)
        except (ImportError, OSError) as e:
            logger.debug("KontenplanParser konnte nicht geladen werden: %s", e)
            return None

    @property
    def default_accounts_validated(self) -> dict[str, str]:
        """Gibt die Default-Accounts normalisiert und gegen den Kontenplan validiert zurück.

        Wenn ein Default ungültig ist, wird ein sicheres Fallback-Konto verwendet und
        eine Warnung geloggt.
        """
        mapping = self.default_accounts
        parser = self._load_kontenplan_parser()

        # Sinnvolle Fallbacks (konservativ)
        fallbacks: dict[str, str] = {
            "elektromaterial": "3400",
            "office": "4930",
            "tools": "4985",
            "assets": "490",
        }

        validated: dict[str, str] = {}
        for key, konto in mapping.items():
            norm = self._normalize_konto(konto)
            # Versuche Validierung gegen Kontenplan (falls verfügbar)
            if parser:
                if parser.ist_gueltig(norm):
                    validated[key] = norm
                elif parser.ist_gueltig(konto):
                    # Falls der Kontenplan originale Form enthält
                    validated[key] = konto
                else:
                    logger.warning(
                        "Ungültiges Default-Konto '%s' für '%s' - nutze Fallback '%s'",
                        konto,
                        key,
                        fallbacks[key],
                    )
                    validated[key] = fallbacks[key]
            else:
                # Ohne Kontenplan nur normalisiert zurückgeben (ohne Validierung)
                if norm:
                    validated[key] = norm
                else:
                    validated[key] = fallbacks[key]

        return validated

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
    # Deutsche NER-Labels für Elektrohandwerk-Rechnungen
    # Artikel- und Positionsdaten
    "ARTIKEL",  # Produktbezeichnungen
    "ARTIKELNUMMER",  # Hersteller-/Bestellnummern
    "RECHNUNGSPOSITION",  # Position auf Rechnung
    "MENGE",  # Anzahl/Stückzahl
    "EINZELPREIS",  # Preise pro Einheit
    "GESAMTPREIS",  # Gesamtbetrag
    "MWST_SATZ",  # Mehrwertsteuersatz
    # Rechnungsinformationen
    "RECHNUNGSNUMMER",  # Rechnungs-ID
    "RECHNUNGSDATUM",  # Ausstellungsdatum
    "LIEFERDATUM",  # Liefertermin
    # Parteien und Kundendaten
    "HÄNDLER",  # Lieferant/Verkäufer
    "KUNDE",  # Käufer/Empfänger
    "KUNDENNUMMER",  # Kundennummer
    # Legacy-Support (English labels für internationale Kompatibilität)
    "INVOICE_NUMBER",  # -> RECHNUNGSNUMMER
    "INVOICE_DATE",  # -> RECHNUNGSDATUM
    "SUPPLIER",  # -> HÄNDLER
    "NET_AMOUNT",  # -> GESAMTPREIS
    "VAT_AMOUNT",  # -> MWST_SATZ
    "GROSS_AMOUNT",  # -> GESAMTPREIS
    "ITEM_DESCRIPTION",  # -> ARTIKEL
    "ITEM_NUMBER",  # -> ARTIKELNUMMER
    "QUANTITY",  # -> MENGE
    "UNIT_PRICE",  # -> EINZELPREIS
    "TOTAL_PRICE",  # -> GESAMTPREIS
    "SKR03_ACCOUNT",  # SKR03 Konto
    "PRODUCT_CATEGORY",  # Produktkategorie
]
