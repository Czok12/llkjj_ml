"""
Bridge-Modul für LLKJJ ML - Zugriff auf zentrale Backend-Settings

Dieses Modul stellt eine kompatible Config-Schnittstelle für das ML-Paket bereit,
die auf die zentralen llkjj_backend.core.settings zugreift.

Nach der Konfigurationszentralisierung wird dieses Modul die Config-Klasse
aus src.config ersetzen.
"""

from pathlib import Path
from typing import Any


# Dynamischer Import der Backend-Settings
def _get_backend_settings() -> Any:
    """Lädt die Backend-Settings, falls verfügbar."""
    try:
        # Versuche zuerst, die Backend-Settings zu importieren
        from llkjj_api.core.settings import settings

        return settings
    except ImportError:
        # Fallback: Mock-Config für lokale Entwicklung
        from pydantic import BaseModel

        class MockConfig(BaseModel):
            google_api_key: str | None = None
            gemini_model: str = "gemini-2.5-flash"
            spacy_model_name: str = "de_core_news_sm"
            sentence_transformer_model: str = (
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )
            training_iterations: int = 100
            batch_size: int = 8
            chroma_collection_name: str = "llkjj_invoices"
            embedding_dimension: int = 384
            max_pdf_pages: int = 50
            max_pdf_size_mb: int = 50
            ocr_language: str = "deu"
            log_level: str = "INFO"
            log_file: str = "logs/llkjj_ml.log"
            umsatzsteuer_regulaer: float = 0.19
            umsatzsteuer_ermaessigt: float = 0.07
            gwg_grenze: float = 800.0
            default_account_elektro: str = "3400"
            default_account_office: str = "4930"
            default_account_tools: str = "4985"
            default_account_assets: str = "0490"
            skr03_regeln_file: str = "config/skr03/skr03_regeln.yaml"
            elektro_lieferanten_file: str = "config/ml/elektro_lieferanten.txt"

            @property
            def project_root(self) -> Path:
                # Go up from llkjj_ml/src/llkjj_ml to llkjj_backend root
                return Path(__file__).parent.parent.parent.parent.parent

            @property
            def data_raw_path(self) -> Path:
                return self.project_root / "data" / "raw"

            @property
            def data_processed_path(self) -> Path:
                return self.project_root / "data" / "processed"

            @property
            def training_data_path(self) -> Path:
                return self.project_root / "data" / "training"

            @property
            def models_path(self) -> Path:
                return self.project_root / "models"

            @property
            def vector_db_path(self) -> Path:
                return self.project_root / "data" / "vectors"

            @property
            def elektro_lieferanten(self) -> list[str]:
                """Load elektro_lieferanten from config file - TASK-028: No hardcoded fallback."""
                try:
                    import yaml

                    # Determine config file path relative to project root
                    config_file = self.project_root / "config" / "ml" / "suppliers.yaml"

                    if config_file.exists():
                        with config_file.open("r", encoding="utf-8") as f:
                            config_data = yaml.safe_load(f)
                            if config_data and "elektro_lieferanten" in config_data:
                                suppliers = config_data["elektro_lieferanten"]
                                if suppliers:  # Only return if non-empty
                                    return list(suppliers)  # Ensure list[str] type
                except Exception as e:
                    import logging

                    logger = logging.getLogger(__name__)
                    logger.warning(f"Failed to load suppliers config: {e}")

                # TASK-028: No hardcoded fallback - return empty list to force proper configuration
                import logging

                logger = logging.getLogger(__name__)
                logger.error(
                    "No elektro_lieferanten configuration found! Please configure suppliers in "
                    "config/ml/suppliers.yaml"
                )
                return []

            @property
            def default_accounts(self) -> dict[str, str]:
                return {
                    "elektromaterial": self.default_account_elektro,
                    "office": self.default_account_office,
                    "tools": self.default_account_tools,
                    "assets": self.default_account_assets,
                }

            def __post_init__(self) -> None:
                pass

        return MockConfig()


class ConfigBridge:
    """
    Bridge-Klasse die eine kompatible Schnittstelle zu den alten Config-Zugriffs-Patterns bereitstellt.
    """

    def __init__(self) -> None:
        self._settings = _get_backend_settings()

        # Wenn Backend-Settings verfügbar sind, verwende ML-spezifische Werte
        if hasattr(self._settings, "ml"):
            self._ml_config = self._settings.ml
        else:
            self._ml_config = self._settings

    # Properties für Kompatibilität mit der alten Config-Klasse

    @property
    def google_api_key(self) -> str | None:
        return getattr(self._ml_config, "google_api_key", None)

    @property
    def gemini_model(self) -> str:
        return getattr(self._ml_config, "gemini_model", "gemini-2.5-flash")

    @property
    def model_name(self) -> str:
        """Alias für gemini_model für Kompatibilität"""
        return self.gemini_model

    @property
    def temperature(self) -> float:
        return getattr(self._ml_config, "temperature", 0.3)

    @property
    def max_output_tokens(self) -> int:
        return getattr(self._ml_config, "max_output_tokens", 8192)

    @property
    def spacy_model_name(self) -> str:
        return getattr(self._ml_config, "spacy_model", "de_core_news_sm")

    @property
    def sentence_transformer_model(self) -> str:
        return getattr(
            self._ml_config,
            "sentence_transformer_model",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        )

    @property
    def training_iterations(self) -> int:
        return getattr(self._ml_config, "training_iterations", 100)

    @property
    def batch_size(self) -> int:
        return getattr(self._ml_config, "batch_size", 8)

    @property
    def chroma_collection_name(self) -> str:
        return getattr(self._ml_config, "chroma_collection_name", "llkjj_invoices")

    @property
    def embedding_dimension(self) -> int:
        return getattr(self._ml_config, "embedding_dimension", 384)

    @property
    def max_pdf_pages(self) -> int:
        return getattr(self._ml_config, "max_pdf_pages", 50)

    @property
    def max_pdf_size_mb(self) -> int:
        return getattr(self._ml_config, "max_pdf_size_mb", 50)

    @property
    def ocr_language(self) -> str:
        return getattr(self._ml_config, "ocr_language", "deu")

    @property
    def log_level(self) -> str:
        return getattr(self._settings, "log_level", "INFO")

    @property
    def log_file(self) -> str:
        return getattr(self._ml_config, "log_file", "logs/llkjj_ml.log")

    @property
    def umsatzsteuer_regulaer(self) -> float:
        return getattr(self._ml_config, "umsatzsteuer_regulaer", 0.19)

    @property
    def umsatzsteuer_ermaessigt(self) -> float:
        return getattr(self._ml_config, "umsatzsteuer_ermaessigt", 0.07)

    @property
    def gwg_grenze(self) -> float:
        return getattr(self._ml_config, "gwg_grenze", 800.0)

    @property
    def default_account_elektro(self) -> str:
        return getattr(self._ml_config, "default_account_elektro", "3400")

    @property
    def default_account_office(self) -> str:
        return getattr(self._ml_config, "default_account_office", "4930")

    @property
    def default_account_tools(self) -> str:
        return getattr(self._ml_config, "default_account_tools", "4985")

    @property
    def default_account_assets(self) -> str:
        return getattr(self._ml_config, "default_account_assets", "0490")

    @property
    def skr03_regeln_file(self) -> str:
        return getattr(
            self._ml_config, "skr03_regeln_file", "config/skr03/skr03_regeln.yaml"
        )

    @property
    def elektro_lieferanten_file(self) -> str:
        return getattr(
            self._ml_config,
            "elektro_lieferanten_file",
            "config/ml/elektro_lieferanten.txt",
        )

    @property
    def gemini_prompt_file(self) -> str:
        return getattr(
            self._ml_config,
            "gemini_prompt_file",
            "config/ml/gemini_extraction_prompt.txt",
        )

    # Pfad-Properties
    @property
    def project_root(self) -> Path:
        if hasattr(self._settings, "project_root"):
            return Path(self._settings.project_root)
        # Go up from llkjj_ml/src/llkjj_ml to llkjj_backend root
        return Path(__file__).parent.parent.parent.parent.parent

    @property
    def data_raw_path(self) -> Path:
        if hasattr(self._ml_config, "data_raw_path"):
            return Path(self._ml_config.data_raw_path)
        return self.project_root / "data" / "raw"

    @property
    def data_processed_path(self) -> Path:
        if hasattr(self._ml_config, "data_processed_path"):
            return Path(self._ml_config.data_processed_path)
        return self.project_root / "data" / "processed"

    @property
    def training_data_path(self) -> Path:
        if hasattr(self._ml_config, "training_data_path"):
            return Path(self._ml_config.training_data_path)
        return self.project_root / "data" / "training"

    @property
    def models_path(self) -> Path:
        if hasattr(self._ml_config, "models_root"):
            return Path(self._ml_config.models_root)
        return self.project_root / "models"

    @property
    def vector_db_path(self) -> Path:
        if hasattr(self._ml_config, "vector_db_path"):
            return Path(self._ml_config.vector_db_path)
        return self.project_root / "data" / "vectors"

    # Kompatibilitäts-Properties
    @property
    def elektro_lieferanten(self) -> list[str]:
        """Load elektro_lieferanten from config file - TASK-028: No hardcoded fallback."""
        # Check if available from backend settings first
        if hasattr(self._ml_config, "elektro_lieferanten"):
            suppliers = getattr(self._ml_config, "elektro_lieferanten", [])
            if suppliers:  # Only return if non-empty
                return suppliers

        # Try to load from suppliers.yaml config file
        try:
            import yaml

            # Determine config file path relative to project root
            config_file = self.project_root / "config" / "ml" / "suppliers.yaml"

            if config_file.exists():
                with config_file.open("r", encoding="utf-8") as f:
                    config_data = yaml.safe_load(f)
                    if config_data and "elektro_lieferanten" in config_data:
                        suppliers = config_data["elektro_lieferanten"]
                        if suppliers:  # Only return if non-empty
                            return list(suppliers)  # Ensure list[str] type
        except Exception as e:
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to load suppliers config: {e}")

        # TASK-028: No hardcoded fallback - return empty list to force proper configuration
        import logging

        logger = logging.getLogger(__name__)
        logger.error(
            "No elektro_lieferanten configuration found! Please configure suppliers in "
            "config/ml/suppliers.yaml or via backend settings."
        )
        return []

    @property
    def default_accounts(self) -> dict[str, str]:
        return {
            "elektromaterial": self.default_account_elektro,
            "office": self.default_account_office,
            "tools": self.default_account_tools,
            "assets": self.default_account_assets,
        }

    def __post_init__(self) -> None:
        """Kompatibilität mit alter Config-Klasse"""
        pass


# Type-Alias für mypy compatibility
ConfigType = type[ConfigBridge]

# Globale Instanz für einfachen Import - als ConfigBridge-Instanz deklariert
config_instance: ConfigBridge = ConfigBridge()

# For compatibility with existing code that expects Config variable
Config: ConfigBridge = config_instance

# Export both for convenient imports
__all__ = ["Config", "ConfigBridge", "ConfigType", "config_instance"]
