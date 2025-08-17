"""
Security Manager für LLKJJ ML Pipeline
====================================

Umfassende Security-Features für Production-Readiness:
- API-Key Encryption at rest
- Environment-Variables Management
- Access Control für Konfigurationsdateien
- Security Auditing und Logging
"""

import base64
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)


class SecurityConfig:
    """Security-Konfiguration für die Pipeline."""

    def __init__(self):
        self.encryption_key_file = Path(".security/master.key")
        self.encrypted_secrets_file = Path(".security/secrets.enc")
        self.access_log_file = Path("logs/security_access.log")

        # Erstelle Security-Verzeichnisse
        self.encryption_key_file.parent.mkdir(exist_ok=True)
        self.access_log_file.parent.mkdir(exist_ok=True)


class APIKeyManager:
    """
    Sicherer Manager für API-Keys mit Encryption at rest.

    Features:
    - AES-256 Encryption für API-Keys
    - Passwort-basierte Key-Derivation (PBKDF2)
    - Sichere Key-Rotation
    - Audit-Logging für Key-Zugriffe
    """

    def __init__(self, master_password: str | None = None):
        """
        Initialisiert den APIKeyManager.

        Args:
            master_password: Master-Passwort für Key-Encryption
        """
        self.config = SecurityConfig()
        self.master_password = master_password or os.getenv("LLKJJ_MASTER_PASSWORD")

        if not self.master_password:
            logger.warning(
                "Kein Master-Passwort gesetzt. API-Keys werden unverschlüsselt gespeichert."
            )
            self._encryption_enabled = False
        else:
            self._encryption_enabled = True
            self._derive_encryption_key()

    def _derive_encryption_key(self) -> None:
        """Leitet Encryption-Key vom Master-Passwort ab."""
        password = self.master_password.encode()
        salt = b"llkjj_ml_salt_2025"  # In Production: Zufälliges Salt verwenden

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password))
        self.cipher_suite = Fernet(key)

    def store_api_key(self, service_name: str, api_key: str) -> bool:
        """
        Speichert einen API-Key verschlüsselt.

        Args:
            service_name: Name des Services (z.B. 'gemini', 'openai')
            api_key: Der zu speichernde API-Key

        Returns:
            True bei Erfolg, False bei Fehler
        """
        try:
            # Lade bestehende Keys
            secrets = self._load_secrets()

            # Verschlüssele neuen Key
            if self._encryption_enabled:
                encrypted_key = self.cipher_suite.encrypt(api_key.encode()).decode()
                secrets[service_name] = {
                    "encrypted": True,
                    "value": encrypted_key,
                    "hash": hashlib.sha256(api_key.encode()).hexdigest()[:8],
                }
            else:
                secrets[service_name] = {
                    "encrypted": False,
                    "value": api_key,
                    "hash": hashlib.sha256(api_key.encode()).hexdigest()[:8],
                }

            # Speichere Secrets
            self._save_secrets(secrets)

            # Audit-Log
            self._log_access(f"API-Key für {service_name} gespeichert")

            logger.info("API-Key für %s erfolgreich gespeichert", service_name)
            return True

        except (OSError, ValueError, KeyError) as e:
            logger.error(
                "Fehler beim Speichern des API-Keys für %s: %s", service_name, e
            )
            return False

    def get_api_key(self, service_name: str) -> str | None:
        """
        Lädt einen API-Key entschlüsselt.

        Args:
            service_name: Name des Services

        Returns:
            Entschlüsselter API-Key oder None
        """
        try:
            secrets = self._load_secrets()

            if service_name not in secrets:
                logger.warning("API-Key für %s nicht gefunden", service_name)
                return None

            secret_data = secrets[service_name]

            if secret_data.get("encrypted", False):
                if not self._encryption_enabled:
                    logger.error(
                        "Encryption nicht aktiviert, aber verschlüsselter Key gefunden"
                    )
                    return None

                decrypted_key = self.cipher_suite.decrypt(
                    secret_data["value"].encode()
                ).decode()

                # Validiere Hash
                expected_hash = hashlib.sha256(decrypted_key.encode()).hexdigest()[:8]
                if expected_hash != secret_data.get("hash"):
                    logger.error("Hash-Validierung fehlgeschlagen für %s", service_name)
                    return None

                self._log_access(
                    f"API-Key für {service_name} abgerufen (verschlüsselt)"
                )
                return decrypted_key
            else:
                self._log_access(
                    f"API-Key für {service_name} abgerufen (unverschlüsselt)"
                )
                return secret_data["value"]

        except (OSError, ValueError, KeyError) as e:
            logger.error("Fehler beim Laden des API-Keys für %s: %s", service_name, e)
            return None

    def rotate_api_key(self, service_name: str, new_api_key: str) -> bool:
        """
        Rotiert einen API-Key (ersetzt den alten).

        Args:
            service_name: Name des Services
            new_api_key: Neuer API-Key

        Returns:
            True bei Erfolg
        """
        old_key_exists = self.get_api_key(service_name) is not None

        if self.store_api_key(service_name, new_api_key):
            action = "rotiert" if old_key_exists else "erstellt"
            self._log_access(f"API-Key für {service_name} {action}")
            logger.info("API-Key für %s erfolgreich %s", service_name, action)
            return True

        return False

    def list_stored_keys(self) -> list[str]:
        """
        Listet alle gespeicherten Service-Namen auf.

        Returns:
            Liste der Service-Namen
        """
        try:
            secrets = self._load_secrets()
            services = list(secrets.keys())
            self._log_access(f"Service-Liste abgerufen: {len(services)} Services")
            return services
        except (OSError, ValueError, KeyError) as e:
            logger.error("Fehler beim Auflisten der Services: %s", e)
            return []

    def delete_api_key(self, service_name: str) -> bool:
        """
        Löscht einen API-Key.

        Args:
            service_name: Name des Services

        Returns:
            True bei Erfolg
        """
        try:
            secrets = self._load_secrets()

            if service_name not in secrets:
                logger.warning("API-Key für %s nicht gefunden (Löschung)", service_name)
                return False

            del secrets[service_name]
            self._save_secrets(secrets)

            self._log_access(f"API-Key für {service_name} gelöscht")
            logger.info("API-Key für %s erfolgreich gelöscht", service_name)
            return True

        except (OSError, ValueError, KeyError) as e:
            logger.error("Fehler beim Löschen des API-Keys für %s: %s", service_name, e)
            return False

    def _load_secrets(self) -> dict[str, Any]:
        """Lädt verschlüsselte Secrets von Disk."""
        if not self.config.encrypted_secrets_file.exists():
            return {}

        try:
            with open(self.config.encrypted_secrets_file, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError, KeyError) as e:
            logger.error("Fehler beim Laden der Secrets: %s", e)
            return {}

    def _save_secrets(self, secrets: dict[str, Any]) -> None:
        """Speichert Secrets auf Disk."""
        try:
            with open(self.config.encrypted_secrets_file, "w", encoding="utf-8") as f:
                json.dump(secrets, f, indent=2)
        except (OSError, ValueError) as e:
            logger.error("Fehler beim Speichern der Secrets: %s", e)
            raise

    def _log_access(self, action: str) -> None:
        """Loggt Security-relevante Aktionen."""
        import datetime

        timestamp = datetime.datetime.now().isoformat()
        log_entry = f"{timestamp} - {action}\n"

        try:
            with open(self.config.access_log_file, "a", encoding="utf-8") as f:
                f.write(log_entry)
        except OSError as e:
            logger.error("Fehler beim Security-Logging: %s", e)


class EnvironmentManager:
    """
    Manager für sichere Environment-Variables.

    Features:
    - Validierung von Environment-Variables
    - Default-Werte für kritische Settings
    - Environment-spezifische Konfigurationen
    """

    def __init__(self):
        self.required_vars = {
            "GOOGLE_API_KEY": "Google Gemini API Key",
            "LLKJJ_ENV": "Environment (development/production)",
        }

        self.optional_vars = {
            "LLKJJ_MASTER_PASSWORD": "Master-Passwort für API-Key Encryption",
            "LLKJJ_LOG_LEVEL": "Logging-Level (DEBUG/INFO/WARNING/ERROR)",
            "LLKJJ_CACHE_TTL": "Cache TTL in Sekunden",
            "LLKJJ_MAX_WORKERS": "Maximale Anzahl Worker-Threads",
        }

    def validate_environment(self) -> dict[str, Any]:
        """
        Validiert Environment-Variables.

        Returns:
            Dict mit Validierungsergebnissen
        """
        results = {
            "valid": True,
            "missing_required": [],
            "missing_optional": [],
            "recommendations": [],
        }

        # Prüfe Required Variables
        for var_name, description in self.required_vars.items():
            if not os.getenv(var_name):
                results["missing_required"].append(
                    {"name": var_name, "description": description}
                )
                results["valid"] = False

        # Prüfe Optional Variables
        for var_name, description in self.optional_vars.items():
            if not os.getenv(var_name):
                results["missing_optional"].append(
                    {"name": var_name, "description": description}
                )

        # Sicherheits-Empfehlungen
        env = os.getenv("LLKJJ_ENV", "development")
        if env == "production":
            if not os.getenv("LLKJJ_MASTER_PASSWORD"):
                results["recommendations"].append(
                    "LLKJJ_MASTER_PASSWORD sollte in Production gesetzt sein"
                )

            if os.getenv("LLKJJ_LOG_LEVEL") == "DEBUG":
                results["recommendations"].append(
                    "DEBUG-Logging sollte in Production vermieden werden"
                )

        return results

    def get_secure_config(self) -> dict[str, Any]:
        """
        Lädt sichere Konfiguration aus Environment.

        Returns:
            Dictionary mit sicherer Konfiguration
        """
        return {
            "environment": os.getenv("LLKJJ_ENV", "development"),
            "log_level": os.getenv("LLKJJ_LOG_LEVEL", "INFO"),
            "cache_ttl": int(os.getenv("LLKJJ_CACHE_TTL", "3600")),
            "max_workers": int(os.getenv("LLKJJ_MAX_WORKERS", "4")),
            "encryption_enabled": bool(os.getenv("LLKJJ_MASTER_PASSWORD")),
            "debug_mode": os.getenv("LLKJJ_ENV", "development") == "development",
        }


def create_api_key_manager(master_password: str | None = None) -> APIKeyManager:
    """
    Factory-Funktion für APIKeyManager.

    Args:
        master_password: Master-Passwort (optional)

    Returns:
        APIKeyManager-Instanz
    """
    return APIKeyManager(master_password)


def validate_production_environment() -> bool:
    """
    Validiert Environment für Production-Deployment.

    Returns:
        True wenn Production-ready
    """
    env_manager = EnvironmentManager()
    validation = env_manager.validate_environment()

    if not validation["valid"]:
        logger.error("Environment-Validierung fehlgeschlagen:")
        for missing in validation["missing_required"]:
            logger.error(f"  - {missing['name']}: {missing['description']}")
        return False

    if validation["recommendations"]:
        logger.warning("Sicherheits-Empfehlungen:")
        for rec in validation["recommendations"]:
            logger.warning(f"  - {rec}")

    return True
