# LLKJJ ML Pipeline - Environment Management

"""
Environment-specific configuration management for LLKJJ ML Pipeline.
Handles Dev/Staging/Production environment separation and secure configuration.
"""

import base64
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from cryptography.fernet import Fernet


class Environment(Enum):
    """Environment types."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class DatabaseConfig:
    """Database configuration."""

    host: str = "localhost"
    port: int = 5432
    name: str = "llkjj_ml"
    user: str = "llkjj_ml"
    password: str = ""
    ssl_mode: str = "prefer"

    def to_dict(self) -> dict[str, Any]:
        return {
            "host": self.host,
            "port": self.port,
            "name": self.name,
            "user": self.user,
            "password": self.password,
            "ssl_mode": self.ssl_mode,
        }


@dataclass
class LoggingConfig:
    """Logging configuration."""

    level: str = "INFO"
    enable_console: bool = True
    enable_file: bool = True
    max_file_size_mb: int = 10
    backup_count: int = 5
    structured_logging: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "enable_console": self.enable_console,
            "enable_file": self.enable_file,
            "max_file_size_mb": self.max_file_size_mb,
            "backup_count": self.backup_count,
            "structured_logging": self.structured_logging,
        }


@dataclass
class MLConfig:
    """Machine Learning configuration."""

    model_cache_size: int = 1000
    batch_size: int = 32
    gpu_enabled: bool = False
    cpu_threads: int = 4
    memory_limit_gb: int = 8
    timeout_seconds: int = 300

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_cache_size": self.model_cache_size,
            "batch_size": self.batch_size,
            "gpu_enabled": self.gpu_enabled,
            "cpu_threads": self.cpu_threads,
            "memory_limit_gb": self.memory_limit_gb,
            "timeout_seconds": self.timeout_seconds,
        }


@dataclass
class SecurityConfig:
    """Security configuration."""

    enable_encryption: bool = True
    api_rate_limit: int = 1000
    session_timeout_minutes: int = 30
    max_upload_size_mb: int = 50
    allowed_file_types: list[str] = field(default_factory=lambda: ["pdf"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "enable_encryption": self.enable_encryption,
            "api_rate_limit": self.api_rate_limit,
            "session_timeout_minutes": self.session_timeout_minutes,
            "max_upload_size_mb": self.max_upload_size_mb,
            "allowed_file_types": self.allowed_file_types,
        }


@dataclass
class EnvironmentConfig:
    """Complete environment configuration."""

    environment: Environment
    debug: bool = False
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    custom_settings: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "environment": self.environment.value,
            "debug": self.debug,
            "database": self.database.to_dict(),
            "logging": self.logging.to_dict(),
            "ml": self.ml.to_dict(),
            "security": self.security.to_dict(),
            "custom_settings": self.custom_settings,
        }


class SecretManager:
    """Secure secret management with encryption."""

    def __init__(self, key_file: Path | None = None):
        self.key_file = key_file or Path(".encryption_key")
        self._key: bytes | None = None
        self._fernet: Fernet | None = None
        self._init_encryption()

    def _init_encryption(self) -> None:
        """Initialize encryption key."""
        if self.key_file.exists():
            with open(self.key_file, "rb") as f:
                self._key = f.read()
        else:
            self._key = Fernet.generate_key()
            with open(self.key_file, "wb") as f:
                f.write(self._key)
            # Secure file permissions
            os.chmod(self.key_file, 0o600)

        self._fernet = Fernet(self._key)

    def encrypt_secret(self, secret: str) -> str:
        """Encrypt a secret value."""
        if self._fernet is None:
            raise RuntimeError("Encryption not initialized")
        encrypted = self._fernet.encrypt(secret.encode())
        return base64.b64encode(encrypted).decode()

    def decrypt_secret(self, encrypted_secret: str) -> str:
        """Decrypt a secret value."""
        if self._fernet is None:
            raise RuntimeError("Encryption not initialized")
        encrypted_bytes = base64.b64decode(encrypted_secret.encode())
        decrypted = self._fernet.decrypt(encrypted_bytes)
        return decrypted.decode()

    def is_encrypted(self, value: str) -> bool:
        """Check if a value is encrypted."""
        try:
            base64.b64decode(value)
            return True
        except Exception:
            return False


class EnvironmentManager:
    """Manages environment-specific configurations."""

    def __init__(self, config_dir: Path | None = None):
        self.config_dir = config_dir or Path("deployment/config")
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.secret_manager = SecretManager()
        self._current_config: EnvironmentConfig | None = None
        self.logger = logging.getLogger(__name__)

    def get_environment_from_string(self, env_str: str) -> Environment:
        """Convert string to Environment enum."""
        env_map = {
            "dev": Environment.DEVELOPMENT,
            "development": Environment.DEVELOPMENT,
            "staging": Environment.STAGING,
            "stage": Environment.STAGING,
            "prod": Environment.PRODUCTION,
            "production": Environment.PRODUCTION,
        }
        return env_map.get(env_str.lower(), Environment.DEVELOPMENT)

    def create_default_configs(self) -> None:
        """Create default configuration files for all environments."""
        # Development configuration
        dev_config = EnvironmentConfig(
            environment=Environment.DEVELOPMENT,
            debug=True,
            logging=LoggingConfig(level="DEBUG", structured_logging=False),
            ml=MLConfig(model_cache_size=100, batch_size=16, memory_limit_gb=4),
            security=SecurityConfig(enable_encryption=False, api_rate_limit=10000),
        )
        self._save_config(dev_config, "development.yaml")

        # Staging configuration
        staging_config = EnvironmentConfig(
            environment=Environment.STAGING,
            debug=False,
            logging=LoggingConfig(level="INFO"),
            ml=MLConfig(model_cache_size=500, batch_size=24, memory_limit_gb=6),
            security=SecurityConfig(api_rate_limit=5000),
        )
        self._save_config(staging_config, "staging.yaml")

        # Production configuration
        prod_config = EnvironmentConfig(
            environment=Environment.PRODUCTION,
            debug=False,
            logging=LoggingConfig(level="WARNING", backup_count=10),
            ml=MLConfig(model_cache_size=1000, batch_size=32, memory_limit_gb=8),
            security=SecurityConfig(api_rate_limit=1000),
        )
        self._save_config(prod_config, "production.yaml")

        # Create environment-specific .env templates
        self._create_env_templates()

    def _create_env_templates(self) -> None:
        """Create .env template files for each environment."""
        templates = {
            "development": """# Development Environment Variables
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=DEBUG

# API Keys (Development)
GOOGLE_API_KEY=your_dev_google_api_key_here
OPENAI_API_KEY=your_dev_openai_api_key_here

# Database (Development)
DATABASE_URL=postgresql://llkjj_ml:password@localhost:5432/llkjj_ml_dev

# ML Configuration
MODEL_CACHE_SIZE=100
BATCH_SIZE=16
MEMORY_LIMIT_GB=4

# Security
ENABLE_ENCRYPTION=false
API_RATE_LIMIT=10000
""",
            "staging": """# Staging Environment Variables
ENVIRONMENT=staging
DEBUG=false
LOG_LEVEL=INFO

# API Keys (Staging)
GOOGLE_API_KEY=your_staging_google_api_key_here
OPENAI_API_KEY=your_staging_openai_api_key_here

# Database (Staging)
DATABASE_URL=postgresql://llkjj_ml:password@staging-db:5432/llkjj_ml_staging

# ML Configuration
MODEL_CACHE_SIZE=500
BATCH_SIZE=24
MEMORY_LIMIT_GB=6

# Security
ENABLE_ENCRYPTION=true
API_RATE_LIMIT=5000
""",
            "production": """# Production Environment Variables
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=WARNING

# API Keys (Production) - Use secure secret management
GOOGLE_API_KEY=your_prod_google_api_key_here
OPENAI_API_KEY=your_prod_openai_api_key_here

# Database (Production)
DATABASE_URL=postgresql://llkjj_ml:secure_password@prod-db:5432/llkjj_ml_prod

# ML Configuration
MODEL_CACHE_SIZE=1000
BATCH_SIZE=32
MEMORY_LIMIT_GB=8

# Security
ENABLE_ENCRYPTION=true
API_RATE_LIMIT=1000
MAX_UPLOAD_SIZE_MB=50
""",
        }

        for env_name, template in templates.items():
            env_file = self.config_dir / f".env.{env_name}"
            if not env_file.exists():
                with open(env_file, "w") as f:
                    f.write(template)
                os.chmod(env_file, 0o600)  # Secure permissions

    def load_config(self, environment: str | Environment) -> EnvironmentConfig:
        """Load configuration for specific environment."""
        if isinstance(environment, str):
            environment = self.get_environment_from_string(environment)

        config_file = self.config_dir / f"{environment.value}.yaml"

        if not config_file.exists():
            self.logger.warning(
                f"Config file {config_file} not found, creating defaults"
            )
            self.create_default_configs()

        with open(config_file, encoding="utf-8") as f:
            config_data = yaml.safe_load(f)

        # Parse configuration
        config = EnvironmentConfig(
            environment=environment,
            debug=config_data.get("debug", False),
            database=DatabaseConfig(**config_data.get("database", {})),
            logging=LoggingConfig(**config_data.get("logging", {})),
            ml=MLConfig(**config_data.get("ml", {})),
            security=SecurityConfig(**config_data.get("security", {})),
            custom_settings=config_data.get("custom_settings", {}),
        )

        # Load environment variables
        self._load_env_variables(environment)

        # Apply environment variable overrides
        self._apply_env_overrides(config)

        self._current_config = config
        return config

    def _load_env_variables(self, environment: Environment) -> None:
        """Load environment-specific .env file."""
        env_file = self.config_dir / f".env.{environment.value}"
        if env_file.exists():
            with open(env_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        os.environ[key.strip()] = value.strip()

    def _apply_env_overrides(self, config: EnvironmentConfig) -> None:
        """Apply environment variable overrides to configuration."""
        # Debug override
        debug_env = os.getenv("DEBUG")
        if debug_env:
            config.debug = debug_env.lower() in ("true", "1", "yes")

        # Logging overrides
        log_level = os.getenv("LOG_LEVEL")
        if log_level:
            config.logging.level = log_level

        # ML overrides
        model_cache_size = os.getenv("MODEL_CACHE_SIZE")
        if model_cache_size:
            config.ml.model_cache_size = int(model_cache_size)
        batch_size = os.getenv("BATCH_SIZE")
        if batch_size:
            config.ml.batch_size = int(batch_size)
        memory_limit = os.getenv("MEMORY_LIMIT_GB")
        if memory_limit:
            config.ml.memory_limit_gb = int(memory_limit)

        # Security overrides
        enable_encryption = os.getenv("ENABLE_ENCRYPTION")
        if enable_encryption:
            config.security.enable_encryption = enable_encryption.lower() in (
                "true",
                "1",
                "yes",
            )
        api_rate_limit = os.getenv("API_RATE_LIMIT")
        if api_rate_limit:
            config.security.api_rate_limit = int(api_rate_limit)
        max_upload_size = os.getenv("MAX_UPLOAD_SIZE_MB")
        if max_upload_size:
            config.security.max_upload_size_mb = int(max_upload_size)

    def _save_config(self, config: EnvironmentConfig, filename: str) -> None:
        """Save configuration to file."""
        config_file = self.config_dir / filename
        with open(config_file, "w", encoding="utf-8") as f:
            yaml.dump(config.to_dict(), f, default_flow_style=False, indent=2)

    def get_secret(self, key: str, default: str | None = None) -> str | None:
        """Get secret value, decrypting if necessary."""
        value = os.getenv(key, default)
        if value and self.secret_manager.is_encrypted(value):
            return self.secret_manager.decrypt_secret(value)
        return value

    def set_secret(self, key: str, value: str, encrypt: bool = True) -> None:
        """Set secret value, encrypting if requested."""
        if (
            encrypt
            and self._current_config
            and self._current_config.security.enable_encryption
        ):
            encrypted_value = self.secret_manager.encrypt_secret(value)
            os.environ[key] = encrypted_value
        else:
            os.environ[key] = value

    def get_current_config(self) -> EnvironmentConfig | None:
        """Get currently loaded configuration."""
        return self._current_config

    def validate_config(self, config: EnvironmentConfig) -> list[str]:
        """Validate configuration and return list of issues."""
        issues: list[str] = []

        # Check required environment variables
        required_vars = ["GOOGLE_API_KEY"]
        if config.environment == Environment.PRODUCTION:
            required_vars.extend(["DATABASE_URL"])

        for var in required_vars:
            if not os.getenv(var):
                issues.append(f"Missing required environment variable: {var}")

        # Validate ML configuration
        if config.ml.memory_limit_gb < 2:
            issues.append("ML memory limit should be at least 2GB")

        if config.ml.batch_size < 1 or config.ml.batch_size > 64:
            issues.append("ML batch size should be between 1 and 64")

        # Validate security configuration
        if config.environment == Environment.PRODUCTION:
            if not config.security.enable_encryption:
                issues.append("Encryption should be enabled in production")

            if config.security.api_rate_limit > 10000:
                issues.append("API rate limit seems too high for production")

        return issues


# Global environment manager instance
_environment_manager: EnvironmentManager | None = None


def get_environment_manager() -> EnvironmentManager:
    """Get global environment manager instance."""
    global _environment_manager
    if _environment_manager is None:
        _environment_manager = EnvironmentManager()
    return _environment_manager


def load_environment_config(environment: str | Environment) -> EnvironmentConfig:
    """Load configuration for specific environment."""
    return get_environment_manager().load_config(environment)


def get_current_environment() -> Environment:
    """Get current environment from environment variable or default."""
    env_str = os.getenv("ENVIRONMENT", "development")
    return get_environment_manager().get_environment_from_string(env_str)


def initialize_environment(
    environment: str | Environment | None = None,
) -> EnvironmentConfig:
    """Initialize environment configuration."""
    if environment is None:
        environment = get_current_environment()

    manager = get_environment_manager()
    config = manager.load_config(environment)

    # Validate configuration
    issues = manager.validate_config(config)
    if issues:
        logger = logging.getLogger(__name__)
        logger.warning("Configuration issues found: %s", issues)

    return config
