from typing import Any

# Minimal stub for pydantic_settings.SettingsConfigDict used in code
SettingsConfigDict = dict[str, Any]

class BaseSettings:
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
