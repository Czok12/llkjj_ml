from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")

class FieldInfo:
    default: Any
    metadata: dict[str, Any]

# pydantic Field returns an object used at runtime but mypy should allow assignment
# to typed class variables, so expose as Any here to suppress incompatible-assignment
def Field(default: Any = ..., *args: Any, **kwargs: Any) -> Any: ...
def validator(
    *args: Any, **kwargs: Any
) -> Callable[[Callable[..., T]], Callable[..., T]]: ...
def field_validator(
    *args: Any, **kwargs: Any
) -> Callable[[Callable[..., T]], Callable[..., T]]: ...

class ValidationError(Exception):
    errors: Any

class BaseModel:
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

    # We expose only the high-level model_dump and model_copy signatures used
    # across the codebase. Keep signatures simple to avoid stub-type conflicts.
    def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]: ...
    def model_copy(self, *args: Any, **kwargs: Any) -> BaseModel: ...
    def model_dump_json(self, *args: Any, **kwargs: Any) -> str: ...
    @classmethod
    def model_validate(cls, data: Any) -> BaseModel: ...

class BaseSettings(BaseModel):
    pass

# pydantic v2 config helpers
ConfigDict = dict[str, Any]

# ValidationInfo / ValidationError helpers
ValidationInfo = Any
