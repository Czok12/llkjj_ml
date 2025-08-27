"""
Resource Manager für LLKJJ ML - Memory und Resource Management
"""

from typing import Any


class ResourceManager:
    """Mock Resource Manager für Migration."""

    def __init__(self):
        self.resources = {}

    def get_resource(self, name: str) -> Any:
        """Get resource by name."""
        return self.resources.get(name)

    def set_resource(self, name: str, resource: Any) -> None:
        """Set resource by name."""
        self.resources[name] = resource


# Singleton instance
_resource_manager: ResourceManager | None = None


def get_resource_manager() -> ResourceManager:
    """Get global resource manager instance."""
    global _resource_manager
    if _resource_manager is None:
        _resource_manager = ResourceManager()
    return _resource_manager
