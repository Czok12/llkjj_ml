"""
Resource Manager für LLKJJ ML - Memory und Resource Management
"""

from typing import Any


class ResourceManager:
    """Mock Resource Manager für Migration."""

    def __init__(self) -> None:
        self.resources: dict[str, Any] = {}

    def get_resource(self, name: str) -> Any:
        """Get resource by name."""
        return self.resources.get(name)

    def set_resource(self, name: str, resource: Any) -> None:
        """Set resource by name."""
        self.resources[name] = resource

    def get_memory_report(self) -> dict[str, Any]:
        """Get memory usage report für Kompatibilität mit Pipeline."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()

        return {
            "memory_rss_mb": memory_info.rss / 1024 / 1024,
            "memory_vms_mb": memory_info.vms / 1024 / 1024,
            "memory_percent": process.memory_percent(),
            "resources_count": len(self.resources),
        }


# Singleton instance
_resource_manager: ResourceManager | None = None


def get_resource_manager() -> ResourceManager:
    """Get global resource manager instance."""
    global _resource_manager
    if _resource_manager is None:
        _resource_manager = ResourceManager()
    return _resource_manager
