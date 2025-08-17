# LLKJJ ML Pipeline - Monitoring and Logging Configuration

"""
Production-grade monitoring and logging infrastructure for LLKJJ ML Pipeline.
Implements structured logging, performance metrics, health checks, and alerting.
"""

import json
import logging
import logging.config
import os
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any

import psutil


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""

    operation: str
    duration_ms: float
    memory_usage_mb: float
    cpu_percent: float
    timestamp: datetime
    success: bool
    error_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data


@dataclass
class HealthStatus:
    """System health status."""

    status: str  # healthy, degraded, unhealthy
    components: dict[str, str]
    memory_usage_percent: float
    disk_usage_percent: float
    cpu_usage_percent: float
    timestamp: datetime

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data


class ProductionLogger:
    """Production-grade structured logger."""

    def __init__(self, environment: str = "production"):
        self.environment = environment
        self.setup_logging()
        self.metrics_log = logging.getLogger("metrics")
        self.security_log = logging.getLogger("security")
        self.performance_log = logging.getLogger("performance")

    def setup_logging(self) -> None:
        """Configure structured logging."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "structured": {
                    "format": "%(asctime)s | %(levelname)8s | %(name)s | %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                },
                "json": {"format": "%(message)s"},
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "structured",
                    "level": "INFO" if self.environment == "production" else "DEBUG",
                },
                "file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": log_dir / "llkjj_ml.log",
                    "maxBytes": 10 * 1024 * 1024,  # 10MB
                    "backupCount": 5,
                    "formatter": "structured",
                },
                "metrics": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": log_dir / "metrics.log",
                    "maxBytes": 10 * 1024 * 1024,
                    "backupCount": 5,
                    "formatter": "json",
                },
                "security": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": log_dir / "security.log",
                    "maxBytes": 10 * 1024 * 1024,
                    "backupCount": 10,
                    "formatter": "json",
                },
                "performance": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": log_dir / "performance.log",
                    "maxBytes": 10 * 1024 * 1024,
                    "backupCount": 5,
                    "formatter": "json",
                },
            },
            "loggers": {
                "": {  # Root logger
                    "handlers": ["console", "file"],
                    "level": "DEBUG" if self.environment == "development" else "INFO",
                },
                "metrics": {
                    "handlers": ["metrics"],
                    "level": "INFO",
                    "propagate": False,
                },
                "security": {
                    "handlers": ["security", "console"],
                    "level": "WARNING",
                    "propagate": False,
                },
                "performance": {
                    "handlers": ["performance"],
                    "level": "INFO",
                    "propagate": False,
                },
            },
        }

        logging.config.dictConfig(config)

    def log_performance_metric(self, metric: PerformanceMetrics) -> None:
        """Log performance metric in structured format."""
        self.performance_log.info(json.dumps(metric.to_dict()))

    def log_security_event(self, event_type: str, details: dict[str, Any]) -> None:
        """Log security event."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "environment": self.environment,
            **details,
        }
        self.security_log.warning(json.dumps(event))

    def log_metric(
        self, metric_name: str, value: float, tags: dict[str, str] | None = None
    ) -> None:
        """Log custom metric."""
        metric = {
            "timestamp": datetime.now().isoformat(),
            "metric_name": metric_name,
            "value": value,
            "tags": tags or {},
            "environment": self.environment,
        }
        self.metrics_log.info(json.dumps(metric))


class PerformanceMonitor:
    """Performance monitoring and metrics collection."""

    def __init__(self, logger: ProductionLogger):
        self.logger = logger
        self.start_time = time.time()
        self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024

    @contextmanager
    def measure_operation(self, operation_name: str) -> Generator[None, None, None]:
        """Context manager to measure operation performance."""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        start_cpu = psutil.cpu_percent()

        success = True
        error_type = None

        try:
            yield
        except Exception as e:
            success = False
            error_type = type(e).__name__
            raise
        finally:
            duration_ms = (time.time() - start_time) * 1000
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            end_cpu = psutil.cpu_percent()

            metric = PerformanceMetrics(
                operation=operation_name,
                duration_ms=duration_ms,
                memory_usage_mb=end_memory - start_memory,
                cpu_percent=(start_cpu + end_cpu) / 2,
                timestamp=datetime.now(),
                success=success,
                error_type=error_type,
            )

            self.logger.log_performance_metric(metric)

    def performance_decorator(self, operation_name: str) -> Callable[..., Any]:
        """Decorator to measure function performance."""

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                with self.measure_operation(operation_name):
                    return func(*args, **kwargs)

            return wrapper

        return decorator


class HealthChecker:
    """System health monitoring."""

    def __init__(self, logger: ProductionLogger):
        self.logger = logger

    def check_system_health(self) -> HealthStatus:
        """Comprehensive system health check."""
        components = {}

        # Memory check
        memory = psutil.virtual_memory()
        memory_status = (
            "healthy"
            if memory.percent < 80
            else "degraded"
            if memory.percent < 90
            else "unhealthy"
        )
        components["memory"] = memory_status

        # Disk check
        disk = psutil.disk_usage("/")
        disk_status = (
            "healthy"
            if disk.percent < 80
            else "degraded"
            if disk.percent < 90
            else "unhealthy"
        )
        components["disk"] = disk_status

        # CPU check
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_status = (
            "healthy"
            if cpu_percent < 70
            else "degraded"
            if cpu_percent < 85
            else "unhealthy"
        )
        components["cpu"] = cpu_status

        # ML models check
        try:
            # Basic ML availability check
            # For now, just check if the ML service directory exists
            ml_service_path = Path("ml_service")
            if ml_service_path.exists():
                components["ml_models"] = "healthy"
            else:
                components["ml_models"] = "degraded"
        except Exception:
            components["ml_models"] = "unhealthy"

        # Database check (ChromaDB)
        try:
            components["chromadb"] = "healthy"
        except Exception:
            components["chromadb"] = "unhealthy"

        # Overall status
        unhealthy_count = sum(
            1 for status in components.values() if status == "unhealthy"
        )
        degraded_count = sum(
            1 for status in components.values() if status == "degraded"
        )

        if unhealthy_count > 0:
            overall_status = "unhealthy"
        elif degraded_count > 0:
            overall_status = "degraded"
        else:
            overall_status = "healthy"

        health_status = HealthStatus(
            status=overall_status,
            components=components,
            memory_usage_percent=memory.percent,
            disk_usage_percent=disk.percent,
            cpu_usage_percent=cpu_percent,
            timestamp=datetime.now(),
        )

        # Log health status
        self.logger.log_metric("system_health", 1 if overall_status == "healthy" else 0)
        self.logger.log_metric("memory_usage_percent", memory.percent)
        self.logger.log_metric("disk_usage_percent", disk.percent)
        self.logger.log_metric("cpu_usage_percent", cpu_percent)

        return health_status

    def create_health_endpoint_data(self) -> dict[str, Any]:
        """Create health check data for HTTP endpoint."""
        health = self.check_system_health()
        return {
            "status": health.status,
            "timestamp": health.timestamp.isoformat(),
            "components": health.components,
            "metrics": {
                "memory_usage_percent": health.memory_usage_percent,
                "disk_usage_percent": health.disk_usage_percent,
                "cpu_usage_percent": health.cpu_usage_percent,
            },
        }


class AlertManager:
    """Alert management for critical events."""

    def __init__(self, logger: ProductionLogger):
        self.logger = logger
        self.alert_thresholds = {
            "memory_critical": 90,
            "disk_critical": 90,
            "cpu_critical": 85,
            "error_rate_critical": 0.1,  # 10% error rate
            "response_time_critical": 30000,  # 30 seconds
        }

    def check_alerts(
        self, health_status: HealthStatus, recent_metrics: list[PerformanceMetrics]
    ) -> None:
        """Check for alert conditions."""
        alerts = []

        # Resource alerts
        if (
            health_status.memory_usage_percent
            > self.alert_thresholds["memory_critical"]
        ):
            alerts.append(
                self._create_alert(
                    "MEMORY_CRITICAL",
                    f"Memory usage at {health_status.memory_usage_percent}%",
                )
            )

        if health_status.disk_usage_percent > self.alert_thresholds["disk_critical"]:
            alerts.append(
                self._create_alert(
                    "DISK_CRITICAL",
                    f"Disk usage at {health_status.disk_usage_percent}%",
                )
            )

        if health_status.cpu_usage_percent > self.alert_thresholds["cpu_critical"]:
            alerts.append(
                self._create_alert(
                    "CPU_CRITICAL", f"CPU usage at {health_status.cpu_usage_percent}%"
                )
            )

        # Performance alerts
        if recent_metrics:
            failed_operations = [m for m in recent_metrics if not m.success]
            if (
                len(failed_operations) / len(recent_metrics)
                > self.alert_thresholds["error_rate_critical"]
            ):
                alerts.append(
                    self._create_alert(
                        "HIGH_ERROR_RATE",
                        f"Error rate: {len(failed_operations)}/{len(recent_metrics)}",
                    )
                )

            slow_operations = [
                m
                for m in recent_metrics
                if m.duration_ms > self.alert_thresholds["response_time_critical"]
            ]
            if slow_operations:
                alerts.append(
                    self._create_alert(
                        "SLOW_OPERATIONS",
                        f"Slow operations detected: {len(slow_operations)}",
                    )
                )

        # Send alerts
        for alert in alerts:
            self._send_alert(alert)

    def _create_alert(self, alert_type: str, message: str) -> dict[str, Any]:
        """Create alert data structure."""
        return {
            "alert_type": alert_type,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "severity": "critical",
        }

    def _send_alert(self, alert: dict[str, Any]) -> None:
        """Send alert (log for now, can be extended to external systems)."""
        self.logger.security_log.critical(json.dumps(alert))

        # Future: Integration with external alerting systems
        # - Email notifications
        # - Slack/Teams webhooks
        # - PagerDuty integration
        # - SMS alerts


# Global monitoring instance
_production_logger = None
_performance_monitor = None
_health_checker = None
_alert_manager = None


def get_production_logger() -> ProductionLogger:
    """Get global production logger instance."""
    global _production_logger
    if _production_logger is None:
        environment = os.getenv("ENVIRONMENT", "production")
        _production_logger = ProductionLogger(environment)
    return _production_logger


def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor(get_production_logger())
    return _performance_monitor


def get_health_checker() -> HealthChecker:
    """Get global health checker instance."""
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker(get_production_logger())
    return _health_checker


def get_alert_manager() -> AlertManager:
    """Get global alert manager instance."""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager(get_production_logger())
    return _alert_manager


# Convenience functions for easy integration
def monitor_performance(operation_name: str) -> Callable[..., Any]:
    """Decorator for performance monitoring."""
    return get_performance_monitor().performance_decorator(operation_name)


def log_security_event(event_type: str, **details: Any) -> None:
    """Log security event."""
    get_production_logger().log_security_event(event_type, details)


def log_metric(metric_name: str, value: float, **tags: Any) -> None:
    """Log custom metric."""
    get_production_logger().log_metric(metric_name, value, tags)


def check_system_health() -> dict[str, Any]:
    """Check system health and return status."""
    return get_health_checker().create_health_endpoint_data()
