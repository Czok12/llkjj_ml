#!/usr/bin/env python3
# LLKJJ ML Pipeline - Health Check Service

"""
Health check endpoints and monitoring service for LLKJJ ML Pipeline.
Provides HTTP endpoints for system health monitoring and alerting.
"""

import asyncio
import logging
import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import psutil
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from deployment.environment import get_current_environment, initialize_environment
from deployment.monitoring import (
    PerformanceMetrics,
    get_alert_manager,
    get_health_checker,
    get_performance_monitor,
    get_production_logger,
)


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str
    timestamp: str
    components: dict[str, str]
    metrics: dict[str, float]
    message: str


class MetricsResponse(BaseModel):
    """Metrics response model."""

    timestamp: str
    system: dict[str, float]
    application: dict[str, Any]


# Global state
health_checker = get_health_checker()
performance_monitor = get_performance_monitor()
alert_manager = get_alert_manager()
logger = get_production_logger()
recent_metrics: list[PerformanceMetrics] = []


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifespan events."""
    # Startup
    logging.getLogger(__name__).info("Starting LLKJJ ML Health Service")

    # Initialize environment
    environment = get_current_environment()
    initialize_environment(environment)
    logging.getLogger(__name__).info("Initialized environment: %s", environment.value)

    # Start background monitoring
    monitoring_task = asyncio.create_task(background_monitoring())

    yield

    # Shutdown
    logging.getLogger(__name__).info("Shutting down LLKJJ ML Health Service")
    monitoring_task.cancel()
    try:
        await monitoring_task
    except asyncio.CancelledError:
        pass


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="LLKJJ ML Pipeline Health Service",
    description="Health monitoring and metrics for LLKJJ ML Pipeline",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check() -> HealthResponse:
    """Comprehensive health check endpoint."""
    try:
        health_status = health_checker.check_system_health()

        # Determine response message
        if health_status.status == "healthy":
            message = "All systems operational"
        elif health_status.status == "degraded":
            message = "Some components degraded"
        else:
            message = "System unhealthy - immediate attention required"

        return HealthResponse(
            status=health_status.status,
            timestamp=health_status.timestamp.isoformat(),
            components=health_status.components,
            metrics={
                "memory_usage_percent": health_status.memory_usage_percent,
                "disk_usage_percent": health_status.disk_usage_percent,
                "cpu_usage_percent": health_status.cpu_usage_percent,
            },
            message=message,
        )
    except Exception as e:
        logging.getLogger(__name__).error("Health check failed: %s", str(e))
        raise HTTPException(status_code=500, detail="Health check failed") from e


@app.get("/health/ready")
async def readiness_check() -> dict[str, str]:
    """Kubernetes-style readiness probe."""
    try:
        # Check if core components are ready
        # For now, just verify the service is up and running

        return {"status": "ready", "timestamp": datetime.now().isoformat()}
    except Exception as e:
        logging.getLogger(__name__).error("Readiness check failed: %s", str(e))
        raise HTTPException(status_code=503, detail="Service not ready") from e


@app.get("/health/live")
async def liveness_check() -> dict[str, str]:
    """Kubernetes-style liveness probe."""
    return {"status": "alive", "timestamp": datetime.now().isoformat()}


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics() -> MetricsResponse:
    """Get system and application metrics."""
    try:
        # System metrics
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        cpu_percent = psutil.cpu_percent(interval=1)

        # Process metrics
        process = psutil.Process()
        process_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Application metrics
        app_metrics: dict[str, Any] = {
            "recent_operations": len(recent_metrics),
            "successful_operations": len([m for m in recent_metrics if m.success]),
            "failed_operations": len([m for m in recent_metrics if not m.success]),
            "average_duration_ms": (
                sum(m.duration_ms for m in recent_metrics) / len(recent_metrics)
                if recent_metrics
                else 0.0
            ),
            "process_memory_mb": process_memory,
        }

        return MetricsResponse(
            timestamp=datetime.now().isoformat(),
            system={
                "memory_usage_percent": memory.percent,
                "memory_available_gb": memory.available / 1024 / 1024 / 1024,
                "disk_usage_percent": disk.percent,
                "disk_free_gb": disk.free / 1024 / 1024 / 1024,
                "cpu_usage_percent": cpu_percent,
            },
            application=app_metrics,
        )
    except Exception as e:
        logging.getLogger(__name__).error("Metrics collection failed: %s", str(e))
        raise HTTPException(status_code=500, detail="Metrics collection failed") from e


@app.post("/test/process")
async def test_processing() -> dict[str, str]:
    """Test endpoint to verify document processing capability."""
    try:
        with performance_monitor.measure_operation("test_processing"):
            # Simulate processing
            await asyncio.sleep(0.1)

            # Record test metric
            test_metric = PerformanceMetrics(
                operation="test_processing",
                duration_ms=100,
                memory_usage_mb=0,
                cpu_percent=psutil.cpu_percent(),
                timestamp=datetime.now(),
                success=True,
            )
            recent_metrics.append(test_metric)

            # Keep only recent metrics (last 100)
            if len(recent_metrics) > 100:
                recent_metrics.pop(0)

        return {"status": "success", "message": "Processing test completed"}
    except Exception as e:
        logging.getLogger(__name__).error("Processing test failed: %s", str(e))
        raise HTTPException(status_code=500, detail="Processing test failed") from e


@app.get("/debug/logs")
async def get_recent_logs(lines: int = 50) -> dict[str, Any]:
    """Get recent log entries for debugging."""
    try:
        log_file = Path("logs/llkjj_ml.log")
        if not log_file.exists():
            return {"logs": [], "message": "No log file found"}

        with open(log_file, encoding="utf-8") as f:
            all_lines = f.readlines()
            recent_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines

        return {
            "logs": [line.strip() for line in recent_lines],
            "total_lines": len(all_lines),
            "returned_lines": len(recent_lines),
        }
    except Exception as e:
        logging.getLogger(__name__).error("Log retrieval failed: %s", str(e))
        raise HTTPException(status_code=500, detail="Log retrieval failed") from e


@app.get("/debug/config")
async def get_configuration() -> dict[str, Any]:
    """Get current configuration (sanitized)."""
    try:
        from deployment.environment import get_environment_manager

        manager = get_environment_manager()
        config = manager.get_current_config()

        if not config:
            raise HTTPException(status_code=404, detail="No configuration loaded")

        # Sanitize sensitive information
        config_dict = config.to_dict()
        config_dict.pop("database", None)  # Remove DB credentials
        config_dict.pop("custom_settings", None)  # Remove potential secrets

        return config_dict
    except Exception as e:
        logging.getLogger(__name__).error("Configuration retrieval failed: %s", str(e))
        raise HTTPException(
            status_code=500, detail="Configuration retrieval failed"
        ) from e


async def background_monitoring() -> None:
    """Background task for continuous monitoring and alerting."""
    global recent_metrics

    while True:
        try:
            # Check system health
            health_status = health_checker.check_system_health()

            # Check for alerts
            alert_manager.check_alerts(health_status, recent_metrics)

            # Log health status
            logger.log_metric("system_health_check", 1)

            # Clean old metrics
            cutoff_time = datetime.now() - timedelta(hours=1)
            recent_metrics = [m for m in recent_metrics if m.timestamp > cutoff_time]

        except Exception as e:
            logging.getLogger(__name__).error("Background monitoring error: %s", str(e))

        # Wait 30 seconds before next check
        await asyncio.sleep(30)


def main() -> None:
    """Run the health check service."""
    import argparse

    parser = argparse.ArgumentParser(description="LLKJJ ML Health Service")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    parser.add_argument("--env", default="production", help="Environment")

    args = parser.parse_args()

    # Initialize environment
    config = initialize_environment(args.env)

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, config.logging.level),
        format="%(asctime)s | %(levelname)8s | %(name)s | %(message)s",
    )

    print("🚀 Starting LLKJJ ML Health Service")
    print(f"Environment: {config.environment.value}")
    print(f"Listening on: http://{args.host}:{args.port}")
    print(f"Health endpoint: http://{args.host}:{args.port}/health")
    print(f"Metrics endpoint: http://{args.host}:{args.port}/metrics")

    # Run the service
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=config.logging.level.lower(),
        access_log=True,
    )


if __name__ == "__main__":
    main()
