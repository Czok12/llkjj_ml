#!/usr/bin/env python3
"""
LLKJJ ML Performance Monitor
Überwacht Pipeline-Performance und erstellt Reports
"""

import json
from datetime import datetime
from pathlib import Path

import psutil


def monitor_memory():
    """Memory Usage Monitoring"""
    process = psutil.Process()
    return {
        "rss": process.memory_info().rss / 1024 / 1024,  # MB
        "vms": process.memory_info().vms / 1024 / 1024,  # MB
        "percent": process.memory_percent(),
    }


def monitor_cpu():
    """CPU Usage Monitoring"""
    return {"percent": psutil.cpu_percent(interval=1), "cores": psutil.cpu_count()}


def create_performance_report():
    """Erstelle Performance Report"""
    timestamp = datetime.now().isoformat()

    report = {
        "timestamp": timestamp,
        "memory": monitor_memory(),
        "cpu": monitor_cpu(),
        "disk_usage": psutil.disk_usage(".").percent,
    }

    # Report speichern
    report_file = Path(".claude/logs/performance_report.json")
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Performance Report erstellt: {report_file}")
    return report


if __name__ == "__main__":
    create_performance_report()
