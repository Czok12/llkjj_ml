"""
ML Service Package - Main Entry Point für CLI
============================================

Ermöglicht Ausführung des ML Service CLI über:
    python -m ml_service
    poetry run python -m ml_service

Weiterleitung an die CLI-Implementierung.
"""

from ml_service.cli import main

if __name__ == "__main__":
    main()
