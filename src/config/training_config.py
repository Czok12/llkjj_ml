"""
Training Configuration für LLKJJ ML spaCy-Training

Konfigurationsdatenklasse für spaCy-Training-Parameter.

Author: LLKJJ ML Team
Version: 2.0.0
Date: 2025-01-25
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

__all__ = ["TrainingConfig"]


@dataclass
class TrainingConfig:
    """Konfiguration für spaCy-Training."""
    
    # Modell-Konfiguration
    model_name: str = "llkjj_skr03_classifier"
    base_model_path: Optional[Path] = None
    
    # Training-Parameter
    max_epochs: int = 50
    min_epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 0.001
    dropout_rate: float = 0.2
    early_stopping_patience: int = 5
    
    # Daten-Parameter
    min_confidence_rating: int = 3
    max_training_items: int = 2000
    validation_split: float = 0.2
    
    # Elektrohandwerk-spezifisch
    elektrohandwerk_optimization: bool = True
    
    # Experiment-Tracking
    mlflow_tracking: bool = False
    experiment_name: str = "llkjj_spacy_training"
    
    def to_dict(self):
        """Konvertiere zu Dictionary."""
        return {
            "model_name": self.model_name,
            "base_model_path": str(self.base_model_path) if self.base_model_path else None,
            "max_epochs": self.max_epochs,
            "min_epochs": self.min_epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "dropout_rate": self.dropout_rate,
            "early_stopping_patience": self.early_stopping_patience,
            "min_confidence_rating": self.min_confidence_rating,
            "max_training_items": self.max_training_items,
            "validation_split": self.validation_split,
            "elektrohandwerk_optimization": self.elektrohandwerk_optimization,
            "mlflow_tracking": self.mlflow_tracking,
            "experiment_name": self.experiment_name
        }