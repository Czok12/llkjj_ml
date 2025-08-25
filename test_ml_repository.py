"""
LLKJJ ML Repository Integration Test

Testet die Integration zwischen MLPlugin v2.0 und PostgreSQL-Repository.

Author: LLKJJ ML Team  
Version: 2.0.0
Date: 2025-01-25
"""

import pytest
import asyncio
from pathlib import Path
from uuid import uuid4
from datetime import datetime

from llkjj_ml_plugin_v2 import MLPlugin
from src.config.training_config import TrainingConfig


class MockRepository:
    """Mock-Repository für Testing."""
    
    def __init__(self):
        self.embeddings = []
        self.training_data = []
        self.models = []
    
    async def store_embedding(self, embedding_data):
        """Store embedding mock."""
        self.embeddings.append(embedding_data)
        return {"success": True, "id": str(uuid4())}
    
    async def get_similar_items(self, embedding, limit=5):
        """Get similar items mock."""
        return []
    
    async def get_pending_training_data(self, limit=100, min_confidence_rating=3):
        """Get training data mock."""
        return self.training_data
    
    async def save_training_feedback(self, feedback_data):
        """Save training feedback mock."""
        self.training_data.append(feedback_data)
        return {"success": True}
    
    async def get_training_readiness(self):
        """Get training readiness mock."""
        return {
            "ready_for_training": len(self.training_data) >= 10,
            "readiness_score": min(1.0, len(self.training_data) / 50.0),
            "readiness_checks": {
                "sufficient_high_confidence_feedback": len(self.training_data) >= 10,
                "diverse_supplier_feedback": True,
                "recent_feedback_activity": True,
                "embedding_diversity": True
            }
        }
    
    async def save_trained_spacy_model(self, model_name, model_path, model_metadata, training_metrics):
        """Save trained model mock."""
        self.models.append({
            "model_name": model_name,
            "model_path": str(model_path),
            "metadata": model_metadata,
            "metrics": training_metrics
        })
        return {"success": True, "model_id": str(uuid4())}
    
    async def get_latest_spacy_model(self, model_type="text_classifier"):
        """Get latest model mock."""
        if self.models:
            return self.models[-1]
        return None


@pytest.fixture
def mock_repository():
    """Repository-Fixture für Tests."""
    return MockRepository()


@pytest.fixture 
def training_config():
    """Training-Config-Fixture."""
    return TrainingConfig(
        model_name="test_model",
        max_epochs=5,
        batch_size=16,
        max_training_items=100
    )


@pytest.mark.asyncio
async def test_ml_plugin_initialization(mock_repository):
    """Test MLPlugin v2.0 Initialisierung."""
    plugin = MLPlugin(
        repository=mock_repository, 
        validate_env=False
    )
    
    assert plugin is not None
    assert plugin.repository == mock_repository
    print("✅ MLPlugin v2.0 erfolgreich initialisiert")


@pytest.mark.asyncio  
async def test_training_feedback_submission(mock_repository):
    """Test Training-Feedback-Einreichung."""
    plugin = MLPlugin(repository=mock_repository, validate_env=False)
    
    feedback_result = await plugin.submit_training_feedback(
        invoice_item_id=uuid4(),
        user_id=uuid4(),
        original_classification="4000",
        corrected_classification="4400",
        feedback_type="manual_correction", 
        user_confidence_rating=5,
        item_description="Elektroinstallation Hauptverteilung",
        supplier_context="Elektro Mustermann GmbH"
    )
    
    assert feedback_result["success"] == True
    assert len(mock_repository.training_data) == 1
    print("✅ Training-Feedback erfolgreich eingereicht")


@pytest.mark.asyncio
async def test_training_readiness(mock_repository):
    """Test Training-Readiness-Assessment."""
    plugin = MLPlugin(repository=mock_repository, validate_env=False)
    
    # Füge Training-Daten hinzu
    for i in range(15):
        await plugin.submit_training_feedback(
            invoice_item_id=uuid4(),
            user_id=uuid4(),
            original_classification="4000",
            corrected_classification="4400", 
            feedback_type="manual_correction",
            user_confidence_rating=4,
            item_description=f"Test Item {i}",
            supplier_context="Test Supplier"
        )
    
    readiness = await plugin.repository.get_training_readiness()
    
    assert readiness["ready_for_training"] == True
    assert readiness["readiness_score"] >= 0.3
    print(f"✅ Training-Readiness: {readiness['readiness_score']:.2f}")


if __name__ == "__main__":
    print("🧪 LLKJJ ML Repository Integration Tests")
    pytest.main([__file__, "-v"])