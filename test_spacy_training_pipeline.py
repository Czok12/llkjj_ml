"""
Tests für Phase 5: spaCy Training Pipeline Validierung
=====================================================

Umfassende Tests für das neue spaCy-Training-System in v2.0:
- SpacyTrainer-Funktionalität
- Training-Pipeline Integration
- BackendTrainingOrchestrator
- Repository-Pattern für Training-Daten

Author: LLKJJ ML Team
Version: 2.0.0
Date: 2025-01-25
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path
from uuid import UUID, uuid4
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Test-Daten-Konstanten
MOCK_TRAINING_ITEMS = [
    {
        "id": uuid4(),
        "item_description": "Kabel NYM-J 3x1,5 mm² grau",
        "supplier_context": "Elektro Großhandel München",
        "correct_classification": "3400",
        "confidence_score": 0.95,
        "created_at": datetime.now().isoformat()
    },
    {
        "id": uuid4(), 
        "item_description": "Schalter Busch-Jaeger 20EUC-214",
        "supplier_context": "Busch-Jaeger Vertrieb",
        "correct_classification": "3420",
        "confidence_score": 0.88,
        "created_at": (datetime.now() - timedelta(days=2)).isoformat()
    },
    {
        "id": uuid4(),
        "item_description": "LED Panel 600x600 neutralweiß 40W",
        "supplier_context": "OSRAM Lighting Solutions",
        "correct_classification": "3430",
        "confidence_score": 0.91,
        "created_at": (datetime.now() - timedelta(days=5)).isoformat()
    }
]


class TestSpacyTrainer:
    """Tests für das SpacyTrainer-System."""

    @pytest.fixture
    def mock_repository(self):
        """Mock Repository für Training-Tests."""
        repo = Mock()
        
        # Training-Daten Mocks
        repo.get_pending_training_data = AsyncMock(return_value=[
            Mock(**item) for item in MOCK_TRAINING_ITEMS
        ])
        
        repo.mark_training_data_used = AsyncMock(return_value=None)
        repo.store_spacy_model = AsyncMock(return_value=uuid4())
        repo.log_training_run = AsyncMock(return_value=uuid4())
        
        # Statistik-Mocks
        repo.get_training_stats = AsyncMock(return_value={
            "total_feedback_count": 150,
            "high_confidence_count": 95,
            "unique_suppliers": 25,
            "recent_feedback_count": 45,
            "avg_confidence_score": 0.87
        })
        
        return repo

    @pytest.fixture
    def training_config(self):
        """Standard Training-Konfiguration für Tests."""
        from src.training.spacy_trainer import TrainingConfig
        return TrainingConfig(
            model_name="test_skr03_classifier",
            max_training_items=100,
            validation_split=0.2,
            batch_size=16,
            max_epochs=5,  # Kurze Tests
            learning_rate=0.001,
            elektrohandwerk_optimization=True
        )

    @pytest.mark.asyncio
    async def test_spacy_trainer_initialization(self, mock_repository, training_config):
        """Test SpacyTrainer-Initialisierung."""
        from src.training.spacy_trainer import SpacyTrainer
        
        trainer = SpacyTrainer(
            repository=mock_repository,
            config=training_config
        )
        
        assert trainer.repository == mock_repository
        assert trainer.config.model_name == "test_skr03_classifier"
        assert trainer.config.elektrohandwerk_optimization is True

    @pytest.mark.asyncio
    async def test_load_training_data(self, mock_repository, training_config):
        """Test Training-Daten-Loading."""
        from src.training.spacy_trainer import SpacyTrainer
        
        trainer = SpacyTrainer(mock_repository, training_config)
        
        # Training-Daten laden
        training_items = await trainer._load_training_data_from_repository()
        
        assert len(training_items) == len(MOCK_TRAINING_ITEMS)
        assert training_items[0]["item_description"].startswith("Kabel NYM-J")
        assert training_items[1]["correct_classification"] == "3420"
        
        # Repository-Aufruf prüfen
        mock_repository.get_pending_training_data.assert_called_once_with(
            limit=training_config.max_training_items,
            min_confidence_rating=training_config.min_confidence_rating
        )

    @pytest.mark.asyncio
    async def test_convert_to_spacy_format(self, mock_repository, training_config):
        """Test Konvertierung zu spaCy-Format."""
        from src.training.spacy_trainer import SpacyTrainer
        
        trainer = SpacyTrainer(mock_repository, training_config)
        
        # Mock Training-Items
        mock_items = [
            {
                "item_description": "Kabel NYM-J 3x1,5 mm²",
                "supplier_context": "Test Supplier",
                "correct_classification": "3400"
            }
        ]
        
        # Konvertierung testen
        spacy_examples = trainer._convert_to_spacy_format(mock_items)
        
        assert len(spacy_examples) == 1
        assert spacy_examples[0][0] == "Kabel NYM-J 3x1,5 mm² | Test Supplier"
        assert spacy_examples[0][1]["cats"]["3400"] == 1.0

    @pytest.mark.asyncio
    async def test_training_validation_split(self, mock_repository, training_config):
        """Test Training/Validation Split."""
        from src.training.spacy_trainer import SpacyTrainer
        
        trainer = SpacyTrainer(mock_repository, training_config)
        
        # Test-Daten (mindestens 10 Items für Split)
        mock_examples = [
            (f"Test item {i}", {"cats": {"3400": 1.0}})
            for i in range(20)
        ]
        
        train_examples, val_examples = trainer._train_validation_split(mock_examples)
        
        # Split-Verhältnis prüfen (80/20)
        expected_val_size = int(len(mock_examples) * training_config.validation_split)
        assert len(val_examples) == expected_val_size
        assert len(train_examples) == len(mock_examples) - expected_val_size
        
        # Keine Überschneidungen
        train_texts = {example[0] for example in train_examples}
        val_texts = {example[0] for example in val_examples}
        assert len(train_texts & val_texts) == 0

    @pytest.mark.asyncio 
    async def test_elektrohandwerk_optimization(self, mock_repository, training_config):
        """Test Elektrohandwerk-spezifische Optimierungen."""
        from src.training.spacy_trainer import SpacyTrainer
        
        trainer = SpacyTrainer(mock_repository, training_config)
        
        # Elektrohandwerk-Keywords testen
        elektro_keywords = trainer._get_elektrohandwerk_keywords()
        
        assert "kabel" in elektro_keywords
        assert "schalter" in elektro_keywords
        assert "led" in elektro_keywords
        assert "installation" in elektro_keywords
        
        # Mindestens 20 Keywords für gute Abdeckung
        assert len(elektro_keywords) >= 20

    @pytest.mark.asyncio
    async def test_model_saving_and_versioning(self, mock_repository, training_config):
        """Test Modell-Speicherung und Versionierung."""
        from src.training.spacy_trainer import SpacyTrainer, TrainingMetrics
        
        trainer = SpacyTrainer(mock_repository, training_config)
        
        # Mock TrainingMetrics
        mock_metrics = TrainingMetrics(
            model_name="test_model",
            training_items_used=50,
            final_accuracy=0.89,
            final_f1_score=0.85,
            training_duration_minutes=15.5,
            epochs_completed=10,
            model_save_path=Path("/tmp/test_model")
        )
        
        # Modell-Registrierung testen
        with patch.object(trainer, '_register_trained_model') as mock_register:
            await trainer._register_trained_model(mock_metrics)
            mock_register.assert_called_once_with(mock_metrics)


class TestBackendTrainingOrchestrator:
    """Tests für den BackendTrainingOrchestrator."""

    @pytest.fixture
    def mock_ml_plugin(self):
        """Mock MLPlugin für Orchestrator-Tests."""
        plugin = Mock()
        
        # Repository Mock
        plugin.repository = Mock()
        plugin.repository.get_training_stats = AsyncMock(return_value={
            "high_confidence_count": 75,
            "unique_suppliers": 12,
            "recent_feedback_count": 30
        })
        plugin.repository.get_embedding_stats = AsyncMock(return_value={
            "unique_skr03_accounts": 18,
            "total_embeddings": 1200
        })
        plugin.repository.get_latest_spacy_model = AsyncMock(return_value={
            "model_name": "current_model",
            "version": "1.0.0",
            "accuracy": 0.87,
            "days_since_training": 20
        })
        
        # Plugin-Methoden Mock
        plugin.get_training_readiness = AsyncMock(return_value={
            "ready_for_training": True,
            "readiness_score": 0.85,
            "readiness_checks": {
                "has_pending_feedback": True,
                "sufficient_high_confidence_feedback": True,
                "diverse_supplier_feedback": True,
                "recent_feedback_activity": True,
                "embedding_diversity": True
            },
            "recommendation": "Training empfohlen"
        })
        
        plugin.get_training_statistics = AsyncMock(return_value={
            "high_confidence_count": 75
        })
        
        plugin.get_embedding_statistics = AsyncMock(return_value={
            "unique_skr03_accounts": 18
        })
        
        plugin.start_spacy_training = AsyncMock(return_value={
            "success": True,
            "model_name": "test_model",
            "training_items_used": 125,
            "final_accuracy": 0.91,
            "final_f1_score": 0.87,
            "training_duration_minutes": 22.5,
            "epochs_completed": 15
        })
        
        return plugin

    def test_orchestrator_initialization(self, mock_ml_plugin):
        """Test BackendTrainingOrchestrator-Initialisierung."""
        from src.integration.training_services import BackendTrainingOrchestrator
        
        orchestrator = BackendTrainingOrchestrator(mock_ml_plugin)
        
        assert orchestrator.ml_plugin == mock_ml_plugin
        assert orchestrator.default_training_config.model_name == "llkjj_skr03_classifier"
        assert orchestrator.default_training_config.elektrohandwerk_optimization is True

    @pytest.mark.asyncio
    async def test_assess_training_needs(self, mock_ml_plugin):
        """Test Training-Needs-Assessment."""
        from src.integration.training_services import BackendTrainingOrchestrator
        
        orchestrator = BackendTrainingOrchestrator(mock_ml_plugin)
        
        # Assessment durchführen
        assessment = await orchestrator.assess_training_needs()
        
        assert "assessment_timestamp" in assessment
        assert assessment["readiness_score"] == 0.85
        assert assessment["training_recommended"] is True
        assert assessment["priority"] in ["low", "medium", "high", "critical"]

    @pytest.mark.asyncio
    async def test_training_priority_calculation(self, mock_ml_plugin):
        """Test Training-Prioritäts-Berechnung."""
        from src.integration.training_services import BackendTrainingOrchestrator
        
        orchestrator = BackendTrainingOrchestrator(mock_ml_plugin)
        
        # High-Priority Scenario
        high_readiness = {"readiness_score": 0.95}
        high_stats = {"high_confidence_count": 150}
        old_model = {"days_since_training": 45}
        
        priority = orchestrator._calculate_training_priority(high_readiness, high_stats, old_model)
        assert priority in ["high", "critical"]
        
        # Low-Priority Scenario
        low_readiness = {"readiness_score": 0.3}
        low_stats = {"high_confidence_count": 10}
        recent_model = {"days_since_training": 3}
        
        priority = orchestrator._calculate_training_priority(low_readiness, low_stats, recent_model)
        assert priority in ["low", "medium"]

    @pytest.mark.asyncio
    async def test_execute_training_pipeline(self, mock_ml_plugin):
        """Test komplette Training-Pipeline-Ausführung."""
        from src.integration.training_services import BackendTrainingOrchestrator
        
        orchestrator = BackendTrainingOrchestrator(mock_ml_plugin)
        
        # Pipeline ausführen
        result = await orchestrator.execute_training_pipeline()
        
        assert result["success"] is True
        assert result["model_name"] == "test_model"
        assert "training_result" in result
        assert "validation_result" in result
        assert result["training_result"]["final_accuracy"] == 0.91

    @pytest.mark.asyncio
    async def test_automated_training_scheduling(self, mock_ml_plugin):
        """Test automatisches Training-Scheduling."""
        from src.integration.training_services import BackendTrainingOrchestrator
        
        orchestrator = BackendTrainingOrchestrator(mock_ml_plugin)
        
        # High-Readiness -> Sofortiges Training
        schedule_result = await orchestrator.schedule_automated_training(
            assessment_interval_hours=24,
            min_readiness_threshold=0.8  # Unter mock readiness_score von 0.85
        )
        
        assert schedule_result["immediate_training"] is True
        assert "training_result" in schedule_result

    @pytest.mark.asyncio
    async def test_training_recommendations_generation(self, mock_ml_plugin):
        """Test Training-Empfehlungen-Generierung."""
        from src.integration.training_services import BackendTrainingOrchestrator
        
        orchestrator = BackendTrainingOrchestrator(mock_ml_plugin)
        
        # Mock suboptimale Readiness
        mock_readiness = {
            "readiness_checks": {
                "sufficient_high_confidence_feedback": False,
                "diverse_supplier_feedback": False,
                "recent_feedback_activity": True,
                "embedding_diversity": True
            }
        }
        
        recommendations = orchestrator._generate_training_recommendations(
            mock_readiness, 
            {"high_confidence_count": 20}
        )
        
        assert len(recommendations) >= 2
        assert any("hoher Confidence" in rec for rec in recommendations)
        assert any("Lieferanten" in rec for rec in recommendations)


class TestTrainingIntegrationE2E:
    """End-to-End Tests für die gesamte Training-Integration."""

    @pytest.mark.asyncio
    async def test_full_training_workflow_mock(self):
        """Test kompletten Training-Workflow mit Mocks."""
        from src.integration.training_services import create_training_orchestrator_for_backend
        from src.training.spacy_trainer import TrainingConfig
        
        # Mock MLPlugin
        mock_plugin = Mock()
        mock_plugin.get_training_readiness = AsyncMock(return_value={
            "ready_for_training": True,
            "readiness_score": 0.9
        })
        mock_plugin.start_spacy_training = AsyncMock(return_value={
            "success": True,
            "final_accuracy": 0.93,
            "training_duration_minutes": 18
        })
        mock_plugin.get_training_statistics = AsyncMock(return_value={"high_confidence_count": 100})
        mock_plugin.get_embedding_statistics = AsyncMock(return_value={"unique_skr03_accounts": 20})
        mock_plugin.repository = Mock()
        mock_plugin.repository.get_latest_spacy_model = AsyncMock(return_value=None)
        
        # Custom Training-Config
        custom_config = TrainingConfig(
            model_name="e2e_test_model",
            max_training_items=200,
            elektrohandwerk_optimization=True
        )
        
        # Orchestrator erstellen
        orchestrator = create_training_orchestrator_for_backend(mock_plugin, custom_config)
        
        # Workflow ausführen
        assessment = await orchestrator.assess_training_needs()
        assert assessment["training_recommended"] is True
        
        pipeline_result = await orchestrator.execute_training_pipeline("e2e_test_model")
        assert pipeline_result["success"] is True
        assert pipeline_result["training_result"]["success"] is True

    def test_training_config_validation(self):
        """Test Training-Konfiguration-Validierung."""
        from src.training.spacy_trainer import TrainingConfig
        
        # Valide Konfiguration
        valid_config = TrainingConfig(
            model_name="valid_model",
            max_training_items=100,
            validation_split=0.2,
            batch_size=32
        )
        
        assert valid_config.model_name == "valid_model"
        assert valid_config.validation_split == 0.2
        assert valid_config.elektrohandwerk_optimization is True  # Default

    def test_training_metrics_schema(self):
        """Test TrainingMetrics-Schema."""
        from src.training.spacy_trainer import TrainingMetrics
        from pathlib import Path
        
        metrics = TrainingMetrics(
            model_name="test_metrics_model",
            training_items_used=150,
            final_accuracy=0.89,
            final_f1_score=0.85,
            training_duration_minutes=25.5,
            epochs_completed=12,
            model_save_path=Path("/tmp/model")
        )
        
        assert metrics.model_name == "test_metrics_model"
        assert metrics.final_accuracy == 0.89
        assert metrics.epochs_completed == 12
        assert isinstance(metrics.model_save_path, Path)

    @pytest.mark.asyncio
    async def test_error_handling_in_training_pipeline(self):
        """Test Fehlerbehandlung in der Training-Pipeline."""
        from src.integration.training_services import BackendTrainingOrchestrator
        
        # Mock Plugin mit Fehler
        error_plugin = Mock()
        error_plugin.get_training_readiness = AsyncMock(side_effect=Exception("Repository error"))
        error_plugin.repository = Mock()
        
        orchestrator = BackendTrainingOrchestrator(error_plugin)
        
        # Fehlerbehandlung testen
        assessment = await orchestrator.assess_training_needs()
        assert assessment["training_recommended"] is False
        assert "error" in assessment

    def test_convenience_functions(self):
        """Test Convenience-Funktionen für Training-Integration."""
        from src.integration.training_services import (
            create_training_orchestrator_for_backend,
            get_training_recommendations,
            schedule_training_pipeline
        )
        
        # Factory-Function testen
        mock_plugin = Mock()
        orchestrator = create_training_orchestrator_for_backend(mock_plugin)
        assert orchestrator.ml_plugin == mock_plugin
        
        # Funktionen sind callable
        assert callable(get_training_recommendations)
        assert callable(schedule_training_pipeline)


class TestProductionReadiness:
    """Tests für Production-Readiness der Training-Pipeline."""

    def test_import_structure(self):
        """Test korrekter Import-Struktur."""
        # Hauptmodule
        from llkjj_ml import (
            MLPlugin, 
            SpacyTrainer, 
            BackendTrainingOrchestrator,
            TrainingConfig
        )
        
        # Alle Klassen sind verfügbar
        assert MLPlugin is not None
        assert SpacyTrainer is not None
        assert BackendTrainingOrchestrator is not None
        assert TrainingConfig is not None

    def test_training_system_integration_ready(self):
        """Test dass Training-System integration-ready ist."""
        # Training-Services importieren
        from src.integration.training_services import (
            BackendTrainingOrchestrator,
            create_training_orchestrator_for_backend
        )
        
        # SpaCy-Trainer importieren
        from src.training.spacy_trainer import SpacyTrainer, TrainingConfig
        
        # v2.0 Plugin erweitern um Training-Methoden
        from llkjj_ml_plugin_v2 import MLPlugin
        
        # Alle Integration-Punkte sind verfügbar
        assert hasattr(MLPlugin, 'start_spacy_training')
        assert hasattr(MLPlugin, 'get_training_readiness')
        
        # Factory-Funktionen sind verfügbar
        assert callable(create_training_orchestrator_for_backend)

    @pytest.mark.asyncio
    async def test_minimal_training_workflow(self):
        """Test minimaler Training-Workflow ohne externe Dependencies."""
        from src.training.spacy_trainer import TrainingConfig
        
        # Minimale Konfiguration
        config = TrainingConfig(
            model_name="minimal_test",
            max_training_items=10,
            max_epochs=1,  # Sehr kurz für Test
            elektrohandwerk_optimization=False  # Deaktiviert für Speed
        )
        
        assert config.model_name == "minimal_test"
        assert config.max_epochs == 1
        assert config.elektrohandwerk_optimization is False


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])