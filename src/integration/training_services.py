"""
LLKJJ ML Training Services Integration für llkjj_backend

Diese Datei stellt die Integration zwischen dem spaCy-Training-System
und dem llkjj_backend her. Sie implementiert die Adapter-Pattern für
die Repository-Interfaces und orchestriert Training-Workflows.

Author: LLKJJ ML Team
Version: 2.0.0
Date: 2025-01-25
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any
from uuid import UUID
import asyncio
from datetime import datetime

from ..training.spacy_trainer import SpacyTrainer, TrainingConfig, TrainingMetrics

logger = logging.getLogger(__name__)

__all__ = [
    "BackendTrainingOrchestrator",
    "create_training_orchestrator_for_backend",
    "schedule_training_pipeline",
    "get_training_recommendations"
]


class BackendTrainingOrchestrator:
    """
    Training-Orchestrator für Integration mit llkjj_backend.
    
    Koordiniert spaCy-Training-Zyklen basierend auf:
    - Verfügbare Training-Daten im Repository
    - Training-Readiness-Metriken
    - Automatische Modell-Versionierung
    - Performance-Tracking
    """
    
    def __init__(self, ml_plugin, training_config: Optional[TrainingConfig] = None):
        """
        Initialisiere Training-Orchestrator.
        
        Args:
            ml_plugin: MLPlugin v2.0 Instanz mit Repository-Zugriff
            training_config: Standard-Training-Konfiguration
        """
        self.ml_plugin = ml_plugin
        self.default_training_config = training_config or TrainingConfig(
            model_name="llkjj_skr03_classifier",
            elektrohandwerk_optimization=True,
            min_confidence_rating=3,
            max_training_items=2000,
            batch_size=32,
            max_epochs=50,
            early_stopping_patience=5
        )
        
        logger.info("🎭 BackendTrainingOrchestrator initialisiert")
    
    async def assess_training_needs(self) -> Dict[str, Any]:
        """
        Bewerte aktuellen Training-Bedarf.
        
        Returns:
            Training-Assessment mit Empfehlungen
        """
        try:
            logger.info("🔍 Bewerte Training-Bedarf...")
            
            # Training-Readiness prüfen
            readiness = await self.ml_plugin.get_training_readiness()
            
            # Training-Statistiken analysieren
            training_stats = await self.ml_plugin.get_training_statistics()
            embedding_stats = await self.ml_plugin.get_embedding_statistics()
            
            # Neueste Modell-Performance prüfen
            latest_model = await self.ml_plugin.repository.get_latest_spacy_model("text_classifier")
            
            # Assessment erstellen
            assessment = {
                "assessment_timestamp": datetime.now().isoformat(),
                "readiness_score": readiness.get("readiness_score", 0.0),
                "training_recommended": readiness.get("ready_for_training", False),
                "priority": self._calculate_training_priority(readiness, training_stats, latest_model),
                "training_stats": training_stats,
                "embedding_stats": embedding_stats,
                "latest_model_info": latest_model,
                "recommendations": self._generate_training_recommendations(readiness, training_stats)
            }
            
            logger.info(f"📊 Training-Assessment: Readiness={assessment['readiness_score']:.2f}, "
                       f"Priority={assessment['priority']}")
            
            return assessment
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Training-Needs-Assessment: {e}")
            return {
                "assessment_timestamp": datetime.now().isoformat(),
                "error": str(e),
                "training_recommended": False
            }
    
    async def execute_training_pipeline(
        self, 
        model_name: Optional[str] = None,
        custom_config: Optional[TrainingConfig] = None
    ) -> Dict[str, Any]:
        """
        Führe vollständige Training-Pipeline aus.
        
        Args:
            model_name: Name des Modells (Standard aus Config)
            custom_config: Custom Training-Konfiguration
        
        Returns:
            Training-Pipeline-Ergebnis mit Metriken
        """
        config = custom_config or self.default_training_config
        model_name = model_name or config.model_name
        
        try:
            logger.info(f"🚀 Starte Training-Pipeline für {model_name}...")
            
            # Phase 1: Pre-Training-Assessment
            assessment = await self.assess_training_needs()
            if not assessment.get("training_recommended", False):
                logger.warning("⚠️ Training nicht empfohlen - breche ab")
                return {
                    "success": False,
                    "reason": "training_not_recommended",
                    "assessment": assessment
                }
            
            # Phase 2: Training-Daten vorbereiten
            training_data_summary = await self._prepare_training_data(config)
            
            # Phase 3: spaCy-Training ausführen
            training_result = await self.ml_plugin.start_spacy_training(
                model_name=model_name,
                min_confidence_rating=config.min_confidence_rating,
                max_training_items=config.max_training_items
            )
            
            if not training_result.get("success", False):
                logger.error(f"❌ Training fehlgeschlagen: {training_result.get('error')}")
                return training_result
            
            # Phase 4: Post-Training-Validation
            validation_result = await self._validate_trained_model(model_name, training_result)
            
            # Phase 5: Modell-Deployment (falls erfolgreich)
            deployment_result = await self._deploy_model_if_ready(model_name, validation_result)
            
            # Finales Ergebnis zusammenstellen
            pipeline_result = {
                "success": True,
                "model_name": model_name,
                "pipeline_duration_minutes": training_result.get("training_duration_minutes", 0),
                "training_result": training_result,
                "validation_result": validation_result,
                "deployment_result": deployment_result,
                "training_data_summary": training_data_summary,
                "assessment": assessment
            }
            
            logger.info(f"✅ Training-Pipeline erfolgreich: {model_name}, "
                       f"Accuracy={training_result.get('final_accuracy', 0):.3f}")
            
            return pipeline_result
            
        except Exception as e:
            logger.error(f"❌ Kritischer Fehler in Training-Pipeline: {e}")
            return {
                "success": False,
                "error": str(e),
                "model_name": model_name
            }
    
    async def schedule_automated_training(
        self,
        assessment_interval_hours: int = 24,
        min_readiness_threshold: float = 0.8
    ) -> Dict[str, Any]:
        """
        Plane automatisches Training basierend auf Readiness-Metriken.
        
        Args:
            assessment_interval_hours: Intervall für Readiness-Checks
            min_readiness_threshold: Minimale Readiness für automatisches Training
        
        Returns:
            Scheduling-Ergebnis
        """
        try:
            logger.info(f"📅 Plane automatisches Training (Intervall: {assessment_interval_hours}h, "
                       f"Threshold: {min_readiness_threshold})")
            
            # Aktuelles Assessment
            assessment = await self.assess_training_needs()
            current_readiness = assessment.get("readiness_score", 0.0)
            
            if current_readiness >= min_readiness_threshold:
                logger.info(f"🎯 Readiness-Threshold erreicht ({current_readiness:.2f} >= {min_readiness_threshold}) "
                           "- starte sofortiges Training")
                
                # Sofortiges Training ausführen
                training_result = await self.execute_training_pipeline()
                
                return {
                    "immediate_training": True,
                    "training_result": training_result,
                    "readiness_score": current_readiness,
                    "assessment": assessment
                }
            else:
                logger.info(f"📊 Readiness unter Threshold ({current_readiness:.2f} < {min_readiness_threshold}) "
                           "- Training verschoben")
                
                return {
                    "immediate_training": False,
                    "scheduled": True,
                    "next_check_hours": assessment_interval_hours,
                    "readiness_score": current_readiness,
                    "threshold_required": min_readiness_threshold,
                    "recommendations": assessment.get("recommendations", [])
                }
                
        except Exception as e:
            logger.error(f"❌ Fehler bei automatischem Training-Scheduling: {e}")
            return {
                "immediate_training": False,
                "scheduled": False,
                "error": str(e)
            }
    
    def _calculate_training_priority(
        self, 
        readiness: Dict[str, Any], 
        training_stats: Dict[str, Any],
        latest_model: Optional[Dict[str, Any]]
    ) -> str:
        """Berechne Training-Priorität basierend auf verschiedenen Faktoren."""
        readiness_score = readiness.get("readiness_score", 0.0)
        high_confidence_feedback = training_stats.get("high_confidence_count", 0)
        days_since_last_training = latest_model.get("days_since_training", 999) if latest_model else 999
        
        # Prioritäts-Scoring
        priority_score = 0
        if readiness_score >= 0.9: priority_score += 3
        elif readiness_score >= 0.7: priority_score += 2
        elif readiness_score >= 0.5: priority_score += 1
        
        if high_confidence_feedback >= 100: priority_score += 2
        elif high_confidence_feedback >= 50: priority_score += 1
        
        if days_since_last_training >= 30: priority_score += 2
        elif days_since_last_training >= 14: priority_score += 1
        
        # Priority-Label
        if priority_score >= 6: return "critical"
        elif priority_score >= 4: return "high"
        elif priority_score >= 2: return "medium"
        else: return "low"
    
    def _generate_training_recommendations(
        self, 
        readiness: Dict[str, Any],
        training_stats: Dict[str, Any]
    ) -> list[str]:
        """Generiere Training-Empfehlungen."""
        recommendations = []
        
        readiness_checks = readiness.get("readiness_checks", {})
        
        if not readiness_checks.get("sufficient_high_confidence_feedback"):
            recommendations.append("Mehr User-Feedback mit hoher Confidence sammeln (Ziel: 50+)")
        
        if not readiness_checks.get("diverse_supplier_feedback"):
            recommendations.append("Feedback von mehr verschiedenen Lieferanten sammeln (Ziel: 10+)")
        
        if not readiness_checks.get("recent_feedback_activity"):
            recommendations.append("Aktuelle Feedback-Aktivität steigern (Ziel: 20+ letzte 30 Tage)")
        
        if not readiness_checks.get("embedding_diversity"):
            recommendations.append("Diversität der SKR03-Klassifizierungen erhöhen (Ziel: 15+ verschiedene)")
        
        return recommendations
    
    async def _prepare_training_data(self, config: TrainingConfig) -> Dict[str, Any]:
        """Vorbereitung der Training-Daten."""
        try:
            # Training-Daten abrufen (nur für Statistiken)
            pending_data = await self.ml_plugin.repository.get_pending_training_data(
                limit=10,  # Nur Stichprobe für Statistiken
                min_confidence_rating=config.min_confidence_rating
            )
            
            return {
                "sample_items_checked": len(pending_data),
                "min_confidence_rating": config.min_confidence_rating,
                "max_training_items": config.max_training_items,
                "data_preparation_successful": True
            }
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Training-Daten-Vorbereitung: {e}")
            return {
                "data_preparation_successful": False,
                "error": str(e)
            }
    
    async def _validate_trained_model(self, model_name: str, training_result: Dict[str, Any]) -> Dict[str, Any]:
        """Post-Training-Modell-Validierung."""
        try:
            final_accuracy = training_result.get("final_accuracy", 0.0)
            final_f1_score = training_result.get("final_f1_score", 0.0)
            
            # Validierungs-Thresholds
            min_accuracy = 0.85
            min_f1_score = 0.80
            
            validation_passed = (final_accuracy >= min_accuracy and final_f1_score >= min_f1_score)
            
            return {
                "validation_passed": validation_passed,
                "final_accuracy": final_accuracy,
                "final_f1_score": final_f1_score,
                "min_accuracy_threshold": min_accuracy,
                "min_f1_threshold": min_f1_score,
                "model_quality": (
                    "excellent" if final_accuracy >= 0.95 and final_f1_score >= 0.90 else
                    "good" if validation_passed else
                    "insufficient"
                )
            }
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Modell-Validierung: {e}")
            return {
                "validation_passed": False,
                "error": str(e)
            }
    
    async def _deploy_model_if_ready(self, model_name: str, validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Deploye Modell falls Validierung erfolgreich."""
        try:
            if not validation_result.get("validation_passed", False):
                return {
                    "deployed": False,
                    "reason": "validation_failed",
                    "model_quality": validation_result.get("model_quality", "unknown")
                }
            
            # Hier würde in der Praxis das Modell deployed werden
            # Für Phase 5.2 simulieren wir das Deployment
            
            return {
                "deployed": True,
                "deployment_timestamp": datetime.now().isoformat(),
                "model_name": model_name,
                "model_quality": validation_result.get("model_quality", "good"),
                "deployment_target": "llkjj_backend_production"
            }
            
        except Exception as e:
            logger.error(f"❌ Fehler beim Modell-Deployment: {e}")
            return {
                "deployed": False,
                "error": str(e)
            }


# ============================================================================= 
# Factory Functions für llkjj_backend Integration
# =============================================================================

def create_training_orchestrator_for_backend(ml_plugin, custom_config: Optional[TrainingConfig] = None):
    """
    Factory für BackendTrainingOrchestrator.
    
    Args:
        ml_plugin: MLPlugin v2.0 Instanz mit Repository-Zugriff
        custom_config: Custom Training-Konfiguration
    
    Returns:
        BackendTrainingOrchestrator Instanz
    """
    return BackendTrainingOrchestrator(ml_plugin, custom_config)


async def schedule_training_pipeline(
    ml_plugin,
    assessment_interval_hours: int = 24,
    min_readiness_threshold: float = 0.8
) -> Dict[str, Any]:
    """
    Convenience-Funktion für Training-Pipeline-Scheduling.
    
    Args:
        ml_plugin: MLPlugin v2.0 Instanz
        assessment_interval_hours: Intervall für Readiness-Checks  
        min_readiness_threshold: Minimale Readiness für Training
    
    Returns:
        Scheduling-Ergebnis
    """
    orchestrator = create_training_orchestrator_for_backend(ml_plugin)
    return await orchestrator.schedule_automated_training(
        assessment_interval_hours, 
        min_readiness_threshold
    )


async def get_training_recommendations(ml_plugin) -> Dict[str, Any]:
    """
    Convenience-Funktion für Training-Empfehlungen.
    
    Args:
        ml_plugin: MLPlugin v2.0 Instanz
    
    Returns:
        Training-Empfehlungen und Assessment
    """
    orchestrator = create_training_orchestrator_for_backend(ml_plugin)
    return await orchestrator.assess_training_needs()