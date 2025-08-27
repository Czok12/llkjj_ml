#!/usr/bin/env python3
"""
LLKJJ ML Pipeline - Feedback API Interface
==========================================

🎯 REST-API Interface für das Feedback Learning System

Bietet Endpunkte für:
- User-Feedback-Eingabe für Klassifizierungs-Korrekturen
- Learning-Stats und Dashboard-Integration
- Conflict-Analysis für Qualitätskontrolle

Autor: LLKJJ ML Pipeline Team
Version: 1.0.0 (API Integration)
Datum: 18. August 2025
"""

import logging
from typing import Any, Literal

from pydantic import BaseModel, Field

from llkjj_ml.settings_bridge import Config, ConfigType

from .feedback_learning import FeedbackLearningEngine

logger = logging.getLogger(__name__)


class FeedbackSubmissionRequest(BaseModel):
    """Request-Schema für Feedback-Eingabe."""

    pdf_path: str = Field(..., description="Pfad zur ursprünglichen PDF-Datei")
    item_position: int = Field(..., description="Position der Rechnung (1-basiert)")
    original_classification: dict[str, Any] = Field(
        ..., description="Ursprüngliche SKR03-Klassifizierung"
    )
    corrected_classification: dict[str, Any] = Field(
        ..., description="Benutzer-korrigierte SKR03-Klassifizierung"
    )
    feedback_type: Literal["correction", "confirmation", "rejection"] = Field(
        ..., description="Art des Feedbacks"
    )
    user_id: str = Field(..., description="ID des Benutzers")
    supplier_context: dict[str, Any] = Field(
        default_factory=dict, description="Lieferanten-Kontext"
    )


class FeedbackResponse(BaseModel):
    """Response-Schema für Feedback-API."""

    success: bool = Field(..., description="Erfolg der Operation")
    feedback_id: str = Field(..., description="Eindeutige Feedback-ID")
    message: str = Field(..., description="Status-Nachricht")
    learning_triggered: bool = Field(
        default=False, description="Ob Pattern-Learning ausgelöst wurde"
    )


class LearningStatsResponse(BaseModel):
    """Response-Schema für Learning-Statistiken."""

    total_feedback: int = Field(..., description="Gesamtzahl Feedback-Records")
    feedback_types: dict[str, int] = Field(
        ..., description="Verteilung nach Feedback-Typ"
    )
    top_patterns: list[dict[str, Any]] = Field(..., description="Top-gelernte Pattern")
    recent_activity_7days: int = Field(..., description="Aktivität letzte 7 Tage")
    learning_effectiveness: float = Field(
        ..., description="Lern-Effektivität (0.0-1.0)"
    )


class FeedbackAPI:
    """
    🔄 REST-API Interface für Feedback Learning System

    Bietet strukturierte Endpunkte für Frontend-Integration
    und Backend-Kommunikation mit dem LLKJJ-System.
    """

    def __init__(self, config: ConfigType | None = None):
        self.config = config or Config
        self.feedback_engine = FeedbackLearningEngine(self.config)
        logger.info("🔄 Feedback API initialisiert")

    def submit_feedback(self, request: FeedbackSubmissionRequest) -> FeedbackResponse:
        """
        📝 Endpunkt für User-Feedback-Eingabe

        Args:
            request: Feedback-Submission-Request

        Returns:
            FeedbackResponse: Erfolg-Status und Feedback-ID
        """
        try:
            feedback_id = self.feedback_engine.record_feedback(
                pdf_path=request.pdf_path,
                item_position=request.item_position,
                original_classification=request.original_classification,
                corrected_classification=request.corrected_classification,
                feedback_type=request.feedback_type,
                user_id=request.user_id,
                confidence_before=request.original_classification.get(
                    "confidence", 0.0
                ),
                supplier_context=request.supplier_context,
            )

            return FeedbackResponse(
                success=True,
                feedback_id=feedback_id,
                message="Feedback erfolgreich gespeichert",
                learning_triggered=(request.feedback_type == "correction"),
            )

        except (ValueError, RuntimeError, OSError) as e:
            logger.error("❌ Feedback-Submission fehlgeschlagen: %s", e)
            return FeedbackResponse(
                success=False,
                feedback_id="",
                message=f"Fehler beim Speichern des Feedbacks: {e}",
                learning_triggered=False,
            )

    def get_learning_stats(self) -> LearningStatsResponse:
        """
        📊 Endpunkt für Learning-Statistiken

        Returns:
            LearningStatsResponse: Aktuelle Learning-Statistiken
        """
        try:
            stats = self.feedback_engine.get_learning_stats()

            return LearningStatsResponse(
                total_feedback=stats.get("total_feedback", 0),
                feedback_types=stats.get("feedback_types", {}),
                top_patterns=stats.get("top_patterns", []),
                recent_activity_7days=stats.get("recent_activity_7days", 0),
                learning_effectiveness=stats.get("learning_effectiveness", 0.0),
            )

        except (ValueError, RuntimeError, OSError) as e:
            logger.error("❌ Learning-Stats fehlgeschlagen: %s", e)
            return LearningStatsResponse(
                total_feedback=0,
                feedback_types={},
                top_patterns=[],
                recent_activity_7days=0,
                learning_effectiveness=0.0,
            )

    def get_conflict_analysis(
        self, supplier: str, days_back: int = 30
    ) -> dict[str, Any]:
        """
        ⚠️ Endpunkt für Conflict-Analysis

        Args:
            supplier: Lieferanten-Name
            days_back: Tage zurückblicken

        Returns:
            dict: Conflict-Analysis mit Empfehlungen
        """
        try:
            return self.feedback_engine.get_conflict_analysis(supplier, days_back)
        except (ValueError, RuntimeError, OSError) as e:
            logger.error("❌ Conflict-Analysis fehlgeschlagen: %s", e)
            return {"conflicts": [], "recommendations": [], "error": str(e)}

    def apply_feedback_boost(
        self,
        classification: dict[str, Any],
        supplier_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        🚀 Endpunkt für Feedback-Boost-Anwendung

        Args:
            classification: Original-Klassifizierung
            supplier_context: Lieferanten-Kontext

        Returns:
            dict: Enhanced classification mit Feedback-Boost
        """
        try:
            return self.feedback_engine.apply_feedback_boost(
                classification, supplier_context
            )
        except (ValueError, RuntimeError, OSError) as e:
            logger.error("❌ Feedback-Boost fehlgeschlagen: %s", e)
            return classification


# Convenience Functions für direkte API-Nutzung
def create_feedback_api(config: ConfigType | None = None) -> FeedbackAPI:
    """
    Factory-Function für FeedbackAPI-Instanz.

    Args:
        config: Optionale Konfiguration

    Returns:
        FeedbackAPI: Initialisierte API-Instanz
    """
    return FeedbackAPI(config)


def submit_classification_feedback(
    pdf_path: str,
    item_position: int,
    original_skr03: str,
    corrected_skr03: str,
    feedback_type: Literal["correction", "confirmation", "rejection"],
    user_id: str,
    confidence_before: float = 0.0,
    supplier: str = "",
    description: str = "",
    price: float = 0.0,
) -> str:
    """
    📝 Einfache Feedback-Eingabe ohne komplexe Strukturen

    Args:
        pdf_path: Pfad zur PDF-Datei
        item_position: Position auf der Rechnung
        original_skr03: Original SKR03-Konto
        corrected_skr03: Korrigiertes SKR03-Konto
        feedback_type: Art des Feedbacks
        user_id: Benutzer-ID
        confidence_before: Ursprüngliche Konfidenz
        supplier: Lieferant
        description: Artikel-Beschreibung
        price: Preis

    Returns:
        str: Feedback-ID
    """
    api = create_feedback_api()

    request = FeedbackSubmissionRequest(
        pdf_path=pdf_path,
        item_position=item_position,
        original_classification={
            "skr03_account": original_skr03,
            "description": description,
            "confidence": confidence_before,
            "einzelpreis": price,
        },
        corrected_classification={
            "skr03_account": corrected_skr03,
            "description": description,
            "confidence": 1.0,
            "einzelpreis": price,
        },
        feedback_type=feedback_type,
        user_id=user_id,
        supplier_context={"supplier": supplier},
    )

    response = api.submit_feedback(request)
    return response.feedback_id if response.success else ""


# FastAPI Integration Beispiel - zum Aktivieren in separater Datei verwenden
# from fastapi import FastAPI, HTTPException
#
# app = FastAPI(title="LLKJJ ML Feedback API")
# feedback_api = create_feedback_api()
#
# @app.post("/api/feedback", response_model=FeedbackResponse)
# async def submit_feedback_endpoint(request: FeedbackSubmissionRequest):
#     return feedback_api.submit_feedback(request)
#
# @app.get("/api/learning-stats", response_model=LearningStatsResponse)
# async def get_learning_stats_endpoint():
#     return feedback_api.get_learning_stats()
#
# @app.get("/api/conflicts/{supplier}")
# async def get_conflict_analysis_endpoint(supplier: str, days_back: int = 30):
#     return feedback_api.get_conflict_analysis(supplier, days_back)
