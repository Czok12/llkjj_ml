#!/usr/bin/env python3
from __future__ import annotations

"""
LLKJJ ML Pipeline - Intelligent Feedback Learning System
========================================================

🎯 PRIORITÄT 1.3: RAG-System Intelligence Boost - Feedback Learning

System für:
- Benutzer-Feedback-Integration: Backend-API für Korrektur-Eingaben
- Confidence-Boosting: User-bestätigte Klassifizierungen → Konfidenz 1.0
- Pattern-Learning: Lieferanten-spezifische Klassifizierungs-Pattern erkennen
- Conflict-Resolution: Widersprüchliche Klassifizierungen automatisch kennzeichnen

Autor: LLKJJ ML Pipeline Team
Version: 1.0.0 (Feedback Intelligence)
Datum: 18. August 2025
"""

import json
import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from ..settings_bridge import ConfigBridge, config_instance

logger = logging.getLogger(__name__)


class FeedbackRecord(BaseModel):
    """
    Datenmodell für User-Feedback zu SKR03-Klassifizierungen.
    """

    feedback_id: str = Field(..., description="Eindeutige Feedback-ID (UUID)")
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
    user_id: str = Field(default="system", description="ID des Benutzers")
    confidence_before: float = Field(..., description="Konfidenz vor Korrektur")
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="Feedback-Zeitstempel",
    )
    supplier_context: dict[str, Any] = Field(
        default_factory=dict, description="Lieferanten-Kontext für Pattern Learning"
    )


class PatternLearning(BaseModel):
    """
    Gelerntes Klassifizierungs-Pattern aus User-Feedback.
    """

    pattern_id: str = Field(..., description="Pattern-ID")
    supplier_name: str = Field(..., description="Lieferant")
    item_description_pattern: str = Field(
        ..., description="Regex-Pattern für Artikelbeschreibung"
    )
    price_range_min: float = Field(default=0.0, description="Minimaler Preis")
    price_range_max: float = Field(default=999999.0, description="Maximaler Preis")
    preferred_skr03_account: str = Field(..., description="Bevorzugtes SKR03-Konto")
    confidence_boost: float = Field(
        default=0.1, description="Konfidenz-Boost für dieses Pattern"
    )
    feedback_count: int = Field(default=1, description="Anzahl bestätigender Feedbacks")
    last_updated: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="Letztes Update",
    )


class FeedbackLearningEngine:
    """
    🧠 Intelligentes Feedback-Learning für SKR03-Klassifizierung

    Features:
    - User-Feedback-Collection und Persistierung
    - Pattern-Extraktion aus wiederholten Korrekturen
    - Confidence-Boosting für bestätigte Klassifizierungen
    - Conflict-Detection bei widersprüchlichen Feedbacks
    """

    def __init__(self, config: ConfigBridge | None = None):
        self.config = config or config_instance
        self.feedback_db_path = Path("data/feedback/learning_feedback.db")
        self.feedback_db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._initialize_feedback_database()
        logger.info(
            "🧠 FeedbackLearningEngine initialisiert: %s", self.feedback_db_path
        )

    def _initialize_feedback_database(self) -> None:
        """Initialize SQLite database for feedback storage."""
        with sqlite3.connect(self.feedback_db_path) as conn:
            # Feedback Records Table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback_records (
                    feedback_id TEXT PRIMARY KEY,
                    pdf_path TEXT NOT NULL,
                    item_position INTEGER NOT NULL,
                    original_classification TEXT NOT NULL,
                    corrected_classification TEXT NOT NULL,
                    feedback_type TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    confidence_before REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    supplier_context TEXT NOT NULL
                )
            """
            )

            # Learned Patterns Table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS learned_patterns (
                    pattern_id TEXT PRIMARY KEY,
                    supplier_name TEXT NOT NULL,
                    item_description_pattern TEXT NOT NULL,
                    price_range_min REAL NOT NULL,
                    price_range_max REAL NOT NULL,
                    preferred_skr03_account TEXT NOT NULL,
                    confidence_boost REAL NOT NULL,
                    feedback_count INTEGER NOT NULL,
                    last_updated TEXT NOT NULL
                )
            """
            )

            # Indexes for performance
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_feedback_supplier
                ON feedback_records(supplier_context)
            """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_feedback_timestamp
                ON feedback_records(timestamp)
            """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_pattern_supplier
                ON learned_patterns(supplier_name)
            """
            )

    def record_feedback(
        self,
        pdf_path: str,
        item_position: int,
        original_classification: dict[str, Any],
        corrected_classification: dict[str, Any],
        feedback_type: Literal["correction", "confirmation", "rejection"],
        user_id: str = "system",
        confidence_before: float = 0.0,
        supplier_context: dict[str, Any] | None = None,
    ) -> str:
        """
        Speichere User-Feedback für eine SKR03-Klassifizierung.

        Returns:
            str: Feedback-ID für Tracking
        """
        import uuid

        feedback_id = str(uuid.uuid4())
        feedback = FeedbackRecord(
            feedback_id=feedback_id,
            pdf_path=pdf_path,
            item_position=item_position,
            original_classification=original_classification,
            corrected_classification=corrected_classification,
            feedback_type=feedback_type,
            user_id=user_id,
            confidence_before=confidence_before,
            supplier_context=supplier_context or {},
        )

        with sqlite3.connect(self.feedback_db_path) as conn:
            conn.execute(
                """
                INSERT INTO feedback_records
                (feedback_id, pdf_path, item_position, original_classification,
                 corrected_classification, feedback_type, user_id, confidence_before,
                 timestamp, supplier_context)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    feedback.feedback_id,
                    feedback.pdf_path,
                    feedback.item_position,
                    json.dumps(feedback.original_classification),
                    json.dumps(feedback.corrected_classification),
                    feedback.feedback_type,
                    feedback.user_id,
                    feedback.confidence_before,
                    feedback.timestamp,
                    json.dumps(feedback.supplier_context),
                ),
            )

        logger.info(
            "📝 User-Feedback gespeichert: %s (%s)",
            feedback_id[:8],
            feedback.feedback_type,
        )

        # Trigger pattern learning if correction
        if feedback_type == "correction":
            self._extract_learning_pattern(feedback)

        return feedback_id

    def _extract_learning_pattern(self, feedback: FeedbackRecord) -> None:
        """
        Extrahiere Lern-Pattern aus User-Feedback.
        """
        try:
            # Extract supplier from context
            supplier = feedback.supplier_context.get("supplier", "unknown")
            if supplier == "unknown":
                return

            # Extract description pattern
            description = feedback.corrected_classification.get("description", "")
            if not description:
                return

            # Extract price information
            price = feedback.corrected_classification.get("einzelpreis", 0.0)
            skr03_account = feedback.corrected_classification.get("skr03_account", "")

            if not skr03_account:
                return

            # Create or update pattern
            pattern_id = f"{supplier}_{skr03_account}_{hash(description) % 10000}"

            with sqlite3.connect(self.feedback_db_path) as conn:
                # Check if pattern exists
                existing = conn.execute(
                    "SELECT feedback_count FROM learned_patterns WHERE pattern_id = ?",
                    (pattern_id,),
                ).fetchone()

                if existing:
                    # Update existing pattern
                    new_count = existing[0] + 1
                    confidence_boost = min(0.3, 0.05 * new_count)  # Max 0.3 boost

                    conn.execute(
                        """
                        UPDATE learned_patterns
                        SET feedback_count = ?, confidence_boost = ?, last_updated = ?
                        WHERE pattern_id = ?
                    """,
                        (
                            new_count,
                            confidence_boost,
                            datetime.now().isoformat(),
                            pattern_id,
                        ),
                    )
                    logger.info(
                        "🔄 Pattern aktualisiert: %s (Count: %d, Boost: %.2f)",
                        pattern_id[:12],
                        new_count,
                        confidence_boost,
                    )
                else:
                    # Create new pattern
                    pattern = PatternLearning(
                        pattern_id=pattern_id,
                        supplier_name=supplier,
                        item_description_pattern=description[:100],  # Truncate
                        price_range_min=max(0.0, price * 0.8),
                        price_range_max=price * 1.2,
                        preferred_skr03_account=skr03_account,
                        confidence_boost=0.05,
                        feedback_count=1,
                    )

                    conn.execute(
                        """
                        INSERT INTO learned_patterns
                        (pattern_id, supplier_name, item_description_pattern,
                         price_range_min, price_range_max, preferred_skr03_account,
                         confidence_boost, feedback_count, last_updated)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            pattern.pattern_id,
                            pattern.supplier_name,
                            pattern.item_description_pattern,
                            pattern.price_range_min,
                            pattern.price_range_max,
                            pattern.preferred_skr03_account,
                            pattern.confidence_boost,
                            pattern.feedback_count,
                            pattern.last_updated,
                        ),
                    )
                    logger.info("🆕 Neues Pattern gelernt: %s", pattern_id[:12])

        except Exception as e:
            logger.warning("⚠️ Pattern-Extraktion fehlgeschlagen: %s", e)

    def apply_feedback_boost(
        self,
        classification: dict[str, Any],
        supplier_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Wende gelerntes Feedback auf eine Klassifizierung an.

        Returns:
            dict: Enhanced classification with feedback-based confidence boost
        """
        if not supplier_context:
            return classification

        supplier = supplier_context.get("supplier", "")
        description = classification.get("description", "")
        price = classification.get("einzelpreis", 0.0)
        current_account = classification.get("skr03_account", "")

        if not all([supplier, description, current_account]):
            return classification

        try:
            with sqlite3.connect(self.feedback_db_path) as conn:
                # Find matching patterns
                patterns = conn.execute(
                    """
                    SELECT preferred_skr03_account, confidence_boost, feedback_count
                    FROM learned_patterns
                    WHERE supplier_name = ?
                    AND ? BETWEEN price_range_min AND price_range_max
                    AND (item_description_pattern = ? OR ? LIKE '%' || item_description_pattern || '%')
                    ORDER BY feedback_count DESC
                    LIMIT 3
                """,
                    (supplier, price, description[:100], description[:100]),
                ).fetchall()

                if not patterns:
                    return classification

                # Calculate feedback boost
                total_boost = 0.0
                matching_patterns = 0

                for preferred_account, boost, count in patterns:
                    if preferred_account == current_account:
                        # Boost confidence for matching account
                        total_boost += boost * (
                            count / 10.0
                        )  # Weight by feedback count
                        matching_patterns += 1
                    else:
                        # Slight penalty for different account
                        total_boost -= boost * 0.1

                if matching_patterns > 0:
                    # Apply boost
                    enhanced_classification = classification.copy()
                    original_confidence = enhanced_classification.get("confidence", 0.8)
                    boosted_confidence = min(1.0, original_confidence + total_boost)

                    enhanced_classification["confidence"] = boosted_confidence
                    enhanced_classification["feedback_boost"] = total_boost
                    enhanced_classification["matching_patterns"] = matching_patterns

                    logger.debug(
                        "🚀 Feedback-Boost angewendet: %.2f → %.2f (+%.2f)",
                        original_confidence,
                        boosted_confidence,
                        total_boost,
                    )

                    return enhanced_classification

        except Exception as e:
            logger.warning("⚠️ Feedback-Boost fehlgeschlagen: %s", e)

        return classification

    def get_conflict_analysis(
        self, supplier: str, days_back: int = 30
    ) -> dict[str, Any]:
        """
        Analysiere Konflikte in User-Feedback für einen Lieferanten.

        Returns:
            dict: Conflict analysis with recommendations
        """
        cutoff_date = (datetime.now() - timedelta(days=days_back)).isoformat()

        try:
            with sqlite3.connect(self.feedback_db_path) as conn:
                # Get recent feedback for supplier
                feedbacks = conn.execute(
                    """
                    SELECT corrected_classification, feedback_type, timestamp
                    FROM feedback_records
                    WHERE supplier_context LIKE ?
                    AND timestamp > ?
                    ORDER BY timestamp DESC
                """,
                    (f'%"{supplier}"%', cutoff_date),
                ).fetchall()

                if not feedbacks:
                    return {"conflicts": [], "recommendations": []}

                # Analyze conflicts
                account_corrections: dict[str, int] = {}
                rejection_patterns = []

                for feedback_json, feedback_type, _timestamp in feedbacks:
                    feedback_data = json.loads(feedback_json)
                    account = feedback_data.get("skr03_account", "")

                    if feedback_type == "correction" and account:
                        account_corrections[account] = (
                            account_corrections.get(account, 0) + 1
                        )
                    elif feedback_type == "rejection":
                        rejection_patterns.append(feedback_data)

                # Detect conflicts
                conflicts = []
                if len(account_corrections) > 3:
                    conflicts.append(
                        {
                            "type": "multiple_accounts",
                            "description": f"Zu viele verschiedene Konten für {supplier}: {list(account_corrections.keys())}",
                            "severity": "medium",
                        }
                    )

                # Generate recommendations
                recommendations = []
                if account_corrections:
                    most_common = max(account_corrections.items(), key=lambda x: x[1])[
                        0
                    ]
                    recommendations.append(
                        f"Standardkonto für {supplier}: {most_common} "
                        f"({account_corrections[most_common]} Bestätigungen)"
                    )

                return {
                    "conflicts": conflicts,
                    "recommendations": recommendations,
                    "total_feedbacks": len(feedbacks),
                    "account_distribution": account_corrections,
                }

        except Exception as e:
            logger.error("❌ Conflict-Analysis fehlgeschlagen: %s", e)
            return {"conflicts": [], "recommendations": [], "error": str(e)}

    def get_learning_stats(self) -> dict[str, Any]:
        """
        Statistiken zum Feedback-Learning-System.

        Returns:
            dict: Learning statistics and insights
        """
        try:
            with sqlite3.connect(self.feedback_db_path) as conn:
                # Total feedback count
                total_feedback = conn.execute(
                    "SELECT COUNT(*) FROM feedback_records"
                ).fetchone()[0]

                # Feedback by type
                feedback_types = conn.execute(
                    """
                    SELECT feedback_type, COUNT(*)
                    FROM feedback_records
                    GROUP BY feedback_type
                """
                ).fetchall()

                # Top patterns
                top_patterns = conn.execute(
                    """
                    SELECT supplier_name, preferred_skr03_account, feedback_count, confidence_boost
                    FROM learned_patterns
                    ORDER BY feedback_count DESC
                    LIMIT 10
                """
                ).fetchall()

                # Recent activity
                recent_activity = conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM feedback_records
                    WHERE timestamp > ?
                """,
                    ((datetime.now() - timedelta(days=7)).isoformat(),),
                ).fetchone()[0]

                return {
                    "total_feedback": total_feedback,
                    "feedback_types": dict(feedback_types),
                    "top_patterns": [
                        {
                            "supplier": supplier,
                            "account": account,
                            "count": count,
                            "boost": boost,
                        }
                        for supplier, account, count, boost in top_patterns
                    ],
                    "recent_activity_7days": recent_activity,
                    "learning_effectiveness": min(1.0, total_feedback / 1000.0),
                }

        except Exception as e:
            logger.error("❌ Learning-Stats fehlgeschlagen: %s", e)
            return {"error": str(e)}
