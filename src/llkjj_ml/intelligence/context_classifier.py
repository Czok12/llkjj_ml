#!/usr/bin/env python3
"""
LLKJJ ML Pipeline - Context-Aware SKR03 Klassifizierer (Type-Safe)
==================================================================

🧠 Context-bewusste SKR03-Account-Klassifizierung mit supplier-spezifischen Optimierungen

Features:
- Supplier-Context-Learning (Siemens vs Conrad vs lokaler Elektrohandel)
- Multi-Position-Context-Berücksichtigung für Sammelrechnungen
- Price-Threshold-Logik für Anlagevermögen vs Betriebsausgaben
- Seasonal-Pattern-Learning für wiederkehrende Ausgabentypen
- Konsistenz-Analyse über ähnliche Items hinweg

Autor: LLKJJ ML Pipeline Team
Version: 1.0.0 (Type-Safe Context-Aware Klassifizierer)
Datum: 18. August 2025
"""

import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ContextAwareClassifier:
    """
    Type-Safe Context-bewusste SKR03-Account-Klassifizierung

    Nutzt Supplier-Context, Price-Thresholds und Pattern-Learning
    für präzise Buchungskonten-Zuordnung
    """

    def __init__(self, context_db_path: Path | None = None) -> None:
        """Initialisierung mit Context-Database"""
        self.context_db_path = context_db_path or Path("data/context_patterns.db")
        self.supplier_contexts: dict[str, dict[str, Any]] = {}
        self.price_thresholds = self._load_price_thresholds()
        self._init_context_database()
        self._load_supplier_contexts()

    def _load_price_thresholds(self) -> dict[str, Any]:
        """Lädt Price-Thresholds für verschiedene Kategorien"""
        return {
            "anlagevermögen_min": 800.0,
            "verbrauchsmaterial_max": 50.0,
            "werkzeug_range": (50.0, 500.0),
            "büroausstattung_max": 150.0,
            "software_typical": (100.0, 2000.0),
        }

    def _init_context_database(self) -> None:
        """Initialisiert SQLite Context-Database"""
        try:
            self.context_db_path.parent.mkdir(parents=True, exist_ok=True)

            with sqlite3.connect(self.context_db_path) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS supplier_patterns (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        supplier_name TEXT NOT NULL,
                        item_description TEXT NOT NULL,
                        classified_account TEXT NOT NULL,
                        confidence_score REAL NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """
                )

                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS seasonal_patterns (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        account_pattern TEXT NOT NULL,
                        month INTEGER NOT NULL,
                        frequency_count INTEGER NOT NULL,
                        avg_amount REAL NOT NULL,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """
                )

        except Exception as e:
            logger.error("Fehler beim Initialisieren der Context-Database: %s", e)

    def _load_supplier_contexts(self) -> None:
        """Lädt Supplier-spezifische Contexts"""
        self.supplier_contexts = {
            "SIEMENS": {
                "bias_keywords": ["automation", "steuerung", "software", "lizenz"],
                "preferred_accounts": ["0350", "4680", "6100"],
                "confidence_boost": 0.15,
            },
            "CONRAD": {
                "bias_keywords": ["elektronik", "werkzeug", "verbrauch", "klein"],
                "preferred_accounts": ["6670", "6680", "6100"],
                "confidence_boost": 0.12,
            },
            "GIRA": {
                "bias_keywords": ["schalter", "steckdose", "installation", "gebäude"],
                "preferred_accounts": ["6670", "0350"],
                "confidence_boost": 0.10,
            },
        }

    def classify_with_context(
        self, item: dict[str, Any], all_items: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        """
        Context-bewusste Klassifizierung eines Items

        Args:
            item: Zu klassifizierendes Item mit Basis-Klassifizierung
            all_items: Alle Items der Rechnung für Multi-Position-Context

        Returns:
            Enhanced Klassifizierung mit Context-Adjustments
        """
        base_account = item.get("skr03_account", "")
        base_confidence = item.get("confidence", 0.5)

        # Context-Boosts sammeln
        context_adjustments = []

        # 1. Supplier-Context
        supplier_boost = self._apply_supplier_context(item)
        if supplier_boost != 0.0:
            context_adjustments.append(("supplier_context", supplier_boost))

        # 2. Price-Threshold-Logic
        price_boost = self._apply_price_threshold_logic(item)
        if price_boost != 0.0:
            context_adjustments.append(("price_threshold", price_boost))

        # 3. Multi-Position-Context
        if all_items:
            multi_boost = self._apply_multi_position_context(item, all_items)
            if multi_boost != 0.0:
                context_adjustments.append(("multi_position", multi_boost))

        # 4. Seasonal-Patterns
        seasonal_boost = self._apply_seasonal_patterns(item)
        if seasonal_boost != 0.0:
            context_adjustments.append(("seasonal", seasonal_boost))

        # Gesamte Confidence berechnen
        total_boost = sum(boost for _, boost in context_adjustments)
        final_confidence = min(0.95, max(0.05, base_confidence + total_boost))

        return {
            "skr03_account": base_account,
            "confidence": final_confidence,
            "context_adjustments": context_adjustments,
            "applied_contexts": [name for name, _ in context_adjustments if _ != 0.0],
        }

    def _apply_supplier_context(self, item: dict[str, Any]) -> float:
        """Wendet Supplier-spezifische Context-Logik an"""
        supplier_name = item.get("supplier", "").upper()

        if supplier_name not in self.supplier_contexts:
            return 0.0

        context = self.supplier_contexts[supplier_name]
        bias_keywords = context.get("bias_keywords", [])

        if bias_keywords:
            beschreibung = item.get("beschreibung", "").lower()
            keyword_matches = sum(
                1 for keyword in bias_keywords if keyword in beschreibung
            )

            if keyword_matches > 0:
                boost_value = context.get("confidence_boost", 0.0)
                if isinstance(boost_value, int | float):
                    boost = float(boost_value) * (keyword_matches / len(bias_keywords))
                    return boost

        return 0.0

    def _apply_price_threshold_logic(self, item: dict[str, Any]) -> float:
        """Preis-basierte Klassifizierungs-Logik"""
        price = item.get("einzelpreis", 0.0)
        if not isinstance(price, int | float) or price <= 0:
            return 0.0

        current_account = item.get("skr03_account", "")

        # Anlagevermögen-Prüfung für teure Items
        anlagevermögen_min = self.price_thresholds["anlagevermögen_min"]
        if price >= anlagevermögen_min:
            if current_account.startswith("04") or current_account.startswith("05"):
                return (
                    0.20  # Confidence-Boost für korrekte Anlagevermögen-Klassifizierung
                )
            elif current_account.startswith("34") or current_account.startswith("68"):
                logger.warning(
                    "💰 Hochpreisiger Artikel (%.2f€) als Aufwand klassifiziert: %s",
                    price,
                    item.get("beschreibung", "")[:50],
                )
                return -0.10  # Confidence-Reduction

        # Verbrauchsmaterial-Bias für niedrigpreisige Items
        verbrauchsmaterial_max = self.price_thresholds["verbrauchsmaterial_max"]
        if price <= verbrauchsmaterial_max:
            if current_account.startswith("68") or current_account.startswith("66"):
                return 0.15  # Korrekt als Verbrauchsmaterial klassifiziert

        # Werkzeug-Range
        werkzeug_min, werkzeug_max = self.price_thresholds["werkzeug_range"]
        if werkzeug_min <= price <= werkzeug_max:
            if current_account.startswith("35") or current_account.startswith("667"):
                return 0.12  # Korrekt als Werkzeug/Betriebsmittel klassifiziert

        return 0.0

    def _apply_multi_position_context(
        self, item: dict[str, Any], all_items: list[dict[str, Any]]
    ) -> float:
        """Analysiert Konsistenz über ähnliche Items in der Rechnung"""
        current_account = item.get("skr03_account", "")
        current_description = item.get("beschreibung", "").lower()

        # Finde ähnliche Items basierend auf Beschreibung
        similar_items = []
        for other_item in all_items:
            if other_item == item:
                continue

            other_description = other_item.get("beschreibung", "").lower()
            similarity = self._calculate_description_similarity(
                current_description, other_description
            )

            if similarity > 0.6:  # 60% Ähnlichkeit
                similar_items.append(other_item)

        if len(similar_items) >= 2:  # Mindestens 2 ähnliche Items
            similar_accounts = [item.get("skr03_account", "") for item in similar_items]
            consistent_accounts = [
                acc for acc in similar_accounts if acc == current_account
            ]

            consistency_ratio = len(consistent_accounts) / len(similar_items)

            if consistency_ratio >= 0.8:  # 80% Konsistenz
                return 0.10  # Confidence-Boost für konsistente Klassifizierung
            elif consistency_ratio <= 0.3:  # 30% Konsistenz
                return -0.05  # Confidence-Reduction für inkonsistente Klassifizierung

        return 0.0

    def _apply_seasonal_patterns(self, item: dict[str, Any]) -> float:
        """Wendet saisonale Pattern-Logik an"""
        current_month = datetime.now().month
        current_account = item.get("skr03_account", "")

        try:
            with sqlite3.connect(self.context_db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT frequency_count, avg_amount
                    FROM seasonal_patterns
                    WHERE account_pattern = ? AND month = ?
                """,
                    (current_account, current_month),
                )

                result = cursor.fetchone()
                if result:
                    frequency, avg_amount = result
                    if frequency >= 3:  # Mindestens 3 historische Vorkommen
                        return 0.05  # Leichter Confidence-Boost für saisonale Pattern

        except Exception as e:
            logger.warning("Fehler bei Seasonal-Pattern-Analyse: %s", e)

        return 0.0

    def _calculate_description_similarity(self, desc1: str, desc2: str) -> float:
        """Berechnet Ähnlichkeit zwischen zwei Beschreibungen"""
        if not desc1 or not desc2:
            return 0.0

        words1 = set(desc1.split())
        words2 = set(desc2.split())

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union) if union else 0.0

    def update_supplier_context(
        self, supplier_name: str, successful_patterns: list[dict[str, Any]]
    ) -> None:
        """Aktualisiert Supplier-Context basierend auf erfolgreichen Klassifizierungen"""
        supplier_name = supplier_name.upper()

        if supplier_name not in self.supplier_contexts:
            self.supplier_contexts[supplier_name] = {
                "bias_keywords": [],
                "preferred_accounts": [],
                "confidence_boost": 0.05,
            }

        # Extrahiere häufige Keywords und Accounts
        [pattern.get("beschreibung", "") for pattern in successful_patterns]
        accounts = [pattern.get("skr03_account", "") for pattern in successful_patterns]

        # Aktualisiere Context (vereinfacht)
        context = self.supplier_contexts[supplier_name]
        context["preferred_accounts"] = list(set(accounts))

        logger.info(
            "✅ Updated Supplier-Context für %s: %d erfolgreiche Pattern",
            supplier_name,
            len(successful_patterns),
        )

    def get_context_statistics(self) -> dict[str, Any]:
        """Erstellt Statistiken über Context-Performance"""
        try:
            with sqlite3.connect(self.context_db_path) as conn:
                # Supplier-Pattern-Stats
                cursor = conn.execute(
                    """
                    SELECT supplier_name, COUNT(*), AVG(confidence_score)
                    FROM supplier_patterns
                    GROUP BY supplier_name
                """
                )
                supplier_stats = dict(cursor.fetchall())

                # Seasonal-Pattern-Stats
                cursor = conn.execute(
                    """
                    SELECT COUNT(*), AVG(frequency_count)
                    FROM seasonal_patterns
                """
                )
                seasonal_stats = cursor.fetchone()

                return {
                    "supplier_pattern_count": len(supplier_stats),
                    "supplier_stats": supplier_stats,
                    "seasonal_patterns": seasonal_stats[0] if seasonal_stats else 0,
                    "avg_seasonal_frequency": (
                        seasonal_stats[1] if seasonal_stats else 0.0
                    ),
                    "active_supplier_contexts": list(self.supplier_contexts.keys()),
                    "price_threshold_config": self.price_thresholds,
                }

        except Exception as e:
            logger.error("Fehler bei Context-Statistiken: %s", e)
            return {"error": str(e)}


def enhance_classification_with_context(
    items: list[dict[str, Any]],
    context_classifier: ContextAwareClassifier | None = None,
) -> list[dict[str, Any]]:
    """
    Verbessert Basis-Klassifizierungen mit Context-Awareness

    Args:
        items: Liste von Items mit Basis-Klassifizierung
        context_classifier: Optional ContextAwareClassifier Instanz

    Returns:
        Enhanced Items mit Context-Adjustments
    """
    if not items:
        return []

    if context_classifier is None:
        context_classifier = ContextAwareClassifier()

    enhanced_items = []

    for item in items:
        try:
            enhanced = context_classifier.classify_with_context(item, items)

            # Merge mit Original-Item
            enhanced_item = {**item, **enhanced}
            enhanced_items.append(enhanced_item)

        except Exception as e:
            logger.error(
                "Fehler bei Context-Enhancement für Item %s: %s",
                item.get("beschreibung", "unknown"),
                e,
            )
            enhanced_items.append(item)  # Fallback auf Original

    logger.info(
        "✅ Context-Enhancement abgeschlossen: %d Items verarbeitet",
        len(enhanced_items),
    )
    return enhanced_items


# Alias für Backward Compatibility
ContextClassifier = ContextAwareClassifier


# Export für mypy
__all__ = ["ContextAwareClassifier", "ContextClassifier", "enhance_classification_with_context"]
