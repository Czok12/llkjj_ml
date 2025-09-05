#!/usr/bin/env python3
"""
LLKJJ ML Pipeline - spaCy Annotation Correction
==============================================

Position-korrigierte spaCy-Annotationen für bessere Training-Datenqualität.
Implementiert die Verbesserung aus dem Gemini-Vorschlag für präzisere
Named Entity Recognition (NER) Training-Daten.

Autor: LLKJJ ML Pipeline Team
Version: 4.0.0 (Hybrid Implementation)
Datum: 18. August 2025
"""

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


class SpacyAnnotationCorrector:
    """
    Position-korrigierte spaCy-Annotationen für präzise Training-Daten.

    **VERBESSERUNG aus Gemini-Vorschlag:**
    - Korrigiert Character-Positionen für exakte NER-Training
    - Validiert Annotation-Grenzen gegen Text-Inhalt
    - Behandelt Unicode-Sonderzeichen und deutsche Umlaute
    - Optimiert für Elektrotechnik-Terminologie

    **USAGE:**
    ```python
    corrector = SpacyAnnotationCorrector()

    # Korrigiere einzelne Annotation
    corrected = corrector.correct_annotation(
        text="GIRA Rahmen 55x55mm Standard",
        entity_text="GIRA",
        start_char=0,
        end_char=4,
        label="MARKE"
    )

    # Batch-Korrektur für Training-Pipeline
    annotations = corrector.correct_batch_annotations(
        full_text=invoice_text,
        raw_annotations=gemini_annotations
    )
    ```
    """

    def __init__(self) -> None:
        """Initialize spaCy annotation corrector"""
        # Deutsche Elektrotechnik-Marken für verbesserte Erkennung
        self.known_brands = {
            "GIRA",
            "HAGER",
            "SIEMENS",
            "ABB",
            "SCHNEIDER",
            "PHOENIX",
            "WAGO",
            "BUSCH-JAEGER",
            "JUNG",
            "MERTEN",
            "BERKER",
            "SPELSBERG",
        }

        # Elektrische Kategorien für bessere Klassifizierung
        self.electrical_categories = {
            "SCHALTER",
            "STECKDOSE",
            "RAHMEN",
            "KABEL",
            "LEITUNG",
            "LAMPE",
            "LEUCHTE",
            "LED",
            "TRAFO",
            "SICHERUNG",
            "AUTOMAT",
            "FI",
            "LEITUNGSSCHUTZSCHALTER",
        }

    def correct_annotation(
        self, text: str, entity_text: str, start_char: int, end_char: int, label: str
    ) -> dict[str, Any]:
        """
        Korrigiert eine einzelne spaCy-Annotation für exakte Position.

        Args:
            text: Vollständiger Text-Kontext
            entity_text: Erkannte Entität (z.B. "GIRA")
            start_char: Startposition (möglicherweise ungenau)
            end_char: Endposition (möglicherweise ungenau)
            label: spaCy-Label (z.B. "MARKE", "ARTIKEL")

        Returns:
            Korrigierte Annotation mit präzisen Positionen
        """
        original_positions = (start_char, end_char)

        try:
            # Validiere initiale Positionen
            if start_char < 0 or end_char > len(text) or start_char >= end_char:
                logger.warning(
                    f"⚠️ Ungültige Annotation-Positionen: {start_char}-{end_char} "
                    f"für Text-Länge {len(text)}"
                )
                return self._create_fallback_annotation(entity_text, label, text)

            # Extrahiere Text an angegebener Position
            extracted_text = text[start_char:end_char]

            # Wenn exakte Übereinstimmung, Position ist korrekt
            if extracted_text.strip() == entity_text.strip():
                return {
                    "text": entity_text,
                    "label": label,
                    "start_char": start_char,
                    "end_char": end_char,
                    "confidence": 1.0,
                    "correction_applied": False,
                    "original_positions": original_positions,
                }

            # Position-Korrektur erforderlich
            logger.info(
                f"🔧 Korrigiere Annotation: '{extracted_text}' -> '{entity_text}'"
            )

            # Suche exakte Position der Entität im Text
            corrected_positions = self._find_exact_position(
                text, entity_text, start_char
            )

            if corrected_positions:
                corrected_start, corrected_end = corrected_positions
                confidence = 0.95  # Hohe Konfidenz bei erfolgreicher Korrektur

                logger.info(
                    f"✅ Position korrigiert: {original_positions} -> {corrected_positions}"
                )

                return {
                    "text": entity_text,
                    "label": label,
                    "start_char": corrected_start,
                    "end_char": corrected_end,
                    "confidence": confidence,
                    "correction_applied": True,
                    "original_positions": original_positions,
                }
            else:
                # Fallback: Fuzzy-Matching
                return self._fuzzy_correction(
                    text, entity_text, label, start_char, end_char
                )

        except Exception as e:
            logger.warning(f"⚠️ Annotation-Korrektur fehlgeschlagen: {e}")
            return self._create_fallback_annotation(entity_text, label, text)

    def _find_exact_position(
        self, text: str, entity_text: str, hint_position: int
    ) -> tuple[int, int] | None:
        """
        Findet die exakte Position einer Entität im Text.

        Args:
            text: Vollständiger Text
            entity_text: Zu suchende Entität
            hint_position: Ungefähre Position als Hinweis

        Returns:
            Tuple (start, end) oder None falls nicht gefunden
        """
        # Exakte Suche (case-sensitive)
        exact_pos = text.find(entity_text)
        if exact_pos != -1:
            return (exact_pos, exact_pos + len(entity_text))

        # Case-insensitive Suche
        text_lower = text.lower()
        entity_lower = entity_text.lower()
        case_insensitive_pos = text_lower.find(entity_lower)
        if case_insensitive_pos != -1:
            return (case_insensitive_pos, case_insensitive_pos + len(entity_text))

        # Regex-basierte Suche für Varianten
        pattern = re.escape(entity_text)
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return (match.start(), match.end())

        # Suche in Nähe der ursprünglichen Position
        search_window = 50  # Zeichen um hint_position
        start_search = max(0, hint_position - search_window)
        end_search = min(len(text), hint_position + search_window)
        text_window = text[start_search:end_search]

        window_pos = text_window.find(entity_text)
        if window_pos != -1:
            absolute_start = start_search + window_pos
            return (absolute_start, absolute_start + len(entity_text))

        return None

    def find_exact_position_window_search(
        self, text: str, entity_text: str, start_hint: int = 0, window_size: int = 50
    ) -> tuple[int, int] | None:
        """
        Find exact position using window search.

        Args:
            text: Full text to search in
            entity_text: Entity text to find
            start_hint: Starting position hint
            window_size: Search window size

        Returns:
            (start, end) positions or None
        """
        entity_text = entity_text.strip()

        # Direkte Suche zuerst
        pos = text.find(entity_text, start_hint)
        if pos != -1:
            return (pos, pos + len(entity_text))

        # Fenster-Suche mit Fuzzy-Matching
        for offset in range(-window_size, window_size + 1, 5):
            search_start = max(0, start_hint + offset)
            search_end = min(len(text), search_start + len(entity_text) + window_size)

            window_text = text[search_start:search_end]

            # Fuzzy-Match in Window
            pos = self._fuzzy_find(window_text, entity_text)
            if pos != -1:
                actual_start = search_start + pos
                actual_end = actual_start + len(entity_text)
                return (actual_start, actual_end)

        return None

    def _fuzzy_find(self, text: str, entity_text: str) -> int:
        """Internal fuzzy find method."""
        # Case-insensitive Suche
        text_lower = text.lower()
        entity_lower = entity_text.lower()
        pos = text_lower.find(entity_lower)
        if pos != -1:
            return pos

        # Regex-based fuzzy search
        import re

        pattern = re.escape(entity_text)
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.start()

        return -1

    def fuzzy_correction_known_brand(
        self, entity_text: str, threshold: float = 0.8
    ) -> str:
        """Apply fuzzy correction for known brands."""
        known_brands = ["GIRA", "HAGER", "SIEMENS", "ABB", "SCHNEIDER"]

        entity_upper = entity_text.upper()

        # Exakte Übereinstimmung zuerst
        if entity_upper in known_brands:
            return entity_upper

        # Fuzzy-Matching
        from difflib import SequenceMatcher

        best_match = None
        best_score = 0.0

        for brand in known_brands:
            score = SequenceMatcher(None, entity_upper, brand).ratio()
            if score > best_score and score >= threshold:
                best_score = score
                best_match = brand

        return best_match or entity_text

    def _fuzzy_correction(
        self, text: str, entity_text: str, label: str, start_char: int, end_char: int
    ) -> dict[str, Any]:
        """
        Fuzzy-Matching für ähnliche Entitäten bei nicht-exakter Übereinstimmung.

        Args:
            text: Vollständiger Text
            entity_text: Erkannte Entität
            label: spaCy-Label
            start_char: Ursprüngliche Startposition
            end_char: Ursprüngliche Endposition

        Returns:
            Best-effort korrigierte Annotation
        """
        logger.info(f"🔍 Fuzzy-Matching für Entität: '{entity_text}'")

        # Suche ähnliche Marken
        if label == "MARKE" and entity_text.upper() in self.known_brands:
            # Suche bekannte Marke im Text
            for brand in self.known_brands:
                brand_pos = text.upper().find(brand)
                if brand_pos != -1:
                    return {
                        "text": brand,
                        "label": label,
                        "start_char": brand_pos,
                        "end_char": brand_pos + len(brand),
                        "confidence": 0.8,
                        "correction_applied": True,
                        "original_positions": (start_char, end_char),
                        "fuzzy_matched": True,
                    }

        # Fallback auf ursprüngliche Position mit niedrigerer Konfidenz
        return {
            "text": entity_text,
            "label": label,
            "start_char": start_char,
            "end_char": end_char,
            "confidence": 0.5,
            "correction_applied": False,
            "original_positions": (start_char, end_char),
            "fuzzy_failed": True,
        }

    def _create_fallback_annotation(
        self, entity_text: str, label: str, text: str
    ) -> dict[str, Any]:
        """
        Erstellt Fallback-Annotation bei Korrektur-Fehlern.

        Args:
            entity_text: Entitäts-Text
            label: spaCy-Label
            text: Vollständiger Text-Kontext

        Returns:
            Fallback-Annotation mit niedriger Konfidenz
        """
        return {
            "text": entity_text,
            "label": label,
            "start_char": 0,
            "end_char": len(entity_text),
            "confidence": 0.3,
            "correction_applied": False,
            "fallback_annotation": True,
            "note": "Position konnte nicht korrigiert werden",
        }

    def correct_batch_annotations(
        self, full_text: str, raw_annotations: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Batch-Korrektur für alle Annotationen eines Dokuments.

        Args:
            full_text: Kompletter Text des Dokuments
            raw_annotations: Liste von Rohantationen aus Gemini/spaCy

        Returns:
            Liste korrigierter Annotationen
        """
        if not raw_annotations:
            logger.info("📝 Keine Annotationen zum Korrigieren")
            return []

        logger.info(f"🔧 Korrigiere {len(raw_annotations)} Annotationen...")

        corrected_annotations = []
        correction_stats = {
            "total": len(raw_annotations),
            "corrected": 0,
            "exact_match": 0,
            "fuzzy_matched": 0,
            "fallback": 0,
        }

        for annotation in raw_annotations:
            try:
                # Extrahiere Annotation-Daten
                entity_text = annotation.get("text", "")
                label = annotation.get("label", "UNKNOWN")
                start_char = annotation.get("start_char", 0)
                end_char = annotation.get("end_char", len(entity_text))

                # Korrigiere Annotation
                corrected = self.correct_annotation(
                    full_text, entity_text, start_char, end_char, label
                )

                corrected_annotations.append(corrected)

                # Statistiken aktualisieren
                if corrected.get("correction_applied"):
                    correction_stats["corrected"] += 1
                if corrected.get("fuzzy_matched"):
                    correction_stats["fuzzy_matched"] += 1
                if corrected.get("fallback_annotation"):
                    correction_stats["fallback"] += 1
                if corrected.get("confidence", 0) == 1.0:
                    correction_stats["exact_match"] += 1

            except Exception as e:
                logger.warning(f"⚠️ Fehler bei Annotation-Korrektur: {e}")
                # Fehlerhafte Annotation überspringen
                continue

        # Statistiken loggen
        logger.info(
            f"✅ Annotation-Korrektur abgeschlossen: "
            f"{correction_stats['corrected']}/{correction_stats['total']} korrigiert, "
            f"{correction_stats['exact_match']} exakte Treffer, "
            f"{correction_stats['fuzzy_matched']} Fuzzy-Matches"
        )

        return corrected_annotations

    def validate_annotations(
        self, annotations: list[dict[str, Any]], text: str
    ) -> dict[str, Any]:
        """
        Validiert korrigierte Annotationen für Training-Qualität.

        Args:
            annotations: Liste korrigierter Annotationen
            text: Vollständiger Text

        Returns:
            Validierungsreport mit Qualitätsmetriken
        """
        if not annotations:
            return {
                "valid_count": 0,
                "invalid_count": 0,
                "average_confidence": 0.0,
                "quality_score": 0.0,
                "validation_errors": [],
            }

        validation_errors = []
        valid_count = 0
        total_confidence = 0.0

        for i, annotation in enumerate(annotations):
            try:
                start = annotation.get("start_char", 0)
                end = annotation.get("end_char", 0)
                entity_text = annotation.get("text", "")
                confidence = annotation.get("confidence", 0.0)

                # Validierung 1: Position innerhalb Text
                if start < 0 or end > len(text) or start >= end:
                    validation_errors.append(
                        f"Annotation {i}: Ungültige Position {start}-{end}"
                    )
                    continue

                # Validierung 2: Text-Übereinstimmung
                actual_text = text[start:end]
                if actual_text.strip() != entity_text.strip():
                    validation_errors.append(
                        f"Annotation {i}: Text-Mismatch '{actual_text}' != '{entity_text}'"
                    )
                    continue

                # Validierung 3: Mindest-Konfidenz
                if confidence < 0.3:
                    validation_errors.append(
                        f"Annotation {i}: Zu niedrige Konfidenz {confidence:.2f}"
                    )
                    continue

                valid_count += 1
                total_confidence += confidence

            except Exception as e:
                validation_errors.append(f"Annotation {i}: Validierungsfehler - {e}")

        average_confidence = total_confidence / valid_count if valid_count > 0 else 0.0
        quality_score = valid_count / len(annotations) if annotations else 0.0

        return {
            "valid_count": valid_count,
            "invalid_count": len(annotations) - valid_count,
            "average_confidence": average_confidence,
            "quality_score": quality_score,
            "validation_errors": validation_errors,
        }


# Convenience function für einfache Integration
def correct_spacy_annotations(
    full_text: str, raw_annotations: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """
    Convenience function für spaCy-Annotation-Korrektur.

    Args:
        full_text: Vollständiger Dokumenttext
        raw_annotations: Rohantationen aus Gemini/spaCy

    Returns:
        Position-korrigierte Annotationen
    """
    corrector = SpacyAnnotationCorrector()
    return corrector.correct_batch_annotations(full_text, raw_annotations)
