#!/usr/bin/env python3
"""
LLKJJ ML Pipeline - Feature Engineering Framework
=================================================

Automatisierte Feature-Extraktion und -Optimierung für deutsche
Elektrohandwerk-Rechnungsverarbeitung.

Ziel: Verbesserung der SKR03-Klassifizierung von 85% auf 92%+ Genauigkeit
durch systematische Feature-Engineering Pipeline.

Author: LLKJJ ML Pipeline
Version: 1.0.0
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Optional, Protocol, TypeVar, Union

from pydantic import BaseModel, ConfigDict, Field

# Type definitions
FeatureDict = dict[str, float | int | str | bool | list[Any]]
T = TypeVar("T")

logger = logging.getLogger(__name__)


@dataclass
class FeatureMetadata:
    """Metadata für Feature-Extraktion mit Performance-Tracking"""

    name: str
    extractor_type: str
    computation_time: float = 0.0
    success: bool = True
    error_message: str = ""
    feature_count: int = 0
    importance_score: float = 0.0

    def __post_init__(self) -> None:
        """Post-initialization validation"""
        if self.computation_time < 0:
            raise ValueError("Computation time cannot be negative")


@dataclass
class FeatureExtractionResult:
    """Ergebnis der Feature-Extraktion mit Metadaten"""

    features: FeatureDict = field(default_factory=dict)
    metadata: FeatureMetadata | None = None
    raw_data: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0

    def __post_init__(self) -> None:
        """Initialize metadata if not provided"""
        if self.metadata is None:
            self.metadata = FeatureMetadata(name="unknown", extractor_type="unknown")


class FeatureExtractor(ABC):
    """
    Abstract base class für alle Feature-Extraktoren.

    Alle Feature-Extraktoren müssen diese Schnittstelle implementieren
    für konsistente Integration in die Pipeline.
    """

    def __init__(self, name: str, enabled: bool = True) -> None:
        self.name = name
        self.enabled = enabled
        self.logger = logging.getLogger(f"{__name__}.{name}")

    @abstractmethod
    def extract_features(
        self, invoice_data: dict[str, Any], **kwargs: Any
    ) -> FeatureExtractionResult:
        """
        Extrahiere Features aus Rechnungsdaten.

        Args:
            invoice_data: Strukturierte Rechnungsdaten
            **kwargs: Zusätzliche Parameter für Extraktor

        Returns:
            FeatureExtractionResult mit extrahierten Features
        """
        pass

    @abstractmethod
    def get_feature_names(self) -> list[str]:
        """Gebe Liste aller von diesem Extraktor erzeugten Feature-Namen zurück"""
        pass

    def validate_input(self, invoice_data: dict[str, Any]) -> bool:
        """
        Validiere Eingabedaten für Feature-Extraktion.

        Args:
            invoice_data: Zu validierende Daten

        Returns:
            True wenn Daten gültig sind
        """
        if not isinstance(invoice_data, dict):
            self.logger.error(f"Invalid input type: {type(invoice_data)}")
            return False

        if not invoice_data:
            self.logger.warning("Empty invoice data provided")
            return False

        return True

    def is_enabled(self) -> bool:
        """Prüfe ob Extraktor aktiviert ist"""
        return self.enabled

    def enable(self) -> None:
        """Aktiviere Extraktor"""
        self.enabled = True
        self.logger.info(f"Feature extractor '{self.name}' enabled")

    def disable(self) -> None:
        """Deaktiviere Extraktor"""
        self.enabled = False
        self.logger.info(f"Feature extractor '{self.name}' disabled")


class FeaturePipeline:
    """
    Hauptklasse für Feature-Engineering Pipeline.

    Koordiniert mehrere Feature-Extraktoren und kombiniert deren Ergebnisse
    zu einem einheitlichen Feature-Vektor für ML-Modelle.
    """

    def __init__(self, extractors: list[FeatureExtractor] | None = None) -> None:
        self.extractors: list[FeatureExtractor] = extractors or []
        self.logger = logging.getLogger(f"{__name__}.FeaturePipeline")
        self._feature_cache: dict[str, FeatureExtractionResult] = {}

    def add_extractor(self, extractor: FeatureExtractor) -> None:
        """Füge Feature-Extraktor zur Pipeline hinzu"""
        if not isinstance(extractor, FeatureExtractor):
            raise ValueError("Extractor must inherit from FeatureExtractor")

        self.extractors.append(extractor)
        self.logger.info(f"Added feature extractor: {extractor.name}")

    def remove_extractor(self, name: str) -> bool:
        """
        Entferne Feature-Extraktor aus Pipeline.

        Args:
            name: Name des zu entfernenden Extraktors

        Returns:
            True wenn erfolgreich entfernt
        """
        for i, extractor in enumerate(self.extractors):
            if extractor.name == name:
                del self.extractors[i]
                self.logger.info(f"Removed feature extractor: {name}")
                return True

        self.logger.warning(f"Feature extractor not found: {name}")
        return False

    def extract_all_features(
        self, invoice_data: dict[str, Any], use_cache: bool = True, **kwargs: Any
    ) -> dict[str, FeatureExtractionResult]:
        """
        Extrahiere Features mit allen aktivierten Extraktoren.

        Args:
            invoice_data: Strukturierte Rechnungsdaten
            use_cache: Verwende Caching für Performance
            **kwargs: Zusätzliche Parameter

        Returns:
            Dictionary mit Ergebnissen aller Extraktoren
        """
        results: dict[str, FeatureExtractionResult] = {}

        # Create cache key
        cache_key = self._create_cache_key(invoice_data) if use_cache else None

        for extractor in self.extractors:
            if not extractor.is_enabled():
                self.logger.debug(f"Skipping disabled extractor: {extractor.name}")
                continue

            # Check cache first
            if cache_key and f"{extractor.name}_{cache_key}" in self._feature_cache:
                results[extractor.name] = self._feature_cache[
                    f"{extractor.name}_{cache_key}"
                ]
                self.logger.debug(f"Using cached features for: {extractor.name}")
                continue

            try:
                self.logger.debug(f"Extracting features with: {extractor.name}")

                if not extractor.validate_input(invoice_data):
                    self.logger.error(f"Input validation failed for: {extractor.name}")
                    continue

                import time

                start_time = time.time()

                result = extractor.extract_features(invoice_data, **kwargs)

                computation_time = time.time() - start_time

                # Ensure metadata is not None
                if result.metadata is None:
                    result.metadata = FeatureMetadata(
                        name=extractor.name, extractor_type=extractor.__class__.__name__
                    )

                result.metadata.computation_time = computation_time
                result.metadata.name = extractor.name
                result.metadata.extractor_type = extractor.__class__.__name__
                result.metadata.feature_count = len(result.features)

                results[extractor.name] = result

                # Cache result
                if cache_key:
                    self._feature_cache[f"{extractor.name}_{cache_key}"] = result

                self.logger.info(
                    f"Extracted {len(result.features)} features with {extractor.name} "
                    f"in {computation_time:.3f}s"
                )

            except Exception as e:
                self.logger.error(
                    f"Feature extraction failed for {extractor.name}: {e}"
                )

                # Create error result
                error_result = FeatureExtractionResult(
                    features={},
                    metadata=FeatureMetadata(
                        name=extractor.name,
                        extractor_type=extractor.__class__.__name__,
                        success=False,
                        error_message=str(e),
                    ),
                )
                results[extractor.name] = error_result

        return results

    def combine_features(
        self, extraction_results: dict[str, FeatureExtractionResult]
    ) -> FeatureDict:
        """
        Kombiniere Features von verschiedenen Extraktoren zu einem Vektor.

        Args:
            extraction_results: Ergebnisse aller Extraktoren

        Returns:
            Kombinierter Feature-Vektor
        """
        combined_features: FeatureDict = {}

        for extractor_name, result in extraction_results.items():
            if result.metadata is None or not result.metadata.success:
                self.logger.warning(f"Skipping failed extraction: {extractor_name}")
                continue

            # Prefix feature names with extractor name to avoid conflicts
            for feature_name, feature_value in result.features.items():
                prefixed_name = f"{extractor_name}_{feature_name}"
                combined_features[prefixed_name] = feature_value

        self.logger.info(
            f"Combined {len(combined_features)} features from {len(extraction_results)} extractors"
        )
        return combined_features

    def get_all_feature_names(self) -> list[str]:
        """Gebe alle verfügbaren Feature-Namen zurück"""
        all_names = []
        for extractor in self.extractors:
            extractor_names = extractor.get_feature_names()
            prefixed_names = [f"{extractor.name}_{name}" for name in extractor_names]
            all_names.extend(prefixed_names)
        return all_names

    def get_pipeline_stats(self) -> dict[str, Any]:
        """Gebe Pipeline-Statistiken zurück"""
        return {
            "total_extractors": len(self.extractors),
            "enabled_extractors": sum(1 for e in self.extractors if e.is_enabled()),
            "cache_size": len(self._feature_cache),
            "extractor_names": [e.name for e in self.extractors if e.is_enabled()],
        }

    def clear_cache(self) -> None:
        """Lösche Feature-Cache"""
        cache_size = len(self._feature_cache)
        self._feature_cache.clear()
        self.logger.info(f"Cleared feature cache ({cache_size} entries)")

    def _create_cache_key(self, invoice_data: dict[str, Any]) -> str:
        """Erstelle Cache-Key für Rechnungsdaten"""
        import hashlib

        # Use essential fields for cache key
        key_fields = [
            str(invoice_data.get("rechnung_nummer", "")),
            str(invoice_data.get("lieferant", "")),
            str(invoice_data.get("gesamt_betrag", 0)),
            str(len(invoice_data.get("line_items", []))),
        ]

        key_string = "|".join(key_fields)
        return hashlib.md5(key_string.encode()).hexdigest()[:16]


class FeatureSelector:
    """
    Automatische Feature-Auswahl basierend auf verschiedenen Kriterien.

    Implementiert verschiedene Algorithmen zur Auswahl der wichtigsten Features
    für die SKR03-Klassifizierung.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(f"{__name__}.FeatureSelector")

    def select_by_variance(
        self, features_dict: dict[str, list[float]], threshold: float = 0.01
    ) -> list[str]:
        """
        Selektiere Features basierend auf Varianz.

        Args:
            features_dict: Dictionary mit Feature-Namen und Werten
            threshold: Minimale Varianz

        Returns:
            Liste der ausgewählten Feature-Namen
        """
        if not features_dict:
            return []

        selected = []
        for feature_name, values in features_dict.items():
            if not values or len(values) < 2:
                continue

            # Calculate variance manually
            mean_val = sum(values) / len(values)
            variance = sum((x - mean_val) ** 2 for x in values) / len(values)

            if variance > threshold:
                selected.append(feature_name)

        self.logger.info(
            f"Selected {len(selected)} features by variance (threshold: {threshold})"
        )
        return selected

    def select_by_correlation(
        self,
        features_dict: dict[str, list[float]],
        target_values: list[float],
        threshold: float = 0.1,
    ) -> list[str]:
        """
        Selektiere Features basierend auf Korrelation mit Ziel-Variable.

        Args:
            features_dict: Dictionary mit Feature-Namen und Werten
            target_values: Ziel-Variable Werte
            threshold: Minimale Korrelation

        Returns:
            Liste der ausgewählten Feature-Namen
        """
        if not features_dict or not target_values:
            return []

        selected = []
        n = len(target_values)

        # Calculate target mean
        target_mean = sum(target_values) / n

        for feature_name, feature_values in features_dict.items():
            if len(feature_values) != n:
                continue

            # Calculate correlation manually
            feature_mean = sum(feature_values) / n

            # Numerator and denominators for correlation
            numerator = sum(
                (feature_values[i] - feature_mean) * (target_values[i] - target_mean)
                for i in range(n)
            )

            feature_var = sum((x - feature_mean) ** 2 for x in feature_values)
            target_var = sum((x - target_mean) ** 2 for x in target_values)

            if feature_var == 0 or target_var == 0:
                continue

            correlation = numerator / (feature_var * target_var) ** 0.5

            if abs(correlation) >= threshold:
                selected.append(feature_name)

        self.logger.info(
            f"Selected {len(selected)} features by correlation (threshold: {threshold})"
        )
        return selected

    def select_top_k(
        self, feature_importance: dict[str, float], k: int = 50
    ) -> list[str]:
        """
        Selektiere Top-K Features basierend auf Importance-Scores.

        Args:
            feature_importance: Dictionary mit Feature-Namen und Importance-Scores
            k: Anzahl der zu selektierenden Features

        Returns:
            Liste der Top-K Feature-Namen
        """
        if not feature_importance:
            return []

        # Sort by importance score (descending)
        sorted_features = sorted(
            feature_importance.items(), key=lambda x: x[1], reverse=True
        )

        selected = [feature for feature, _ in sorted_features[:k]]

        self.logger.info(f"Selected top {len(selected)} features by importance")
        return selected


# Pydantic models for configuration
class FeatureEngineeringConfig(BaseModel):
    """Konfiguration für Feature-Engineering Pipeline"""

    model_config = ConfigDict(validate_assignment=True)

    # Extractor settings
    enable_text_features: bool = Field(
        default=True, description="Text-Features aktivieren"
    )
    enable_layout_features: bool = Field(
        default=True, description="Layout-Features aktivieren"
    )
    enable_domain_features: bool = Field(
        default=True, description="Domain-Features aktivieren"
    )
    enable_temporal_features: bool = Field(
        default=True, description="Temporal-Features aktivieren"
    )

    # Performance settings
    use_caching: bool = Field(default=True, description="Feature-Caching aktivieren")
    max_cache_size: int = Field(default=1000, description="Maximale Cache-Größe")
    parallel_extraction: bool = Field(default=False, description="Parallele Extraktion")

    # Feature selection
    auto_feature_selection: bool = Field(
        default=True, description="Automatische Feature-Auswahl"
    )
    max_features: int = Field(default=100, description="Maximale Anzahl Features")
    min_variance_threshold: float = Field(default=0.01, description="Minimale Varianz")
    min_correlation_threshold: float = Field(
        default=0.05, description="Minimale Korrelation"
    )

    # Quality settings
    min_confidence: float = Field(
        default=0.6, ge=0.0, le=1.0, description="Minimale Konfidenz"
    )
    validate_features: bool = Field(default=True, description="Feature-Validierung")


# Export main classes
__all__ = [
    "FeatureExtractor",
    "FeaturePipeline",
    "FeatureSelector",
    "FeatureExtractionResult",
    "FeatureMetadata",
    "FeatureEngineeringConfig",
    "FeatureDict",
]
