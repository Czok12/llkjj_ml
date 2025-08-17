#!/usr/bin/env python3
"""
LLKJJ ML Pipeline - Feature Engineering Integration (Simplified)
===============================================================

Vereinfachte ML-Integration für Feature Engineering Pipeline.
Stellt Basis-Funktionalität für Feature-Extraktion und SKR03-Training bereit.

Author: LLKJJ ML Pipeline
Version: 1.0.0 (Simplified)
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from src.features import FeaturePipeline
from src.features.domain_extractors import (
    ElectricalContractorExtractor,
    TemporalFeatureExtractor,
)
from src.features.layout_extractors import (
    DocumentQualityExtractor,
    LayoutFeatureExtractor,
)
from src.features.text_extractors import GermanNERExtractor, TextFeatureExtractor

logger = logging.getLogger(__name__)


class FeatureEngineeringMLIntegration:
    """
    Vereinfachte ML-Integration für Feature Engineering Pipeline.

    Stellt Basis-Funktionalität bereit:
    - Feature-Pipeline-Setup
    - Feature-Extraktion für Dokumente
    - Datenexport für Training
    """

    def __init__(
        self, model_path: Path = Path("output_model/feature_enhanced_model.pkl")
    ) -> None:
        self.model_path = model_path

        # Feature Pipeline Setup
        self.feature_pipeline = self._setup_feature_pipeline()

        # Performance Tracking
        self.baseline_accuracy: float = 0.85  # Aktueller Stand aus copilot-instructions
        self.target_accuracy: float = 0.92  # Ziel aus User-Request

        logger.info("Feature Engineering ML Integration (Simplified) initialisiert")

    def _setup_feature_pipeline(self) -> FeaturePipeline:
        """Setup der vollständigen Feature Pipeline"""

        # Alle Feature-Extraktoren
        extractors = [
            TextFeatureExtractor(name="text_features"),
            GermanNERExtractor(name="ner_features"),
            LayoutFeatureExtractor(name="layout_features"),
            DocumentQualityExtractor(name="quality_features"),
            ElectricalContractorExtractor(name="electrical_features"),
            TemporalFeatureExtractor(name="temporal_features"),
        ]

        # Feature Pipeline mit allen Extraktoren
        pipeline = FeaturePipeline(extractors=extractors)

        logger.info(f"Feature Pipeline mit {len(extractors)} Extraktoren erstellt")
        return pipeline

    def extract_features_for_document(
        self, invoice_data: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Extrahiere Features für ein einzelnes Dokument

        Args:
            invoice_data: Rechnungsdaten

        Returns:
            Extrahierte Features als Dictionary
        """
        try:
            # Feature-Extraktion über Pipeline
            extraction_results = self.feature_pipeline.extract_all_features(
                invoice_data
            )
            combined_features = self.feature_pipeline.combine_features(
                extraction_results
            )

            # Pipeline-Statistiken
            pipeline_stats = self.feature_pipeline.get_pipeline_stats()

            result = {
                "success": True,
                "features": combined_features,
                "feature_count": len(combined_features),
                "extraction_results": {
                    name: {
                        "success": (
                            result.metadata.success if result.metadata else False
                        ),
                        "feature_count": len(result.features),
                        "computation_time": (
                            result.metadata.computation_time if result.metadata else 0.0
                        ),
                    }
                    for name, result in extraction_results.items()
                },
                "pipeline_stats": pipeline_stats,
            }

            logger.info(
                f"Features erfolgreich extrahiert: {len(combined_features)} Features"
            )
            return result

        except Exception as e:
            logger.error(f"Feature-Extraktion fehlgeschlagen: {e}")
            return {
                "success": False,
                "error": str(e),
                "features": {},
                "feature_count": 0,
            }

    def extract_features_batch(
        self, invoice_data_list: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Extrahiere Features für mehrere Dokumente

        Args:
            invoice_data_list: Liste von Rechnungsdaten

        Returns:
            Liste von Feature-Extraktions-Ergebnissen
        """
        logger.info(f"Batch-Feature-Extraktion für {len(invoice_data_list)} Dokumente")

        results = []
        successful_count = 0

        for i, invoice_data in enumerate(invoice_data_list):
            result = self.extract_features_for_document(invoice_data)

            # Dokument-Index hinzufügen
            result["document_index"] = i

            if result["success"]:
                successful_count += 1

            results.append(result)

        success_rate = (
            successful_count / len(invoice_data_list) if invoice_data_list else 0
        )
        logger.info(
            f"Batch-Extraktion abgeschlossen: {successful_count}/{len(invoice_data_list)} erfolgreich ({success_rate:.1%})"
        )

        return results

    def export_training_data(
        self,
        invoice_data_list: list[dict[str, Any]],
        skr03_labels: list[str],
        output_path: Path = Path("data/training/feature_enhanced_training.jsonl"),
    ) -> dict[str, Any]:
        """
        Exportiere Trainingsdaten mit Features

        Args:
            invoice_data_list: Liste von Rechnungsdaten
            skr03_labels: Entsprechende SKR03-Kontennummern
            output_path: Ausgabepfad für Trainingsdaten

        Returns:
            Export-Statistiken
        """
        if len(invoice_data_list) != len(skr03_labels):
            raise ValueError("Anzahl Dokumente und Labels stimmt nicht überein")

        logger.info(f"Exportiere Trainingsdaten für {len(invoice_data_list)} Dokumente")

        # Features extrahieren
        feature_results = self.extract_features_batch(invoice_data_list)

        # Trainingsdaten zusammenstellen
        training_data = []
        successful_exports = 0

        for i, (feature_result, skr03_label) in enumerate(
            zip(feature_results, skr03_labels, strict=False)
        ):
            if feature_result["success"]:
                training_entry = {
                    "document_id": i,
                    "skr03_label": skr03_label,
                    "features": feature_result["features"],
                    "feature_count": feature_result["feature_count"],
                    "original_data": {
                        "lieferant": invoice_data_list[i].get("lieferant", ""),
                        "gesamt_betrag": invoice_data_list[i].get("gesamt_betrag", 0),
                        "rechnung_nummer": invoice_data_list[i].get(
                            "rechnung_nummer", ""
                        ),
                    },
                }
                training_data.append(training_entry)
                successful_exports += 1
            else:
                logger.warning(f"Feature-Extraktion für Dokument {i} fehlgeschlagen")

        # Export
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for entry in training_data:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        # Statistiken
        stats = {
            "total_documents": len(invoice_data_list),
            "successful_exports": successful_exports,
            "success_rate": successful_exports / len(invoice_data_list),
            "total_features": len(training_data[0]["features"]) if training_data else 0,
            "output_path": str(output_path),
            "export_timestamp": datetime.now().isoformat(),
        }

        logger.info(
            f"Trainingsdaten exportiert: {successful_exports}/{len(invoice_data_list)} erfolgreich"
        )
        logger.info(f"Ausgabe: {output_path}")

        return stats

    def analyze_feature_extraction(
        self, feature_results: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Analysiere Feature-Extraktions-Ergebnisse

        Args:
            feature_results: Ergebnisse der Feature-Extraktion

        Returns:
            Analyse-Ergebnisse
        """
        successful_results = [r for r in feature_results if r["success"]]

        if not successful_results:
            return {"error": "Keine erfolgreichen Feature-Extraktionen"}

        # Feature-Anzahl-Statistiken
        feature_counts = [r["feature_count"] for r in successful_results]

        # Extraktor-Performance
        extractor_stats = {}
        for result in successful_results:
            for extractor_name, extractor_result in result.get(
                "extraction_results", {}
            ).items():
                if extractor_name not in extractor_stats:
                    extractor_stats[extractor_name] = {
                        "success_count": 0,
                        "total_features": 0,
                        "total_time": 0.0,
                    }

                if extractor_result["success"]:
                    extractor_stats[extractor_name]["success_count"] += 1
                    extractor_stats[extractor_name]["total_features"] += (
                        extractor_result["feature_count"]
                    )
                    extractor_stats[extractor_name]["total_time"] += (
                        extractor_result.get("computation_time", 0.0)
                    )

        # Durchschnittswerte berechnen
        for _extractor_name, stats in extractor_stats.items():
            success_count = stats["success_count"]
            if success_count > 0:
                stats["avg_features"] = stats["total_features"] / success_count
                stats["avg_time"] = stats["total_time"] / success_count
                stats["success_rate"] = success_count / len(feature_results)
            else:
                stats["avg_features"] = 0.0
                stats["avg_time"] = 0.0
                stats["success_rate"] = 0.0

        return {
            "total_documents": len(feature_results),
            "successful_extractions": len(successful_results),
            "success_rate": len(successful_results) / len(feature_results),
            "feature_count_stats": {
                "min": min(feature_counts),
                "max": max(feature_counts),
                "avg": sum(feature_counts) / len(feature_counts),
                "total_unique_features": len(
                    set().union(*[r["features"].keys() for r in successful_results])
                ),
            },
            "extractor_performance": extractor_stats,
        }

    def get_pipeline_info(self) -> dict[str, Any]:
        """Gebe Pipeline-Informationen zurück"""
        return {
            "pipeline_stats": self.feature_pipeline.get_pipeline_stats(),
            "all_feature_names": self.feature_pipeline.get_all_feature_names(),
            "baseline_accuracy": self.baseline_accuracy,
            "target_accuracy": self.target_accuracy,
            "accuracy_gap": self.target_accuracy - self.baseline_accuracy,
        }


def create_feature_pipeline() -> FeatureEngineeringMLIntegration:
    """
    Factory Function für Feature Pipeline

    Returns:
        Vorkonfigurierte Feature Engineering Integration
    """
    integration = FeatureEngineeringMLIntegration()

    logger.info("Feature Engineering Pipeline erstellt")
    return integration
