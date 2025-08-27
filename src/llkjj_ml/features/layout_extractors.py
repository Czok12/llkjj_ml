#!/usr/bin/env python3
"""
LLKJJ ML Pipeline - Layout Feature Extractors
=============================================

Layout- und strukturelle Feature-Extraktion für PDF-Rechnungen.
Analysiert Dokumentstruktur, Tabellen, und visuelle Eigenschaften.

Author: LLKJJ ML Pipeline
Version: 1.0.0
"""

import logging
from typing import Any

from . import FeatureExtractionResult, FeatureExtractor, FeatureMetadata

logger = logging.getLogger(__name__)


class LayoutFeatureExtractor(FeatureExtractor):
    """
    Extrahiert layout-basierte Features aus PDF-Dokumenten.

    Features:
    - Tabellen-Analyse (Anzahl, Größe, Komplexität)
    - Text-Block-Verteilung
    - Seitenlayout-Eigenschaften
    - Strukturelle Muster
    """

    def __init__(self, name: str = "layout_features", enabled: bool = True) -> None:
        super().__init__(name, enabled)

    def extract_features(
        self, invoice_data: dict[str, Any], **kwargs: Any
    ) -> FeatureExtractionResult:
        """Extrahiere alle Layout-Features"""

        features: dict[str, float | int | str | bool | list[Any]] = {}

        # Basis-Layout-Features
        features.update(self._extract_document_structure(invoice_data))

        # Tabellen-Features
        features.update(self._extract_table_features(invoice_data))

        # Text-Block-Features
        features.update(self._extract_text_block_features(invoice_data))

        # Seiten-Features
        features.update(self._extract_page_features(invoice_data))

        return FeatureExtractionResult(
            features=features,
            metadata=FeatureMetadata(
                name=self.name,
                extractor_type=self.__class__.__name__,
                success=True,
                feature_count=len(features),
            ),
        )

    def _extract_document_structure(
        self, invoice_data: dict[str, Any]
    ) -> dict[str, float | int | str | bool | list[Any]]:
        """Grundlegende Dokumentstruktur"""
        features: dict[str, float | int | str | bool | list[Any]] = {}

        # Anzahl verschiedener Elemente
        features["page_count"] = float(len(invoice_data.get("pages", [1])))
        features["total_elements"] = float(len(invoice_data.get("elements", [])))

        # Document type indicators
        raw_text = invoice_data.get("raw_text", "")
        features["has_header"] = bool(
            "rechnung" in raw_text.lower() or "invoice" in raw_text.lower()
        )
        features["has_footer"] = bool(
            "seite" in raw_text.lower() or "page" in raw_text.lower()
        )
        features["has_logo"] = bool(
            "logo" in str(invoice_data.get("metadata", {})).lower()
        )

        # Content distribution
        text_length = len(raw_text)
        page_count = float(len(invoice_data.get("pages", [1])))
        features["text_density"] = float(text_length / max(page_count, 1))

        return features

    def _extract_table_features(
        self, invoice_data: dict[str, Any]
    ) -> dict[str, float | int | str | bool | list[Any]]:
        """Tabellen-spezifische Features"""
        features: dict[str, float | int | str | bool | list[Any]] = {}

        tables = invoice_data.get("tables", [])
        features["table_count"] = float(len(tables))

        if not tables:
            # Default values when no tables
            features.update(
                {
                    "avg_table_rows": 0.0,
                    "avg_table_cols": 0.0,
                    "max_table_size": 0.0,
                    "table_complexity_score": 0.0,
                    "has_main_table": False,
                    "table_text_ratio": 0.0,
                }
            )
            return features

        # Analyze each table
        row_counts = []
        col_counts = []
        table_sizes = []
        table_text_lengths = []

        for table in tables:
            rows = 1  # Default value
            cols = 1  # Default value

            if isinstance(table, dict):
                # Extract table dimensions
                if "rows" in table:
                    rows = len(table["rows"])
                elif "content" in table:
                    # Estimate rows from content
                    content = str(table["content"])
                    rows = content.count("\n") + 1

                row_counts.append(rows)

                # Estimate columns (simplified)
                if "columns" in table:
                    cols = len(table["columns"])
                elif "content" in table:
                    # Estimate from content structure
                    lines = str(table["content"]).split("\n")
                    if lines:
                        # Count separators in first line
                        separators = max(
                            lines[0].count("|"),
                            lines[0].count("\t"),
                            lines[0].count("  "),  # Multiple spaces
                        )
                        cols = separators + 1

                col_counts.append(cols)
                table_sizes.append(rows * cols)

                # Table text content length
                if "content" in table:
                    table_text_lengths.append(len(str(table["content"])))
                else:
                    table_text_lengths.append(0)

        # Calculate aggregate features
        features["avg_table_rows"] = (
            float(sum(row_counts) / len(row_counts)) if row_counts else 0.0
        )
        features["avg_table_cols"] = (
            float(sum(col_counts) / len(col_counts)) if col_counts else 0.0
        )
        features["max_table_size"] = float(max(table_sizes)) if table_sizes else 0.0

        # Table complexity score (based on size and structure)
        complexity_scores = []
        for i, table in enumerate(tables):
            if i < len(row_counts) and i < len(col_counts):
                complexity = float(row_counts[i] * col_counts[i])
                # Bonus for nested structures or special formatting
                if isinstance(table, dict) and "formatting" in table:
                    complexity *= 1.5
                complexity_scores.append(complexity)

        features["table_complexity_score"] = (
            float(sum(complexity_scores) / len(complexity_scores))
            if complexity_scores
            else 0.0
        )

        # Main table detection (largest table is usually the itemized list)
        max_table_size = features["max_table_size"]
        if isinstance(max_table_size, int | float):
            features["has_main_table"] = max_table_size > 10.0
        else:
            features["has_main_table"] = False

        # Table text ratio
        total_table_text = sum(table_text_lengths)
        total_document_text = len(invoice_data.get("raw_text", ""))
        features["table_text_ratio"] = (
            float(total_table_text / total_document_text)
            if total_document_text > 0
            else 0.0
        )

        return features

    def _extract_text_block_features(
        self, invoice_data: dict[str, Any]
    ) -> dict[str, float | int | str | bool | list[Any]]:
        """Text-Block-Verteilung und -Struktur"""
        features: dict[str, float | int | str | bool | list[Any]] = {}

        # Text blocks analysis
        raw_text = invoice_data.get("raw_text", "")

        # Split into logical blocks (paragraphs)
        text_blocks = [
            block.strip() for block in raw_text.split("\n\n") if block.strip()
        ]
        features["text_block_count"] = float(len(text_blocks))

        if text_blocks:
            block_lengths = [len(block) for block in text_blocks]
            avg_length = sum(block_lengths) / len(block_lengths)
            features["avg_block_length"] = float(avg_length)
            features["max_block_length"] = float(max(block_lengths))
            features["min_block_length"] = float(min(block_lengths))

            # Block length variance (indicates structure uniformity)
            variance = sum(
                (length - avg_length) ** 2 for length in block_lengths
            ) / len(block_lengths)
            features["block_length_variance"] = variance

        else:
            features.update(
                {
                    "avg_block_length": 0.0,
                    "max_block_length": 0.0,
                    "min_block_length": 0.0,
                    "block_length_variance": 0.0,
                }
            )

        # Line-based analysis
        lines = [line.strip() for line in raw_text.split("\n") if line.strip()]
        features["line_count"] = float(len(lines))

        if lines:
            line_lengths = [len(line) for line in lines]
            features["avg_line_length"] = float(sum(line_lengths) / len(line_lengths))

            # Short lines (headers, labels)
            short_lines = [line for line in lines if len(line) < 20]
            features["short_line_ratio"] = float(len(short_lines) / len(lines))

            # Long lines (descriptions, paragraphs)
            long_lines = [line for line in lines if len(line) > 80]
            features["long_line_ratio"] = float(len(long_lines) / len(lines))

        else:
            features.update(
                {
                    "avg_line_length": 0.0,
                    "short_line_ratio": 0.0,
                    "long_line_ratio": 0.0,
                }
            )

        return features

    def _extract_page_features(
        self, invoice_data: dict[str, Any]
    ) -> dict[str, float | int | str | bool | list[Any]]:
        """Seiten-bezogene Features"""
        features: dict[str, float | int | str | bool | list[Any]] = {}

        # Page metadata
        metadata = invoice_data.get("metadata", {})

        # Document format indicators
        features["is_scanned"] = bool(metadata.get("scanned", False))
        features["is_digital"] = not features["is_scanned"]

        # Quality indicators
        quality_score = invoice_data.get("quality_score", 1.0)
        features["quality_score"] = float(quality_score)
        features["high_quality"] = quality_score > 0.8
        features["low_quality"] = quality_score < 0.5

        # Content organization
        raw_text = invoice_data.get("raw_text", "")

        # Header content (first 20% of text)
        text_length = len(raw_text)
        if text_length > 0:
            header_text = raw_text[: int(text_length * 0.2)]
            footer_text = raw_text[int(text_length * 0.8) :]

            # Header characteristics
            features["header_has_company"] = bool(
                any(
                    keyword in header_text.lower()
                    for keyword in ["gmbh", "ag", "e.k.", "ug"]
                )
            )
            features["header_has_address"] = bool(
                len(
                    [
                        match
                        for match in header_text
                        if match.isdigit() and len(match) == 5
                    ]
                )
                > 0  # PLZ
            )

            # Footer characteristics
            features["footer_has_page_number"] = bool(
                any(
                    keyword in footer_text.lower()
                    for keyword in ["seite", "page", "blatt"]
                )
            )
            features["footer_has_contact"] = bool(
                any(
                    keyword in footer_text.lower()
                    for keyword in ["tel", "fax", "mail", "@"]
                )
            )

        else:
            features.update(
                {
                    "header_has_company": False,
                    "header_has_address": False,
                    "footer_has_page_number": False,
                    "footer_has_contact": False,
                }
            )

        return features

    def get_feature_names(self) -> list[str]:
        """Gebe alle Layout Feature-Namen zurück"""
        return [
            # Document structure
            "page_count",
            "total_elements",
            "has_header",
            "has_footer",
            "has_logo",
            "text_density",
            # Table features
            "table_count",
            "avg_table_rows",
            "avg_table_cols",
            "max_table_size",
            "table_complexity_score",
            "has_main_table",
            "table_text_ratio",
            # Text block features
            "text_block_count",
            "avg_block_length",
            "max_block_length",
            "min_block_length",
            "block_length_variance",
            "line_count",
            "avg_line_length",
            "short_line_ratio",
            "long_line_ratio",
            # Page features
            "is_scanned",
            "is_digital",
            "quality_score",
            "high_quality",
            "low_quality",
            "header_has_company",
            "header_has_address",
            "footer_has_page_number",
            "footer_has_contact",
        ]


class DocumentQualityExtractor(FeatureExtractor):
    """
    Extrahiert Qualitäts-Indikatoren für Dokument-Verarbeitung.

    Features:
    - OCR-Qualität
    - Text-Klarheit
    - Struktur-Konsistenz
    - Verarbeitungs-Schwierigkeit
    """

    def __init__(self, name: str = "quality_features", enabled: bool = True) -> None:
        super().__init__(name, enabled)

    def extract_features(
        self, invoice_data: dict[str, Any], **kwargs: Any
    ) -> FeatureExtractionResult:
        """Extrahiere Qualitäts-Features"""

        features: dict[str, float | int | str | bool | list[Any]] = {}

        # OCR-Qualitäts-Features
        features.update(self._extract_ocr_quality(invoice_data))

        # Text-Konsistenz-Features
        features.update(self._extract_text_consistency(invoice_data))

        # Struktur-Klarheit-Features
        features.update(self._extract_structure_clarity(invoice_data))

        # Verarbeitungs-Schwierigkeit
        features.update(self._extract_processing_difficulty(invoice_data))

        return FeatureExtractionResult(
            features=features,
            metadata=FeatureMetadata(
                name=self.name,
                extractor_type=self.__class__.__name__,
                success=True,
                feature_count=len(features),
            ),
        )

    def _extract_ocr_quality(
        self, invoice_data: dict[str, Any]
    ) -> dict[str, float | int | str | bool | list[Any]]:
        """OCR-Qualitäts-Indikatoren"""
        features: dict[str, float | int | str | bool | list[Any]] = {}

        raw_text = invoice_data.get("raw_text", "")

        # Character-level quality indicators
        if raw_text:
            # Unusual character patterns (OCR errors)
            unusual_chars = len(
                [c for c in raw_text if ord(c) > 127 and c not in "äöüÄÖÜß"]
            )
            features["unusual_char_ratio"] = float(unusual_chars / len(raw_text))

            # Broken words (spaces in middle of words)
            import re

            broken_word_patterns = [
                r"\b[a-zäöü]\s+[a-zäöü]+\b",  # Single char followed by space
                r"\b[A-ZÄÖÜ]\s+[a-zäöü]+\b",  # Capital followed by lowercase
            ]
            broken_words = sum(
                len(re.findall(pattern, raw_text)) for pattern in broken_word_patterns
            )
            features["broken_word_count"] = float(broken_words)

            # Repetitive characters (OCR artifacts)
            repetitive_chars = len(re.findall(r"(.)\1{3,}", raw_text))
            features["repetitive_char_count"] = float(repetitive_chars)

            # Missing spaces (words run together)
            missing_space_indicators = len(re.findall(r"[a-z][A-Z]", raw_text))
            features["missing_space_indicators"] = float(missing_space_indicators)

        else:
            features.update(
                {
                    "unusual_char_ratio": 0.0,
                    "broken_word_count": 0.0,
                    "repetitive_char_count": 0.0,
                    "missing_space_indicators": 0.0,
                }
            )

        # Confidence from extraction process
        features["extraction_confidence"] = float(
            invoice_data.get("quality_score", 1.0)
        )

        return features

    def _extract_text_consistency(
        self, invoice_data: dict[str, Any]
    ) -> dict[str, float | int | str | bool | list[Any]]:
        """Text-Konsistenz-Analyse"""
        features: dict[str, float | int | str | bool | list[Any]] = {}

        raw_text = invoice_data.get("raw_text", "")

        if not raw_text:
            features.update(
                {
                    "language_consistency": 1.0,
                    "formatting_consistency": 1.0,
                    "encoding_issues": 0.0,
                }
            )
            return features

        # Language consistency (should be primarily German)
        german_indicators = ["der", "die", "das", "und", "mit", "für", "von", "zu"]
        german_count = sum(1 for word in german_indicators if word in raw_text.lower())
        total_words = len(raw_text.split())
        features["language_consistency"] = float(german_count / max(total_words, 1))

        # Formatting consistency
        lines = raw_text.split("\n")
        if lines:
            # Check for consistent indentation
            indented_lines = [
                line for line in lines if line.startswith("  ") or line.startswith("\t")
            ]
            features["formatting_consistency"] = float(
                1.0 - (len(indented_lines) / len(lines))
            )
        else:
            features["formatting_consistency"] = 1.0

        # Encoding issues
        encoding_issue_patterns = ["â€", "Ã", "â", "€œ", "€"]
        encoding_issues = sum(
            pattern in raw_text for pattern in encoding_issue_patterns
        )
        features["encoding_issues"] = float(encoding_issues)

        return features

    def _extract_structure_clarity(
        self, invoice_data: dict[str, Any]
    ) -> dict[str, float | int | str | bool | list[Any]]:
        """Struktur-Klarheits-Analyse"""
        features: dict[str, float | int | str | bool | list[Any]] = {}

        # Clear section separation
        raw_text = invoice_data.get("raw_text", "")
        sections = raw_text.split("\n\n")
        features["clear_sections"] = float(
            len(sections) > 3
        )  # Good documents have clear sections

        # Table structure clarity
        tables = invoice_data.get("tables", [])
        if tables:
            # Check if main table has clear structure
            main_table = max(
                tables, key=lambda t: len(str(t)) if isinstance(t, dict) else 0
            )
            if isinstance(main_table, dict):
                has_headers = "header" in str(main_table).lower()
                has_rows = "row" in str(main_table).lower() or "\n" in str(
                    main_table.get("content", "")
                )
                features["table_structure_clarity"] = float(has_headers and has_rows)
            else:
                features["table_structure_clarity"] = 0.5
        else:
            features["table_structure_clarity"] = 0.0

        # Key information detectability
        key_patterns = [
            r"rechnung\w*nummer",
            r"datum",
            r"betrag",
            r"summe",
            r"mwst",
            r"steuer",
        ]

        import re

        detected_keys = sum(
            1 for pattern in key_patterns if re.search(pattern, raw_text.lower())
        )
        features["key_info_detectability"] = float(detected_keys / len(key_patterns))

        return features

    def _extract_processing_difficulty(
        self, invoice_data: dict[str, Any]
    ) -> dict[str, float | int | str | bool | list[Any]]:
        """Verarbeitungs-Schwierigkeit-Indikatoren"""
        features: dict[str, float | int | str | bool | list[Any]] = {}

        # Complexity indicators
        raw_text = invoice_data.get("raw_text", "")

        # Text complexity
        words = raw_text.split()
        if words:
            avg_word_length = sum(len(word) for word in words) / len(words)
            features["text_complexity"] = float(avg_word_length / 10.0)  # Normalize
        else:
            features["text_complexity"] = 0.0

        # Layout complexity
        table_count = len(invoice_data.get("tables", []))
        page_count = len(invoice_data.get("pages", [1]))
        features["layout_complexity"] = float(
            (table_count + page_count) / 5.0
        )  # Normalize

        # Processing confidence (inverse of difficulty)
        extraction_confidence = invoice_data.get("quality_score", 1.0)
        features["processing_confidence"] = float(extraction_confidence)
        features["processing_difficulty"] = float(1.0 - extraction_confidence)

        # Multi-language indicators (increase difficulty)
        english_indicators = ["invoice", "total", "amount", "date", "number"]
        english_count = sum(
            1 for word in english_indicators if word in raw_text.lower()
        )
        features["multi_language_complexity"] = float(
            english_count / len(english_indicators)
        )

        return features

    def get_feature_names(self) -> list[str]:
        """Gebe alle Quality Feature-Namen zurück"""
        return [
            # OCR quality
            "unusual_char_ratio",
            "broken_word_count",
            "repetitive_char_count",
            "missing_space_indicators",
            "extraction_confidence",
            # Text consistency
            "language_consistency",
            "formatting_consistency",
            "encoding_issues",
            # Structure clarity
            "clear_sections",
            "table_structure_clarity",
            "key_info_detectability",
            # Processing difficulty
            "text_complexity",
            "layout_complexity",
            "processing_confidence",
            "processing_difficulty",
            "multi_language_complexity",
        ]
