#!/usr/bin/env python3
"""
Debug Tabellen-Extraktion - Sonepar Test PDF
===========================================

Analysiert detailliert was in den Tabellen extrahiert wird
und warum so wenige Artikel erkannt werden.
"""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.append("/Users/czok/Skripte/llkjj_v0.1/llkjj_ml")

from src.extraction.docling_processor import AdvancedDoclingProcessor

# Setup logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def debug_table_extraction():
    """Debug Tabellen-Extraktion für Sonepar PDF"""

    pdf_path = Path(
        "/Users/czok/Skripte/llkjj_v0.1/llkjj_ml/test_pdfs/Sonepar_test3.pdf"
    )

    print(f"🔍 Analysiere Tabellen-Extraktion: {pdf_path.name}")

    # Initialize Docling processor
    processor = AdvancedDoclingProcessor(
        use_gpu=True,
        enable_table_structure=True,
        enable_ocr=True,
        ocr_engine="rapid",
        table_mode="accurate",
        german_optimized=True,
    )

    # Extract tables
    result, quality_score = processor.process(str(pdf_path))

    print(f"Result keys: {list(result.keys())}")

    raw_text = result.get("content", "")
    structured_data = result.get("structured_data", {})
    tables = structured_data.get("tables", [])

    print("\n📊 TABELLEN-ANALYSE:")
    print(f"   Anzahl Tabellen: {len(tables)}")
    print(f"   Qualitäts-Score: {quality_score:.1%}")
    print(f"   Text-Länge: {len(raw_text)} Zeichen")

    # Analyze each table
    for idx, table in enumerate(tables):
        print(f"\n📋 TABELLE {idx + 1}:")
        headers = table.get("headers", [])
        rows = table.get("rows", [])

        print(f"   Headers ({len(headers)}): {headers}")
        print(f"   Rows: {len(rows)}")

        # Show first few rows
        for row_idx, row in enumerate(rows[:5]):  # Erste 5 Zeilen
            print(f"   Row {row_idx + 1}: {row}")

        if len(rows) > 5:
            print(f"   ... und {len(rows) - 5} weitere Zeilen")

    # Show a sample of the raw text
    print("\n📄 RAW TEXT SAMPLE (erste 1000 Zeichen):")
    print(raw_text[:1000])
    print("...")


if __name__ == "__main__":
    debug_table_extraction()
