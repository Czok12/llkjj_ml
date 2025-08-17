#!/usr/bin/env python3
"""
Debug Script für offizielle Docling Table API
Basiert auf IBM Docling Documentation und Beispielen

Verwendet die korrekte conv_res.document.tables API statt manueller Parsing.
"""

import logging
from pathlib import Path

import pandas as pd
from docling.document_converter import DocumentConverter

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def debug_docling_official():
    """Debug der offiziellen Docling Table API für Sonepar PDF"""

    input_pdf = Path("test_pdfs/Sonepar_test3.pdf")
    if not input_pdf.exists():
        logger.error(f"PDF nicht gefunden: {input_pdf}")
        return

    logger.info(f"=== Docling Official API Debug für {input_pdf} ===")

    # Docling DocumentConverter erstellen
    converter = DocumentConverter()

    # PDF konvertieren
    logger.info("Starte PDF Konvertierung mit Docling...")
    result = converter.convert(input_pdf)

    logger.info(f"Konvertierung Status: {result.status}")
    logger.info(f"Dokument: {result.document}")

    # Tabellen aus DoclingDocument extrahieren (offizielle API)
    tables = result.document.tables
    logger.info("\n=== OFFIZIELLE DOCLING TABLES API ===")
    logger.info(f"Anzahl Tabellen gefunden: {len(tables)}")

    total_rows = 0
    all_articles = []

    for table_ix, table in enumerate(tables):
        logger.info(f"\n--- Tabelle {table_ix + 1} ---")

        # Tabelle als DataFrame exportieren (offizielle Methode)
        try:
            df = table.export_to_dataframe()
            logger.info(f"DataFrame Shape: {df.shape}")
            logger.info(f"DataFrame Columns: {df.columns.tolist()}")
            logger.info(f"DataFrame Index: {df.index.tolist()}")

            # DataFrame Details
            logger.info("\nDataFrame Head:")
            logger.info(df.head())

            # DataFrame als Markdown anzeigen
            logger.info(f"\nTabelle {table_ix + 1} als Markdown:")
            print(df.to_markdown())

            # Alle Zellen zählen für Artikel-Extraktion
            non_empty_cells = 0
            for col in df.columns:
                for val in df[col]:
                    if pd.notna(val) and str(val).strip():
                        non_empty_cells += 1
                        # Artikel-ähnliche Zellen sammeln
                        cell_content = str(val).strip()
                        if cell_content and cell_content not in ["", "nan", "NaN"]:
                            all_articles.append(
                                {
                                    "table_id": table_ix + 1,
                                    "content": cell_content,
                                    "type": "cell_content",
                                }
                            )

            logger.info(
                f"Nicht-leere Zellen in Tabelle {table_ix + 1}: {non_empty_cells}"
            )
            total_rows += len(df)

            # CSV Export für Debug
            csv_file = f"debug_table_{table_ix + 1}_official.csv"
            df.to_csv(csv_file, index=True)
            logger.info(f"Tabelle gespeichert als: {csv_file}")

        except Exception as e:
            logger.error(
                f"Fehler beim DataFrame Export für Tabelle {table_ix + 1}: {e}"
            )

    # Zusammenfassung
    logger.info("\n=== ZUSAMMENFASSUNG ===")
    logger.info(f"Total Tabellen: {len(tables)}")
    logger.info(f"Total Zeilen: {total_rows}")
    logger.info(f"Total Artikel-Zellen gesammelt: {len(all_articles)}")

    # Artikel anzeigen
    if all_articles:
        logger.info(f"\n=== EXTRAHIERTE ARTIKEL ({len(all_articles)}) ===")
        for i, article in enumerate(all_articles[:20], 1):  # Erste 20 anzeigen
            logger.info(f"{i:2d}. [T{article['table_id']}] {article['content']}")

        if len(all_articles) > 20:
            logger.info(f"... und {len(all_articles) - 20} weitere Artikel")

    # Rohe Tabellen-Informationen anzeigen
    logger.info("\n=== ROHE TABELLEN-OBJEKTE ===")
    for table_ix, table in enumerate(tables):
        logger.info(f"Tabelle {table_ix + 1}: {type(table)}")
        if hasattr(table, "data"):
            logger.info(f"  - data: {table.data}")
        if hasattr(table, "model_dump"):
            try:
                table_dict = table.model_dump()
                logger.info(f"  - Keys: {list(table_dict.keys())}")
            except Exception as e:
                logger.info(f"  - model_dump() Error: {e}")

    return all_articles


if __name__ == "__main__":
    articles = debug_docling_official()
    print(f"\nDEBUG ABGESCHLOSSEN: {len(articles) if articles else 0} Artikel gefunden")
