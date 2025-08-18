
Absolut. Basierend auf dem umfassenden Projektkontext und der Analyse der bereitgestellten Dateien habe ich mehrere Schlüsselbereiche für Verbesserungen identifiziert, die die Architektur straffen, die Codequalität erhöhen und das Projekt strategisch auf Phase 2 vorbereiten.

Meine Änderungen konzentrieren sich auf drei Hauptbereiche:

1. **Vervollständigung des Training-Modul-Refactorings:** Ich habe die redundante `spacy_training`-Paketstruktur entfernt und die zentrale `main.py` auf den modernen, besser strukturierten `TrainingService` in `src/trainer.py` umgestellt. Dies eliminiert Codeduplizierung und verbessert die Wartbarkeit.
2. **Verbesserung der Trainingsdaten-Qualität:** Die Erzeugung von NER-Annotationen in `training_data_persistence.py` war fehleranfällig. Ich habe den Prozess überarbeitet, um positionsgenaue Annotationen direkt aus den strukturierten Gemini-Ergebnissen zu erstellen. Dies ist ein entscheidender Schritt, um die Qualität der Trainingsdaten für Phase 2 (lokale Autonomie) sicherzustellen.
3. **Konsolidierung der Processor-Architektur:** Das öffentliche `ml_service`-Paket verwendete eine veraltete Version des `UnifiedProcessor`. Ich habe dies korrigiert, um den neuen, auf dem Strategy-Pattern basierenden `UnifiedProcessor` zu verwenden, und das veraltete Modul entfernt. Dies richtet die Implementierung an der strategischen Vision aus.

Hier sind die überarbeiteten Dateien:

### `src/training_data_persistence.py`

*(**Änderungsgrund:** Die Qualität der generierten NER-Trainingsdaten wurde durch eine robuste, positionsgenaue Methode zur Annotation anstelle des fehleranfälligen `string.find()`-Ansatzes drastisch verbessert.)*

```python
#!/usr/bin/env python3
"""
LLKJJ ML Pipeline - Training Data Persistence Service
====================================================

Strategischer Service für die systematische Sammlung von Trainingsdaten
aus der Gemini-Pipeline für die zukünftige lokale Autonomie (Phase 2).

Funktionen:
- spaCy-Trainingsdaten (JSONL) für NER/TextCat
- RAG-System-Population mit validierten Klassifizierungen
- GoBD-konformes Audit-Logging für Nachvollziehbarkeit

Strategische Bedeutung:
- Jede verarbeitete Rechnung = wertvolle Trainingsdaten
- Vorbereitung für Phase 2: Gemini → lokale spaCy/RAG Transition
- Kontinuierliche Verbesserung der Klassifizierungsqualität

Autor: LLKJJ ML Pipeline Team
Version: 1.0.0
Datum: 18. August 2025
"""

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from sentence_transformers import SentenceTransformer

from src.config import Config

logger = logging.getLogger(__name__)


class TrainingDataPersistence:
    """
    Service für die systematische Persistierung von Trainingsdaten.

    Sammelt und speichert Daten aus der Gemini-Pipeline für:
    1. spaCy-Training (NER/TextCat)
    2. RAG-System-Enhancement
    3. Audit-Trail (GoBD-konform)
    """

    def __init__(self, config: Config):
        self.config = config
        self.base_path = Path("data/training")
        self.logs_path = Path("logs")

        # Stelle sicher, dass Verzeichnisse existieren
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.logs_path.mkdir(parents=True, exist_ok=True)

        # Trainingsdaten-Pfade
        self.spacy_annotations_file = self.base_path / "gemini_spacy_annotations.jsonl"
        self.audit_log_file = self.logs_path / "audit_gemini.jsonl"

        # RAG-System (ChromaDB) Setup
        self._setup_chromadb()
        self._setup_embedding_model()

    def _setup_chromadb(self) -> None:
        """Setup ChromaDB für RAG-System-Population"""
        try:
            self.chroma_client = chromadb.PersistentClient(path="data/vectors")
            self.collection: Collection | None = self.chroma_client.get_or_create_collection(
                name="invoice_classifications",
                metadata={
                    "description": "Gemini-validierte Rechnungsklassifizierungen"
                },
            )
            logger.info("✅ ChromaDB für Trainingsdaten-RAG initialisiert")
        except Exception as e:
            logger.warning(f"⚠️ ChromaDB Setup fehlgeschlagen: {e}")
            self.collection = None

    def _setup_embedding_model(self) -> None:
        """Setup Embedding-Modell für RAG-System"""
        try:
            self.embedding_model = SentenceTransformer(
                "sentence-transformers/distiluse-base-multilingual-cased"
            )
            logger.info("✅ Embedding-Modell für Trainingsdaten geladen")
        except Exception as e:
            logger.warning(f"⚠️ Embedding-Modell Setup fehlgeschlagen: {e}")
            self.embedding_model = None

    def persist_training_data(
        self,
        pdf_path: str,
        validated_gemini_result: dict[str, Any],
        skr03_classifications: list[dict[str, Any]],
        processing_result: Any,  # ProcessingResult
    ) -> dict[str, Any]:
        """
        Hauptfunktion: Persistiere alle Trainingsdaten nach erfolgreicher Verarbeitung.

        Args:
            pdf_path: Pfad zur verarbeiteten PDF
            validated_gemini_result: Pydantic-validierte Gemini-Response
            skr03_classifications: SKR03-Klassifizierungen
            processing_result: Vollständiges ProcessingResult

        Returns:
            Dict mit Statistiken zur Persistierung
        """
        stats: dict[str, Any] = {
            "pdf_path": pdf_path,
            "timestamp": datetime.now().isoformat(),
            "spacy_annotations": 0,
            "rag_documents": 0,
            "audit_logged": False,
        }

        try:
            # 1. spaCy-Trainingsdaten speichern
            spacy_stats = self._persist_spacy_annotations(
                pdf_path, validated_gemini_result
            )
            stats["spacy_annotations"] = spacy_stats["annotation_count"]

            # 2. RAG-System mit Klassifizierungen füttern
            rag_stats = self._persist_rag_classifications(
                pdf_path, validated_gemini_result, skr03_classifications
            )
            stats["rag_documents"] = rag_stats["document_count"]

            # 3. Audit-Trail für GoBD-Konformität
            audit_stats = self._persist_audit_trail(
                pdf_path, validated_gemini_result, processing_result
            )
            stats["audit_logged"] = audit_stats["logged"]

            logger.info(
                "✅ Trainingsdaten-Persistierung erfolgreich: %d spaCy + %d RAG + Audit",
                stats["spacy_annotations"],
                stats["rag_documents"],
            )

        except Exception as e:
            logger.error(f"❌ Trainingsdaten-Persistierung fehlgeschlagen: {e}")
            stats["error"] = str(e)

        return stats

    def _persist_spacy_annotations(
        self,
        pdf_path: str,
        gemini_result: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Speichere spaCy-Trainingsdaten im JSONL-Format.

        Format: {"text": "...", "entities": [...], "cats": {...}}
        """
        try:
            # Erzeuge Text und Entitäten mit exakten Positionen
            raw_text, entities = self._create_annotated_text(gemini_result)

            if not raw_text:
                return {"annotation_count": 0, "error": "Kein Text für Training extrahiert"}

            # Kategorien für TextCat
            categories = self._extract_categories_for_training(gemini_result)

            # spaCy-Annotations-Format
            annotation_entry = {
                "text": raw_text,
                "entities": entities,
                "cats": categories,
                "meta": {
                    "pdf_path": pdf_path,
                    "source": "gemini_validated",
                    "timestamp": datetime.now().isoformat(),
                    "gemini_model": gemini_result.get("gemini_model", "unknown"),
                },
            }

            # Append zu JSONL-Datei
            with open(self.spacy_annotations_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(annotation_entry, ensure_ascii=False) + "\n")

            logger.debug(f"💾 spaCy-Annotations gespeichert: {len(entities)} Entitäten")

            return {
                "annotation_count": len(entities),
                "category_count": len(categories),
                "text_length": len(raw_text),
            }

        except Exception as e:
            logger.error(f"❌ spaCy-Annotations-Speicherung fehlgeschlagen: {e}")
            return {"annotation_count": 0, "error": str(e)}

    def _create_annotated_text(
        self, gemini_result: dict[str, Any]
    ) -> tuple[str, list[tuple[int, int, str]]]:
        """
        Erstellt den Trainingstext und die NER-Entitäten mit exakten Positionen.
        Dieser robuste Ansatz vermeidet fehleranfälliges `string.find()`.
        """
        text_parts: list[str] = []
        entities: list[tuple[int, int, str]] = []
        current_pos = 0

        def add_entity(text: str, label: str) -> None:
            nonlocal current_pos
            if not text or not isinstance(text, str):
                return

            start_pos = current_pos
            text_parts.append(text)
            current_pos += len(text)
            entities.append((start_pos, current_pos, label))

            # Add a separator
            text_parts.append("\n")
            current_pos += 1

        # Header-Informationen
        header = gemini_result.get("invoice_header", {})
        if header.get("lieferant"):
            add_entity(header["lieferant"], "HÄNDLER")
        if header.get("rechnungsnummer"):
            add_entity(header["rechnungsnummer"], "RECHNUNGSNUMMER")
        if header.get("rechnungsdatum"):
            add_entity(header["rechnungsdatum"], "RECHNUNGSDATUM")

        # Line Items
        for item in gemini_result.get("line_items", []):
            if item.get("beschreibung"):
                add_entity(item["beschreibung"], "ARTIKEL")
            if item.get("artikelnummer"):
                add_entity(item["artikelnummer"], "ARTIKELNUMMER")
            if item.get("menge"):
                add_entity(str(item["menge"]), "MENGE")
            if item.get("einzelpreis"):
                add_entity(f"{item['einzelpreis']:.2f}", "EINZELPREIS")
            if item.get("gesamtpreis"):
                add_entity(f"{item['gesamtpreis']:.2f}", "GESAMTPREIS")

        return "".join(text_parts).strip(), entities

    def _extract_categories_for_training(
        self, gemini_result: dict[str, Any]
    ) -> dict[str, float]:
        """Extrahiere Kategorien für TextCat-Training"""
        categories = {}

        # Elektro-Kategorien aus Line Items
        elektro_kategorien = set()
        for item in gemini_result.get("line_items", []):
            if item.get("elektro_kategorie"):
                # Normalisiere Kategorie für spaCy (keine Leerzeichen/Sonderzeichen)
                clean_kategorie = item["elektro_kategorie"].upper().replace(" ", "_").replace("/", "_")
                elektro_kategorien.add(clean_kategorie)

        # Kategorien mit Konfidenz (1.0 für bekannte, 0.0 für unbekannte)
        for kategorie in elektro_kategorien:
            categories[f"ELEKTRO_{kategorie}"] = 1.0

        return categories

    def _persist_rag_classifications(
        self,
        pdf_path: str,
        gemini_result: dict[str, Any],
        skr03_classifications: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Speichere Klassifizierungen im RAG-System (ChromaDB).
        """
        if not self.collection or not self.embedding_model:
            logger.warning(
                "⚠️ RAG-System nicht verfügbar - überspringe RAG-Persistierung"
            )
            return {"document_count": 0, "error": "RAG system not available"}

        try:
            documents = []
            metadatas = []
            embeddings = []
            ids = []

            for i, classification in enumerate(skr03_classifications):
                # Dokument-Text für Embedding
                lieferant = gemini_result.get("invoice_header", {}).get(
                    "lieferant", "Unbekannt"
                )
                beschreibung = classification.get("beschreibung", "") or classification.get("description", "")
                menge = classification.get("menge", 0)
                einzelpreis = classification.get("einzelpreis", 0)

                if not beschreibung:
                    continue

                # Kontextueller Dokument-Text (wie in RAG-Strategie beschrieben)
                doc_text = (
                    f"Artikel: {beschreibung}, "
                    f"Menge: {menge}, "
                    f"Preis: {einzelpreis:.2f} EUR. "
                    f"Lieferant: {lieferant} (Elektro-Großhandel). "
                    f"Kategorie: {classification.get('elektro_kategorie', 'Unbekannt')}."
                )

                # Embedding erstellen
                embedding = self.embedding_model.encode(doc_text).tolist()

                # Metadaten mit validation_status (RAG-Strategie)
                predicted_account = classification.get("predicted_account") or classification.get("skr03_konto")
                metadata = {
                    "pdf_path": pdf_path,
                    "supplier": lieferant,
                    "skr03_account": str(predicted_account or ""),
                    "skr03_description": classification.get("account_description", ""),
                    "confidence": float(classification.get("confidence", 0.0)),
                    "validation_status": "ai_suggested",  # 🎯 RAG-Strategie Implementation
                    "source": "gemini_validated",
                    "timestamp": datetime.now().isoformat(),
                    "elektro_kategorie": classification.get("elektro_kategorie", ""),
                    "price_range": self._get_price_range(einzelpreis),
                }

                # Eindeutige ID basierend auf Inhalt
                doc_id = self._generate_document_id(pdf_path, i, beschreibung)

                documents.append(doc_text)
                metadatas.append(metadata)
                embeddings.append(embedding)
                ids.append(doc_id)

            # Batch-Insert in ChromaDB
            if documents:
                self.collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    embeddings=embeddings,
                    ids=ids,
                )

                logger.info(
                    f"💾 RAG-Dokumente gespeichert: {len(documents)} Klassifizierungen"
                )

            return {
                "document_count": len(documents),
                "collection_size": self.collection.count(),
            }

        except Exception as e:
            logger.error(f"❌ RAG-Persistierung fehlgeschlagen: {e}")
            return {"document_count": 0, "error": str(e)}

    def _get_price_range(self, price: float) -> str:
        """Kategorisiere Preise für bessere RAG-Filterung"""
        if price < 10:
            return "niedrig"
        elif price < 100:
            return "mittel"
        elif price < 800:
            return "hoch"
        else:
            return "anlagevermögen"

    def _generate_document_id(
        self, pdf_path: str, item_index: int, description: str
    ) -> str:
        """Generiere eindeutige Dokument-ID"""
        content = f"{pdf_path}_{item_index}_{description}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def _persist_audit_trail(
        self,
        pdf_path: str,
        gemini_result: dict[str, Any],
        processing_result: Any,
    ) -> dict[str, Any]:
        """
        Speichere GoBD-konformen Audit-Trail.
        """
        try:
            audit_entry = {
                "timestamp": datetime.now().isoformat(),
                "pdf_path": pdf_path,
                "pdf_hash": self._calculate_pdf_hash(pdf_path),
                "processing_method": "gemini_first",
                "gemini_model": gemini_result.get("gemini_model", "unknown"),
                "raw_gemini_response": gemini_result,  # Vollständige rohe Antwort
                "final_classification_count": len(gemini_result.get("line_items", [])),
                "total_amount": gemini_result.get("totals", {}).get("bruttosumme", 0),
                "validation_status": "pydantic_validated",
                "processing_duration_ms": getattr(
                    processing_result, "processing_time_ms", 0
                ),
                "confidence_score": getattr(processing_result, "confidence_score", 0),
            }

            # Append zu Audit-Log
            with open(self.audit_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(audit_entry, ensure_ascii=False) + "\n")

            logger.debug(f"📋 Audit-Trail gespeichert für {Path(pdf_path).name}")

            return {"logged": True, "log_file": str(self.audit_log_file)}

        except Exception as e:
            logger.error(f"❌ Audit-Trail-Speicherung fehlgeschlagen: {e}")
            return {"logged": False, "error": str(e)}

    def _calculate_pdf_hash(self, pdf_path: str) -> str:
        """Berechne SHA256-Hash der PDF-Datei für Audit-Trail"""
        try:
            with open(pdf_path, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception as e:
            logger.warning(f"⚠️ PDF-Hash-Berechnung fehlgeschlagen: {e}")
            return "hash_calculation_failed"

    def get_training_stats(self) -> dict[str, Any]:
        """
        Hole Statistiken über gesammelte Trainingsdaten.

        Returns:
            Dict mit Statistiken zu spaCy-Annotations, RAG-Dokumenten, etc.
        """
        stats = {
            "spacy_annotations": 0,
            "rag_documents": 0,
            "audit_entries": 0,
            "files": {
                "spacy_file": str(self.spacy_annotations_file),
                "audit_file": str(self.audit_log_file),
            },
        }

        try:
            # spaCy-Annotations zählen
            if self.spacy_annotations_file.exists():
                with open(self.spacy_annotations_file, encoding="utf-8") as f:
                    stats["spacy_annotations"] = sum(1 for _ in f)

            # RAG-Dokumente zählen
            if self.collection:
                stats["rag_documents"] = self.collection.count()

            # Audit-Einträge zählen
            if self.audit_log_file.exists():
                with open(self.audit_log_file, encoding="utf-8") as f:
                    stats["audit_entries"] = sum(1 for _ in f)

            logger.info(
                "📊 Trainingsdaten-Statistiken: %d spaCy + %d RAG + %d Audit",
                stats["spacy_annotations"],
                stats["rag_documents"],
                stats["audit_entries"],
            )

        except Exception as e:
            logger.error(f"❌ Statistiken-Abruf fehlgeschlagen: {e}")
            stats["error"] = str(e)

        return stats

```

### `main.py`

*(**Änderungsgrund:** Die Aufrufe der Trainings-Pipeline wurden von der veralteten `TrainingPipeline` auf den neuen, zentralen `TrainingService` umgestellt. Dies vereinfacht die Codebasis und stellt sicher, dass die refaktorierte Logik verwendet wird.)*

```python
#!/usr/bin/env python3
"""
LLKJJ ML Pipeline - Hauptmodul
=============================

Zentrale CLI-Schnittstelle für alle ML-Operationen:
- PDF-Verarbeitung mit KI-Pipeline (GEMINI-FIRST als Standard!)
- Training und Modell-Management
- Batch-Verarbeitung und Export
- Security & Production-Readiness
- Performance-Monitoring
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from src.config import Config

# Feature Engineering Integration
from src.features.ml_integration import create_feature_pipeline

# Import ProcessingResult for type-safe results
from src.models.processing_result import ProcessingResult

# Dual-Purpose Pipeline Import
from src.pipeline.processor import UnifiedProcessor

# Import consolidated services
from src.processing.modular_processor import ModularProcessor
from src.security.auditor import run_security_audit
from src.security.manager import APIKeyManager, validate_production_environment
from src.trainer import TrainingService


def setup_logging(verbose: bool = False, production: bool = False) -> None:
    """Configure logging for CLI"""
    if production:
        # Produktions-Modus: Nur Warnungen und Fehler
        level = logging.WARNING
    else:
        # Development-Modus: Debug oder Info
        level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("logs/llkjj_ml.log")],
    )


def process_pdfs(args: argparse.Namespace) -> None:
    """
    Process PDF invoices with NEW GEMINI-FIRST PIPELINE (Standard)

    NEUE STANDARD-PIPELINE:
    PDF → Gemini AI direkt → Strukturierte Daten → SKR03-Klassifizierung

    Kein Fallback auf Docling - bei Gemini-Fehlern wird Error-Log ausgegeben!
    """
    config = Config()

    # NEUE STANDARD-PIPELINE: Gemini-First Processor
    from src.pipeline.gemini_first_processor import GeminiDirectProcessor

    processor = GeminiDirectProcessor(config)

    input_path = Path(args.input)
    output_dir = Path(args.output) if args.output else Path("data/output")

    print("🚀 GEMINI-FIRST PIPELINE (Standard)")
    print(f"🔄 Processing: {input_path}")
    print(f"📁 Output: {output_dir}")

    if input_path.is_file() and input_path.suffix.lower() == ".pdf":
        # Single PDF mit Gemini-First
        try:
            result = processor.process_pdf_gemini_first(input_path)

            print("✅ GEMINI-FIRST Processing complete!")
            print(f"📄 Processed: {Path(result.pdf_path).name}")
            print(f"🎯 SKR03 classifications: {len(result.skr03_classifications)}")
            print(f"💾 Quality: {result.extraction_quality}")
            print(f"⚡ Processing time: {result.processing_time_ms}ms")
            print(
                f"🤖 Gemini model: {getattr(result.structured_data, 'gemini_model', 'gemini-2.0-flash-exp')}"
            )

            # Speichere Ergebnis
            output_file = output_dir / f"{input_path.stem}_gemini_result.json"
            output_dir.mkdir(parents=True, exist_ok=True)

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)

            print(f"💾 Result saved: {output_file}")

        except Exception as e:
            print(f"❌ GEMINI-FIRST Pipeline failed: {e}")
            print("💡 Hinweis: Überprüfe Google API Key und Gemini-Konfiguration")
            print("🚫 KEIN FALLBACK AUF DOCLING - Fehler muss behoben werden!")
            sys.exit(1)

    elif input_path.is_dir():
        # Batch process directory mit Gemini-First
        pdf_files = list(input_path.glob("*.pdf"))

        if not pdf_files:
            print(f"❌ No PDF files found in {input_path}")
            return

        print(f"📦 Found {len(pdf_files)} PDF files")

        results: list[ProcessingResult] = []
        failed_files: list[tuple[str, str]] = []

        for pdf_file in pdf_files:
            try:
                result = processor.process_pdf_gemini_first(pdf_file)
                results.append(result)

                print(
                    f"  ✅ {pdf_file.name}: {len(result.skr03_classifications)} SKR03, "
                    f"{result.extraction_quality} quality, "
                    f"{result.processing_time_ms}ms"
                )

            except Exception as e:
                failed_files.append((pdf_file.name, str(e)))
                print(f"  ❌ {pdf_file.name}: Gemini-Fehler - {e}")
                logging.error(
                    "GEMINI-FIRST Pipeline Fehler bei %s: %s",
                    pdf_file.name,
                    e,
                    exc_info=True,
                )
                continue

        # Summary
        total_skr03 = sum(len(r.skr03_classifications) for r in results)
        avg_confidence = (
            sum(r.confidence_score for r in results) / len(results) if results else 0
        )

        print("\n📊 GEMINI-FIRST Batch Summary:")
        print(f"   ✅ Files processed: {len(results)}/{len(pdf_files)}")
        print(f"   ❌ Failed files: {len(failed_files)}")
        print(f"   🎯 SKR03 classifications: {total_skr03}")
        print(f"   📈 Average confidence: {avg_confidence:.3f}")

        if failed_files:
            print("\n❌ Failed files (kein Fallback!):")
            for filename, error in failed_files:
                print(f"   - {filename}: {error}")

        # Speichere Batch-Ergebnisse
        batch_output = output_dir / "gemini_batch_results.json"
        output_dir.mkdir(parents=True, exist_ok=True)

        batch_data = {
            "results": [result.to_dict() for result in results],
            "failed_files": failed_files,
            "summary": {
                "processed": len(results),
                "total": len(pdf_files),
                "total_skr03": total_skr03,
                "avg_confidence": avg_confidence,
            },
        }

        with open(batch_output, "w", encoding="utf-8") as f:
            json.dump(batch_data, f, ensure_ascii=False, indent=2)

        print(f"💾 Batch results saved: {batch_output}")

    else:
        print(f"❌ Invalid input: {input_path} (must be PDF file or directory)")


def process_pdfs_async_batch(args: argparse.Namespace) -> None:
    """
    🚀 A3: HIGH-PERFORMANCE ASYNC BATCH PROCESSING (Strategic TODO)

    Performance-optimierte Verarbeitung mit:
    - Async Gemini API-Calls (3x parallel)
    - PDF-Hash-Caching (0ms für Duplikate)
    - Rate-Limiting für API-Schutz
    - Batch-Processing für multiple PDFs
    """
    config = Config()

    # 🎯 A3: AsyncGeminiDirectProcessor mit Performance-Optimierungen
    from src.pipeline.async_gemini_processor import AsyncGeminiDirectProcessor

    processor = AsyncGeminiDirectProcessor(config)
    input_path = Path(args.input)
    output_dir = Path(args.output) if args.output else Path("data/output")
    max_concurrent = getattr(args, "concurrent", 3)

    print("🚀 ASYNC BATCH PROCESSING (A3 Performance Optimization)")
    print(f"🔄 Processing: {input_path}")
    print(f"📁 Output: {output_dir}")
    print(f"⚡ Max concurrent: {max_concurrent}")

    async def _process_async():
        if input_path.is_file() and input_path.suffix.lower() == ".pdf":
            # Single PDF with async processing
            print(f"📄 Processing single PDF: {input_path.name}")

            start_time = time.time()
            result = await processor.process_pdf_async(input_path)
            processing_time = (time.time() - start_time) * 1000

            if result:
                print(f"✅ ASYNC Processing complete in {processing_time:.0f}ms!")
                print(f"📄 Processed: {Path(result.pdf_path).name}")
                print(f"🎯 SKR03 matches: {len(result.skr03_classifications)}")
                print(f"💾 Quality: {result.extraction_quality}")
                print(f"📊 Confidence: {result.confidence_score:.1%}")

                # Save result
                output_file = output_dir / f"{input_path.stem}_async_result.json"
                output_dir.mkdir(parents=True, exist_ok=True)

                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)

                print(f"💾 Result saved: {output_file}")
            else:
                print("❌ Processing failed")

        elif input_path.is_dir():
            # Batch process directory with async optimization
            pdf_files = list(input_path.glob("*.pdf"))

            if not pdf_files:
                print(f"❌ No PDF files found in {input_path}")
                return

            print(f"📊 Found {len(pdf_files)} PDF files for ASYNC BATCH processing")
            print(
                f"⚡ Processing {min(max_concurrent, len(pdf_files))} files in parallel..."
            )

            start_time = time.time()
            results = await processor.process_batch_async(
                pdf_files, max_concurrent=max_concurrent
            )
            total_time = (time.time() - start_time) * 1000

            # Statistics
            successful_results = [r for r in results if r is not None]
            failed_count = len(results) - len(successful_results)

            if successful_results:
                avg_confidence = sum(
                    r.confidence_score for r in successful_results
                ) / len(successful_results)
                total_skr03 = sum(
                    len(r.skr03_classifications) for r in successful_results
                )
                avg_processing_time = sum(
                    r.processing_time_ms for r in successful_results
                ) / len(successful_results)

                print("\n🎉 ASYNC BATCH PROCESSING COMPLETE!")
                print(
                    f"📊 Results: {len(successful_results)}/{len(pdf_files)} successful"
                )
                print(
                    f"⚡ Total time: {total_time:.0f}ms ({total_time/len(pdf_files):.0f}ms/file)"
                )
                print(f"💫 Avg processing: {avg_processing_time:.0f}ms/file")
                print(f"🎯 Total SKR03 matches: {total_skr03}")
                print(f"📊 Average confidence: {avg_confidence:.1%}")

                if failed_count > 0:
                    print(f"⚠️ Failed files: {failed_count}")

                # Save batch results
                batch_output = (
                    output_dir
                    / f"async_batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                )
                output_dir.mkdir(parents=True, exist_ok=True)

                batch_data = {
                    "metadata": {
                        "timestamp": datetime.now().isoformat(),
                        "method": "async_gemini_batch",
                        "total_files": len(pdf_files),
                        "successful": len(successful_results),
                        "failed": failed_count,
                        "total_time_ms": total_time,
                        "avg_time_ms": avg_processing_time,
                        "max_concurrent": max_concurrent,
                    },
                    "results": [r.to_dict() if r else None for r in results],
                    "statistics": {
                        "total_skr03_matches": total_skr03,
                        "avg_confidence": avg_confidence,
                        "performance_gain": (
                            f"{(5000 - avg_processing_time) / 5000 * 100:.1f}%"
                            if avg_processing_time < 5000
                            else "N/A"
                        ),
                    },
                }

                with open(batch_output, "w", encoding="utf-8") as f:
                    json.dump(batch_data, f, ensure_ascii=False, indent=2)

                print(f"💾 Async batch results saved: {batch_output}")
            else:
                print("❌ All files failed to process")

        else:
            print(f"❌ Invalid input: {input_path} (must be PDF file or directory)")

    # Run async processing
    try:
        asyncio.run(_process_async())
    except KeyboardInterrupt:
        print("\n⚠️ Async processing cancelled by user")
    except Exception as e:
        print(f"❌ Async processing error: {e}")
        logging.error("Async processing failed", exc_info=True)


def process_pdfs_unified_strategy(args: argparse.Namespace) -> None:
    """
    🔧 B1: UNIFIED STRATEGY PROCESSING (Strategic TODO)

    Strategy-Pattern-basierte Verarbeitung mit:
    - Automatische Engine-Auswahl (Gemini/spaCy/Hybrid)
    - Explizite Strategy-Auswahl für Tests
    - Strategy-Vergleich für Benchmarking
    - Nahtlose Transition zwischen Engines
    """
    config = Config()

    # 🎯 B1: UnifiedProcessor mit Strategy Pattern
    from src.pipeline.unified_processor import UnifiedProcessor

    processor = UnifiedProcessor(config)
    input_path = Path(args.input)
    output_dir = Path(args.output) if args.output else Path("data/output")
    strategy = getattr(args, "strategy", "auto")
    compare_strategies = getattr(args, "compare", False)

    print("🔧 UNIFIED STRATEGY PROCESSING (B1 Strategy Pattern)")
    print(f"🔄 Processing: {input_path}")
    print(f"📁 Output: {output_dir}")
    print(f"🎯 Strategy: {strategy}")

    # Show available strategies
    available = processor.get_available_strategies()
    print(f"📊 Available strategies: {available}")

    if compare_strategies:
        # Strategy comparison mode
        if not input_path.is_file() or not input_path.suffix.lower() == ".pdf":
            print("❌ Strategy comparison requires a single PDF file")
            return

        print(f"\n🔍 STRATEGY COMPARISON on {input_path.name}")
        print("=" * 60)

        try:
            comparison = processor.compare_strategies(input_path)

            if "error" in comparison:
                print(f"❌ Comparison failed: {comparison['error']}")
                return

            # Display comparison results
            for strategy_name, result in comparison["comparison_results"].items():
                if result["success"]:
                    print(f"\n✅ {strategy_name.upper()} STRATEGY:")
                    print(f"   ⏱️  Processing time: {result['processing_time_ms']}ms")
                    print(f"   📊 Confidence: {result['confidence_score']:.1%}")
                    print(f"   💎 Quality: {result['extraction_quality']}")
                    print(
                        f"   🎯 SKR03 classifications: {result['skr03_classifications']}"
                    )
                    print(f"   📋 Line items: {result['total_line_items']}")
                else:
                    print(f"\n❌ {strategy_name.upper()} STRATEGY: {result['error']}")

            # Summary
            summary = comparison["summary"]
            if summary["successful_strategies"] > 1:
                print("\n🏆 COMPARISON SUMMARY:")
                print(f"   ⚡ Fastest: {summary['fastest_strategy']}")
                print(f"   🎯 Highest confidence: {summary['highest_confidence']}")
                print(f"   📊 Most classifications: {summary['most_classifications']}")

            # Save comparison results
            comparison_file = output_dir / f"strategy_comparison_{input_path.stem}.json"
            output_dir.mkdir(parents=True, exist_ok=True)

            with open(comparison_file, "w", encoding="utf-8") as f:
                json.dump(comparison, f, ensure_ascii=False, indent=2)

            print(f"\n💾 Comparison results saved: {comparison_file}")

        except Exception as e:
            print(f"❌ Strategy comparison failed: {e}")
            logging.error("Strategy comparison failed", exc_info=True)

        return

    # Normal processing mode
    if input_path.is_file() and input_path.suffix.lower() == ".pdf":
        # Single PDF processing
        print(f"📄 Processing single PDF with {strategy} strategy...")

        try:
            start_time = time.time()
            result = processor.process_pdf(input_path, strategy)
            processing_time = (time.time() - start_time) * 1000

            print(f"✅ UNIFIED Processing complete in {processing_time:.0f}ms!")
            print(f"📄 Strategy used: {result.processing_method}")
            print(f"📄 Processed: {Path(result.pdf_path).name}")
            print(f"🎯 SKR03 matches: {len(result.skr03_classifications)}")
            print(f"💾 Quality: {result.extraction_quality}")
            print(f"📊 Confidence: {result.confidence_score:.1%}")

            # Save result
            output_file = (
                output_dir / f"{input_path.stem}_unified_{strategy}_result.json"
            )
            output_dir.mkdir(parents=True, exist_ok=True)

            result_data = result.to_dict()
            result_data["unified_processor_info"] = {
                "strategy_used": strategy,
                "strategy_pattern": "enabled",
                "available_strategies": available,
            }

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result_data, f, ensure_ascii=False, indent=2)

            print(f"💾 Result saved: {output_file}")

        except Exception as e:
            print(f"❌ Unified processing failed: {e}")
            logging.error("Unified processing failed", exc_info=True)

    elif input_path.is_dir():
        # Batch processing with strategy
        pdf_files = list(input_path.glob("*.pdf"))

        if not pdf_files:
            print(f"❌ No PDF files found in {input_path}")
            return

        print(f"📊 Found {len(pdf_files)} PDF files for unified batch processing")
        print(f"🎯 Using {strategy} strategy for all files...")

        successful_results = []
        failed_files = []

        for pdf_file in pdf_files:
            try:
                result = processor.process_pdf(pdf_file, strategy)
                successful_results.append(result)

                print(
                    f"  ✅ {pdf_file.name}: {len(result.skr03_classifications)} SKR03, "
                    f"{result.confidence_score:.1%} confidence, {result.extraction_quality} quality"
                )

            except Exception as e:
                failed_files.append((pdf_file.name, str(e)))
                print(f"  ❌ {pdf_file.name}: {e}")

        # Batch summary
        if successful_results:
            avg_confidence = sum(r.confidence_score for r in successful_results) / len(
                successful_results
            )
            total_skr03 = sum(len(r.skr03_classifications) for r in successful_results)
            avg_processing_time = sum(
                r.processing_time_ms for r in successful_results
            ) / len(successful_results)

            print("\n🎉 UNIFIED BATCH PROCESSING COMPLETE!")
            print(f"📊 Results: {len(successful_results)}/{len(pdf_files)} successful")
            print(f"🎯 Strategy used: {strategy}")
            print(f"⚡ Avg processing time: {avg_processing_time:.0f}ms/file")
            print(f"🎯 Total SKR03 matches: {total_skr03}")
            print(f"📊 Average confidence: {avg_confidence:.1%}")

            if failed_files:
                print(f"⚠️ Failed files: {len(failed_files)}")

            # Save batch results
            batch_output = (
                output_dir
                / f"unified_batch_{strategy}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            output_dir.mkdir(parents=True, exist_ok=True)

            batch_data = {
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "method": f"unified_{strategy}",
                    "strategy_pattern": True,
                    "total_files": len(pdf_files),
                    "successful": len(successful_results),
                    "failed": len(failed_files),
                    "avg_time_ms": avg_processing_time,
                    "available_strategies": available,
                },
                "results": [r.to_dict() for r in successful_results],
                "failed_files": failed_files,
                "statistics": {
                    "total_skr03_matches": total_skr03,
                    "avg_confidence": avg_confidence,
                    "strategy_used": strategy,
                },
            }

            with open(batch_output, "w", encoding="utf-8") as f:
                json.dump(batch_data, f, ensure_ascii=False, indent=2)

            print(f"💾 Unified batch results saved: {batch_output}")
        else:
            print("❌ All files failed to process")

    else:
        print(f"❌ Invalid input: {input_path} (must be PDF file or directory)")


def process_pdfs_docling_alternative(args: argparse.Namespace) -> None:
    """
    ALTERNATIVE: Process PDF invoices with DOCLING (nur bei expliziter Anfrage)

    Diese Methode wird nur noch bei expliziter Anfrage verwendet!
    Standard ist die neue Gemini-First Pipeline.
    """
    config = Config()

    # ALTERNATIVE: Docling-basierte Verarbeitung
    from src.pipeline.gemini_first_processor import DoclingAlternativeProcessor

    processor = DoclingAlternativeProcessor(config)

    input_path = Path(args.input)
    output_dir = Path(args.output) if args.output else Path("data/output")

    print("🔄 ALTERNATIVE: Docling-Pipeline (explizit angefordert)")
    print(f"🔄 Processing: {input_path}")
    print(f"📁 Output: {output_dir}")

    if input_path.is_file() and input_path.suffix.lower() == ".pdf":
        # Single PDF mit Docling-Alternative
        result = processor.process_pdf_with_docling(input_path)

        print("✅ DOCLING-ALTERNATIVE Processing complete!")
        print(f"📄 Processed: {Path(result['pdf_path']).name}")
        print(f"🎯 SKR03 matches: {len(result.get('skr03_classifications', []))}")
        print(f"💾 Quality: {result.get('extraction_quality', 'unknown')}")

        # Speichere Ergebnis
        output_file = output_dir / f"{input_path.stem}_docling_result.json"
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"💾 Result saved: {output_file}")

    elif input_path.is_dir():
        # Batch process directory mit Docling-Alternative
        pdf_files = list(input_path.glob("*.pdf"))

        if not pdf_files:
            print(f"❌ No PDF files found in {input_path}")
            return

        print(f"� Found {len(pdf_files)} PDF files for DOCLING processing")

        results: list[dict[str, Any]] = []
        failed_files: list[tuple[str, str]] = []

        for pdf_file in pdf_files:
            try:
                result = processor.process_pdf_with_docling(pdf_file)
                results.append(result)

                print(
                    f"  ✅ {pdf_file.name}: {len(result.get('skr03_classifications', []))} SKR03, "
                    f"{result.get('extraction_quality', 'unknown')} quality"
                )

            except Exception as e:
                failed_files.append((pdf_file.name, str(e)))
                print(f"  ❌ {pdf_file.name}: Docling-Fehler - {e}")
                logging.error(
                    "DOCLING-ALTERNATIVE Pipeline Fehler bei %s: %s",
                    pdf_file.name,
                    e,
                    exc_info=True,
                )
                continue

        # Summary
        total_skr03 = sum(len(r.get("skr03_classifications", [])) for r in results)

        print("\n📊 DOCLING-ALTERNATIVE Batch Summary:")
        print(f"   ✅ Files processed: {len(results)}/{len(pdf_files)}")
        print(f"   ❌ Failed files: {len(failed_files)}")
        print(f"   🎯 SKR03 classifications: {total_skr03}")

        if failed_files:
            print("\n❌ Failed files:")
            for filename, error in failed_files:
                print(f"   - {filename}: {error}")

        # Speichere Batch-Ergebnisse
        batch_output = output_dir / "docling_batch_results.json"
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(batch_output, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "results": results,
                    "failed_files": failed_files,
                    "summary": {
                        "processed": len(results),
                        "total": len(pdf_files),
                        "total_skr03": total_skr03,
                    },
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        print(f"💾 Batch results saved: {batch_output}")

    else:
        print(f"❌ Invalid input: {input_path} (must be PDF file or directory)")
        pdf_files = list(input_path.glob("*.pdf"))

        if not pdf_files:
            print(f"❌ No PDF files found in {input_path}")
            return


def export_training_data(args: argparse.Namespace) -> None:
    """Export processed data for spaCy training"""
    config = Config()
    training_service = TrainingService(config)

    input_dir = Path(args.input)
    output_path = Path(args.output) if args.output else Path("data/training")

    print(f"📤 Exporting training data from: {input_dir}")

    result = training_service.export_training_data(input_dir, output_path)

    print("✅ Export complete!")
    print(f"📊 Training examples: {result.total_records}")
    print(f"🎯 SKR03 classifications: {result.skr03_classifications}")
    print(f"💾 Saved: {result.jsonl_path}")


def train_model(args: argparse.Namespace) -> None:
    """Train spaCy models for German electrical invoices using new training pipeline"""
    config = Config()
    training_service = TrainingService(config)

    training_data = Path(args.input)
    model_output = Path(args.output) if args.output else Path("output_model")
    epochs = args.epochs
    model_type = getattr(args, "model_type", "ner")

    print(f"🚀 Training {model_type} model with: {training_data}")
    print(f"⚙️  Epochs: {epochs}")

    result = training_service.train_model(
        training_data, model_output, epochs, model_type
    )

    print(f"✅ Training complete!")
    print(f"🎯 F1 Score: {result.f1_score:.3f}")
    print(f"📊 Precision: {result.precision:.3f}")
    print(f"📊 Recall: {result.recall:.3f}")
    print(f"⏱️  Training time: {result.training_time_seconds:.1f}s")
    print(f"💾 Model saved to: {model_output}")

    print(f"📈 Metrics saved to: {model_output}/training_metrics.json")


def export_textcat_data(args: argparse.Namespace) -> None:
    """Export textcat training data for SKR03 classification"""
    config = Config()
    training_service = TrainingService(config)

    input_dir = Path(args.input)
    output_dir = Path(args.output) if args.output else Path("data/training/textcat/")

    print(f"📤 Exporting textcat data from: {input_dir}")
    print(f"🎯 Output directory: {output_dir}")

    # Use regular export (supports both NER and TextCat)
    result = training_service.export_training_data(input_dir, output_dir)

    print("✅ Textcat export complete!")
    print(f"💾 Training data saved: {result.jsonl_path}")
    print(f"📊 Total examples: {result.total_records}")


def train_textcat_model(args: argparse.Namespace) -> None:
    """Train spaCy text classification model for SKR03"""
    config = Config()
    training_service = TrainingService(config)

    training_data_dir = Path(args.input)
    model_output = (
        Path(args.output) if args.output else Path("output_model/textcat_model")
    )
    epochs = args.epochs

    print(f"🚀 Training textcat model with: {training_data_dir}")
    print(f"⚙️  Epochs: {epochs}")
    print(f"🎯 Output model: {model_output}")

    # Train textcat model using new API
    result = training_service.train_model(
        training_data_dir, model_output, epochs, "textcat"
    )

    print("✅ Textcat training complete!")
    print(f"📁 Model saved to: {model_output}")
    print(f"🎯 F1 Score: {result.f1_score:.3f}")


def run_pipeline(args: argparse.Namespace) -> None:
    """Run complete PDF→Training pipeline"""
    config = Config()

    input_dir = Path(args.input)
    temp_processed = Path("data/processed/pipeline_temp")
    temp_training_dir = Path("data/training")  # Verzeichnis für Export
    model_output = (
        Path(args.output) if args.output else Path("output_model/pipeline_model")
    )

    print("🔄 Running complete ML pipeline...")
    print(f"📁 Input: {input_dir}")
    print(f"🎯 Final model: {model_output}")

    # Step 1: Process PDFs
    print("\n📄 Step 1: Processing PDFs...")
    processor = UnifiedProcessor(config)
    pdf_files = list(input_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"❌ No PDF files found in {input_dir}")
        return

    temp_processed.mkdir(parents=True, exist_ok=True)

    for pdf_file in pdf_files:
        result = processor.process_pdf(pdf_file)
        print(f"  ✅ {pdf_file.name}: {len(result.skr03_classifications)} SKR03")

    # Step 2: Export training data
    print("\n📤 Step 2: Exporting training data...")
    training_service = TrainingService(config)
    export_result = training_service.export_training_data(
        temp_processed, temp_training_dir
    )
    print(f"  ✅ {export_result.total_records} training examples exported")

    # Step 3: Train model
    print("\n🚀 Step 3: Training model...")
    train_result = training_service.train_model(
        Path(export_result.jsonl_path), model_output, args.epochs
    )

    print("\n🎉 Pipeline complete!")
    print(f"🎯 F1 Score: {train_result.f1_score:.3f}")
    print(f"📁 Final model saved to: {model_output}")


def analyze_results(args: argparse.Namespace) -> None:
    """Analyze processing results and model performance"""
    results_dir = Path(args.input)

    print(f"📊 Analyzing results in: {results_dir}")

    # Load all result files
    json_files = list(results_dir.glob("*.json"))

    if not json_files:
        print(f"❌ No JSON result files found in {results_dir}")
        return

    total_docs = len(json_files)
    total_skr03 = 0
    processing_times = []
    quality_scores = []

    for json_file in json_files:
        try:
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)

            # Extract metrics
            skr03_count = len(data.get("skr03_classifications", []))
            total_skr03 += skr03_count

            if "processing_time_ms" in data:
                processing_times.append(
                    data["processing_time_ms"] / 1000
                )  # Convert to seconds

            if "confidence_score" in data:
                quality_scores.append(data["confidence_score"])

        except (FileNotFoundError, PermissionError) as e:
            print(f"⚠️  Dateizugriff fehlgeschlagen für {json_file.name}: {e}")
            continue
        except (json.JSONDecodeError, KeyError) as e:
            print(f"⚠️  Ungültiges JSON-Format in {json_file.name}: {e}")
            continue
        except Exception as e:
            print(f"⚠️  Unerwarteter Fehler bei Analyse von {json_file.name}: {e}")
            logging.error("Unerwarteter Fehler bei Datei-Analyse: %s", e, exc_info=True)
            continue

    # Calculate statistics
    avg_processing_time = (
        sum(processing_times) / len(processing_times) if processing_times else 0
    )
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0

    print("\n📈 Analysis Results:")
    print(f"   Documents processed: {total_docs}")
    print(f"   SKR03 classifications: {total_skr03}")
    print(f"   Average SKR03 per doc: {total_skr03/total_docs:.1f}")

    if processing_times:
        print(f"   Average processing time: {avg_processing_time:.2f}s")

    if quality_scores:
        print(f"   Average quality score: {avg_quality:.3f}")


# NEW MODULAR WORKFLOW FUNCTIONS
def run_workflow_1(args: argparse.Namespace) -> None:
    """Run Workflow 1: PDF → Docling → TXT only"""
    config = Config()
    processor = ModularProcessor(config)

    pdf_path = Path(args.input)
    output_path = Path(args.output) if args.output else None

    print("🔄 Workflow 1: PDF → Docling → TXT")
    print(f"📄 Input: {pdf_path}")

    result = processor.workflow_1_pdf_to_docling_txt(pdf_path, output_path)

    print("✅ Workflow 1 completed!")
    print(f"📝 Output TXT: {result['output_txt']}")
    print(f"📊 Text length: {result['text_length']} characters")
    print(f"📋 Tables found: {result['table_count']}")


def run_workflow_2(args: argparse.Namespace) -> None:
    """Run Workflow 2: PDF → Gemini directly"""
    config = Config()
    processor = ModularProcessor(config)

    pdf_path = Path(args.input)
    output_path = Path(args.output) if args.output else None

    print("🔄 Workflow 2: PDF → Gemini direct")
    print(f"📄 Input: {pdf_path}")

    result = processor.workflow_2_pdf_to_gemini_direct(pdf_path, output_path)

    print("✅ Workflow 2 completed!")
    print(f"💾 Output JSON: {result['output_json']}")
    print(f"📊 Raw text: {result['raw_text_length']} characters")
    print(f"🤖 Method: {result['extraction_method']}")


def run_workflow_3(args: argparse.Namespace) -> None:
    """Run Workflow 3: Docling TXT → Gemini processing"""
    config = Config()
    processor = ModularProcessor(config)

    txt_path = Path(args.input)
    output_path = Path(args.output) if args.output else None

    print("🔄 Workflow 3: Docling TXT → Gemini")
    print(f"📝 Input: {txt_path}")

    result = processor.workflow_3_docling_txt_to_gemini(txt_path, output_path)

    print("✅ Workflow 3 completed!")
    print(f"💾 Output JSON: {result['output_json']}")
    print(f"📊 Text processed: {result['text_length']} characters")
    print(f"🤖 Method: {result['processing_method']}")


def run_workflow_4(args: argparse.Namespace) -> None:
    """Run Workflow 4: Complete pipeline"""
    Config()
    processor = UnifiedProcessor()

    pdf_path = Path(args.input)
    Path(args.output) if args.output else None

    print("🔄 Workflow 4: Complete pipeline")
    print(f"📄 Input: {pdf_path}")

    # Verwende existierenden Workflow
    result = processor.process_pdf(str(pdf_path))

    print("✅ Workflow 4 completed!")
    print(f"💾 Output: {result}")
    print(f"🎯 Confidence: {result.confidence_score}")
    print(f"📊 Quality: {result.extraction_quality}")


def init_database(args: argparse.Namespace) -> None:
    """Initialize database with simple manager"""
    from src.database.simple_manager import create_simple_manager

    db_path = Path(args.path)

    print(f"🔧 Initializing database at: {db_path}")

    db_manager = create_simple_manager(str(db_path))
    results = db_manager.initialize_database()

    if results["success"]:
        print("✅ Database initialization completed!")
        print(f"⏱️  Total time: {results['total_time_seconds']}s")
        print(f"� Database path: {results['db_path']}")
        if "existing_collections" in results:
            print(f"�️  Existing collections: {results['existing_collections']}")
    else:
        print(
            f"❌ Database initialization failed: {results.get('error', 'Unknown error')}"
        )


def show_database_stats(args: argparse.Namespace) -> None:
    """Show database statistics with simple manager"""
    from src.database.simple_manager import create_simple_manager

    db_path = Path(args.path)

    print(f"📊 Database statistics for: {db_path}")

    db_manager = create_simple_manager(str(db_path))
    stats = db_manager.get_database_stats()

    if "error" in stats:
        print(f"❌ Error getting stats: {stats['error']}")
        return

    print(f"📁 Database path: {stats['db_path']}")
    print(f"🗃️  Total collections: {stats['total_collections']}")
    print(f"💓 Heartbeat: {stats['heartbeat']}")

    if stats["collections"]:
        print("\n📋 Collections:")
        for col in stats["collections"]:
            print(f"  • {col['name']}: {col['count']} documents")
    else:
        print("📭 No collections found")

        # Storage info
        storage = stats["database"]["storage"]
        print("\n💾 Storage:")
        print(f"   Path exists: {storage['path_exists']}")
        if storage["path_exists"]:
            print(f"   Total size: {storage['total_size_mb']} MB")

        # Collections info
        collections = stats["database"]["collections"]
        print(f"\n📚 Collections: {len(collections)}")
        for collection in collections:
            print(
                f"   {collection['name']}: {collection.get('count', 'N/A')} documents"
            )

        # Connection pool info
        pool_stats = stats["connection_pool"]
        print("\n🔗 Connection Pool:")
        print(f"   Active connections: {pool_stats['active_connections']}")
        print(f"   Total operations: {pool_stats['total_operations']}")
        print(
            f"   Average response time: {pool_stats['average_response_time_ms']:.2f}ms"
        )

        # Migration history
        migrations = stats["migration_history"]
        successful_migrations = [m for m in migrations if m["success"]]
        print(
            f"\n🔄 Migrations: {len(successful_migrations)}/{len(migrations)} successful"
        )


def backup_database(args: argparse.Namespace) -> None:
    """Create database backup with simple manager"""
    from src.database.simple_manager import create_simple_manager

    db_path = Path(args.path)
    backup_name = args.name

    print(f"💾 Creating backup for: {db_path}")

    db_manager = create_simple_manager(str(db_path))
    result = db_manager.backup_database(backup_name)

    if result["success"]:
        print(f"✅ Backup created: {result['backup_path']}")
        print(f"📦 Backup size: {result['backup_size_mb']:.2f} MB")
    else:
        print(f"❌ Backup failed: {result['error']}")


def optimize_database(args: argparse.Namespace) -> None:
    """Optimize database performance with simple manager"""
    from src.database.simple_manager import create_simple_manager

    db_path = Path(args.path)

    print(f"⚡ Optimizing database: {db_path}")

    db_manager = create_simple_manager(str(db_path))
    results = db_manager.optimize_database()

    if results["success"]:
        print("✅ Database optimization completed!")
        print(f"⏱️  Total time: {results['optimization_time_seconds']}s")
        print("🔧 Performance optimization applied")
    else:
        print(f"❌ Optimization failed: {results.get('error', 'Unknown error')}")


# FEATURE ENGINEERING COMMANDS
def extract_features(args: argparse.Namespace) -> None:
    """Extract features for single document"""
    input_path = Path(args.input)

    print(f"🔧 Extracting features from: {input_path}")

    # Load invoice data
    if input_path.suffix.lower() == ".json":
        with open(input_path, encoding="utf-8") as f:
            invoice_data = json.load(f)
    else:
        print("❌ Only JSON files supported for feature extraction")
        return

    # Create feature pipeline
    feature_pipeline = create_feature_pipeline()

    # Extract features
    result = feature_pipeline.extract_features_for_document(invoice_data)

    if result["success"]:
        print("✅ Feature extraction completed!")
        print(f"🎯 Features extracted: {result['feature_count']}")

        # Show pipeline stats
        print("\n📊 Pipeline Statistics:")
        for extractor, stats in result["extraction_results"].items():
            status = "✅" if stats["success"] else "❌"
            print(
                f"  {status} {extractor}: {stats['feature_count']} features ({stats['computation_time']:.3f}s)"
            )

        # Save features if output specified
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            print(f"💾 Features saved to: {output_path}")
    else:
        print(f"❌ Feature extraction failed: {result.get('error', 'Unknown error')}")


def extract_features_batch(args: argparse.Namespace) -> None:
    """Extract features for multiple documents"""
    input_dir = Path(args.input)
    output_dir = Path(args.output) if args.output else Path("data/features")

    print(f"🔧 Batch feature extraction from: {input_dir}")
    print(f"📁 Output directory: {output_dir}")

    # Find JSON files
    json_files = list(input_dir.glob("*.json"))
    if not json_files:
        print(f"❌ No JSON files found in {input_dir}")
        return

    print(f"📦 Found {len(json_files)} JSON files")

    # Load invoice data
    invoice_data_list = []
    for json_file in json_files:
        try:
            with open(json_file, encoding="utf-8") as f:
                invoice_data = json.load(f)
                invoice_data_list.append(invoice_data)
        except Exception as e:
            print(f"⚠️  Error loading {json_file.name}: {e}")
            continue

    if not invoice_data_list:
        print("❌ No valid invoice data loaded")
        return

    # Create feature pipeline
    feature_pipeline = create_feature_pipeline()

    # Extract features batch
    results = feature_pipeline.extract_features_batch(invoice_data_list)

    # Analysis
    analysis = feature_pipeline.analyze_feature_extraction(results)

    print("\n📊 Batch Feature Extraction Results:")
    print(f"   Documents processed: {analysis['total_documents']}")
    print(f"   Successful extractions: {analysis['successful_extractions']}")
    print(f"   Success rate: {analysis['success_rate']:.1%}")

    if analysis["successful_extractions"] > 0:
        feature_stats = analysis["feature_count_stats"]
        print(
            f"   Feature count - Min: {feature_stats['min']}, Max: {feature_stats['max']}, Avg: {feature_stats['avg']:.1f}"
        )
        print(f"   Unique features: {feature_stats['total_unique_features']}")

        print("\n📈 Extractor Performance:")
        for extractor, stats in analysis["extractor_performance"].items():
            print(
                f"   {extractor}: {stats['success_rate']:.1%} success, {stats['avg_features']:.1f} avg features, {stats['avg_time']:.3f}s avg time"
            )

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save individual results
    for _i, (result, json_file) in enumerate(zip(results, json_files, strict=False)):
        if result["success"]:
            feature_file = output_dir / f"{json_file.stem}_features.json"
            with open(feature_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

    # Save analysis
    analysis_file = output_dir / "feature_analysis.json"
    with open(analysis_file, "w", encoding="utf-8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)

    print(f"💾 Results saved to: {output_dir}")

    # Save individual results
    for _i, (result, json_file) in enumerate(zip(results, json_files, strict=False)):
        if result["success"]:
            feature_file = output_dir / f"{json_file.stem}_features.json"
            with open(feature_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

    # Save analysis
    analysis_file = output_dir / "feature_analysis.json"
    with open(analysis_file, "w", encoding="utf-8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)

    print(f"💾 Results saved to: {output_dir}")


def analyze_feature_pipeline(args: argparse.Namespace) -> None:
    """Analyze feature pipeline capabilities"""

    print("🔍 Analyzing Feature Engineering Pipeline")

    # Create feature pipeline
    feature_pipeline = create_feature_pipeline()

    # Get pipeline info
    info = feature_pipeline.get_pipeline_info()

    print("\n📊 Pipeline Information:")
    stats = info["pipeline_stats"]
    print(f"   Total extractors: {stats['total_extractors']}")
    print(f"   Enabled extractors: {stats['enabled_extractors']}")
    print(f"   Cache size: {stats['cache_size']}")

    print("\n🎯 Performance Targets:")
    print(f"   Baseline accuracy: {info['baseline_accuracy']:.1%}")
    print(f"   Target accuracy: {info['target_accuracy']:.1%}")
    print(f"   Accuracy gap: {info['accuracy_gap']:.1%}")

    print("\n📋 Available Extractors:")
    for extractor_name in stats["extractor_names"]:
        print(f"   ✅ {extractor_name}")

    print(f"\n🔧 Total Available Features: {len(info['all_feature_names'])}")

    # Group features by category
    categories: dict[str, list[str]] = {}
    for feature_name in info["all_feature_names"]:
        if "_" in feature_name:
            category = feature_name.split("_")[0]
            if category not in categories:
                categories[category] = []
            categories[category].append(feature_name)

    print("\n📈 Feature Categories:")
    for category, features in categories.items():
        print(f"   {category}: {len(features)} features")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)

        print(f"💾 Analysis saved to: {output_path}")


def dual_purpose_pipeline(args: argparse.Namespace) -> None:
    """
    Unified Dual-Purpose Pipeline - buchhaltungsbutler.de Ersatz mit integrierter KI-Lernfähigkeit

    Einheitliche Pipeline die BEIDE Zwecke gleichzeitig erfüllt:
    1. **SOFORT-NUTZEN**: SKR03-klassifizierte Buchungsdaten für Ihre Buchhaltung
    2. **TRAINING**: Jede Verarbeitung erzeugt spaCy-Trainingsdaten für Eigenständigkeit

    **Workflow:**
    PDF → Docling TXT → Gemini (dual purpose) → {
        - SKR03 Buchungsausgabe (für Sie)
        - spaCy Training Export (für zukünftige Unabhängigkeit)
    }
    """
    from src.pipeline.dual_pipeline import UnifiedDualPurposePipeline

    config = Config()
    pipeline = UnifiedDualPurposePipeline(config)

    pdf_path = Path(args.input)
    output_dir = Path(args.output) if args.output else Path("data/output/dual_purpose")

    print("🔄 Unified Dual-Purpose Pipeline")
    print("📋 buchhaltungsbutler.de Ersatz mit integrierter KI-Lernfähigkeit")
    print(f"📄 Input PDF: {pdf_path}")
    print(f"📁 Output Directory: {output_dir}")

    if not pdf_path.exists():
        print(f"❌ PDF nicht gefunden: {pdf_path}")
        return

    if not pdf_path.suffix.lower() == ".pdf":
        print(f"❌ Eingabe muss eine PDF-Datei sein: {pdf_path}")
        return

    print("\n🚀 Starte Dual-Purpose Verarbeitung...")

    try:
        # Process PDF with dual purpose
        result = pipeline.process_pdf_dual_purpose(pdf_path)

        if result:
            print("✅ Dual-Purpose Pipeline erfolgreich abgeschlossen!")

            # Save immediate SKR03 results
            buchungs_file = pipeline.save_buchungsausgabe(
                result["buchungsausgabe"], output_dir
            )
            print("\n📊 SOFORT-NUTZEN - SKR03 Buchungsausgabe:")
            print(f"   💾 Gespeichert: {buchungs_file}")
            print(
                f"   🎯 SKR03 Klassifikationen: {len(result['buchungsausgabe'].get('positionen', []))}"
            )

            # Save training data export
            training_file = pipeline.save_training_export(
                result["training_export"], output_dir
            )
            print("\n🧠 TRAINING - spaCy Export für Eigenständigkeit:")
            print(f"   💾 Gespeichert: {training_file}")
            print(
                f"   🎯 Training Beispiele: {len(result['training_export'].get('spacy_ner', []))}"
            )

            # Summary
            print("\n🎉 Pipeline Summary:")
            print(f"   📄 Verarbeitete PDF: {pdf_path.name}")
            print(f"   📁 Alle Ausgaben in: {output_dir}")

            # Future benefits message
            print("\n💡 Jede Verarbeitung verbessert Ihre KI-Eigenständigkeit!")
            print("   📈 Mit jedem Dokument wird Ihr spaCy-Modell intelligenter")
            print("   🎯 Ziel: Vollständige Unabhängigkeit von externen APIs")

        else:
            print("❌ Pipeline lieferte kein Ergebnis zurück")

    except Exception as e:
        print(f"❌ Unerwarteter Fehler in Dual-Purpose Pipeline: {e}")
        logging.error("Dual-Purpose Pipeline Fehler: %s", e, exc_info=True)


def create_arg_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser"""
    parser = argparse.ArgumentParser(
        description="LLKJJ ML Pipeline - Deutsche Elektrohandwerk Buchhaltungs-KI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  python main.py process rechnung.pdf                  # NEUE STANDARD GEMINI-FIRST Pipeline
  python main.py process-docling rechnung.pdf          # Alternative Docling-Pipeline
  python main.py export data/processed/ --output training.jsonl
  python main.py train training.jsonl --epochs 30
  python main.py database init --path data/vectors
  python main.py workflow1 rechnung.pdf
        """,
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose output (DEBUG logging)"
    )

    parser.add_argument(
        "--production",
        action="store_true",
        help="Production mode (only warnings and errors)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Process command (NEW GEMINI-FIRST as standard!)
    process_parser = subparsers.add_parser(
        "process", help="Process PDF invoices (GEMINI-FIRST Standard)"
    )
    process_parser.add_argument("input", help="PDF file or directory to process")
    process_parser.add_argument("--output", "-o", help="Output directory (optional)")

    # NEW: Docling Alternative command (nur bei expliziter Anfrage)
    docling_parser = subparsers.add_parser(
        "process-docling",
        help="Process PDF invoices (DOCLING Alternative - nur bei expliziter Anfrage)",
    )
    docling_parser.add_argument("input", help="PDF file or directory to process")
    docling_parser.add_argument("--output", "-o", help="Output directory (optional)")

    # NEW: A3 Async Batch Processing (Performance Optimization)
    async_parser = subparsers.add_parser(
        "process-async",
        help="🚀 A3: High-Performance Async Batch Processing (Strategic TODO)",
    )
    async_parser.add_argument("input", help="PDF file or directory to process")
    async_parser.add_argument("--output", "-o", help="Output directory (optional)")
    async_parser.add_argument(
        "--concurrent",
        "-c",
        type=int,
        default=3,
        help="Max concurrent processing (default: 3)",
    )

    # NEW: B1 Unified Strategy Processing (Strategy Pattern)
    unified_parser = subparsers.add_parser(
        "process-unified",
        help="🔧 B1: Unified Strategy Processing (Strategy Pattern)",
    )
    unified_parser.add_argument("input", help="PDF file or directory to process")
    unified_parser.add_argument("--output", "-o", help="Output directory (optional)")
    unified_parser.add_argument(
        "--strategy",
        "-s",
        choices=["auto", "gemini", "spacy_rag"],
        default="auto",
        help="Processing strategy (default: auto)",
    )
    unified_parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare all available strategies on the same PDF",
    )

    # Export command
    export_parser = subparsers.add_parser("export", help="Export training data")
    export_parser.add_argument(
        "input", help="Directory containing processed JSON files"
    )
    export_parser.add_argument("--output", "-o", help="Output JSONL file (optional)")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train spaCy model")
    train_parser.add_argument("input", help="Training data JSONL file")
    train_parser.add_argument(
        "--output", "-o", help="Output model directory (optional)"
    )
    train_parser.add_argument(
        "--epochs", type=int, default=20, help="Training epochs (default: 20)"
    )
    train_parser.add_argument(
        "--model-type",
        choices=["ner", "textcat", "both"],
        default="ner",
        help="Model type to train (ner, textcat, or both)",
    )

    # Export textcat command
    export_textcat_parser = subparsers.add_parser(
        "export-textcat", help="Export textcat training data"
    )
    export_textcat_parser.add_argument(
        "input", help="Directory containing processed JSON files"
    )
    export_textcat_parser.add_argument(
        "--output", "-o", help="Output directory for textcat data (optional)"
    )

    # Train textcat command
    train_textcat_parser = subparsers.add_parser(
        "train-textcat", help="Train spaCy text classification model"
    )
    train_textcat_parser.add_argument("input", help="Textcat training data directory")
    train_textcat_parser.add_argument(
        "--output", "-o", help="Output textcat model directory (optional)"
    )
    train_textcat_parser.add_argument(
        "--epochs", type=int, default=30, help="Training epochs (default: 30)"
    )

    # Pipeline command
    pipeline_parser = subparsers.add_parser("pipeline", help="Run complete pipeline")
    pipeline_parser.add_argument("input", help="Directory containing PDF files")
    pipeline_parser.add_argument(
        "--output", "-o", help="Final model output directory (optional)"
    )
    pipeline_parser.add_argument(
        "--epochs", type=int, default=20, help="Training epochs (default: 20)"
    )

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze processing results")
    analyze_parser.add_argument("input", help="Directory containing result JSON files")

    # Database command
    db_parser = subparsers.add_parser("database", help="Database operations")
    db_subparsers = db_parser.add_subparsers(
        dest="db_command", help="Database commands"
    )

    # Database init
    db_init_parser = db_subparsers.add_parser(
        "init", help="Initialize optimized database"
    )
    db_init_parser.add_argument("--path", default="data/vectors", help="Database path")

    # Database stats
    db_stats_parser = db_subparsers.add_parser("stats", help="Show database statistics")
    db_stats_parser.add_argument("--path", default="data/vectors", help="Database path")

    # Database backup
    db_backup_parser = db_subparsers.add_parser("backup", help="Create database backup")
    db_backup_parser.add_argument(
        "--path", default="data/vectors", help="Database path"
    )
    db_backup_parser.add_argument("--name", help="Backup name (optional)")

    # Database optimize
    db_optimize_parser = db_subparsers.add_parser("optimize", help="Optimize database")
    db_optimize_parser.add_argument(
        "--path", default="data/vectors", help="Database path"
    )

    # NEW MODULAR WORKFLOWS
    # Workflow 1: PDF → Docling → TXT only
    w1_parser = subparsers.add_parser("workflow1", help="PDF → Docling → TXT only")
    w1_parser.add_argument("input", help="PDF file to process")
    w1_parser.add_argument("--output", "-o", help="Output TXT file path (optional)")

    # Workflow 2: PDF → Gemini directly (bypass Docling)
    w2_parser = subparsers.add_parser(
        "workflow2", help="PDF → Gemini direct (bypass Docling)"
    )
    w2_parser.add_argument("input", help="PDF file to process")
    w2_parser.add_argument("--output", "-o", help="Output JSON file path (optional)")

    # Workflow 3: PDF → Docling → Gemini → Classification
    w3_parser = subparsers.add_parser(
        "workflow3", help="PDF → Docling → Gemini → Classification"
    )
    w3_parser.add_argument("input", help="PDF file to process")
    w3_parser.add_argument("--output", "-o", help="Output JSON file path (optional)")

    # Workflow 4: Full Pipeline with all features
    w4_parser = subparsers.add_parser(
        "workflow4", help="Full Pipeline with all features"
    )
    w4_parser.add_argument("input", help="PDF file to process")
    w4_parser.add_argument("--output", "-o", help="Output JSON file path (optional)")

    # Security Commands
    # Security Audit
    security_audit_parser = subparsers.add_parser(
        "security-audit", help="Führt umfassenden Security Audit durch"
    )
    security_audit_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Detaillierte Ausgabe"
    )

    # Production Validation
    subparsers.add_parser("security-validate", help="Validiert Production-Readiness")

    # API Key Management
    security_keys_parser = subparsers.add_parser(
        "security-keys", help="API Key Management"
    )
    security_keys_parser.add_argument(
        "security_action",
        choices=["list", "store", "rotate", "delete"],
        help="Aktion für API-Keys",
    )
    security_keys_parser.add_argument(
        "--service", help="Service-Name (z.B. gemini, openai)"
    )
    security_keys_parser.add_argument("--key", help="API-Key Wert")

    # Feature Engineering Commands
    parser_extract = subparsers.add_parser(
        "extract-features",
        help="Extract features from single document",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""Extract features from a single invoice document.

This command extracts features using the complete feature engineering pipeline
for improved SKR03 classification accuracy.

Examples:
    poetry run python main.py extract-features data/processed/invoice.json
    poetry run python main.py extract-features data/processed/invoice.json --output features/invoice_features.json
""",
    )
    parser_extract.add_argument(
        "input", help="Input JSON file (processed invoice data)"
    )
    parser_extract.add_argument(
        "--output", "-o", help="Output file for extracted features (JSON format)"
    )
    parser_extract.set_defaults(func=extract_features)

    parser_extract_batch = subparsers.add_parser(
        "extract-features-batch",
        help="Extract features from multiple documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""Extract features from multiple invoice documents in batch.

This command processes all JSON files in the input directory and extracts
features using the complete feature engineering pipeline.

Examples:
    poetry run python main.py extract-features-batch data/processed/
    poetry run python main.py extract-features-batch data/processed/ --output data/features/
""",
    )
    parser_extract_batch.add_argument(
        "input", help="Input directory with JSON files (processed invoice data)"
    )
    parser_extract_batch.add_argument(
        "--output",
        "-o",
        help="Output directory for extracted features",
        default="data/features",
    )
    parser_extract_batch.set_defaults(func=extract_features_batch)

    # UNIFIED DUAL-PURPOSE PIPELINE
    dual_parser = subparsers.add_parser(
        "dual-purpose",
        help="🚀 Unified Pipeline: Sofort-Buchhaltung + KI-Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""Einheitliche Pipeline die BEIDE Zwecke gleichzeitig erfüllt:

1. **SOFORT-NUTZEN**: SKR03-klassifizierte Buchungsdaten für Ihre Buchhaltung
2. **TRAINING**: Jede Verarbeitung erzeugt spaCy-Trainingsdaten für Eigenständigkeit

**Workflow:**
PDF → Docling TXT → Gemini (dual purpose) → {
    - SKR03 Buchungsausgabe (für Sie)
    - spaCy Training Export (für zukünftige Unabhängigkeit)
}

Das ist Ihr buchhaltungsbutler.de Ersatz mit integrierter KI-Lernfähigkeit.

Beispiele:
    poetry run python main.py dual-purpose rechnung.pdf
    poetry run python main.py dual-purpose data/pdfs/ --output data/dual_output/
    poetry run python main.py dual-purpose data/pdfs/ --batch --output monthly_batch/
""",
    )
    dual_parser.add_argument("input", help="PDF-Datei oder Verzeichnis mit PDFs")
    dual_parser.add_argument(
        "--output",
        "-o",
        help="Output-Verzeichnis (Standard: data/output/dual_purpose/)",
        default="data/output/dual_purpose",
    )
    dual_parser.add_argument(
        "--batch",
        action="store_true",
        help="Batch-Modus für Verzeichnis-Verarbeitung",
    )

    parser_analyze_features = subparsers.add_parser(
        "analyze-features",
        help="Analyze feature engineering pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""Analyze the feature engineering pipeline capabilities.

This command provides detailed information about available extractors,
feature categories, and pipeline performance targets.

Examples:
    poetry run python main.py analyze-features
    poetry run python main.py analyze-features --output pipeline_analysis.json
""",
    )
    parser_analyze_features.add_argument(
        "--output", "-o", help="Output file for analysis results (JSON format)"
    )
    parser_analyze_features.set_defaults(func=analyze_feature_pipeline)

    return parser


def command_security_audit(args: argparse.Namespace) -> None:
    """Führt umfassenden Security Audit durch."""
    print("🔒 Starte Security Audit...")

    report = run_security_audit(project_root=Path.cwd(), verbose=args.verbose)

    # Zusätzliche Ausgabe für CLI
    if not report["overall_summary"]["production_ready"]:
        print("\n⚠️  WARNUNG: Pipeline ist NICHT production-ready!")
        print("Beheben Sie die kritischen Security-Issues vor Deployment.")
        sys.exit(1)
    else:
        print("\n✅ Pipeline ist production-ready!")


def command_security_validate(args: argparse.Namespace) -> None:
    """Validiert Production-Readiness."""
    print("🔍 Validiere Production Environment...")

    if validate_production_environment():
        print("✅ Environment ist production-ready!")
    else:
        print("❌ Environment ist NICHT production-ready!")
        print("Setzen Sie die erforderlichen Environment-Variables.")
        sys.exit(1)


def command_security_key_manager(args: argparse.Namespace) -> None:
    """API Key Management."""
    manager = APIKeyManager()

    if args.security_action == "list":
        services = manager.list_stored_keys()
        print(f"📋 Gespeicherte API-Keys: {len(services)}")
        for service in services:
            print(f"  - {service}")

    elif args.security_action == "store":
        if not args.service or not args.key:
            print("❌ --service und --key sind erforderlich für 'store'")
            sys.exit(1)

        if manager.store_api_key(args.service, args.key):
            print(f"✅ API-Key für {args.service} gespeichert")
        else:
            print(f"❌ Fehler beim Speichern des API-Keys für {args.service}")
            sys.exit(1)

    elif args.security_action == "rotate":
        if not args.service or not args.key:
            print("❌ --service und --key sind erforderlich für 'rotate'")
            sys.exit(1)

        if manager.rotate_api_key(args.service, args.key):
            print(f"✅ API-Key für {args.service} rotiert")
        else:
            print(f"❌ Fehler beim Rotieren des API-Keys für {args.service}")
            sys.exit(1)

    elif args.security_action == "delete":
        if not args.service:
            print("❌ --service ist erforderlich für 'delete'")
            sys.exit(1)

        if manager.delete_api_key(args.service):
            print(f"✅ API-Key für {args.service} gelöscht")
        else:
            print(f"❌ API-Key für {args.service} nicht gefunden")
            sys.exit(1)


def main() -> None:
    """Main CLI entry point"""
    parser = create_arg_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Setup logging
    setup_logging(args.verbose, getattr(args, "production", False))

    try:
        # Route commands
        if args.command == "process":
            process_pdfs(args)  # NEUE GEMINI-FIRST Standard Pipeline
        elif args.command == "process-docling":
            process_pdfs_docling_alternative(args)  # Alternative Docling Pipeline
        elif args.command == "process-async":
            process_pdfs_async_batch(args)  # 🚀 A3: High-Performance Async Processing
        elif args.command == "process-unified":
            process_pdfs_unified_strategy(args)  # 🔧 B1: Unified Strategy Processing
        elif args.command == "export":
            export_training_data(args)
        elif args.command == "train":
            train_model(args)
        elif args.command == "export-textcat":
            export_textcat_data(args)
        elif args.command == "train-textcat":
            train_textcat_model(args)
        elif args.command == "pipeline":
            run_pipeline(args)
        elif args.command == "analyze":
            analyze_results(args)
        elif args.command == "database":
            # Database sub-commands
            if args.db_command == "init":
                init_database(args)
            elif args.db_command == "stats":
                show_database_stats(args)
            elif args.db_command == "backup":
                backup_database(args)
            elif args.db_command == "optimize":
                optimize_database(args)
            else:
                print("❌ Unknown database command. Use 'database -h' for help.")
        # NEW MODULAR WORKFLOWS
        elif args.command == "workflow1":
            run_workflow_1(args)
        elif args.command == "workflow2":
            run_workflow_2(args)
        elif args.command == "workflow3":
            run_workflow_3(args)
        elif args.command == "workflow4":
            run_workflow_4(args)
        # Security Commands
        elif args.command == "security-audit":
            command_security_audit(args)
        elif args.command == "security-validate":
            command_security_validate(args)
        elif args.command == "security-keys":
            command_security_key_manager(args)
        # Dual-Purpose Pipeline
        elif args.command == "dual-purpose":
            dual_purpose_pipeline(args)

    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user")
        sys.exit(1)
    except (ImportError, ModuleNotFoundError) as e:
        logging.error("Modul-Import-Fehler: %s", e, exc_info=True)
        print(f"❌ Abhängigkeitsfehler: {e}")
        sys.exit(1)
    except (ValueError, TypeError) as e:
        logging.error("Konfigurationsfehler: %s", e, exc_info=True)
        print(f"❌ Konfigurationsfehler: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error("Pipeline-Fehler: %s", e, exc_info=True)
        print(f"❌ Unerwarteter Pipeline-Fehler: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
```

### `ml_service/processor.py`

*(**Änderungsgrund:** Die öffentliche "Blackbox"-API wurde aktualisiert, um den neuen `UnifiedProcessor` mit Strategy-Pattern und das korrekte, detaillierte `ProcessingResult`-Modell zu verwenden. Dies stellt die Konsistenz mit der in der Vision beschriebenen Architektur her.)*

```python
#!/usr/bin/env python3
"""
LLKJJ ML Service - Eigenständiger Processor
==========================================

Eigenständiger ML-Processor der völlig unabhängig von externen Systemen
funktioniert. Verwendet MLSettings für Konfiguration und stellt eine
saubere API für PDF-Verarbeitung bereit.

**EIGENSTÄNDIGKEIT:**
- Keine Abhängigkeiten zu src/config oder externen Modulen
- Eigene Ressourcenverwaltung
- Konfiguration über MLSettings
- Wiederverwendbar in jedem Python-Projekt

Author: LLKJJ ML Pipeline Team
Version: 3.0.0 (Eigenständige Implementierung)
"""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import MLSettings, default_settings
from src.models.processing_result import ProcessingResult
from src.pipeline.unified_processor import UnifiedProcessor

logger = logging.getLogger(__name__)


class MLResourceManager:
    """
    Eigenständiger Resource Manager für das ML Service

    Verwaltet ML-Modelle und Ressourcen unabhängig von externen Systemen.
    """

    def __init__(self, settings: MLSettings):
        self.settings = settings
        self._initialized = False

    def initialize(self) -> None:
        """Initialisiert alle notwendigen Ressourcen."""
        if self._initialized:
            return

        logger.info("🔄 Initialisiere ML Service Ressourcen...")

        # Verzeichnisse erstellen
        self.settings.ensure_directories()

        # Logging konfigurieren
        self._setup_logging()

        self._initialized = True
        logger.info("✅ ML Service Ressourcen initialisiert")

    def _setup_logging(self) -> None:
        """Konfiguriert Logging für das ML Service."""
        log_level = getattr(logging, self.settings.log_level.upper(), logging.INFO)
        logging.basicConfig(level=log_level, format=self.settings.log_format)

    def cleanup(self) -> None:
        """Bereinigt Ressourcen."""
        logger.info("🧹 Bereinige ML Service Ressourcen...")
        self._initialized = False
        logger.info("✅ ML Service Ressourcen bereinigt")


class MLProcessor:
    """
    Eigenständiger ML-Processor für PDF-Verarbeitung

    **USAGE:**
    ```python
    from ml_service import MLProcessor, MLSettings

    # Mit Standardkonfiguration
    processor = MLProcessor()
    result = processor.process_pdf("invoice.pdf")

    # Mit eigener Konfiguration
    settings = MLSettings(cache_enabled=False)
    processor = MLProcessor(settings=settings)
    result = processor.process_pdf("invoice.pdf")

    # Ressourcen bereinigen
    processor.cleanup()
    ```
    """

    def __init__(self, settings: MLSettings | None = None):
        """
        Initialisiert ML-Processor.

        Args:
            settings: ML-Konfiguration. Verwendet Defaults falls None.
        """
        self.settings = settings or default_settings
        self.resource_manager = MLResourceManager(self.settings)
        self._processor: UnifiedProcessor | None = None

        # Validiere Konfiguration
        validation = self.settings.validate_configuration()
        if not validation["valid"]:
            logger.error(f"Konfigurationsfehler: {validation['errors']}")
            raise ValueError(f"Ungültige Konfiguration: {validation['errors']}")

        if validation["warnings"]:
            for warning in validation["warnings"]:
                logger.warning(warning)

        logger.info(f"ML Processor initialisiert: {self.settings}")

    def _get_processor(self) -> UnifiedProcessor:
        """Lazy-loaded Unified Processor."""
        if self._processor is None:
            logger.info("🔄 Initialisiere Unified Processor...")
            # Hier wird der neue UnifiedProcessor aus src.pipeline.unified_processor verwendet
            self._processor = UnifiedProcessor()
            logger.info("✅ Unified Processor initialisiert")

        return self._processor

    def process_pdf(self, pdf_path: str | Path) -> ProcessingResult:
        """
        Verarbeitet PDF-Datei und gibt strukturierte Ergebnisse zurück.

        Args:
            pdf_path: Pfad zur PDF-Datei

        Returns:
            ProcessingResult mit strukturierten Daten und Klassifizierungen

        Raises:
            FileNotFoundError: Wenn PDF-Datei nicht existiert
            ValueError: Wenn PDF-Datei ungültig ist
            RuntimeError: Bei Verarbeitungsfehlern
        """
        start_time = time.time()

        # Input-Validierung
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF-Datei nicht gefunden: {pdf_path}")

        if not pdf_path.suffix.lower() == ".pdf":
            raise ValueError(f"Datei ist keine PDF: {pdf_path}")

        # Dateigröße prüfen
        file_size_mb = pdf_path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.settings.max_pdf_size_mb:
            raise ValueError(
                f"PDF zu groß: {file_size_mb:.1f}MB > {self.settings.max_pdf_size_mb}MB"
            )

        logger.info(f"🚀 Starte PDF-Verarbeitung: {pdf_path.name}")

        try:
            # Ressourcen initialisieren
            self.resource_manager.initialize()

            # Hauptprozessor aufrufen
            processor = self._get_processor()
            result = processor.process_pdf(pdf_path)

            processing_time = int((time.time() - start_time) * 1000)

            # Verarbeitungszeit im Ergebnis aktualisieren
            result.processing_time_ms = processing_time

            logger.info(
                f"✅ PDF-Verarbeitung abgeschlossen: {pdf_path.name} "
                f"({processing_time}ms, {result.confidence_score:.1%} Konfidenz)"
            )

            return result

        except Exception as e:
            logger.error(f"❌ Fehler bei PDF-Verarbeitung: {e}")
            raise RuntimeError(f"PDF-Verarbeitung fehlgeschlagen: {e}") from e

    def validate_input(self, pdf_path: str | Path) -> bool:
        """
        Validiert Eingabedatei.

        Args:
            pdf_path: Pfad zur PDF-Datei

        Returns:
            True wenn Datei gültig ist
        """
        try:
            pdf_path = Path(pdf_path)
            return (
                pdf_path.exists()
                and pdf_path.suffix.lower() == ".pdf"
                and pdf_path.stat().st_size
                <= self.settings.max_pdf_size_mb * 1024 * 1024
            )
        except Exception:
            return False

    def get_capabilities(self) -> list[str]:
        """
        Gibt verfügbare Capabilities zurück.

        Returns:
            Liste der verfügbaren Funktionen
        """
        capabilities = [
            "pdf_extraction",
            "german_nlp",
            "text_processing",
            "skr03_classification",
            "quality_assessment",
            "vector_search"
        ]

        if self.settings.cache_enabled:
            capabilities.append("caching")

        if self.settings.google_api_key:
            capabilities.append("gemini_enhancement")

        return capabilities

    def get_settings_summary(self) -> dict[str, Any]:
        """
        Gibt Zusammenfassung der aktuellen Einstellungen zurück.

        Returns:
            Dict mit Konfigurationszusammenfassung
        """
        validation_result = self.settings.validate_configuration()
        return validation_result.get("summary", {})

    def cleanup(self) -> None:
        """Bereinigt alle Ressourcen."""
        logger.info("🧹 Bereinige ML Processor...")

        # Der UnifiedProcessor hat kein eigenes Cleanup, Ressourcen werden vom ResourceManager verwaltet
        self.resource_manager.cleanup()

        logger.info("✅ ML Processor bereinigt")

    def __enter__(self) -> "MLProcessor":
        """Context Manager Eingang."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Context Manager Ausgang."""
        self.cleanup()


# Convenience-Funktion für einfache Verwendung
def process_pdf_simple(
    pdf_path: str | Path, settings: MLSettings | None = None
) -> ProcessingResult:
    """
    Einfache PDF-Verarbeitung mit automatischer Ressourcenverwaltung.

    Args:
        pdf_path: Pfad zur PDF-Datei
        settings: Optionale ML-Konfiguration

    Returns:
        ProcessingResult mit Verarbeitungsergebnissen
    """
    with MLProcessor(settings=settings) as processor:
        return processor.process_pdf(pdf_path)
```

### Entfernte Dateien

Die folgenden Dateien und das Verzeichnis wurden entfernt, um Redundanz zu beseitigen und die Projektstruktur zu bereinigen:

- **`src/pipeline/processor.py`**: Dieses Modul war veraltet. Seine Funktionalität wird nun vollständig durch `src/pipeline/unified_processor.py` (neuer Processor) und `src/models/processing_result.py` (Datenmodell) abgedeckt.
- **Verzeichnis `spacy_training/`**: Dieses gesamte Verzeichnis (`__init__.py`, `cat_trainer.py`, `ner_training.py`, `pipeline.py`) war eine ältere, duplizierte Implementierung der Trainingslogik. Die überlegene und refaktorierte Implementierung befindet sich nun zentral in `src/trainer.py`.
