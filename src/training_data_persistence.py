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
            self.collection = self.chroma_client.get_or_create_collection(
                name="invoice_classifications",
                metadata={
                    "description": "Gemini-validierte Rechnungsklassifizierungen"
                },
            )
            logger.info("✅ ChromaDB für Trainingsdaten-RAG initialisiert")
        except Exception as e:
            logger.warning(f"⚠️ ChromaDB Setup fehlgeschlagen: {e}")
            # Die Attribute wurden bereits bei der erfolgreichen Initialisierung gesetzt
            # Keine erneute Definition erforderlich

    def _setup_embedding_model(self) -> None:
        """Setup Embedding-Modell für RAG-System"""
        try:
            self.embedding_model = SentenceTransformer(
                "sentence-transformers/distiluse-base-multilingual-cased"
            )
            logger.info("✅ Embedding-Modell für Trainingsdaten geladen")
        except Exception as e:
            logger.warning(f"⚠️ Embedding-Modell Setup fehlgeschlagen: {e}")
            self.embedding_model = None  # type: ignore[assignment]

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
                pdf_path, validated_gemini_result, processing_result
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
        processing_result: Any,
    ) -> dict[str, Any]:
        """
        Speichere spaCy-Trainingsdaten im JSONL-Format.

        Format: {"text": "...", "entities": [...], "cats": {...}}
        """
        try:
            # Text aus Gemini-Result extrahieren
            raw_text = self._extract_training_text(gemini_result)

            # Entitäten aus Rechnungsdaten extrahieren
            entities = self._extract_entities_for_training(gemini_result)

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

    def _extract_training_text(self, gemini_result: dict[str, Any]) -> str:
        """Extrahiere Volltext für spaCy-Training"""
        text_parts: list[str] = []

        # Header-Informationen
        header = gemini_result.get("invoice_header", {})
        if header.get("lieferant"):
            text_parts.append(f"Lieferant: {header['lieferant']}")
        if header.get("rechnungsnummer"):
            text_parts.append(f"Rechnung: {header['rechnungsnummer']}")

        # Line Items
        for item in gemini_result.get("line_items", []):
            if item.get("beschreibung"):
                text_parts.append(item["beschreibung"])

        return "\n".join(text_parts)

    def _extract_entities_for_training(
        self, gemini_result: dict[str, Any]
    ) -> list[tuple[int, int, str]]:
        """Extrahiere NER-Entitäten für spaCy-Training"""
        entities = []
        text = self._extract_training_text(gemini_result)

        # Einfache Entitäts-Extraktion basierend auf bekannten Mustern
        # (In Produktion würde hier eine sophisticated NER-Pipeline stehen)

        header = gemini_result.get("invoice_header", {})
        line_items = gemini_result.get("line_items", [])

        current_pos = 0

        # Lieferant
        if header.get("lieferant"):
            lieferant = header["lieferant"]
            start_pos = text.find(lieferant, current_pos)
            if start_pos != -1:
                entities.append((start_pos, start_pos + len(lieferant), "LIEFERANT"))
                current_pos = start_pos + len(lieferant)

        # Artikel-Beschreibungen
        for item in line_items:
            if item.get("beschreibung"):
                beschreibung = item["beschreibung"]
                start_pos = text.find(beschreibung, current_pos)
                if start_pos != -1:
                    entities.append(
                        (start_pos, start_pos + len(beschreibung), "ARTIKEL")
                    )
                    current_pos = start_pos + len(beschreibung)

        return entities

    def _extract_categories_for_training(
        self, gemini_result: dict[str, Any]
    ) -> dict[str, float]:
        """Extrahiere Kategorien für TextCat-Training"""
        categories = {}

        # Elektro-Kategorien aus Line Items
        elektro_kategorien = set()
        for item in gemini_result.get("line_items", []):
            if item.get("elektro_kategorie"):
                elektro_kategorien.add(item["elektro_kategorie"])

        # Kategorien mit Konfidenz (1.0 für bekannte, 0.0 für unbekannte)
        for kategorie in elektro_kategorien:
            categories[f"ELEKTRO_{kategorie.upper()}"] = 1.0

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
                beschreibung = classification.get("beschreibung", "")
                menge = classification.get("menge", 0)
                einzelpreis = classification.get("einzelpreis", 0)

                # Kontextueller Dokument-Text (wie in RAG-Strategie beschrieben)
                doc_text = (
                    f"Artikel: {beschreibung}, "
                    f"Menge: {menge}, "
                    f"Preis: {einzelpreis:.2f} EUR. "
                    f"Lieferant: {lieferant} (Elektro-Großhandel). "
                    f"Kategorie: {classification.get('elektro_kategorie', 'Unbekannt')}."
                )

                # Embedding erstellen
                embedding = self.embedding_model.encode(doc_text).tolist()  # type: ignore[misc]

                # Metadaten mit validation_status (RAG-Strategie)
                metadata = {
                    "pdf_path": pdf_path,
                    "supplier": lieferant,
                    "skr03_account": classification.get("predicted_account", ""),
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
                self.collection.add(  # type: ignore[misc]
                    documents=documents,
                    metadatas=metadatas,  # type: ignore[arg-type]
                    embeddings=embeddings,  # type: ignore[arg-type]
                    ids=ids,
                )

                logger.info(
                    f"💾 RAG-Dokumente gespeichert: {len(documents)} Klassifizierungen"
                )

            return {
                "document_count": len(documents),
                "collection_size": self.collection.count() if self.collection else 0,
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
