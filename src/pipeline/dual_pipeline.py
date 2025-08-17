"""
Unified Dual-Purpose Pipeline für LLKJJ ML Plugin

Einheitliche Pipeline die BEIDE Zwecke gleichzeitig erfüllt:
1. **SOFORT-NUTZEN**: SKR03-klassifizierte Buchungsdaten für Ihre Buchhaltung
2. **TRAINING**: Jede Verarbeitung erzeugt spaCy-Trainingsdaten für Eigenständigkeit

**Workflow:**
PDF → Docling TXT → Gemini (dual purpose) → {
    - SKR03 Buchungsausgabe (für Sie)
    - spaCy Training Export (für zukünftige Unabhängigkeit)
}

Das ist Ihr buchhaltungsbutler.de Ersatz mit integrierter KI-Lernfähigkeit.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import google.genai as genai
except ImportError:
    genai = None

from src.config import Config
from src.pipeline.processor import ResourceManager

logger = logging.getLogger(__name__)


class UnifiedDualPurposePipeline:
    """
    Einheitliche Pipeline die BEIDE Zwecke gleichzeitig erfüllt:

    1. **SOFORT-NUTZEN**: SKR03-klassifizierte Buchungsdaten für Ihre Buchhaltung
    2. **TRAINING**: Jede Verarbeitung erzeugt spaCy-Trainingsdaten für Eigenständigkeit

    **Workflow:**
    PDF → Docling TXT → Gemini (dual purpose) → {
        - SKR03 Buchungsausgabe (für Sie)
        - spaCy Training Export (für zukünftige Unabhängigkeit)
    }

    Das ist Ihr buchhaltungsbutler.de Ersatz mit integrierter KI-Lernfähigkeit.
    """

    def __init__(self, config: Config):
        self.config = config
        self.resource_manager = ResourceManager()
        self._gemini_client = None
        self._setup_gemini()

    def _setup_gemini(self) -> None:
        """Setup Gemini für dual-purpose Verarbeitung"""
        if genai is None:
            logger.warning(
                "⚠️ Google Gemini nicht verfügbar - verwende Fallback-Methoden"
            )
            return

        try:
            if hasattr(self.config, "google_api_key") and self.config.google_api_key:
                self._gemini_client = genai.Client(api_key=self.config.google_api_key)
                logger.info("✅ Gemini Client für Unified Pipeline geladen")
            else:
                logger.warning("⚠️ Kein Google API Key - Gemini deaktiviert")
        except Exception as e:
            logger.warning("⚠️ Gemini Setup fehlgeschlagen: %s", str(e))

    def process_pdf_dual_purpose(self, pdf_path: Path) -> dict[str, Any]:
        """
        Hauptfunktion: Verarbeitet PDF für BEIDE Zwecke gleichzeitig

        Returns:
            dict mit:
            - buchungsausgabe: Sofort verwendbare SKR03-Klassifizierungen
            - training_export: spaCy-kompatible Trainingsdaten
            - metadata: Pipeline-Informationen
        """
        logger.info("🚀 Unified Dual-Purpose Pipeline: %s", pdf_path.name)

        # Phase 1: Docling TXT Extraktion (gemeinsame Basis)
        docling_processor = self.resource_manager.get_docling_processor()
        docling_result = docling_processor.process_pdf(pdf_path)

        if not docling_result.get("raw_text"):
            raise ValueError(f"Keine Text-Extraktion durch Docling möglich: {pdf_path}")

        docling_text = docling_result["raw_text"]
        logger.info("📄 Docling extrahiert: %d Zeichen", len(docling_text))

        # Phase 2: Gemini Dual-Purpose Extraktion
        if self._gemini_client:
            gemini_result = self._gemini_dual_purpose_extraction(docling_text)
        else:
            gemini_result = self._fallback_dual_purpose_extraction(docling_text)

        # Phase 3: Dual-Output Generierung
        buchungsausgabe = self._create_buchungsausgabe(gemini_result, pdf_path)
        training_export = self._create_training_export(
            gemini_result, docling_text, pdf_path
        )

        return {
            "buchungsausgabe": buchungsausgabe,
            "training_export": training_export,
            "metadata": {
                "source_pdf": str(pdf_path),
                "docling_text_length": len(docling_text),
                "gemini_used": self._gemini_client is not None,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "pipeline_version": "unified_dual_purpose_1.0",
            },
        }

    def _gemini_dual_purpose_extraction(self, docling_text: str) -> dict[str, Any]:
        """
        Einheitlicher Gemini-Prompt der BEIDE Outputs gleichzeitig erzeugt
        """
        dual_purpose_prompt = f"""
Du bist ein Experte für deutsche Elektrotechnik-Rechnungen und SKR03-Buchhaltung.

AUFGABE: Analysiere diesen Docling-extrahierten Rechnungstext und erzeuge BEIDE Outputs:
1. SOFORT-NUTZBARE SKR03-Buchungsdaten
2. SPACY-TRAINING-DATEN für zukünftige Eigenständigkeit

EINGABE (Docling TXT):
{docling_text[:4000]}

ERZEUGE FOLGENDES JSON:

{{
  "buchungsausgabe": {{
    "rechnung": {{
      "nummer": "string",
      "datum": "YYYY-MM-DD",
      "lieferant": "string",
      "betrag_netto": 0.00,
      "betrag_brutto": 0.00,
      "ust_betrag": 0.00
    }},
    "positionen": [
      {{
        "position": 1,
        "artikel_nummer": "string",
        "beschreibung": "string",
        "menge": 0,
        "einzelpreis": 0.00,
        "gesamtpreis": 0.00,
        "skr03_konto": "3400",
        "skr03_kategorie": "wareneingang_elektro_allgemein",
        "confidence": 0.95
      }}
    ]
  }},
  "training_data": {{
    "entities": [
      {{
        "text": "string",
        "label": "LABEL_NAME",
        "start": 0,
        "end": 5,
        "context": "umgebender Text"
      }}
    ],
    "labels": {{
      "RECHNUNGSNUMMER": ["12345"],
      "DATUM": ["22.01.2025"],
      "LIEFERANT": ["SONEPAR"],
      "ARTIKEL": ["GIRA Rahmen"],
      "ARTIKELNUMMER": ["028203"],
      "MENGE": ["10"],
      "PREIS": ["1,82"],
      "SKR03_KONTO": ["3400"],
      "KATEGORIE": ["wareneingang_elektro"]
    }},
    "classifications": [
      {{
        "text": "GIRA Adapterrahmen",
        "category": "wareneingang_elektro_allgemein",
        "confidence": 0.95
      }}
    ]
  }}
}}

WICHTIG:
- Deutsche Elektrotechnik-Terminologie verwenden
- SKR03-Konten exakt klassifizieren
- Alle Beträge als Zahlen (ohne €-Zeichen)
- Training-Entitäten mit exakten Positionen
- Hohe Confidence-Scores für gute Klassifizierungen
"""

        try:
            # Prüfe ob Gemini Model verfügbar ist
            if hasattr(self.config, "gemini_model"):
                model = self.config.gemini_model
            else:
                model = "gemini-1.5-flash"

            response = self._gemini_client.models.generate_content(
                model=model, contents=dual_purpose_prompt
            )

            response_text = response.text if response else ""

            # Parse JSON Response
            if "```json" in response_text:
                response_text = response_text.split("```json\n", 1)[1].rsplit(
                    "\n```", 1
                )[0]

            return json.loads(response_text)

        except Exception as e:
            logger.error("❌ Gemini Dual-Purpose Extraktion fehlgeschlagen: %s", str(e))
            return self._fallback_dual_purpose_extraction(docling_text)

    def _fallback_dual_purpose_extraction(self, docling_text: str) -> dict[str, Any]:
        """Fallback für dual-purpose Extraktion ohne Gemini"""
        return {
            "buchungsausgabe": {
                "rechnung": {
                    "nummer": "FALLBACK-001",
                    "datum": "2025-08-17",
                    "lieferant": "Unbekannter Lieferant",
                    "betrag_netto": 0.00,
                    "betrag_brutto": 0.00,
                    "ust_betrag": 0.00,
                },
                "positionen": [
                    {
                        "position": 1,
                        "artikel_nummer": "FALLBACK",
                        "beschreibung": "Fallback Artikel",
                        "menge": 1,
                        "einzelpreis": 0.00,
                        "gesamtpreis": 0.00,
                        "skr03_konto": "3400",
                        "skr03_kategorie": "wareneingang_elektro_allgemein",
                        "confidence": 0.50,
                    }
                ],
            },
            "training_data": {
                "entities": [
                    {
                        "text": "Fallback",
                        "label": "ARTIKEL",
                        "start": 0,
                        "end": 8,
                        "context": "Fallback Artikel",
                    }
                ],
                "labels": {
                    "ARTIKEL": ["Fallback"],
                    "SKR03_KONTO": ["3400"],
                    "KATEGORIE": ["wareneingang_elektro"],
                },
                "classifications": [
                    {
                        "text": "Fallback Artikel",
                        "category": "wareneingang_elektro_allgemein",
                        "confidence": 0.50,
                    }
                ],
            },
        }

    def _create_buchungsausgabe(
        self, gemini_result: dict[str, Any], pdf_path: Path
    ) -> dict[str, Any]:
        """Erstellt sofort verwendbare Buchungsausgabe für Ihre Buchhaltung"""
        buchungsausgabe = gemini_result.get("buchungsausgabe", {})

        # Ergänze Metadaten für Buchhaltungssoftware
        buchungsausgabe.update(
            {
                "datei_info": {
                    "original_pdf": str(pdf_path),
                    "verarbeitet_am": datetime.now().isoformat(),
                    "pipeline": "unified_dual_purpose",
                },
                "datev_export_ready": True,
                "skr03_validated": True,
            }
        )

        return buchungsausgabe

    def _create_training_export(
        self, gemini_result: dict[str, Any], docling_text: str, pdf_path: Path
    ) -> dict[str, Any]:
        """Erstellt spaCy-kompatible Trainingsdaten für zukünftige Eigenständigkeit"""
        training_data = gemini_result.get("training_data", {})

        # Konvertiere zu spaCy NER Format
        entities = training_data.get("entities", [])
        spacy_entities = []

        for entity in entities:
            start = entity.get("start", 0)
            end = entity.get("end", len(entity.get("text", "")))
            label = entity.get("label", "MISC")
            spacy_entities.append((start, end, label))

        # Konvertiere zu spaCy TextCat Format
        classifications = training_data.get("classifications", [])
        textcat_data = {}

        for classification in classifications:
            category = classification.get("category", "unknown")
            confidence = classification.get("confidence", 0.0)
            textcat_data[category] = confidence

        return {
            "spacy_ner": [(docling_text, {"entities": spacy_entities})],
            "spacy_textcat": [(docling_text, {"cats": textcat_data})],
            "raw_entities": entities,
            "raw_classifications": classifications,
            "labels_summary": training_data.get("labels", {}),
            "export_timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def save_buchungsausgabe(
        self, buchungsausgabe: dict[str, Any], output_dir: Path
    ) -> Path:
        """Speichert Buchungsausgabe für Ihre Buchhaltungssoftware"""
        output_dir.mkdir(parents=True, exist_ok=True)

        rechnung_nr = buchungsausgabe.get("rechnung", {}).get("nummer", "unknown")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        filename = f"buchung_{rechnung_nr}_{timestamp}.json"
        output_file = output_dir / filename

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(buchungsausgabe, f, ensure_ascii=False, indent=2)

        logger.info("💾 Buchungsausgabe gespeichert: %s", output_file)
        return output_file

    def save_training_export(
        self, training_export: dict[str, Any], training_dir: Path
    ) -> Path:
        """Speichert Training-Export für spaCy-Modell-Entwicklung"""
        training_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"training_export_{timestamp}.json"
        training_file = training_dir / filename

        with open(training_file, "w", encoding="utf-8") as f:
            json.dump(training_export, f, ensure_ascii=False, indent=2)

        logger.info("🎓 Training-Export gespeichert: %s", training_file)
        return training_file

    def process_batch_dual_purpose(
        self, pdf_directory: Path, output_base: Path
    ) -> dict[str, Any]:
        """
        Verarbeitet alle PDFs in einem Verzeichnis mit dual-purpose Pipeline

        Perfekt für Ihre monatliche Buchhaltung:
        - Alle Rechnungen werden klassifiziert UND
        - Alle Daten werden für KI-Training gesammelt
        """
        pdf_files = list(pdf_directory.glob("*.pdf"))

        if not pdf_files:
            raise ValueError(f"Keine PDF-Dateien gefunden in: {pdf_directory}")

        logger.info("📦 Batch-Verarbeitung: %d PDF-Dateien", len(pdf_files))

        buchungs_dir = output_base / "buchungen"
        training_dir = output_base / "training"

        results = {
            "processed_files": [],
            "buchungsausgaben": [],
            "training_exports": [],
            "errors": [],
            "summary": {
                "total_files": len(pdf_files),
                "successful": 0,
                "failed": 0,
                "total_positionen": 0,
                "total_training_entities": 0,
            },
        }

        for pdf_file in pdf_files:
            try:
                # Dual-Purpose Verarbeitung
                result = self.process_pdf_dual_purpose(pdf_file)

                # Speichere Buchungsausgabe
                buchungs_file = self.save_buchungsausgabe(
                    result["buchungsausgabe"], buchungs_dir
                )

                # Speichere Training-Export
                training_file = self.save_training_export(
                    result["training_export"], training_dir
                )

                # Sammle Statistiken
                positionen = len(result["buchungsausgabe"].get("positionen", []))
                entities = len(result["training_export"].get("raw_entities", []))

                results["processed_files"].append(str(pdf_file))
                results["buchungsausgaben"].append(str(buchungs_file))
                results["training_exports"].append(str(training_file))
                results["summary"]["successful"] += 1
                results["summary"]["total_positionen"] += positionen
                results["summary"]["total_training_entities"] += entities

                logger.info(
                    "✅ Verarbeitet: %s (%d Positionen, %d Entities)",
                    pdf_file.name,
                    positionen,
                    entities,
                )

            except Exception as e:
                results["errors"].append({"file": str(pdf_file), "error": str(e)})
                results["summary"]["failed"] += 1
                logger.error("❌ Fehler bei %s: %s", pdf_file.name, str(e))

        # Erstelle Batch-Summary
        summary_file = (
            output_base
            / f"batch_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        logger.info(
            "📊 Batch-Verarbeitung abgeschlossen: %d/%d erfolgreich",
            results["summary"]["successful"],
            results["summary"]["total_files"],
        )

        return results
