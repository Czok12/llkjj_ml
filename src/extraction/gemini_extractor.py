"""
Gemini 2.5 Pro Integration für LLKJJ ML Pipeline

Intelligente Extraktion von steuer- und fibu-relevanten Daten
aus PDF-Rechnungen für Elektrotechnik-Handwerk UG.
"""

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


@dataclass
class GeminiExtractionResult:
    """Ergebnis der Gemini-Extraktion"""

    # Rechnungsmetadaten
    invoice_number: str
    invoice_date: str
    supplier: str
    supplier_number: str | None = None
    order_number: str | None = None

    # Finanzielle Daten
    net_amount: float = 0.0
    vat_amount: float = 0.0
    gross_amount: float = 0.0
    currency: str = "EUR"

    # Einzelpositionen
    line_items: list[dict[str, Any]] | None = None

    # Gemini Metadaten
    confidence_score: float = 0.0
    processing_time_ms: int = 0
    model_version: str = "gemini-2.5-pro"

    def __post_init__(self) -> None:
        if self.line_items is None:
            self.line_items = []


class GeminiExtractor:
    """Gemini 2.5 Pro Client für Rechnungsextraktion"""

    def __init__(
        self, api_key: str, model: str | None = None, prompt_path: str | None = None
    ):
        """
        Initialisiert Gemini Client

        Args:
            api_key: Google AI API Key
            model: Gemini Modell Version
            prompt_path: Pfad zur Prompt
        """
        self.api_key = api_key
        # Always ensure self.model is str, never None
        self.model = (
            model if model is not None else os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
        )
        self.client = genai.Client(api_key=api_key)

        # Prompt aus Datei laden
        self.extraction_prompt = self._load_extraction_prompt(prompt_path)
        logger.info("GeminiExtractor initialisiert mit %s", self.model)

    def _load_extraction_prompt(self, prompt_path: str | None = None) -> str:
        """Lädt den Prompt aus Datei, fallback auf Standardtext"""
        prompt_file = prompt_path or os.path.join(
            os.path.dirname(__file__), "gemini_prompt.txt"
        )
        try:
            with open(prompt_file, encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logger.warning("Prompt konnte nicht geladen werden: %s. Nutze Fallback.", e)
            return "Analysiere den folgenden Rechnungstext:"

    def extract_from_text(self, text: str) -> GeminiExtractionResult:
        """
        Extrahiert Daten aus Rechnungstext mit Gemini 2.5 Pro

        Args:
            text: PDF-Text der Rechnung

        Returns:
            GeminiExtractionResult: Strukturierte Rechnungsdaten
        """
        logger.info("Starte Gemini-Extraktion...")
        start_time = time.time()

        try:
            # Echte Gemini API-Aufruf
            response = self.client.models.generate_content(
                model=self.model,
                contents=self.extraction_prompt + text,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                ),
            )

            # Parse JSON Response
            response_text = response.text or "{}"
            result_data = json.loads(response_text)
            processing_time_ms = int((time.time() - start_time) * 1000)

            # Validiere und strukturiere Ergebnis
            result = GeminiExtractionResult(
                invoice_number=result_data.get("invoice_number", ""),
                invoice_date=result_data.get("invoice_date", ""),
                supplier=result_data.get("supplier", ""),
                supplier_number=result_data.get("supplier_number"),
                order_number=result_data.get("order_number"),
                net_amount=float(result_data.get("net_amount", 0)),
                vat_amount=float(result_data.get("vat_amount", 0)),
                gross_amount=float(result_data.get("gross_amount", 0)),
                currency=result_data.get("currency", "EUR"),
                line_items=result_data.get("line_items", []),
                confidence_score=float(result_data.get("confidence_score", 0.9)),
                processing_time_ms=processing_time_ms,
                model_version=str(self.model),
            )

            logger.info(f"Gemini-Extraktion erfolgreich: {result.invoice_number}")
            return result

        except Exception as e:
            logger.error(f"Fehler bei Gemini-Extraktion: {e}")
            # Fallback für Fehlerfall
            return GeminiExtractionResult(
                invoice_number="ERROR-001",
                invoice_date="2024-08-14",
                supplier="Fehler bei Extraktion",
                confidence_score=0.0,
            )

    def batch_extract(self, texts: list[str]) -> list[GeminiExtractionResult]:
        """
        Batch-Extraktion für mehrere Rechnungen

        Args:
            texts: Liste von PDF-Texten

        Returns:
            List[GeminiExtractionResult]: Extraktionsergebnisse
        """
        logger.info(f"Starte Batch-Extraktion für {len(texts)} Rechnungen")

        results = []
        for i, text in enumerate(texts):
            try:
                result = self.extract_from_text(text)
                results.append(result)
                logger.info(f"Batch {i+1}/{len(texts)} erfolgreich")

            except Exception as e:
                logger.error(f"Fehler bei Batch {i+1}: {e}")
                # Füge Fehler-Ergebnis hinzu
                error_result = GeminiExtractionResult(
                    invoice_number=f"ERROR-{i+1:03d}",
                    invoice_date="2024-08-14",
                    supplier="Batch-Extraktion fehlgeschlagen",
                    confidence_score=0.0,
                )
                results.append(error_result)

        logger.info(f"Batch-Extraktion abgeschlossen: {len(results)} Ergebnisse")
        return results

    def validate_extraction(self, result: GeminiExtractionResult) -> dict[str, Any]:
        """
        Validiert Extraktionsergebnis auf Plausibilität

        Args:
            result: Gemini Extraktionsergebnis

        Returns:
            Dict mit Validierungsergebnissen
        """
        validation: dict[str, object] = {"is_valid": True, "warnings": [], "errors": []}

        # Grundvalidierung
        if not result.invoice_number:
            errors = validation["errors"]
            if isinstance(errors, list):
                errors.append("Rechnungsnummer fehlt")
            validation["is_valid"] = False

        if not result.supplier:
            errors = validation["errors"]
            if isinstance(errors, list):
                errors.append("Lieferant fehlt")
            validation["is_valid"] = False

        if result.net_amount <= 0:
            warnings = validation["warnings"]
            if isinstance(warnings, list):
                warnings.append("Nettobetrag ist 0 oder negativ")

        # Umsatzsteuer-Plausibilität (19% oder 7%)
        expected_vat_19 = round(result.net_amount * 0.19, 2)
        expected_vat_7 = round(result.net_amount * 0.07, 2)

        if result.vat_amount > 0:
            if not (
                abs(result.vat_amount - expected_vat_19) < 0.01
                or abs(result.vat_amount - expected_vat_7) < 0.01
            ):
                warnings = validation["warnings"]
                if isinstance(warnings, list):
                    warnings.append("Unplausible Umsatzsteuer")

        # Confidence Score Check
        if result.confidence_score < 0.8:
            warnings = validation["warnings"]
            if isinstance(warnings, list):
                warnings.append(f"Niedrige Confidence: {result.confidence_score}")

        return validation


# Utility Functions
def create_gemini_extractor() -> GeminiExtractor:
    """Factory Function für GeminiExtractor"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set.")
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
    return GeminiExtractor(api_key=api_key, model=model)


def format_extraction_for_accounting(result: GeminiExtractionResult) -> dict[str, Any]:
    """
    Formatiert Extraktionsergebnis für Buchhaltungssoftware

    Args:
        result: Gemini Extraktionsergebnis

    Returns:
        Dict: Buchhaltungs-kompatibles Format
    """
    accounting_format: dict[str, object] = {
        "document_type": "incoming_invoice",
        "document_number": result.invoice_number,
        "document_date": result.invoice_date,
        "supplier": {"name": result.supplier, "number": result.supplier_number},
        "amounts": {
            "net": result.net_amount,
            "vat": result.vat_amount,
            "gross": result.gross_amount,
            "currency": result.currency,
        },
        "line_items": [],
        "skr03_suggestions": {},
    }

    # Verarbeite Einzelpositionen
    if result.line_items:
        for item in result.line_items:
            accounting_item = {
                "description": item.get("description"),
                "quantity": item.get("quantity"),
                "unit_price": item.get("unit_price"),
                "total_net": item.get("total_price"),
                "vat_rate": item.get("vat_rate"),
                "suggested_account": item.get("skr03_account"),
                "category": item.get("product_category"),
            }
            line_items = accounting_format["line_items"]
            if isinstance(line_items, list):
                line_items.append(accounting_item)

            # SKR03 Zusammenfassung
            account = item.get("skr03_account")
            skr03_suggestions = accounting_format["skr03_suggestions"]
            if isinstance(skr03_suggestions, dict) and account:
                if account in skr03_suggestions:
                    current_value = skr03_suggestions[account]
                    if isinstance(current_value, int | float):
                        skr03_suggestions[account] = current_value + item.get(
                            "total_price", 0
                        )
                else:
                    skr03_suggestions[account] = item.get("total_price", 0)

    return accounting_format


if __name__ == "__main__":
    # Test der Gemini-Extraktion
    from dotenv import load_dotenv

    load_dotenv()

    extractor = create_gemini_extractor()

    test_text = """
    RECHNUNG
    Conrad Electronic SE
    Rechnungsnummer: RE-2024-0815
    Datum: 14.08.2024

    Pos. Artikel                           Menge  Einzelpreis  Gesamtpreis
    1    Installationsleitung NYM-J        100m   1,25€        125,00€
    2    SCHUKO Steckdose weiß            5 St.   6,36€         31,78€

    Nettobetrag:     156,78€
    Umsatzsteuer 19%: 29,79€
    Bruttobetrag:    186,57€
    """

    result = extractor.extract_from_text(test_text)
    validation = extractor.validate_extraction(result)

    print("Gemini Extraktionsergebnis:")
    print(f"Rechnung: {result.invoice_number}")
    print(f"Lieferant: {result.supplier}")
    print(
        f"Betrag: {result.net_amount}€ + {result.vat_amount}€ = {result.gross_amount}€"
    )
    print(f"Positionen: {len(result.line_items) if result.line_items else 0}")
    print(f"Confidence: {result.confidence_score}")
    print(f"Validierung: {'✓' if validation['is_valid'] else '✗'}")
