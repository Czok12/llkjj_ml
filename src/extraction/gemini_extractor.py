"""
Gemini 2.5 Pro Integration für LLKJJ ML Pipeline

Intelligente Extraktion von steuer- und fibu-relevanten Daten
aus PDF-Rechnungen für Elektrotechnik-Handwerk UG.
"""

import json
import logging
from dataclasses import dataclass
from typing import Any

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

    def __init__(self, api_key: str, model: str = "gemini-2.5-pro"):
        """
        Initialisiert Gemini Client

        Args:
            api_key: Google AI API Key
            model: Gemini Modell Version
        """
        self.api_key = api_key
        self.model = model
        self.client = None  # Wird später mit google.generativeai initialisiert

        # Elektrotechnik-spezifische Prompts
        self.extraction_prompt = self._create_extraction_prompt()

        logger.info(f"GeminiExtractor initialisiert mit {model}")

    def _create_extraction_prompt(self) -> str:
        """Erstellt optimierten Prompt für Elektrotechnik-Rechnungen"""
        return """
Du bist ein KI-Assistent für ein deutsches Elektrotechnik-Handwerksunternehmen.
Analysiere diese PDF-Rechnung und extrahiere alle steuer- und finanzbuchungsrelevanten Daten.

WICHTIGE HINWEISE:
- Das Unternehmen führt doppelte Buchführung nach SKR03
- Fokus auf Elektromaterial, Werkzeuge und Fremdleistungen
- Deutsche Umsatzsteuer (19% Regelsteuersatz, 7% ermäßigt)
- GWG-Grenze: 800€ für Anlagegüter vs. Betriebsausgaben

EXTRAHIERE FOLGENDE DATEN:

## RECHNUNGSKOPF:
- Rechnungsnummer (eindeutig)
- Rechnungsdatum (Format: YYYY-MM-DD)
- Lieferant/Firma (vollständiger Name)
- Lieferantennummer (falls vorhanden)
- Auftragsnummer/Bestellnummer (falls vorhanden)

## FINANZIELLE DATEN:
- Nettobetrag gesamt
- Umsatzsteuer gesamt (19% oder 7%)
- Bruttobetrag gesamt
- Währung (meist EUR)

## EINZELPOSITIONEN:
Für jede Rechnungsposition extrahiere:
- Positionsnummer (falls vorhanden)
- Artikelbezeichnung (vollständig)
- Artikelnummer/SKU/Waren-ID (falls vorhanden)
- Menge
- Einheit (Stück, Meter, kg, etc.)
- Einzelpreis (netto)
- Gesamtpreis der Position (netto)
- Umsatzsteuersatz (19% oder 7%)
- Produktkategorie bestimmen:
  * "elektromaterial" - Kabel, Schalter, Steckdosen, Sicherungen, Leuchten, Installationsmaterial
  * "werkzeug_klein" - Handwerkzeug, Messgeräte unter 800€
  * "anlagegueter" - Maschinen, Bohrmaschinen, teure Werkzeuge über 800€
  * "bueroausstattung" - Büromaterial, Computer, Software
  * "fremdleistung" - Montage, Installation durch Dritte
  * "betriebsausstattung" - Fahrzeuge, Einrichtungen

## SKR03 KONTIERUNG (KORREKTE KONTEN):
Ordne jede Position automatisch einem SKR03-Konto zu:
- 3400: Wareneingang 19% Vorsteuer (Elektromaterial, Verbrauchsmaterial)
- 3410: Wareneingang 7% Vorsteuer (reduzierte USt, falls vorhanden)
- 4985: Werkzeuge und Kleingeräte (unter 800€, Sofortaufwand)
- 0200: Technische Anlagen und Maschinen (über 800€, aktivierungspflichtig)
- 0210: Maschinen (Bohrmaschinen, größere Elektrowerkzeuge)
- 0400: Betriebsausstattung (Büroausstattung)
- 0420: Büroeinrichtung (Computer, Möbel)
- 0440: Werkzeuge (aktivierungspflichtige Werkzeuge über 800€)
- 4400: Fremdleistungen (externe Montage, Subunternehmer)

## LIEFERANTEN-KLASSIFIKATION:
Erkenne bekannte Elektro-Großhändler:
- Rexel, Conrad, ELV, Elektro Nord
- Wago, Phoenix Contact, Siemens, ABB
- Schneider Electric, Legrand, Hager
- Gira, Jung, Busch-Jaeger, Berker

Gib das Ergebnis als strukturiertes JSON zurück mit folgendem Schema:

{
    "invoice_number": "string",
    "invoice_date": "YYYY-MM-DD",
    "supplier": "string",
    "supplier_number": "string|null",
    "order_number": "string|null",
    "net_amount": number,
    "vat_amount": number,
    "gross_amount": number,
    "currency": "EUR",
    "line_items": [
        {
            "position_number": "string|null",
            "description": "string",
            "item_number": "string|null",
            "product_id": "string|null",
            "quantity": number,
            "unit": "string",
            "unit_price": number,
            "total_price": number,
            "vat_rate": number,
            "product_category": "elektromaterial|werkzeug_klein|anlagegueter|bueroausstattung|fremdleistung|betriebsausstattung",
            "skr03_account": "3400|3410|4985|0200|0210|0400|0420|0440|4400"
        }
    ],
    "confidence_score": number,
    "supplier_type": "elektro_grosshandel|local_supplier|unknown"
}

Analysiere den folgenden Rechnungstext:
"""

    def extract_from_text(self, text: str) -> GeminiExtractionResult:
        """
        Extrahiert Daten aus Rechnungstext mit Gemini 2.5 Pro

        Args:
            text: PDF-Text der Rechnung

        Returns:
            GeminiExtractionResult: Strukturierte Rechnungsdaten
        """
        logger.info("Starte Gemini-Extraktion...")

        try:
            # Placeholder - wird später mit echter Gemini API implementiert
            # response = self.client.generate_content(self.extraction_prompt + text)

            # Mock Response für Entwicklung
            mock_response = self._create_mock_response(text)

            # Parse JSON Response
            result_data = json.loads(mock_response)

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
                processing_time_ms=150,  # Mock-Wert
                model_version=self.model,
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

    def _create_mock_response(self, text: str) -> str:
        """
        Erstellt Mock-Response für Entwicklung
        (wird später durch echte Gemini API ersetzt)
        """
        # Erkennt einige Grundmuster im Text
        mock_data: dict[str, object] = {
            "invoice_number": "RE-2024-001",
            "invoice_date": "2024-08-14",
            "supplier": "Elektro Conrad GmbH & Co. KG",
            "supplier_number": "10001",
            "order_number": None,
            "net_amount": 156.78,
            "vat_amount": 29.79,
            "gross_amount": 186.57,
            "currency": "EUR",
            "line_items": [
                {
                    "position_number": "1",
                    "description": "Installationsleitung NYM-J 3x1,5 mm²",
                    "item_number": "NYM-J-3x1.5",
                    "product_id": "11020304",
                    "quantity": 100.0,
                    "unit": "m",
                    "unit_price": 1.25,
                    "total_price": 125.00,
                    "vat_rate": 0.19,
                    "product_category": "elektromaterial",
                    "skr03_account": "3400",
                },
                {
                    "position_number": "2",
                    "description": "Steckdose SCHUKO weiß",
                    "item_number": "SCHUKO-W-01",
                    "product_id": "22030405",
                    "quantity": 5.0,
                    "unit": "Stk",
                    "unit_price": 6.36,
                    "total_price": 31.78,
                    "vat_rate": 0.19,
                    "product_category": "elektromaterial",
                    "skr03_account": "3400",
                },
            ],
            "confidence_score": 0.92,
            "supplier_type": "elektro_grosshandel",
        }

        # Einfache Textanalyse für bessere Mock-Daten
        if "conrad" in text.lower():
            mock_data["supplier"] = "Conrad Electronic SE"
            mock_data["supplier_type"] = "elektro_grosshandel"
        elif "rexel" in text.lower():
            mock_data["supplier"] = "Rexel Germany GmbH & Co. KG"
        elif "wago" in text.lower():
            mock_data["supplier"] = "WAGO Kontakttechnik GmbH & Co. KG"

        # Werkzeug-Erkennung für korrekte SKR03-Kontierung
        if any(
            word in text.lower()
            for word in ["bohrmaschine", "säge", "schleifer", "winkelschleifer"]
        ):
            # Teure Werkzeuge über 800€ -> Anlagegut
            line_items = mock_data["line_items"]
            if isinstance(line_items, list):
                line_items.append(
                    {
                        "position_number": "3",
                        "description": "Akku-Bohrschrauber 18V Professional",
                        "item_number": "GSR18V-28",
                        "product_id": "33040506",
                        "quantity": 1.0,
                        "unit": "Stk",
                        "unit_price": 899.00,
                        "total_price": 899.00,
                        "vat_rate": 0.19,
                        "product_category": "anlagegueter",
                        "skr03_account": "0210",
                    }
                )
            net_amount = mock_data["net_amount"]
            if isinstance(net_amount, int | float):
                mock_data["net_amount"] = net_amount + 899.00
            vat_amount = mock_data["vat_amount"]
            if isinstance(vat_amount, int | float):
                mock_data["vat_amount"] = vat_amount + 170.81
            gross_amount = mock_data["gross_amount"]
            if isinstance(gross_amount, int | float):
                mock_data["gross_amount"] = gross_amount + 1069.81

        return json.dumps(mock_data, ensure_ascii=False, indent=2)

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

        if (
            abs(result.vat_amount - expected_vat_19) > 0.50
            and abs(result.vat_amount - expected_vat_7) > 0.50
        ):
            warnings = validation["warnings"]
            if isinstance(warnings, list):
                warnings.append(f"Unplausible Umsatzsteuer: {result.vat_amount}€")

        # SKR03 Konten prüfen
        valid_accounts = [
            "3400",
            "3410",
            "4985",
            "0200",
            "0210",
            "0400",
            "0420",
            "0440",
            "4400",
        ]
        if result.line_items:
            for item in result.line_items:
                if item.get("skr03_account") not in valid_accounts:
                    warnings = validation["warnings"]
                    if isinstance(warnings, list):
                        warnings.append(
                            f"Unbekanntes SKR03 Konto: {item.get('skr03_account')}"
                        )

        # Confidence Score
        if result.confidence_score < 0.7:
            warnings = validation["warnings"]
            if isinstance(warnings, list):
                warnings.append(f"Niedrige Confidence: {result.confidence_score}")

        return validation


# Utility Functions
def create_gemini_extractor(api_key: str) -> GeminiExtractor:
    """Factory Function für GeminiExtractor"""
    return GeminiExtractor(api_key=api_key)


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
    extractor = GeminiExtractor(api_key="test-key")

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
