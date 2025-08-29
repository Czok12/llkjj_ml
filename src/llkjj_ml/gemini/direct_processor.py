"""
Gemini Direct PDF Processor - ohne Docling Pipeline

Kostengünstigere und einfachere Alternative zur Docling → Gemini Pipeline.
Verarbeitet PDFs direkt mit Gemini ohne Zwischenschritte.

Author: LLKJJ ML Team
Version: 1.0.0
Date: 2025-01-25
"""

import asyncio
import base64
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from llkjj_ml.settings_bridge import Config, ConfigBridge

logger = logging.getLogger(__name__)


class GeminiDirectConfig(BaseModel):
    """Konfiguration für Gemini-Direct-Processing."""

    model_name: str = "gemini-2.5-flash"
    max_pdf_size_mb: int = 20
    temperature: float = 0.1  # Niedrige Temperatur für konsistente Klassifizierung
    max_output_tokens: int = 8192
    api_key: str | None = None


class InvoiceItem(BaseModel):
    """Rechnungsposition-Ergebnis von Gemini."""

    position: int
    article_number: str | None = None
    description: str
    quantity: float
    unit_price: float | None = None
    total_price: float | None = None
    skr03_account: str
    skr03_category: str
    classification_confidence: float
    classification_reasoning: str


class GeminiDirectResult(BaseModel):
    """Ergebnis der Gemini-Direct-Verarbeitung."""

    success: bool
    invoice_data: dict[str, Any]
    invoice_items: list[InvoiceItem]
    processing_time_ms: int
    token_usage: dict[str, Any]
    errors: list[str] = []


class GeminiDirectProcessor:
    """
    Direkter PDF → Gemini-Processor ohne Docling-Zwischenschritt.

    Kostengünstiger und einfacher als Docling → Gemini Pipeline.
    Optimiert für deutsche Elektrohandwerk-Rechnungen mit SKR03-Klassifizierung.
    """

    def __init__(self, config: ConfigBridge | GeminiDirectConfig | None = None):
        """
        Initialisiere den GeminiDirectProcessor.

        Args:
            config: Konfiguration für ML-Pipeline (optional)
        """
        self.config = config or Config
        self.logger = logging.getLogger(__name__)

        # Client-Reuse: Lazy Loading für bessere Performance
        self._client: Any | None = None

        # Validierung der Gemini-API-Konfiguration
        api_key = getattr(self.config, "google_api_key", None) or getattr(
            self.config, "api_key", None
        )
        if not api_key:
            self.logger.warning("⚠️ Keine Google API Key konfiguriert")
            self._is_available = False
        else:
            self._is_available = True
            self.logger.info("✅ GeminiDirectProcessor bereit")

        # Performance-Monitoring
        self.processing_times: list[float] = []
        self.error_count = 0

    def _initialize_gemini_client(self) -> None:
        """Initialize Gemini Client with proper error handling (Lazy Loading)."""
        # Client-Reuse: Nur initialisieren wenn noch nicht vorhanden
        if self._client is not None:
            return

        try:
            from google import genai

            # Use API key from config or environment
            api_key = getattr(self.config, "google_api_key", None) or getattr(
                self.config, "api_key", None
            )
            if not api_key:
                import os

                api_key = os.getenv("GEMINI_API_KEY")

            if not api_key:
                raise RuntimeError(
                    "GEMINI_API_KEY nicht verfügbar - weder in config noch environment"
                )

            self._client = genai.Client(api_key=api_key)
            logger.debug("✅ Gemini Client erfolgreich initialisiert (Lazy Loading)")

        except ImportError:
            raise RuntimeError(
                "google-genai Package nicht installiert. Run: pip install google-genai"
            ) from None
        except Exception as e:
            raise RuntimeError(f"Fehler bei Gemini Client Initialisierung: {e}") from e

    async def process_pdf_direct(self, pdf_path: Path) -> GeminiDirectResult:
        """
        Verarbeite PDF direkt mit Gemini (ohne Docling).

        Args:
            pdf_path: Pfad zur PDF-Datei

        Returns:
            GeminiDirectResult mit extrahierten Daten
        """
        start_time = time.time()

        result = GeminiDirectResult(
            success=False,
            invoice_data={},
            invoice_items=[],
            processing_time_ms=0,
            token_usage={},
        )

        try:
            # PDF-Validierung
            if not self._validate_pdf(pdf_path):
                result.errors.append("PDF-Validierung fehlgeschlagen")
                return result

            # PDF zu Base64 konvertieren
            pdf_base64 = self._pdf_to_base64(pdf_path)

            # Gemini-Prompt für Elektrohandwerk-Rechnungen
            prompt = self._build_extraction_prompt()

            # Gemini API-Call
            logger.info(f"🤖 Sende PDF an Gemini: {pdf_path.name}")
            gemini_response = await self._call_gemini_api(pdf_base64, prompt)

            if not gemini_response:
                result.errors.append("Gemini API-Call fehlgeschlagen")
                return result

            # Response parsen
            parsed_data = self._parse_gemini_response(gemini_response)

            result.success = True
            result.invoice_data = parsed_data.get("invoice_data", {})
            result.invoice_items = [
                InvoiceItem(**item) for item in parsed_data.get("invoice_items", [])
            ]
            result.token_usage = parsed_data.get("token_usage", {})
            result.processing_time_ms = int((time.time() - start_time) * 1000)

            logger.info(
                f"✅ Gemini-Processing erfolgreich: {len(result.invoice_items)} Items, "
                f"{result.processing_time_ms}ms"
            )

            return result

        except Exception as e:
            error_type = type(e).__name__
            error_msg = f"Fehler bei Gemini-Direct-Processing ({error_type}): {e}"

            # Spezifische Fehlerbehandlung für besseres Debugging
            if "json" in str(e).lower() or "JSONDecodeError" in error_type:
                logger.error(f"🔧 JSON-Parse-Fehler: {e}")
                result.errors.append(f"JSON-Parsing fehlgeschlagen: {e}")
            elif "network" in str(e).lower() or "connection" in str(e).lower():
                logger.error(f"🌐 Netzwerk-Fehler: {e}")
                result.errors.append(f"Netzwerkfehler - Internetverbindung prüfen: {e}")
            elif "api_key" in str(e).lower() or "authentication" in str(e).lower():
                logger.error(f"🔑 API-Key-Fehler: {e}")
                result.errors.append(f"API-Key ungültig oder fehlt: {e}")
            else:
                logger.error(f"❌ {error_msg}")
                result.errors.append(error_msg)

            result.processing_time_ms = int((time.time() - start_time) * 1000)
            return result

    def _validate_pdf(self, pdf_path: Path) -> bool:
        """Validiere PDF-Datei."""
        try:
            if not pdf_path.exists():
                logger.error(f"PDF nicht gefunden: {pdf_path}")
                return False

            file_size_mb = pdf_path.stat().st_size / (1024 * 1024)
            if file_size_mb > self.config.max_pdf_size_mb:
                logger.error(
                    f"PDF zu groß: {file_size_mb:.1f}MB > {self.config.max_pdf_size_mb}MB"
                )
                return False

            if pdf_path.suffix.lower() != ".pdf":
                logger.error(f"Datei ist keine PDF: {pdf_path}")
                return False

            return True

        except Exception as e:
            logger.error(f"PDF-Validierung fehlgeschlagen: {e}")
            return False

    def _pdf_to_base64(self, pdf_path: Path) -> str:
        """Konvertiere PDF zu Base64."""
        with open(pdf_path, "rb") as pdf_file:
            return base64.b64encode(pdf_file.read()).decode("utf-8")

    def _build_extraction_prompt(self) -> str:
        """
        Erstelle optimierten Prompt für Elektrohandwerk-Rechnungen.

        Returns:
            Detaillierter Prompt für SKR03-Klassifizierung
        """
        return """
Du bist ein Experte für deutsche Finanzbuchhaltung im Elektrohandwerk mit Spezialisierung auf SKR03-Kontenplan.

AUFGABE: Analysiere diese Eingangsrechnung und klassifiziere alle Positionen nach SKR03.

ELEKTROHANDWERK-KONTEXT:
- Typische Lieferanten: Rexel, Sonepar, Conrad, ELV, WAGO, Phoenix Contact, Siemens, ABB, Schneider Electric
- Hauptkategorien: Elektroinstallationsmaterial, Kabel, Leuchten, Schaltanlagen, Messgeräte, Werkzeuge

SKR03-HAUPTKONTEN für Elektrohandwerk:
- 3400: Wareneingänge Elektromaterial (Standardkonto)
- 3300: Rohstoffe und Einzelteile
- 4930: Bürobedarf/Software
- 4985: Werkzeuge/Kleingeräte (unter 800€)
- 0490: Anlagegüter (über 800€)
- 4200: Gas/Strom/Wasser
- 4210: Telekommunikation
- 4240: Reparaturen/Wartung

KLASSIFIZIERUNGSREGELN:
1. Elektromaterial → 3400 (Standardkonto)
2. Kleingeräte/Werkzeuge < 800€ → 4985
3. Anlagegüter > 800€ → 0490
4. Büromaterial → 4930
5. Bei Unsicherheit → 3400 (Standardkonto)

ANTWORTFORMAT (JSON):
{
  "invoice_data": {
    "supplier": "Lieferant Name",
    "invoice_number": "Rechnungsnummer",
    "invoice_date": "YYYY-MM-DD",
    "total_amount": 0.00,
    "currency": "EUR"
  },
  "invoice_items": [
    {
      "position": 1,
      "article_number": "Art-Nr oder null",
      "description": "Vollständige Artikelbezeichnung",
      "quantity": 0.0,
      "unit_price": 0.00,
      "total_price": 0.00,
      "skr03_account": "3400",
      "skr03_category": "wareneingang_elektro_allgemein",
      "classification_confidence": 0.95,
      "classification_reasoning": "Kurze Begründung für SKR03-Zuordnung"
    }
  ]
}

WICHTIG:
- Jeden Artikel einzeln klassifizieren
- Confidence zwischen 0.0-1.0
- Bei Unsicherheit: 3400 verwenden
- Deutsche Artikelbezeichnungen beibehalten
- Preise aus Netto-Beträgen (ohne MwSt.)
"""

    async def _call_gemini_api(
        self, pdf_base64: str, prompt: str
    ) -> dict[str, Any] | None:
        """Rufe Gemini API mit PDF und Prompt auf - mit Retry Logic."""
        max_retries = 3

        for attempt in range(max_retries):
            try:
                # Stelle sicher, dass Client initialisiert ist
                if not self._client:
                    self._initialize_gemini_client()

                # Type-Guard: Client muss nach Initialisierung vorhanden sein
                if not self._client:
                    raise RuntimeError("Client-Initialisierung fehlgeschlagen")

                # Gemini API-Call mit PDF
                response = self._client.models.generate_content(
                    model=self.config.model_name,
                    contents=[
                        {
                            "parts": [
                                {"text": prompt},
                                {
                                    "inline_data": {
                                        "mime_type": "application/pdf",
                                        "data": pdf_base64,
                                    }
                                },
                            ]
                        }
                    ],
                    config={
                        "temperature": self.config.temperature,
                        "max_output_tokens": self.config.max_output_tokens,
                        "candidate_count": 1,
                    },
                )

                # Erfolgreicher API-Call
                logger.debug(f"✅ Gemini API-Call erfolgreich (Versuch {attempt + 1})")
                return {
                    "content": response.text,
                    "usage": getattr(response, "usage", {}),
                    "finish_reason": getattr(response, "finish_reason", "unknown"),
                }

            except Exception as e:
                error_type = type(e).__name__

                # Bei letztem Versuch: Fehler werfen
                if attempt == max_retries - 1:
                    logger.error(
                        f"❌ Gemini API nach {max_retries} Versuchen fehlgeschlagen: {error_type}: {e}"
                    )
                    return None

                # Exponential Backoff: 2, 4, 8 Sekunden
                wait_time = 2 ** (attempt + 1)

                # Spezifische Behandlung für Rate-Limiting
                if "429" in str(e) or "rate" in str(e).lower():
                    wait_time = 10  # Längere Wartezeit bei Rate-Limiting
                    logger.warning(
                        f"⏳ Rate-Limit erreicht - Retry {attempt + 1}/{max_retries} nach {wait_time}s"
                    )
                else:
                    logger.warning(
                        f"⚠️ Gemini API Fehler ({error_type}) - Retry {attempt + 1}/{max_retries} nach {wait_time}s: {e}"
                    )

                await asyncio.sleep(wait_time)

        # Fallback wenn alle Retries fehlschlagen
        return None

    def _parse_gemini_response(self, gemini_response: dict[str, Any]) -> dict[str, Any]:
        """Parse Gemini-Response zu strukturierten Daten."""
        try:
            content = gemini_response.get("content", "")

            # JSON aus Response extrahieren (zwischen ```json und ```)
            json_match = re.search(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL)
            if not json_match:
                # Fallback: Versuche gesamten Content als JSON zu parsen
                json_match = re.search(r"(\{.*\})", content, re.DOTALL)

            if not json_match:
                raise ValueError("Kein JSON in Gemini-Response gefunden")

            json_str = json_match.group(1)
            parsed_data: dict[str, Any] = json.loads(json_str)

            # Token-Usage hinzufügen
            parsed_data["token_usage"] = gemini_response.get("usage", {})

            # Validierung der Datenstruktur
            self._validate_parsed_data(parsed_data)

            return parsed_data

        except Exception as e:
            logger.error(f"Fehler beim Parsen der Gemini-Response: {e}")
            # Fallback-Response
            return {
                "invoice_data": {},
                "invoice_items": [],
                "token_usage": gemini_response.get("usage", {}),
                "parsing_error": str(e),
            }

    def _validate_parsed_data(self, data: dict[str, Any]) -> None:
        """Validiere geparste Daten."""
        required_fields = ["invoice_data", "invoice_items"]
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Fehlendes Feld in Response: {field}")

        # Validiere invoice_items
        for i, item in enumerate(data.get("invoice_items", [])):
            required_item_fields = ["position", "description", "skr03_account"]
            for field in required_item_fields:
                if field not in item:
                    logger.warning(f"Fehlendes Feld in Item {i}: {field}")

    async def classify_with_rag_context(
        self, item_description: str, rag_context: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Klassifiziere Item mit RAG-Kontext.

        Args:
            item_description: Artikelbeschreibung
            rag_context: Kontext von ähnlichen Items

        Returns:
            Enhanced classification
        """
        try:
            # RAG-Enhanced Prompt
            context_items = rag_context.get("similar_items", [])
            context_text = "\n".join(
                [
                    f"- {item['description']} (Lieferant: {item['supplier']}) → SKR03: {item['skr03_account']} (Confidence: {item['confidence']:.2f})"
                    for item in context_items[:3]  # Top 3 ähnliche Items
                ]
            )

            prompt = f"""
Klassifiziere diesen Artikel mit Hilfe ähnlicher Artikel aus der Historie:

AKTUELLER ARTIKEL: {item_description}

ÄHNLICHE ARTIKEL (Referenz):
{context_text}

Berücksichtige die SKR03-Klassifizierungen der ähnlichen Artikel, aber entscheide basierend auf dem aktuellen Artikel.

ANTWORT (JSON):
{{
  "skr03_account": "3400",
  "skr03_category": "wareneingang_elektro_allgemein",
  "confidence": 0.95,
  "reasoning": "Begründung basierend auf ähnlichen Artikeln"
}}
"""

            if not self._client:
                self._initialize_gemini_client()

            # Type-Guard: Client muss nach Initialisierung vorhanden sein
            if not self._client:
                raise RuntimeError("Client-Initialisierung fehlgeschlagen")

            response = self._client.models.generate_content(
                model=self.config.model_name,
                contents=[{"parts": [{"text": prompt}]}],
                config={
                    "temperature": 0.05,  # Sehr niedrig für konsistente Klassifizierung
                    "max_output_tokens": 512,
                },
            )

            # Response parsen
            content = response.text

            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if json_match:
                result_data: dict[str, Any] = json.loads(json_match.group(0))
                result_data["success"] = True
                return result_data

            return {"success": False, "error": "Kein JSON in RAG-Response"}

        except Exception as e:
            error_type = type(e).__name__

            # Spezifische Fehlerbehandlung für RAG-Classification
            if "json" in str(e).lower():
                logger.error(f"🔧 RAG JSON-Parse-Fehler: {e}")
                error_msg = f"JSON-Parsing bei RAG-Classification fehlgeschlagen: {e}"
            elif "timeout" in str(e).lower():
                logger.error(f"⏰ RAG Timeout-Fehler: {e}")
                error_msg = f"Timeout bei RAG-Classification: {e}"
            else:
                logger.error(f"❌ RAG Classification Fehler ({error_type}): {e}")
                error_msg = f"RAG Classification fehlgeschlagen ({error_type}): {e}"

            return {"success": False, "error": error_msg}


# =============================================================================
# Factory & Convenience Functions
# =============================================================================


def create_gemini_direct_processor(
    api_key: str | None = None, model_name: str = "gemini-2.5-flash"
) -> GeminiDirectProcessor:
    """
    Factory für GeminiDirectProcessor.

    Args:
        api_key: Optional Gemini API Key (sonst aus Environment)
        model_name: Gemini Model zu verwenden

    Returns:
        Configured GeminiDirectProcessor
    """
    config = GeminiDirectConfig(api_key=api_key, model_name=model_name)
    return GeminiDirectProcessor(config)


async def process_pdf_with_gemini(
    pdf_path: Path, api_key: str | None = None
) -> GeminiDirectResult:
    """
    Convenience-Funktion für direktes PDF-Processing.

    Args:
        pdf_path: Pfad zur PDF-Datei
        api_key: Optional Gemini API Key

    Returns:
        GeminiDirectResult
    """
    processor = create_gemini_direct_processor(api_key)
    return await processor.process_pdf_direct(pdf_path)
