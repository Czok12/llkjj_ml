"""
AI-Batch-Processing-Schemas für Eingangsrechnungen

Definiert Schemas für die Batch-Verarbeitung von PDF-Dokumenten
durch verschiedene AI-Services (Gemini, OpenAI, Claude, etc.)
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AIProvider(str, Enum):
    """Unterstützte AI-Provider für PDF-Extraktion"""
    
    GEMINI = "gemini"
    OPENAI = "openai"
    CLAUDE = "claude"
    AZURE_OPENAI = "azure_openai"
    CUSTOM = "custom"


class ProcessingStatus(str, Enum):
    """Status der Batch-Verarbeitung"""
    
    PENDING = "pending"
    PROCESSING = "processing" 
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIALLY_FAILED = "partially_failed"


class DocumentInput(BaseModel):
    """Einzelnes PDF-Dokument für AI-Verarbeitung"""
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
    )
    
    document_id: str = Field(..., description="Eindeutige Dokument-ID")
    filename: str = Field(..., description="Originaler Dateiname")
    content_base64: str = Field(..., description="PDF-Inhalt als Base64")
    content_type: str = Field(default="application/pdf", description="MIME-Type")
    file_size_bytes: int = Field(..., gt=0, description="Dateigröße in Bytes")
    
    # Optionale Metadaten
    upload_timestamp: Optional[datetime] = Field(None, description="Upload-Zeitstempel")
    source_system: Optional[str] = Field(None, description="Quellsystem")
    priority: int = Field(default=5, ge=1, le=10, description="Verarbeitungspriorität (1=hoch, 10=niedrig)")


class AIProcessingConfig(BaseModel):
    """Konfiguration für AI-Provider"""
    
    provider: AIProvider = Field(..., description="AI-Provider")
    model_name: str = Field(..., description="Spezifisches Modell")
    api_key_reference: Optional[str] = Field(None, description="API-Key-Referenz")
    
    # Provider-spezifische Einstellungen
    temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="Kreativitäts-Parameter")
    max_tokens: int = Field(default=4096, gt=0, description="Maximale Token-Anzahl")
    timeout_seconds: int = Field(default=120, gt=0, description="Timeout in Sekunden")
    
    # Retry-Konfiguration
    max_retries: int = Field(default=3, ge=0, description="Maximale Wiederholungen")
    retry_delay_seconds: int = Field(default=5, ge=1, description="Verzögerung zwischen Wiederholungen")


class BatchProcessingRequest(BaseModel):
    """Haupt-Request für AI-Batch-Processing"""
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
    )
    
    batch_id: str = Field(default_factory=lambda: str(uuid4()), description="Eindeutige Batch-ID")
    request_timestamp: datetime = Field(default_factory=datetime.utcnow, description="Request-Zeitstempel")
    
    documents: List[DocumentInput] = Field(..., min_length=1, max_length=100, description="PDF-Dokumente")
    ai_config: AIProcessingConfig = Field(..., description="AI-Provider-Konfiguration")
    
    # Optionale Batch-Einstellungen
    parallel_processing: bool = Field(default=True, description="Parallele Verarbeitung aktivieren")
    callback_url: Optional[str] = Field(None, description="Callback-URL für Ergebnisse")
    notification_email: Optional[str] = Field(None, description="E-Mail für Benachrichtigungen")
    
    @field_validator("documents")
    @classmethod
    def validate_documents_limit(cls, v: List[DocumentInput]) -> List[DocumentInput]:
        """Validiere Anzahl der Dokumente"""
        if len(v) > 100:
            raise ValueError("Maximal 100 Dokumente pro Batch erlaubt")
        return v


class InvoiceExtractionResult(BaseModel):
    """Extrahierte Rechnungsdaten von AI-Service"""
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        use_enum_values=True,
    )
    
    # Rechnungskopfdaten
    invoice_number: Optional[str] = Field(None, description="Rechnungsnummer")
    supplier_name: Optional[str] = Field(None, description="Lieferantenname")
    supplier_vat_id: Optional[str] = Field(None, description="USt-ID Lieferant")
    supplier_address: Optional[str] = Field(None, description="Lieferantenadresse")
    
    # Rechnungsempfänger
    customer_name: Optional[str] = Field(None, description="Kundenname")
    customer_vat_id: Optional[str] = Field(None, description="USt-ID Kunde")
    customer_address: Optional[str] = Field(None, description="Kundenadresse")
    
    # Datums- und Betragsfelder
    invoice_date: Optional[str] = Field(None, description="Rechnungsdatum (YYYY-MM-DD)")
    due_date: Optional[str] = Field(None, description="Fälligkeitsdatum (YYYY-MM-DD)")
    service_date_from: Optional[str] = Field(None, description="Leistungsdatum von")
    service_date_to: Optional[str] = Field(None, description="Leistungsdatum bis")
    
    net_amount: Optional[Decimal] = Field(None, description="Nettobetrag")
    tax_amount: Optional[Decimal] = Field(None, description="Steuerbetrag")
    gross_amount: Optional[Decimal] = Field(None, description="Bruttobetrag")
    currency_code: str = Field(default="EUR", description="Währung")
    
    # Positionen
    line_items: List[Dict[str, Any]] = Field(default_factory=list, description="Rechnungspositionen")
    
    # Zusatzinformationen
    payment_terms: Optional[str] = Field(None, description="Zahlungsbedingungen")
    delivery_note_number: Optional[str] = Field(None, description="Lieferscheinnummer")
    order_number: Optional[str] = Field(None, description="Bestellnummer")
    project_number: Optional[str] = Field(None, description="Projektnummer")
    
    # Extrahierter Volltext
    extracted_text: Optional[str] = Field(None, description="Vollständiger extrahierter Text")
    
    # Konfidenz-Werte
    confidence_scores: Dict[str, float] = Field(default_factory=dict, description="Konfidenz pro Feld")
    overall_confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Gesamt-Konfidenz")


class ProcessingError(BaseModel):
    """Fehlerinformationen bei der Verarbeitung"""
    
    error_code: str = Field(..., description="Fehlercode")
    error_message: str = Field(..., description="Fehlermeldung")
    error_details: Optional[Dict[str, Any]] = Field(None, description="Zusätzliche Fehlerdetails")
    occurred_at: datetime = Field(default_factory=datetime.utcnow, description="Fehlerzeitpunkt")


class DocumentProcessingResult(BaseModel):
    """Verarbeitungsergebnis für ein einzelnes Dokument"""
    
    document_id: str = Field(..., description="Dokument-ID")
    filename: str = Field(..., description="Dateiname")
    status: ProcessingStatus = Field(..., description="Verarbeitungsstatus")
    
    # Erfolgreiche Extraktion
    extracted_data: Optional[InvoiceExtractionResult] = Field(None, description="Extrahierte Daten")
    
    # Fehlerbehandlung
    errors: List[ProcessingError] = Field(default_factory=list, description="Aufgetretene Fehler")
    
    # Verarbeitungsmetadaten
    processing_started_at: Optional[datetime] = Field(None, description="Verarbeitungsbeginn")
    processing_completed_at: Optional[datetime] = Field(None, description="Verarbeitungsende")
    processing_duration_seconds: Optional[float] = Field(None, description="Verarbeitungsdauer")
    
    ai_provider: AIProvider = Field(..., description="Verwendeter AI-Provider")
    ai_model: str = Field(..., description="Verwendetes AI-Modell")
    api_cost_estimate: Optional[Decimal] = Field(None, description="Geschätzte API-Kosten")


class BatchProcessingResponse(BaseModel):
    """Antwort der Batch-Verarbeitung"""
    
    batch_id: str = Field(..., description="Batch-ID")
    overall_status: ProcessingStatus = Field(..., description="Gesamt-Status")
    
    # Timing
    started_at: datetime = Field(..., description="Batch-Start")
    completed_at: Optional[datetime] = Field(None, description="Batch-Ende")
    total_duration_seconds: Optional[float] = Field(None, description="Gesamtdauer")
    
    # Statistiken
    total_documents: int = Field(..., description="Anzahl Dokumente gesamt")
    successful_documents: int = Field(default=0, description="Erfolgreich verarbeitet")
    failed_documents: int = Field(default=0, description="Fehlgeschlagen")
    
    # Ergebnisse
    results: List[DocumentProcessingResult] = Field(..., description="Verarbeitungsergebnisse")
    
    # Kosten und Performance
    total_api_cost_estimate: Optional[Decimal] = Field(None, description="Geschätzte Gesamtkosten")
    average_processing_time: Optional[float] = Field(None, description="Durchschnittliche Verarbeitungszeit")
    
    # Callback-Information
    callback_sent: bool = Field(default=False, description="Callback gesendet")
    callback_sent_at: Optional[datetime] = Field(None, description="Callback-Zeitpunkt")


class BatchProcessingStats(BaseModel):
    """Statistiken für Batch-Processing"""
    
    batch_id: str = Field(..., description="Batch-ID")
    
    # Dokument-Statistiken
    total_documents: int = Field(..., description="Anzahl Dokumente")
    successful_extractions: int = Field(..., description="Erfolgreiche Extraktionen")
    failed_extractions: int = Field(..., description="Fehlgeschlagene Extraktionen")
    success_rate: float = Field(..., description="Erfolgsrate (0.0 - 1.0)")
    
    # Zeit-Statistiken
    total_processing_time: float = Field(..., description="Gesamtverarbeitungszeit (Sekunden)")
    average_processing_time: float = Field(..., description="Durchschnittliche Zeit pro Dokument")
    min_processing_time: float = Field(..., description="Minimale Verarbeitungszeit")
    max_processing_time: float = Field(..., description="Maximale Verarbeitungszeit")
    
    # Kosten-Statistiken
    total_estimated_cost: Optional[Decimal] = Field(None, description="Geschätzte Gesamtkosten")
    average_cost_per_document: Optional[Decimal] = Field(None, description="Durchschnittliche Kosten pro Dokument")
    
    # Konfidenz-Statistiken
    average_confidence: float = Field(..., description="Durchschnittliche Konfidenz")
    min_confidence: float = Field(..., description="Minimale Konfidenz")
    max_confidence: float = Field(..., description="Maximale Konfidenz")
    
    # Provider-Informationen
    ai_provider: AIProvider = Field(..., description="Verwendeter AI-Provider")
    ai_model: str = Field(..., description="Verwendetes AI-Modell")