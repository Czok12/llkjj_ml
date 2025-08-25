"""
Accounting API Models

Pydantic models for ML Plugin accounting API responses.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class InvoiceResponse(BaseModel):
    """Response model for invoice processing results."""

    processing_id: str
    filename: str
    status: str
    message: str
    supplier: str | None = None
    invoice_number: str | None = None
    total_amount: float | None = None
    line_items: list[dict[str, Any]] | None = None
    suggested_bookings: list[dict[str, Any]] | None = None
    confidence_score: float | None = None
    processing_time_ms: int | None = None


class BookingResponse(BaseModel):
    """Response model for booking operations."""

    processing_id: str
    booking_id: str
    status: str
    message: str
    approved_items: list[dict[str, Any]]
    total_amount: float
    booking_date: datetime
    skr03_accounts: list[str]


class CorrectionRequest(BaseModel):
    """Request model for booking corrections."""

    processing_id: str
    user_id: str
    corrections: list[dict[str, Any]]
    reason: str | None = None
    auto_approve: bool = False


class AnalyticsResponse(BaseModel):
    """Response model for analytics dashboard."""

    total_invoices_processed: int
    processing_accuracy: float
    avg_processing_time_ms: int
    time_saved_hours: float
    cost_savings_euro: float
    user_corrections_rate: float
    confidence_distribution: dict[str, int]
    top_suppliers: list[dict[str, Any]]
    supplier_accuracy: dict[str, float]
    most_used_accounts: list[dict[str, Any]]
    daily_processing_volume: list[dict[str, Any]]
    accuracy_trend: list[dict[str, Any]]


class LearningFeedback(BaseModel):
    """Model for user feedback to improve ML accuracy."""

    processing_id: str
    user_id: str
    feedback_type: str = Field(..., description="'correction', 'approval', 'rejection'")
    original_prediction: dict[str, Any]
    corrected_values: dict[str, Any] | None = None
    feedback_notes: str | None = None
    timestamp: datetime = Field(default_factory=datetime.now)
