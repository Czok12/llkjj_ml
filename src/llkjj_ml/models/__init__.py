"""
LLKJJ ML Pipeline - Pydantic Data Models

This package contains all Pydantic BaseModel definitions for structured data validation
in the German electrical contractor invoice processing pipeline.

Models Overview:
- invoice.py: German invoice data structures and validation
- skr03.py: SKR03 accounting classification models
- base.py: Shared base models and utilities

German Business Logic:
All models are optimized for German electrical contractor businesses:
- German VAT rates (7%, 19%)
- SKR03 chart of accounts validation
- German address and postal code formats
- Currency formatting (EUR)
- German supplier and customer data patterns

Usage:
```python
from .invoice import InvoiceHeader, LineItem
from .skr03 import SKR03Classification

# Create validated invoice data
invoice = InvoiceHeader(
    rechnung_nummer="RE-2025-001",
    lieferant="Elektro Müller GmbH",
    datum="2025-01-16",
    netto_betrag=1250.00,
    mwst_satz=19
)
```
"""

from .gemini_schemas import GeminiExtractionResult, GeminiInvoiceHeader
from .processing_result import ProcessingResult, create_processing_result

__all__ = [
    "ProcessingResult",
    "create_processing_result",
    "GeminiExtractionResult",
    "GeminiInvoiceHeader",
]
