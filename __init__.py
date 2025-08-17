#!/usr/bin/env python3
"""
LLKJJ ML Plugin - Public Package Interface
==========================================

This package provides a self-contained ML processing plugin for
German electrical trade invoice processing with SKR03 integration.

**BLACKBOX INTERFACE:**
```python
from llkjj_ml_plugin import MLPlugin, ProcessingResult, process_pdf_simple

# Method 1: Plugin instance (recommended for multiple files)
plugin = MLPlugin()
result = plugin.process_pdf("invoice.pdf")
plugin.cleanup()

# Method 2: One-shot processing (simple scripts)
result = process_pdf_simple("invoice.pdf")

# Access results
print(f"Quality: {result.extraction_quality}")
print(f"Classifications: {len(result.skr03_classifications)}")
```

**NO EXTERNAL DEPENDENCIES:**
This plugin is completely self-contained and has no knowledge
of external systems, DMS plugins, or other components.

Author: LLKJJ ML Pipeline Team
Version: 3.0.0 (Blackbox Release)
"""

# Import public API components
from llkjj_ml_plugin import MLPlugin, ProcessingResult, process_pdf_simple

# Version information
__version__ = "3.0.0"
__author__ = "LLKJJ ML Pipeline Team"
__description__ = "Self-contained ML plugin for German invoice processing"

# Public API exports
__all__ = [
    "MLPlugin",
    "ProcessingResult",
    "process_pdf_simple",
    "__version__",
]

# Plugin metadata for external systems
PLUGIN_INFO = {
    "name": "LLKJJ ML Plugin",
    "version": __version__,
    "type": "blackbox",
    "input_format": "pdf",
    "output_format": "structured_json",
    "language": "german",
    "domain": "electrical_trade",
    "accounting_system": "skr03",
    "dependencies": "self_contained",
}
