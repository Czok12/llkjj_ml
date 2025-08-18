# Context-Aware Klassifizierung - Implementierungsdetails

## Übersicht

Die Context-Aware Klassifizierung erweitert das Standard-RAG-System um intelligente Kontext-Analyse für verbesserte SKR03-Klassifizierung basierend auf:

- **Lieferanten-spezifische Pattern**: Verschiedene Lieferanten → verschiedene Klassifizierungs-Biases
- **Preis-Threshold-Logik**: Automatische Anlagevermögen/Verbrauchsmaterial-Unterscheidung
- **Multi-Position-Konsistenz**: Ähnliche Artikel auf derselben Rechnung → konsistente Kontierung
- **Temporal-Analyse**: Saisonale Kaufmuster für bessere Kontierung

## Implementierung

### ContextAwareClassifier (`src/intelligence/context_classifier.py`)

**Kernfunktionalität:**
```python
def enhance_classification_with_context(
    self, item: dict[str, Any], supplier_name: str, invoice_context: dict[str, Any]
) -> dict[str, Any]:
```

**Features:**

1. **Supplier-Context-Logic**:
   - Sonepar → Elektro-Bias (Konto 3400)
   - Amazon → Verbrauchsmaterial-Bias (Konto 4000)
   - Würth → Werkzeug-Bias (Konto 4200)
   - Famo → Installation-Bias (Konto 3400)

2. **Price-Threshold-Logic**:
   - Preis > 500€ → Anlagevermögen-Prüfung
   - Preis < 50€ → Verbrauchsmaterial-Bias
   - 50€ ≤ Preis ≤ 500€ → Werkzeug/Betriebsmittel

3. **Multi-Position-Context**:
   - Ähnliche Beschreibungen → konsistente Klassifizierung
   - Validierung über alle Rechnungspositionen

4. **Temporal-Pattern-Analysis**:
   - Saisonale Trends (Q4 = Weihnachtsgeschäft)
   - Monatliche Kaufmuster-Analyse

### SQLite-Persistenz

**Tabellen:**
- `supplier_patterns`: Lieferanten-spezifische Klassifizierungs-Pattern
- `position_contexts`: Multi-Position-Konsistenz-Tracking
- `temporal_patterns`: Saisonale/monatliche Trends

**Datenbank-Pfad:** `data/context/classification_context.db`

## Integration

### GeminiDirectProcessor Integration

Die Context-Aware Enhancement ist in `_enhance_with_rag_system` integriert:

```python
# 1. Standard RAG-Enhancement (SKR03Manager)
enhanced_item = skr03_manager.klassifiziere_artikel(artikel_text)

# 2. Context-Aware Intelligence Enhancement
context_enhanced_item = self.context_classifier.enhance_classification_with_context(
    item=enhanced_item,
    supplier_name=item.get("lieferant", ""),
    invoice_context={
        "datum": item.get("datum"),
        "total_amount": item.get("gesamtbetrag", 0),
        "all_items": line_items
    }
)
```

## Konfidenz-Verbesserung

**Context-Boosts werden kombiniert:**
- Supplier-Boost: bis zu +0.2 Konfidenz
- Price-Adjustment: bis zu +0.15 Konfidenz
- Consistency-Boost: bis zu +0.1 Konfidenz
- Seasonal-Adjustment: bis zu +0.05 Konfidenz

**Finale Konfidenz:** `min(1.0, original_confidence + total_boost)`

## Debugging & Analytics

**Metadaten für Analyse:**
```python
enhanced_item["context_analysis"] = {
    "supplier_boost": 0.15,
    "price_adjustment": 0.1,
    "consistency_boost": 0.05,
    "seasonal_adjustment": 0.0,
    "total_context_boost": 0.3,
    "original_confidence": 0.7
}
```

## Performance

- **Initialisierung**: Einmalig beim Processor-Start
- **Enhancement-Zeit**: ~10ms pro Rechungsposition
- **Speicher-Overhead**: ~5MB SQLite-Database
- **Thread-Safe**: Concurrent processing support

## Tests

**Test-Integration:** `test_context_integration.py`
- Standalone ContextAwareClassifier Test
- GeminiDirectProcessor Integration Test
- Context Enhancement Validation

**Ausführung:**
```bash
poetry run python test_context_integration.py
```

## Nächste Schritte

1. **Feedback-Learning**: User-Korrekturen → Pattern-Learning
2. **Advanced-Patterns**: Komplexere Lieferanten-Rules
3. **Machine-Learning**: Automatische Pattern-Discovery
4. **Multi-Language**: Englische Rechnungen support

---

*Version: 1.0 | Datum: 18. August 2025*
*Teil des LLKJJ ML Pipeline v4.0.0*
