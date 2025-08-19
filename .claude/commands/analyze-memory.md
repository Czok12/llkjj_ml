# Speicher-Performance Analyse

Analysiere Speicherverbrauch und Performance der ML-Pipeline.

## Schritte:

1. ResourceManager Singleton Status prüfen
2. spaCy-Modell Memory Usage analysieren
3. ChromaDB Vector Store Größe messen
4. Gemini API Response Caching überprüfen
5. Docling PDF-Processing Memory Patterns
6. Memory Leaks identifizieren
7. Performance-Bottlenecks lokalisieren
8. Optimierungsempfehlungen erstellen

## Verwendung:

```
/analyze-memory
/analyze-memory --detailed
/analyze-memory --profile-run test_pdfs/large_document.pdf
```

## Parameter:

$ARGUMENTS - Optionale Flags oder Test-Datei für Profiling

## Analysebereiche:

- ResourceManager Modell-Caching
- spaCy Pipeline Memory Usage
- ChromaDB Vector Embeddings Storage
- Gemini API Response Caching
- Memory-intensive Operations Tracking

## Ausgabe:

- Memory Usage Report
- Performance Metrics
- Optimization Recommendations
- Memory Leak Detection

**IMPORTANT**: Proaktive Memory-Optimierung für langfristige Pipeline-Stabilität.
