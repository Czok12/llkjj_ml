# Repository Deep Analysis für ML-Kontext

Führe eine umfassende Repository-Analyse durch, um Claude mit vollständigem ML-Pipeline-Kontext zu versorgen.

## Schritte:

1. **Architektur-Überblick**:

   - Analysiere src/ Struktur und ML-Komponenten
   - Verstehe GeminiDirectProcessor Workflow
   - Erkenne ResourceManager Patterns
   - Identifiziere SKR03Manager Integration

2. **Dependency-Analyse**:

   - Poetry Dependencies und ML-Frameworks
   - spaCy, ChromaDB, Docling Versionen
   - Gemini API Integration
   - Test-Framework Setup

3. **Data Pipeline Mapping**:

   - Input: test_pdfs/ Struktur
   - Processing: ML-Pipeline Flow
   - Storage: ChromaDB Vectors, SKR03 Data
   - Output: Structured Invoice Data

4. **Configuration Analysis**:

   - pyproject.toml ML-spezifische Settings
   - VS Code Workspace Optimizations
   - Environment Variables und Secrets

5. **Testing Strategy**:
   - Unit Tests für ML-Komponenten
   - Integration Tests für Pipeline
   - Performance Tests und Benchmarks

## Verwendung:

```
/primer
/primer --detailed
/primer --focus=ml-pipeline
```

## Parameter:

$ARGUMENTS - Optionale Fokus-Bereiche

## Ausgabe:

- Vollständiger Architektur-Überblick
- ML-Pipeline Flow-Diagramm
- Dependency-Matrix
- Testing-Status
- Nächste Entwicklungsschritte

**IMPORTANT**: Erstelle vollständigen Kontext für effiziente nachfolgende ML-Entwicklung.
