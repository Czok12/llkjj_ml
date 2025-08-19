# Claude Code Best Practices für LLKJJ ML Pipeline 2025

## Übersicht

Dieses Dokument sammelt die neuesten Best Practices für Claude Code CLI Agent und AI-Coding-Unterstützung, speziell optimiert für das LLKJJ ML-Projekt (Deutsche Elektrohandwerk-Buchhaltungs-Pipeline).

## 1. Grundlegende Setup-Optimierung

### 1.1 CLAUDE.md Konfiguration (Bereits implementiert)

```bash
# Prüfe aktuelle CLAUDE.md
cat CLAUDE.md
```

**Best Practices für CLAUDE.md:**

- ✅ Projektspezifische Kontext-Regeln
- ✅ Poetry-Befehle dokumentiert
- ✅ Testing-Workflow definiert
- ✅ Deutsche ML-Domänen-Spezifika

**Erweiterte CLAUDE.md Optimierungen:**

```markdown
# Power Keywords für bessere Claude-Performance

IMPORTANT: Immer Poetry für alle Python-Befehle verwenden
Proactively: SKR03-Klassifizierungsqualität überwachen
Ultra-think: Für komplexe Architekturentscheidungen
```

### 1.2 Permission Management

```bash
# Claude Code Permission Setup
claude
/permissions
```

**Empfohlene Permissions für LLKJJ ML:**

```json
{
  "allowedTools": [
    "Edit",
    "Read",
    "Write",
    "Bash(poetry:*)",
    "Bash(git:*)",
    "Bash(pytest:*)",
    "Bash(mypy:*)",
    "Bash(ruff:*)",
    "Bash(black:*)",
    "mcp__*"
  ]
}
```

## 2. ML-spezifische MCP Server Integration

### 2.1 Essential MCP Servers für ML-Pipeline

**Context7 MCP Server** (Höchste Priorität):

```bash
# Installation
claude mcp add context7 -- npx -y @context7/mcp-server
```

**Nutzen:** Up-to-date Dokumentation für spaCy, Docling, ChromaDB, PyTorch

**Filesystem MCP Server**:

```bash
# Installation
claude mcp add filesystem -- npx -y @anthropic-ai/mcp-server-filesystem
```

**Nutzen:** Sichere Dateisystem-Operationen in data/, src/, tests/

**Python Analysis MCP**:

```bash
# Installation (falls verfügbar)
claude mcp add python-analysis -- npx -y python-analysis-mcp
```

**Nutzen:** Code-Analyse, Dependency-Tracking, Type-Safety

### 2.2 MCP Konfiguration für LLKJJ ML

```json
// .claude/settings.local.json
{
  "mcpServers": {
    "context7": {
      "command": "npx",
      "args": ["-y", "@context7/mcp-server"]
    },
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@anthropic-ai/mcp-server-filesystem",
        "/Users/czok/Skripte/llkjj_v0.1/llkjj_ml"
      ]
    }
  }
}
```

## 3. Custom Slash Commands für ML-Workflows

### 3.1 ML Pipeline Commands

**PDF Processing Test**:

```bash
# .claude/commands/test-pdf-pipeline.md
```

```markdown
# Test PDF Processing Pipeline

Führe einen vollständigen Test der PDF-Verarbeitungs-Pipeline durch:

1. Verwende Test-PDF aus test_pdfs/
2. Führe GeminiDirectProcessor aus
3. Validiere SKR03-Klassifizierung
4. Prüfe ChromaDB-Integration
5. Erstelle Qualitätsbericht

**Verwendung**: /test-pdf-pipeline test_pdfs/Sonepar_test3.pdf
```

**SKR03 Validation**:

```bash
# .claude/commands/validate-skr03.md
```

```markdown
# SKR03 Klassifizierung Validieren

Validiere SKR03-Kontierung für Elektrohandwerk:

1. Lese aktuelle skr03_regeln.yaml
2. Teste gegen neue Rechnungsdaten
3. Berechne Konfidenz-Scores
4. Identifiziere Verbesserungsmöglichkeiten

**Verwendung**: /validate-skr03 $ARGUMENTS
```

### 3.2 Development Commands

**Memory Analysis**:

```bash
# .claude/commands/analyze-memory.md
```

```markdown
# Speicher-Performance Analyse

Analysiere Speicherverbrauch der ML-Pipeline:

1. ResourceManager Singleton Status
2. spaCy-Modell Memory Usage
3. ChromaDB Vector Store Size
4. Gemini API Response Caching
5. Optimierungsempfehlungen

**Verwendung**: /analyze-memory
```

## 4. Subagents für ML-Spezialisierung

### 4.1 ML Quality Assurance Agent

```yaml
# .claude/agents/ml-quality-assurance.md
---
name: ml-quality-assurance
description: "ML Quality Specialist. Proactively überwacht SKR03-Klassifizierungsgenauigkeit und Pipeline-Performance."
tools: Read, Grep, Bash(poetry run pytest:*)
---

Du bist ein ML-Qualitätsspezialist für die LLKJJ-Pipeline.

## Hauptverantwortungen:
1. SKR03-Klassifizierungsgenauigkeit >92% sicherstellen
2. Performance-Metriken überwachen (<30s pro Dokument)
3. Speicher-Lecks in ResourceManager identifizieren
4. Test-Coverage für ML-Komponenten validieren

## Tools:
- pytest für automatisierte Tests
- Grep für Log-Analyse
- Performance-Benchmarking-Scripts

Sei proaktiv bei der Qualitätssicherung!
```

### 4.2 German NLP Specialist

```yaml
# .claude/agents/german-nlp-specialist.md
---
name: german-nlp-specialist
description: "Deutsche NLP-Spezialist. Proactively optimiert spaCy-Modelle und deutsche Textverarbeitung."
tools: Read, Edit, Bash(poetry run python:*)
---

Du bist ein Experte für deutsche NLP-Verarbeitung.

## Hauptverantwortungen:
1. spaCy de_core_news_sm Optimierung
2. Deutsche Elektrotechnik-Terminologie
3. Named Entity Recognition für Rechnungen
4. Fehleranalyse bei deutscher Textextraktion

## Domänen-Expertise:
- Elektrohandwerk-Begriffe
- Deutsche Rechnungsformate
- SKR03-Buchhaltungsklassifikation
```

## 5. Erweiterte Workflows für ML-Development

### 5.1 PRP Framework für ML-Features

**Template für ML-Feature-Development**:

```markdown
# INITIAL.md Template für ML-Features

## FEATURE

[Neue ML-Funktionalität beschreiben]

## EXAMPLES

- Ähnliche Pipeline-Komponente: `src/processor.py`
- ML-Modell-Pattern: `src/skr03_manager.py`
- Test-Pattern: `tests/test_gemini_processor.py`
- Config-Pattern: `src/config.py`

## DOCUMENTATION

- spaCy Docs: https://spacy.io/usage/
- ChromaDB Docs: https://docs.trychroma.com/
- Docling Docs: https://ds4sd.github.io/docling/

## OTHER CONSIDERATIONS

- Deutsche Sprachoptimierung
- Memory-Performance (ResourceManager)
- SKR03-Kompatibilität
- Gemini API Rate Limits
```

### 5.2 Safe YOLO Mode für ML-Experimente

**Dev Container Setup** (bereits konfiguriert):

```bash
# Sichere ML-Experimente
code --install-extension ms-vscode-remote.remote-containers
# F1 -> "Dev Containers: Reopen in Container"
# Im Container:
claude --dangerously-skip-permissions
```

**Anwendungsfälle:**

- Neue ML-Modell-Experimente
- Batch-Processing von Test-PDFs
- ChromaDB Schema-Migrationen
- Performance-Optimierung Tests

## 6. Git Worktrees für Parallel-Development

### 6.1 ML-Feature Parallel Development

```bash
# Parallele ML-Feature-Entwicklung
git worktree add ../llkjj_ml_gemini_v2 feature/gemini-v2
git worktree add ../llkjj_ml_spacy_optimization feature/spacy-optimization
git worktree add ../llkjj_ml_chromadb_migration feature/chromadb-migration

# Separate Claude-Instanzen
cd ../llkjj_ml_gemini_v2 && claude &
cd ../llkjj_ml_spacy_optimization && claude &
cd ../llkjj_ml_chromadb_migration && claude &
```

**Ideal für:**

- A/B-Testing verschiedener ML-Approaches
- Parallele Feature-Entwicklung
- Performance-Optimierung ohne Main-Branch-Störung

## 7. Performance-optimierte Workflows

### 7.1 Context Management

```bash
# Zwischen ML-Tasks Context leeren
/clear
```

**Wann Context leeren:**

- Nach PDF-Batch-Processing
- Zwischen verschiedenen ML-Experimenten
- Nach umfangreichen Debugging-Sessions
- Vor Performance-Tests

### 7.2 Iterative ML-Development

**Test-Driven ML Development**:

```bash
# 1. ML-Tests schreiben
claude "Schreibe Tests für neue SKR03-Klassifizierungsregel"

# 2. Tests laufen lassen (sollten fehlschlagen)
poetry run pytest tests/test_new_classification.py -v

# 3. Implementation bis Tests bestehen
claude "Implementiere SKR03-Regel bis alle Tests bestehen"

# 4. Commit
git add . && git commit -m "Add new SKR03 classification rule"
```

## 8. ML-spezifische GitHub Integration

### 8.1 Automated ML Issue Resolution

```bash
# GitHub CLI Setup für ML-Issues
gh auth login

# Custom Command für ML-Bug-Fixes
/fix-ml-issue 123
```

**ML Issue Types:**

- SKR03-Klassifizierungsfehler
- Performance-Probleme
- Memory-Leaks in ResourceManager
- spaCy-Modell-Updates

### 8.2 ML Performance Monitoring

```bash
# Performance-Report erstellen
claude "Erstelle Performance-Report für letzte 10 PDF-Verarbeitungen"

# Automatisches Issue bei Performance-Degradation
claude "Erstelle GitHub Issue wenn Verarbeitungszeit >30s"
```

## 9. Headless Mode für ML-Automation

### 9.1 Batch PDF Processing

```bash
# Headless PDF-Batch-Processing
claude -p "Verarbeite alle PDFs in test_pdfs/ und erstelle Qualitätsbericht" \
  --allowedTools Edit,Bash(poetry:*) \
  --output-format stream-json
```

### 9.2 ML Model Training Automation

```bash
# Automatisches spaCy-Model-Training
claude -p "Trainiere spaCy-Modell mit neuen Annotationen und validiere Performance" \
  --allowedTools Edit,Bash(poetry:*),Write
```

## 10. LLKJJ-spezifische Optimierungen

### 10.1 Deutsche Elektrohandwerk-Spezialisierung

**Context Engineering für deutsche Elektrobranche:**

```markdown
# Domänen-spezifische Prompts

"Analysiere diese Rechnung unter Berücksichtigung deutscher Elektrohandwerk-Standards"
"Klassifiziere nach SKR03 für deutsche UG (haftungsbeschränkt)"
"Verwende deutsche Elektrotechnik-Terminologie"
```

### 10.2 SKR03 Validation Workflows

```bash
# SKR03-Konsistenz prüfen
claude "Validiere alle SKR03-Klassifizierungen gegen aktuelle Regeln"

# SKR03-Regel-Updates
claude "Update SKR03-Regeln für neue Elektrokomponenten"
```

## 11. Monitoring und Debugging

### 11.1 ML Pipeline Debugging

```bash
# Debug-Modus für ML-Pipeline
export LLKJJ_DEBUG=true
claude "Analysiere PDF mit vollständigem Debug-Output"
```

### 11.2 Performance Profiling

```bash
# Memory Profiling
claude "Führe Memory-Profiling der ResourceManager-Klasse durch"

# Timing Analysis
claude "Analysiere Timing-Bottlenecks in GeminiDirectProcessor"
```

## 12. Enterprise Best Practices (2025)

### 12.1 Security & Compliance

```bash
# Security Audit mit Claude
claude "Führe Security-Audit der Gemini API Integration durch"

# Compliance Check
claude "Prüfe DSGVO-Konformität der PDF-Verarbeitung"
```

### 12.2 Documentation Standards

```bash
# Auto-Documentation
claude "Aktualisiere API-Dokumentation basierend auf Code-Änderungen"

# Architecture Decision Records
claude "Erstelle ADR für neue ML-Architektur-Entscheidung"
```

## 13. Tipps für maximale Effizienz

### 13.1 Spezifische Instructions

❌ **Schlecht**: "Teste das ML-System"
✅ **Gut**: "Teste GeminiDirectProcessor mit Sonepar_test3.pdf, validiere SKR03-Klassifizierung 4400, prüfe Konfidenz-Score >0.85"

### 13.2 Visual Context für ML

```bash
# Screenshots für UI-Development
claude "Implementiere Dashboard-UI basierend auf diesem Mock" [drag & drop image]

# Diagramme für Architektur
claude "Optimiere ML-Pipeline basierend auf diesem Architektur-Diagramm" [paste architecture diagram]
```

### 13.3 Error Recovery

```bash
# Bei ML-Pipeline-Fehlern
ESC # Unterbreche Claude
"Analysiere den Fehler und implementiere robuste Fehlerbehandlung"

# History Navigation
ESC ESC # Springe zurück und editiere vorherigen Prompt
```

## 14. Qualitätssicherung

### 14.1 Code Quality Gates

```bash
# Pre-Commit Quality Check
poetry run mypy src/
poetry run ruff check src/
poetry run black --check src/
poetry run pytest tests/ --cov=src/ --cov-report=html
```

### 14.2 ML Quality Metrics

```bash
# SKR03 Accuracy Check
claude "Berechne SKR03-Klassifizierungsgenauigkeit der letzten 100 Rechnungen"

# Performance Benchmarks
claude "Führe Performance-Benchmark gegen Baseline durch"
```

---

## Zusammenfassung

Diese Best Practices ermöglichen es dir, Claude Code optimal für die LLKJJ ML-Pipeline zu nutzen:

1. **Setup**: CLAUDE.md, Permissions, MCP Servers konfiguriert
2. **Workflows**: Custom Commands, Subagents, PRP Framework
3. **Development**: Test-Driven, Parallel Development, Safe Experimentation
4. **Quality**: Automated Testing, Performance Monitoring, Code Quality
5. **Spezialisierung**: Deutsche Elektrobranche, SKR03, ML-Pipeline-spezifisch

**Nächste Schritte:**

1. MCP Servers installieren (`Context7`, `Filesystem`)
2. Custom Commands implementieren (`/test-pdf-pipeline`, `/validate-skr03`)
3. ML-Subagents konfigurieren
4. Parallel Worktrees für Feature-Development einrichten

Mit dieser Konfiguration wird Claude Code zu einem mächtigen Werkzeug für die autonome Entwicklung und Optimierung der LLKJJ ML-Pipeline!
