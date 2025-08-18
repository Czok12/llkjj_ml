---
applyTo: '**'
---

# User Memory

## User Preferences
 - Programming languages: Python
 - Code style preferences: Use type annotations, follow PEP 484, fix mypy errors
 - Development environment: VS Code, macOS, zsh
 - Communication style: Concise, step-by-step, progress updates
 - Preference: Use plain `python` instead of `poetry run`, keep poetry venv outside workspace

## Project Context
- Current project type: LLKJJ ML-Pipeline (Workspace 2) - Teil eines 4-Workspace-Systems
- Tech stack: Python 3.10+, spaCy, ChromaDB, Docling 2.44.0, Google Gemini 2.5 Pro, mypy, Poetry
- Architecture patterns: Plugin-Architektur, Gemini-First-Pipeline, RAG-System (ChromaDB)
- Key requirements: >92% SKR03-Klassifizierungsgenauigkeit, Type safety, Sicherheit bei Finanzdaten
- Primary mission: PDF → strukturierte, SKR03-klassifizierte Rechnungsdaten
- Strategic approach: 2-Phasen-Strategie (Phase 1: Gemini-First, Phase 2: Lokale autonome KI)

## Coding Patterns
- Use explicit type annotations (Python 3.10+ syntax: dict[str, Any], str | int)
- Fix all mypy errors with --strict compliance
- Follow KISS-Prinzip and Single Responsibility Principle
- Prefer PEP 484 and Context7 best practices
- Use Pydantic for data validation and schemas
- Deutsche Optimierung für Elektrotechnik-Domäne
- Document fixes in memory

## Architecture Knowledge
- 4-Workspace-System: Backend (Core), ML-Pipeline (Workspace 2), E-Invoice, Export
- Workspace-Isolation: Strikte Trennung, keine direkten Backend-Abhängigkeiten
- Einzige Schnittstelle: ProcessingResult-Schema als Datenvertrag
- Kern-Komponenten: GeminiDirectProcessor, ResourceManager (Singleton), QualityAssessor, SKR03Manager
- Autonome Persistenz: ChromaDB in data/vectors (kein PostgreSQL-Zugriff)
- Sicherheit: JWT + Argon2, SecurityAuditor (Bandit/Safety), AES-256 für API-Keys

## Context7 Research History
- None yet (will update after research)

## Conversation History
- Task: Fix mypy errors and missing stubs in 15 files (113 errors)
- Plan: Research, analyze, fix incrementally, test, update memory
- Task (2025-08-18): Dokumentations-Update basierend auf neuer README.md
- Erfolg: Copilot-instructions.md und memory.instruction.md aktualisiert
- Neue Erkenntnisse: 4-Workspace-System, Gemini-First-Pipeline, erweiterte Sicherheitsrichtlinien

## Analysis-Log: 2025-08-17

- Task: Review repository and compare `TODO.md` against codebase state.
- Files inspected: `TODO.md`, `README.md`, `main.py`, `pyproject.toml`,
	`src/pipeline/processor.py`, `src/processing/modular_processor.py`,
	`src/trainer.py`, `src/security/manager.py`, `src/security/auditor.py`,
	`src/database/simple_manager.py`.
- Key findings (concise):
	- Most Phase 1-5 features named in `TODO.md` have corresponding implementations
		in `src/` (UnifiedProcessor, ModularProcessor, TrainingService, APIKeyManager,
		SimpleDatabaseManager, security auditor exist).
	- `pyproject.toml` lists expected dependencies (docling, chromadb, spacy, pymupdf,
		google-genai) and dev tools (mypy, ruff, pytest).
	- Several high-level `TODO.md` claims are optimistic (e.g., "All tests pass").
		I did not run tests or linters yet; verification required.
	- Phase 7/8 items (FastAPI, CI/CD, Docker, AutoML, Multi-Tenant) are still
		backlog items and not implemented in the codebase.

- Next recommended actions:
	1. Run `poetry run pytest` and `poetry run mypy .` to verify claimed green status.
	2. Update `TODO.md` to mark only verified items as done and move optimistic
		 claims into "requires verification".
	3. Create small verification CI job to run tests and linters automatically.

Recorded: 2025-08-17 by automated assistant

## Dokumentations-Update: 2025-08-18

**Task**: Aktualisierung der copilot-instructions.md und memory.instruction.md basierend auf neuer README.md Dokumentation

**Wichtige Änderungen**:
- **4-Workspace-System**: Vollständige Architektur mit Backend (Core), ML-Pipeline (Workspace 2), E-Invoice, Export
- **Gemini-First-Pipeline**: 2-Phasen-Strategie mit detailliertem Workflow (7 Schritte)
- **Kern-Komponenten**: GeminiDirectProcessor, ResourceManager (Singleton), QualityAssessor, SKR03Manager
- **Sicherheitsrichtlinien**: JWT + Argon2, SecurityAuditor, AES-256-Verschlüsselung für API-Keys
- **Design-Entscheidungen**: ProcessingResult als stabile Schnittstelle, ChromaDB für RAG-System
- **Strategische Ausrichtung**: >92% SKR03-Klassifizierungsgenauigkeit, Deutsche Elektrotechnik-Optimierung

**Erfolgreich aktualisiert**:
1. ✅ copilot-instructions.md - Erweitert um vollständige Architektur und neue Komponenten
2. ✅ memory.instruction.md - Aktualisiert mit Projektkontext und Architektur-Wissen

Recorded: 2025-08-18 by automated assistant

## Notes
- Start with type annotation and stub errors, then address assignment and attribute errors
- Update memory after each major step
