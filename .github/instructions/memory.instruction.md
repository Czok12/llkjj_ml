---
applyTo: '**'
---

# User Memory

## User Preferences
- Programming languages: Python
- Code style preferences: Use type annotations, follow PEP 484, fix mypy errors
- Development environment: VS Code, macOS, zsh
- Communication style: Concise, step-by-step, progress updates

## Project Context
- Current project type: Python ML pipeline and extraction toolkit
- Tech stack: Python, spaCy, custom extractors, mypy for type checking
- Architecture patterns: Modular, src/ structure, config-driven
- Key requirements: Type safety, mypy compliance, maintainability

## Coding Patterns
- Use explicit type annotations
- Fix all mypy errors
- Prefer PEP 484 and Context7 best practices
- Document fixes in memory

## Context7 Research History
- None yet (will update after research)

## Conversation History
- Task: Fix mypy errors and missing stubs in 15 files (113 errors)
- Plan: Research, analyze, fix incrementally, test, update memory

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

## Projektkontext Vergleich (automatisch protokolliert)
- Datum: 2025-08-17
- Vergleich von: `Projektkontext01.md` vs. `Projektkontext02.md`
- Ergebnis: Empfehlung zur Verwendung und Platzierung der Dateien im Repo

Kurzfassung:
- `Projektkontext02.md` ist besser geeignet als übergeordnetes Vision/Manifest für das Core-Backend und als generelles Projektmanifest. Es ist umfassend, strukturiert, und beschreibt Architekturprinzipien, Komponenten, Workflows und Governance (Sicherheit, Tests, CI/CD) klar.
- `Projektkontext01.md` ist besser geeignet als konkretes, AI-/Entwickler-orientiertes Manifest für das ML-Plugin (Workspace 2). Es enthält präzisere technische Vorgaben, Tool- und Kommandohinweise (Poetry), ML-Stack-Details und direkt anwendbare Best-Practices.

Empfehlungen:
1. Verwende `Projektkontext02.md` als Haupt-Vision/Manifest im Repository-Root oder in `docs/` (z. B. `docs/vision_manifest.md`).
2. Platziere `Projektkontext01.md` innerhalb des ML-Plugin-Workspaces (z. B. `src/` oder `ml_service/README_ml_manifest.md`) als detailliertes Entwickler- und AI-Kontextdokument.
3. Ergänze in der gewählten Haupt-Manifestdatei eine maschinenlesbare YAML-Frontmatter (keys: `title`, `version`, `last_updated`, `scope`, `audience`) zur einfachen Erfassung durch AI-Assistenten.
4. Optional: Erzeuge eine kurze, synoptische `manifest_summary.json` mit Kernfeldern für schnelle maschinelle Nutzung.

Diese Empfehlung wurde automatisch erstellt und im Memory protokolliert.

## Notes
- Start with type annotation and stub errors, then address assignment and attribute errors
- Update memory after each major step
