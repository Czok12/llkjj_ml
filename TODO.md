# LLKJJ ML Pipeline - Roadmap & TODO (Konsolidierung 18.08.2025)

## 🚨 **PRIORITÄT 1: PROJEKTSTRUKTUR-KONSOLIDIERUNG (KISS-PRINZIP)**

### **Problem Analyse**
- **Redundante Struktur**: `src/` UND `ml_service/` innerhalb eines Plugin-Pakets
- **Abhängigkeits-Chaos**: `ml_service/` importiert von `src/` (4 Imports gefunden)
- **Gegen KISS-Prinzip**: Zwei Quellcode-Verzeichnisse für ein Paket
- **Wartungslast**: Doppelte Implementierungen und unklare Zuständigkeiten

### **Konsolidierungsplan: Alles → `src/`**

#### **Phase 1: Struktur-Analyse & Backup (2h)**
- [ ] **Abhängigkeits-Mapping**: Vollständige Analyse aller Import-Beziehungen
  - [ ] `grep -r "from ml_service" src/` → Rückwärts-Dependencies prüfen
  - [ ] `grep -r "from src" ml_service/` → Vorwärts-Dependencies dokumentieren
  - [ ] Zirkuläre Imports identifizieren und dokumentieren
- [ ] **Backup erstellen**: `git branch backup-before-consolidation`
- [ ] **Funktionalitäts-Audit**: Was macht `ml_service/` was `src/` nicht kann?

#### **Phase 2: ml_service/ → src/ Migration (4h)**
- [ ] **CLI Migration**: `ml_service/cli.py` → `src/cli/ml_service_cli.py`
  - [ ] Imports auf src/-Struktur umstellen
  - [ ] `__main__.py` Funktionalität nach `src/cli/` verschieben
- [ ] **Processor Migration**: `ml_service/processor.py` → `src/processor/ml_service_processor.py`
  - [ ] 4 src/-Imports auflösen (bereits dokumentiert)
  - [ ] MLSettings Integration mit bestehender src/config.py
  - [ ] ProcessingResult Deduplizierung
- [ ] **Config Migration**: `ml_service/config.py` → `src/config/ml_service_config.py`
  - [ ] MLSettings mit bestehender Config-Klasse mergen
  - [ ] Environment-Variable Handling vereinheitlichen

#### **Phase 3: pyproject.toml für src/-Layout (1h)**
- [ ] **Poetry Konfiguration**:
  ```toml
  packages = [{ include = "llkjj_ml", from = "src" }]
  ```
- [ ] **Entry Points aktualisieren**:
  ```toml
  [project.scripts]
  llkjj-ml = "src.main:main"
  ```
- [ ] **Package-Struktur validieren**: `poetry install` testen

#### **Phase 4: Import-Cleanup & Tests (2h)**
- [ ] **Import-Pfade reparieren**: Alle `from ml_service.` → `from src.`
- [ ] **main.py aktualisieren**: CLI-Integration für konsolidierte Struktur
- [ ] **Tests reparieren**: `ml_service/tests/` → `tests/ml_service/`
- [ ] **Funktionalitäts-Test**: Vollständige Pipeline-Validation

#### **Phase 5: Aufräumen & Dokumentation (1h)**
- [ ] **ml_service/ Verzeichnis entfernen**: Nach erfolgreicher Migration
- [ ] **README.md aktualisieren**: Neue src/-Struktur dokumentieren
- [ ] **API_DOCUMENTATION.py**: Import-Pfade korrigieren
- [ ] **Commit & Tag**: `git tag v4.0.0-consolidated`

### **Erfolgs-Kriterien**
- ✅ Ein einziges Quellcode-Verzeichnis: `src/`
- ✅ Keine Import-Abhängigkeiten zwischen ehemaligen Verzeichnissen
- ✅ Alle Tests bestehen nach Konsolidierung
- ✅ CLI-Funktionalität vollständig erhalten
- ✅ Poetry build/install funktioniert einwandfrei

### **Rollback-Plan**
- **Git Branch**: `backup-before-consolidation` für sofortigen Rollback
- **Validierungs-Skript**: `poetry run python -c "from src import *; print('Import OK')"`

---

## 📋 **PROJEKTSTATUS-ÜBERSICHT** (nach Konsolidierung)

**Aktuelle Version:** 3.0.0 → 4.0.0 (Konsolidierte KISS-Architektur)
```markdown
- [ ] Sprint 1 — Critical Foundation (26h)
  - [ ] Learning Rate Optimization (6h) — `spacy_training/ner_training.py`, `spacy_training/cat_trainer.py`
    - [ ] Integrate ProductionLearningRateScheduler into `BaseTrainer`
    - [ ] Add `_update_learning_rate()` hook and call before each epoch
    - [ ] Update spaCy optimizer learn_rate from scheduler
    - [ ] Run LR demo and compare convergence on Sonepar sample invoices
  - [ ] Deutsche Rechnungs-Augmentation (12h) — `spacy_training/pipeline.py`, `unified_processor.py`
    - [ ] Implement `GermanElektroAugmenter` (synonyms, date/currency formats)
    - [ ] Simulate OCR noise (character swaps, missing umlauts)
    - [ ] Integrate augmentation pipeline with Gemini synthetic data generation
    - [ ] Generate 500+ augmented training samples and save to `data/training/augmented/`
  - [ ] SKR03 Business Metrics (8h) — `spacy_training/ner_training.py`, `spacy_training/cat_trainer.py`
    - [ ] Extend `TrainingMetrics` with `invoice_completeness`, `skr03_accuracy`, `entity_coverage`
    - [ ] Implement invoice completeness & date-consistency checks
    - [ ] Add per-entity confusion matrix output and persistence
    - [ ] Wire metrics into existing logging and `save_training_metrics()`

- [ ] Sprint 2 — High-Value Improvements (22h)
  - [ ] Smart Checkpointing & Early Stopping (4h) — `src/trainer.py`
    - [ ] Implement checkpoint save/resume in `BaseTrainer`
    - [ ] Add early stopping based on validation loss patience
    - [ ] Add CLI flag to resume from checkpoint
  - [ ] Stratified Cross-Validation (8h) — `spacy_training/pipeline.py`
    - [ ] Implement stratified K-Fold wrapper for NER/TextCat
    - [ ] Produce aggregated CV reports (mean/std of metrics)
    - [ ] Integrate CV into `TrainingPipeline.run_full_pipeline()` optional mode
  - [ ] Gemini Training Enhancement (10h) — `unified_processor.py`, training pipeline
    - [ ] Add Gemini prompts to synthesize annotated invoice examples
    - [ ] Implement quality checks and automatic correction suggestions
    - [ ] Integrate synthesized data into augmentation pipeline with provenance

- [ ] Sprint 3 — Performance Boost (18h)
  - [ ] Transformer-Integration (16h) — `spacy_training/ner_training.py`
    - [ ] Evaluate `spacy-transformers` with a German BERT model
    - [ ] Implement optional transformer-backed pipeline (config toggle)
    - [ ] Benchmark accuracy vs inference speed and memory
  - [ ] Hyperparameter Automation (2h setup) — `spacy_training/pipeline.py`
    - [ ] Add minimal Optuna integration for LR & batch size search
    - [ ] Persist best-trial params and integrate into training config

- [ ] Validation & Deliverables
  - [ ] After Sprint 1: compare baseline vs new models (F1, time)
  - [ ] After Sprint 2: validate resume/checkpoint behavior and CV results
  - [ ] After Sprint 3: run transformer benchmark and decide rollout

- [ ] Immediate / Pre-work (do these before Sprint 1)
  - [ ] Backup current models to `data/models_backup/` (mandatory)
  - [ ] Run `poetry run python production_lr_scheduler.py` to verify scheduler
  - [ ] Prepare a small Sonepar sample set in `test_pdfs/` for quick iteration

- [ ] Validation commands (examples)
  - [ ] `poetry run python main.py process test_pdfs/Sonepar_test3.pdf --validate-metrics`
  - [ ] `poetry run python demo_enhanced_training.py --compare-baseline`

- [ ] Risk & Rollback
  - [ ] Each sprint must include a rollback test (load previous model + metrics)
  - [ ] Store checkpoints and tags in `data/models_backup/` with timestamped folders

- [ ] Notes
  - Estimated total effort (Sprints 1-3): ~66h (focused) — can be reduced to ~48h if Transformer step deferred
  - Keep new features opt-in via config flags to preserve backward compatibility
  - Prefer incremental PRs per subtask for easy review and rollback

```
```markdown
