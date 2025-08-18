# LLKJJ ML Pipeline - Roadmap & TODO (Gemini-Strategie 18.08.2025)

## 🎯 **PRIORITÄT 1: GEMINI-PIPELINE PRODUKTIONSREIF MACHEN (STRATEGIC PRIORITY)**

### **Strategische Vision: Phase 1 → 2 Transition**
- **Phase 1 (JETZT)**: Gemini AI als produktive Intelligence-Engine
- **Phase 2 (SPÄTER)**: Nahtloser Übergang zu lokaler spaCy/RAG-Autonomie
- **Kernziel**: Jede verarbeitete Rechnung = Trainingsdaten für zukünftige Unabhängigkeit

### **Sofort-Umsetzung (A-Priorität)**

#### **A1: Pydantic-Validierung für Gemini-Antworten (KRITISCH)**
- [x] **Schema-Definition**: `src/models/gemini_schemas.py` erstellen ✅
  - [x] `GeminiInvoiceHeader(BaseModel)` - Rechnungskopf-Schema ✅
  - [x] `GeminiLineItem(BaseModel)` - Rechnungspositions-Schema ✅
  - [x] `GeminiExtractionResult(BaseModel)` - Vollständiges Response-Schema ✅
- [x] **Integration in GeminiDirectProcessor**: Sofortige Validierung nach API-Response ✅
  - [x] `validated_data = GeminiExtractionResult(**json.loads(response_text))` ✅
  - [x] Fehlerbehandlung für ungültige Gemini-Responses ✅
- [x] **Qualitätssicherung**: Nur validierte Daten in ProcessingResult ✅

#### **A2: Trainingsdaten-Persistierung (DATENSCHATZ)**
- [x] **spaCy-Training-Export**: Nach jeder erfolgreichen Verarbeitung ✅
  - [x] JSONL-Format: `data/training/gemini_spacy_annotations.jsonl` ✅
  - [x] Annotationen + raw_text für zukünftiges NER/TextCat-Training ✅
- [x] **RAG-System-Population**: ChromaDB mit Gemini-Klassifizierungen ✅
  - [x] Jede Position → ChromaDB-Dokument mit Metadatum `"source": "gemini_validated"` ✅
  - [x] Embedding-Vektor für Ähnlichkeitssuche ✅
- [x] **Audit-Trail**: GoBD-konforme Speicherung in `logs/audit_gemini.jsonl` ✅

#### **A3: Performance-Optimierung**
- [x] **Async Gemini-Processing**: `AsyncGeminiDirectProcessor` ✅
  - [x] `asyncio.gather()` für Batch-Verarbeitung ✅
  - [x] `asyncio.Semaphore(3)` für API-Rate-Limiting ✅
- [x] **PDF-Hash-Caching**: SQLite-Cache gegen Duplikate ✅
  - [x] SHA256-Hash vor API-Call ✅
  - [x] Cache-Hit → sofortiges Ergebnis (0ms statt 5000ms) ✅

### **Architektur-Vorbereitung für Phase 2 (B-Priorität)**

#### **B1: Strategy-Pattern für nahtlose Transition**
- [x] **Abstrakte ProcessingStrategy**: Interface für alle Engines ✅
- [x] **GeminiStrategy**: Aktueller GeminiDirectProcessor ✅
- [x] **UnifiedProcessor**: Engine-Auswahl zur Laufzeit ✅
- [x] **Vorbereitung SpacyRagStrategy**: Platzhalter für Phase 2 ✅

#### **B2: spaCy-Training-Pipeline**
- [ ] **Automated Training**: Trigger bei X gesammelten Beispielen
- [ ] **Model-Versioning**: Inkrementelle Verbesserung lokaler Modelle
- [ ] **Performance-Benchmarking**: Gemini vs. spaCy Genauigkeitsvergleich

#### **B3: RAG-System Optimierung (INTELLIGENTES GEDÄCHTNIS)**
**Strategische Bedeutung**: Das RAG-System ist das Herzstück der zukünftigen autonomen Pipeline - von einfacher Ähnlichkeitssuche zum intelligenten Langzeitgedächtnis.

##### **Phase 1: Robuste Dateneinspeisung (Sofort)**
- [ ] **Validierungs-Status in ChromaDB-Metadaten**
  - [ ] `validation_status`: `"ai_suggested"` | `"user_confirmed"` | `"user_corrected"` | `"system_flagged"`
  - [ ] Grundlage für Qualitätsgewichtung und Selbstkorrektur
- [ ] **Feedback-Loop Implementation**
  - [ ] Backend-Endpunkt `/feedback` für Benutzerkorrekturen
  - [ ] ChromaDB-Update mit korrigierten Klassifizierungen
  - [ ] Konfidenz auf 1.0 setzen für `user_corrected`-Einträge
- [ ] **Kontext-Anreicherung der Vektor-Dokumente**
  - [ ] Von: `"Lieferant: X | Artikel: Y"`
  - [ ] Zu: `"Artikel: Y, Menge: Z, Preis: N EUR. Lieferant: X (Typ). Kategorie: K."`
  - [ ] Embedding versteht Kontext: Einzelwerkzeug vs. Verbrauchsmaterial

##### **Phase 2: Intelligenter Abruf (Hybrid Search)**
- [ ] **Mehrstufige Suche mit Metadaten-Filterung**
  - [ ] ChromaDB `where`-Filter für Lieferanten-spezifische Suche
  - [ ] Gewichtung: `user_corrected/confirmed` Einträge × 1.5 Faktor
  - [ ] Vorfilterung reduziert "false positives" drastisch
- [ ] **Dynamische Ähnlichkeitsschwelle**
  - [ ] Hohe Schwelle (0.7) bei vielen validierten Lieferanten-Daten
  - [ ] Niedrige Schwelle (0.5) bei unbekannten Artikeln
  - [ ] Adaptive Präzision vs. Recall-Balance

##### **Phase 3: Explainable AI & Selbstkorrektur**
- [ ] **Erweiterte Begründungs-Engine**
  - [ ] XAI-Reasoning: "Vorschlag 3400. Regel-Konfidenz 0.8 + 3 ähnliche bestätigte Sonepar-Buchungen (Ø 0.85)"
  - [ ] Frontend-Integration für Benutzervertrauen
- [ ] **Proaktives Inkonsistenz-Flagging**
  - [ ] Top-3 RAG-Treffer → 3 verschiedene SKR03-Konten = `system_flagged`
  - [ ] Automatische Konfidenz-Reduktion bei Mehrdeutigkeit
  - [ ] "Warnung: Ähnliche Artikel unterschiedlich kontiert. Bitte prüfen."

**Strategischer Nutzen**:
- ✅ Kontinuierliches Lernen aus Benutzerfeedback
- ✅ Selbstheilende Datenbasis ("vergiftete" Daten werden korrigiert)
- ✅ Kontextbewusste Klassifizierung (Anlagevermögen vs. Verbrauchsmaterial)
- ✅ Vorbereitung für Phase 2: Lokales "intelligentes Gedächtnis" ohne Gemini-Abhängigkeit

---

## 🏗️ **PRIORITÄT 2: STRUKTUR-KONSOLIDIERUNG (KANN WARTEN)**

### **Problem Analyse (Ursprünglicher Plan)**
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
