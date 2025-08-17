
# Das LLKJJ-Projekt: Manifest, Vision und technische Blaupause

---
title: "LLKJJ Manifest"
version: "1.0"
last_updated: "2025-08-17"
scope: "repository"
audience: ["core","ml","devops","ai-agent"]
entry_points:
   ml_pipeline: "src/pipeline/dual_pipeline.py"
   cli: "main.py"
   training: "src/trainer.py"
policies:
   poetry_cli: "poetry run ... is required"
   language: "de"
   type_safety: "mypy --strict"
---

**Version 1.0 (Stand: August 2025)**

## 1. Vision und Leitbild (Das Manifest)

**LLKJJ (Lernende Laufende Komfortable Kompetente Judikative)** ist eine moderne, KI-gestützte und sichere Buchhaltungs- und Dokumentenmanagement-Plattform, die als persönliche, überlegene Alternative zu kommerziellen Lösungen wie Buchhaltungsbutler.de, Lexware und Sevdesk konzipiert ist.

**Unser Ziel** ist es, den gesamten administrativen Overhead für kleine deutsche Handwerksbetriebe – initial mit Fokus auf die **Elektrotechnik-Branche** – durch intelligente Automatisierung zu eliminieren. Das System agiert als digitaler Finanzassistent, der nicht nur Daten verarbeitet, sondern den Geschäftskontext versteht, proaktiv lernt und Compliance-Anforderungen (GoBD, XRechnung) von Grund auf erfüllt.

Wir bauen kein weiteres Abrechnungstool. Wir bauen eine **autonome Buchhaltungs-Engine**, die durch Präzision, Sicherheit und eine durchdachte, wartbare Architektur besticht.

## 2. Architektonische Grundpfeiler

Die Entwicklung von LLKJJ folgt konsequent vier Kernprinzipien, die sich in jedem Modul widerspiegeln:

1. **🧠 KI-First und Daten-Getrieben:**
   Das Herzstück ist eine hochentwickelte, lernfähige KI-Pipeline. Wir verlassen uns nicht auf starre Regeln, sondern auf eine Kombination aus modernsten ML-Techniken:

   * **State-of-the-Art Extraktion:** Einsatz von `Docling` mit optimierten Engines (`PaddleOCR/Tesseract`) und `TableFormer` zur präzisen Erfassung von Rechnungsdaten (`docling_processor.py`).
   * **Intelligente Klassifizierung:** Eine Multi-Layer-Strategie aus regelbasiertem Matching (`skr03_manager.py`), Vektorsuche mit **Retrieval-Augmented Generation (RAG)** über `ChromaDB` (`processor.py`, `classifier.py`) und systematischem **Feature Engineering** (`ml_integration.py`), um die SKR03-Kontierungsgenauigkeit von 85% auf über 92% zu steigern.
   * **Kontinuierliches Lernen:** Feedback-Schleifen (`fast_rule_booking_service.py`) ermöglichen es dem System, aus Benutzerkorrekturen zu lernen und sich ständig zu verbessern.
2. **🔌 Modulare Plugin-Architektur (KISS-Prinzip):**
   Das System ist als Kern-Backend mit andockbaren, in sich geschlossenen Plugins konzipiert. Dies gewährleistet maximale Flexibilität, Wartbarkeit und unabhängige Entwicklung.

   * **ML-Pipeline als Blackbox:** Die gesamte ML-Verarbeitung ist als eigenständiges Plugin (`llkjj_ml_plugin`) gekapselt, das eine klar definierte Schnittstelle (`package_api.py`, `ProcessingResult`) bereitstellt.
   * **Backend-Services:** Eine saubere Trennung von API (`routers/`), Geschäftslogik (`services/`) und Datenzugriff (`models/`, `repository_service.py`) nach dem **Single Responsibility Principle**.
   * **Externe Fachmodule:** Komplexe Domänen wie **E-Invoicing** (`einvoice/`) und **Datenexport** (`export_plugin.py`) sind als unabhängige, wiederverwendbare Plugins realisiert, die der Core nur bei Bedarf einbindet.
3. **🛡️ Sicherheit und Compliance by Design:**
   Sicherheit ist keine nachträgliche Ergänzung, sondern ein fundamentaler Bestandteil der Architektur.

   * **Proaktives Auditing:** Ein integrierter `SecurityAuditor` (`auditor.py`) prüft kontinuierlich den Code auf Schwachstellen (`Bandit`), verwundbare Abhängigkeiten (`Safety`) und hartcodierte Geheimnisse.
   * **Secrets Management:** API-Keys und sensible Daten werden mittels `cryptography.Fernet` verschlüsselt gespeichert (`security/manager.py`).
   * **GoBD-Konformität:** Ein lückenloses Audit-System (`audit.py`, `compliance_reporting.py`) protokolliert alle relevanten Datenänderungen und stellt die Unveränderlichkeit der Buchungsdaten sicher.
   * **Strikte Datenvalidierung:** `Pydantic v2` wird durchgängig für eine strenge, typsichere Validierung aller Datenstrukturen genutzt, inklusive deutscher Geschäftsregeln (USt-ID, PLZ, SKR03-Konten) (`models/invoice.py`, `models/skr03.py`).
4. **⚡ Performance und Skalierbarkeit:**
   Das System ist für den reaktionsschnellen und effizienten Umgang mit großen Datenmengen ausgelegt.

   * **Asynchrone Verarbeitung:** Das Backend nutzt `FastAPI` und `asyncio` (`async_processor.py`), um I/O-intensive Operationen wie PDF-Verarbeitung und API-Aufrufe parallel auszuführen.
   * **Optimierter Datenbankzugriff:** `ChromaDBConnectionPool` (`connection_pool.py`) und `ChromaDBBatchProcessor` (`batch_processor.py`) minimieren Latenzen und maximieren den Durchsatz bei der Vektorsuche.
   * **Multi-Level-Caching:** Ein intelligentes Caching-System (`caching.py`, `performance_cache_manager.py`) für Embeddings und Klassifizierungsergebnisse reduziert redundante Berechnungen und beschleunigt wiederkehrende Anfragen.

## 3. Systemübersicht: Komponenten-Architektur

Das LLKJJ-System besteht aus einem zentralen Backend, das durch spezialisierte, in sich geschlossene Plugins erweitert wird.

```
                  +--------------------------+
                  |     Frontend (Web UI)    |
                  +-------------+------------+
                                | (REST API)
+-------------------------------+--------------------------------+
|                       LLKJJ Backend (FastAPI)                    |
|                                                                |
|  +----------------+   +----------------------+   +-------------+  |
|  |  API Routers   |-->|   Service Schicht    |-->|  Datenbank  |  |
|  | (auth, docs,   |   | (Business Logic)     |   | (PostgreSQL)|  |
|  |  business...)  |   | (Auth, Creditor,     |   | (SQLAlchemy)|  |
|  +----------------+   |  Dashboard Services) |   +-------------+  |
|                       +-----------+----------+                   |
|                                   |                              |
|          +------------------------+-------------------------+   |
|          | (Definierte Schnittstellen / Plugins)             |   |
|          v                        v                          v   |
| +-----------------+   +-------------------------+   +-------------------+ |
| |  ML-Plugin      |   |   E-Invoice Plugin      |   |   Export Plugin   | |
| | (Docling, RAG,  |   | (XRechnung, UBL, PDF/A-3)|   | (DATEV, CSV, JSON)| |
| |  SKR03-Klass.)  |   +-------------------------+   +-------------------+ |
| | +-------------+ |                                                    |
| | |  Vector DB  | |                                                    |
| | | (ChromaDB)  | |                                                    |
| | +-------------+ |                                                    |
| +-----------------+                                                    |
+------------------------------------------------------------------------+
```

## 4. Detaillierte Komponentenbeschreibung

### 4.1. Backend-Architektur (FastAPI)

- **Zweck:** Dient als zentraler API-Server, der Anfragen vom Frontend entgegennimmt, die Geschäftslogik orchestriert und mit den Plugins kommuniziert.
- **Kerntechnologien:** FastAPI, Pydantic v2, SQLAlchemy 2.0 (async).
- **Struktur (`api/`, `core/`, `services/`):**
  - **`api/main.py`:** Einstiegspunkt der Anwendung.
  - **`api/routers/`:** Modulare Endpunkte, gruppiert nach Domänen (z.B. `documents`, `business`, `features`, `auth`), was die Wartbarkeit erhöht.
  - **`core/services/`:** Kapselung der Geschäftslogik. Services wie `CreditorService` oder `DashboardService` sind von der API-Schicht entkoppelt.
  - **`core/models/`:** Definition der SQLAlchemy-Datenbankmodelle.
  - **`core/schemas/`:** Definition der Pydantic-API-Datenmodelle (`unified/` für das Zielschema).

### 4.2. ML-Pipeline (Das Herzstück)

- **Zweck:** Vollautomatische Extraktion, Analyse und Klassifizierung von Rechnungsdokumenten.
- **Schlüsseldateien:** `processor.py`, `async_processor.py`, `package_api.py`.
- **Workflow:**
  1. **Orchestrierung (`UnifiedProcessor`):** Koordiniert den gesamten ML-Ablauf. Nutzt einen `ResourceManager` als Singleton, um schwere Modelle (`SentenceTransformer`, `DoclingProcessor`) nur einmal zu laden und Memory Leaks zu verhindern.
  2. **Extraktion (`DataExtractor`, `docling_processor.py`):** Eine hochoptimierte Pipeline zur Extraktion von Text, Tabellen und Layout-Informationen aus PDFs. Nutzt `Docling` mit intelligentem Fallback zwischen verschiedenen OCR-Engines, um maximale Robustheit zu gewährleisten.
  3. **Klassifizierung (`DataClassifier`, `skr03_manager.py`):** Das Gehirn der Pipeline. Nutzt eine hybride Strategie:
     * **Regelwerk:** Ein YAML-basiertes Regelwerk (`skr03_regeln.yaml`) für bekannte Keywords.
     * **RAG-System:** Findet über die Vektordatenbank `ChromaDB` ähnliche, bereits validierte Buchungsbeispiele, um unbekannte Artikel intelligent zu klassifizieren.
     * **Kombination:** Eine Logik (`combine_classification_results`) wägt die Ergebnisse beider Systeme intelligent ab.
  4. **Qualitätssicherung (`QualityAssessor`):** Bewertet die Qualität jeder Extraktion und Klassifizierung und berechnet einen Konfidenz-Score.
- **Öffentliche Schnittstelle:** `ProcessingResult` ist der garantierte, stabile Datenvertrag, den das Plugin nach außen bereitstellt.

### 4.3. Datenmodelle & Validierung (Pydantic)

- **Zweck:** Sicherstellung der Datenintegrität im gesamten System durch strikte, typsichere Validierung.
- **Schlüsseldateien:** `models/invoice.py`, `models/skr03.py`, `models/validation.py`.
- **Features:**
  - **Geschäftsregeln:** Validatoren prüfen deutsche Formate wie USt-ID, PLZ und SKR03-Kontonummern.
  - **Finanzlogik:** Konsistenzprüfungen zwischen Einzelpreisen, Mengen, Steuern und Gesamtbeträgen sind direkt in den Modellen verankert.
  - **Benutzerfreundliche Fehler:** Ein `GermanValidationError`-Handler (`validation.py`) übersetzt technische Pydantic-Fehler in verständliche deutsche Meldungen für das Frontend.
  - **Performance:** Die bewusste Migration von `dataclasses` zu `Pydantic v2` wurde durch Benchmarks (`benchmark.py`) validiert und bietet neben der Validierung auch Performance-Vorteile.

### 4.4. E-Invoice-System

- **Zweck:** Erstellung und Validierung von gesetzeskonformen elektronischen Rechnungen nach deutschen und europäischen Standards.
- **Schlüsseldateien:** `einvoice/`-Verzeichnis.
- **Features:**
  - **Standard-Unterstützung:** Generiert XML für **XRechnung (CII)** und **UBL**.
  - **Factur-X:** Kombiniert Rechnungs-PDFs mit eingebettetem XML zu **PDF/A-3** konformen Hybrid-Dokumenten (`pdf_xml_combiner.py`).
  - **Validierung:** Umfassende Validierung gegen XSD-Schemas (`xsd_validator.py`) und optional gegen den offiziellen **KoSIT-Validator** (`kosit_validator.py`).

## 5. Kern-Workflows

### Workflow 1: Standard-Rechnungsverarbeitung (PDF → Buchungsvorschlag)

1. **Upload:** Ein Benutzer lädt eine PDF-Rechnung über den `/api/v1/documents/upload`-Endpunkt hoch.
2. **Persistierung & Tasking:** Der `dms_router` speichert die Datei sicher, erstellt einen initialen DB-Eintrag mit Status `PENDING` und stößt eine Hintergrundaufgabe (`process_document_ml`) an.
3. **ML-Pipeline Start:** Die Hintergrundaufgabe ruft den `UnifiedProcessor` (oder `AsyncUnifiedProcessor`) des ML-Plugins mit dem Dateipfad auf.
4. **Extraktion:** Der `DataExtractor` nutzt den `AdvancedDoclingProcessor`, um Text, Tabellen und Metadaten zu extrahieren. Bei Bedarf wird auf alternative OCR-Engines zurückgegriffen.
5. **Klassifizierung:** Der `DataClassifier` analysiert jede Rechnungsposition. Er nutzt den `SKR03Manager` für Keyword-Treffer und fragt parallel die `ChromaDB` nach ähnlichen Positionen ab (RAG).
6. **Ergebnis-Konsolidierung:** Die Ergebnisse werden zu einem `ProcessingResult`-Objekt zusammengefügt, und der `QualityAssessor` berechnet einen finalen Konfidenz-Score.
7. **DB-Update:** Die extrahierten und klassifizierten Daten, inklusive der SKR03-Kontovorschläge, werden in der PostgreSQL-Datenbank gespeichert und der Dokumentenstatus auf `PROCESSED` gesetzt.
8. **Feedback (Optional):** Korrigiert ein Benutzer einen Vorschlag, lernt der `FastRuleBookingService` eine neue Regel für zukünftige, schnellere Klassifizierungen.

### Workflow 2: E-Rechnung-Erstellung

1. **API-Aufruf:** Das Frontend sendet strukturierte Rechnungsdaten an den `/api/v1/einvoice/generate`-Endpunkt. Die Daten werden durch das strikte `OutgoingEInvoiceData`-Schema validiert.
2. **XML-Generierung:** Der `einvoice`-Service nutzt den `XRechnungCIIGeneratorV2` (oder `UBLInvoiceGenerator`), um ein standardkonformes XML zu erstellen.
3. **PDF-Generierung (Optional):** Parallel wird aus denselben Daten über den `EnhancedPDFService` und Jinja2-Templates eine visuelle PDF-Rechnung erzeugt.
4. **Factur-X Kombination (Optional):** Falls ein Hybrid-Dokument gewünscht ist, bettet der `FacturXGenerator` das generierte XML in das PDF ein und stellt die PDF/A-3-Konformität sicher.
5. **Antwort:** Die API liefert das generierte XML (und optional das PDF) an das Frontend zurück.

## 6. Entwicklungsphilosophie und nächste Schritte

- **Strikte Typisierung:** MyPy wird im `strict`-Modus verwendet, um die Codequalität und Zuverlässigkeit zu maximieren.
- **Testabdeckung:** Die bestehende Test-Suite wird kontinuierlich erweitert, um alle Kern-Workflows mit End-to-End-Tests abzudecken. Ziel ist eine Testabdeckung von >80%.
- **CI/CD:** Eine CI/CD-Pipeline wird eingerichtet, um bei jedem Commit automatisch Tests, Linting und Security-Audits durchzuführen.
- **Dokumentation:** Die API-Dokumentation wird über OpenAPI automatisch generiert (`generate_typescript_sdk.py`), um das Frontend-Team zu unterstützen.

Dieses Manifest dient als lebendiges Dokument. Jede neue Funktion und jedes neue Plugin wird anhand dieser Vision und den hier definierten architektonischen Prinzipien bewertet.
