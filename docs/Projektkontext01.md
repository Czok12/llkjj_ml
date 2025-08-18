
## Das LLKJJ Projekt: Vision, Architektur & Technisches Manifest (Interne Entwicklung)

**Version 1.1 (Stand: August 2025)**

### 1. Vision & Manifest (Das "Warum")

Das Ziel des LLKJJ-Projekts ist die Schaffung einer persönlichen, hochautomatisierten und intelligenten Buchhaltungs- und Dokumentenmanagement-Engine. Es ist als private, überlegene Alternative zu Standardlösungen wie Buchhaltungsbutler.de konzipiert und exakt auf die Bedürfnisse eines deutschen Elektrohandwerk-Betriebs zugeschnitten.

Es geht nicht darum, eine weitere App zu bauen, sondern eine **maßgeschneiderte, hochintegrierte und intelligente Engine**, die den administrativen Aufwand auf ein absolutes Minimum reduziert.

**Leitprinzipien:**

1. **Autonomie:** Das System soll lernen, Kontexte verstehen und eigenständig korrekte Entscheidungen treffen, um manuelle Eingriffe zu minimieren.
2. **Präzision:** Maximale Genauigkeit bei der Datenextraktion und SKR03-Klassifizierung ist das oberste Ziel (>92% Genauigkeit).
3. **Sicherheit:** Da es um sensible Finanzdaten geht, hat die Sicherheit auf allen Ebenen – von der Code-Analyse bis zur Datenspeicherung – höchste Priorität.
4. **Wartbarkeit:** Eine saubere, entkoppelte Architektur (KISS-Prinzip) ist entscheidend, um das System langfristig pflegen und erweitern zu können.

### 2. Die Kernarchitektur (Das "Wie" – High Level)

Die Architektur basiert auf vier Säulen, die konsequent durchgesetzt werden:

1. **Intelligente Kern-Pipeline (KI-First):** Das Herzstück ist eine ML-Pipeline, die über starre Regeln hinausgeht. Sie nutzt `Docling` für die Extraktion, `ChromaDB` für RAG-basierte Klassifizierung und ein ausgefeiltes **Feature Engineering**, um den Kontext jeder Rechnung zu verstehen.
2. **Modulare & Entkoppelte Architektur (KISS & Plugins):** Das System ist in einen zentralen **Backend-Kern** und spezialisierte, austauschbare **Plugins** (`llkjj_ml_plugin`, `llkjj_efaktura`, `llkjj_export`) unterteilt. Diese kommunizieren ausschließlich über klar definierte Schnittstellen (Datenverträge wie `ProcessingResult`).
3. **Sicherheit & Compliance by Design:** Sicherheit ist integraler Bestandteil, nicht optional. Dies wird durch proaktives Auditing (`SecurityAuditor`), verschlüsseltes Secret-Management und GoBD-konforme Protokollierung (`AuditLog`) sichergestellt.
4. **Performance & Skalierbarkeit (Asynchron & Optimiert):** Die Nutzung von `FastAPI`, `asyncio`, Datenbank-Connection-Pooling und Multi-Level-Caching stellt sicher, dass das System auch bei großen Datenmengen reaktionsschnell bleibt.

### 3. System- und Workspace-Übersicht (Das "Was" und "Wo")

Das System ist in klar voneinander getrennte Workspaces (Plugins/Module) aufgeteilt, die jeweils eine spezifische Aufgabe erfüllen.

```
+---------------------------------------------------------------------------------+
|                                 LLKJJ-System                                    |
|                                                                                 |
|  +---------------------------------------------------------------------------+  |
|  |                Workspace 1: Das Backend (Core-System)                     |  |
|  |---------------------------------------------------------------------------|  |
|  | [FastAPI] -> API Routers -> [Service Schicht] -> [Repository] -> PostgreSQL |  |
|  |      ^                (Business Logic)      (SQLAlchemy)      (Struktur)  |  |
|  |      |                                                                    |  |
|  |      +------------------(Definierte Schnittstellen)-----------------------+  |
|  +------------------------------------|----------------------------------------+  |
|                                       |                                         |
|      +--------------------------------+--------------------------------+        |
|      |                                |                                |        |
|      v                                v                                v        |
| +-------------------------+  +-------------------------+  +-------------------------+ |
| | Workspace 2:            |  | Workspace 3:            |  | Workspace 4:            | |
| | ML-Pipeline             |  | E-Invoice-System        |  | Export-System           | |
| | (`llkjj_ml_plugin`)     |  | (`llkjj_efaktura`)      |  | (`llkjj_export`)        | |
| |-------------------------|  |-------------------------|  |-------------------------| |
| | - OCR (Docling)         |  | - XRechnung / UBL XML   |  | - DATEV CSV Export      | |
| | - Klassifizierung (RAG) |  | - PDF/A-3 Generierung   |  | - JSON / Standard CSV   | |
| | - Feature Engineering   |  | - KoSIT-Validierung     |  | - Universelle Schemas   | |
| | - Vector DB (ChromaDB)  |  +-------------------------+  +-------------------------+ |
| +-------------------------+                                                         |
+-------------------------------------------------------------------------------------+
```

- **Workspace 1: Das Backend (Core-System)**

  - **Zweck:** Dient als zentraler Orchestrator, stellt die REST-API bereit, verwaltet Benutzer, speichert strukturierte Daten und kommuniziert mit den Plugins.
  - **Verantwortlichkeiten:** API-Endpunkte (`api/routers`), Geschäftslogik (`core/services`), Datenbankmodelle (`core/models`), Datenvalidierung (`core/schemas/unified`).
  - **Kennt:** Die Schnittstellen der Plugins, aber nicht deren interne Implementierung.
- **Workspace 2: Die ML-Pipeline (`llkjj_ml_plugin`)**

  - **Zweck:** Das "Gehirn" des Systems. Verarbeitet eine PDF-Datei und gibt ein strukturiertes, klassifiziertes Ergebnis zurück.
  - **Verantwortlichkeiten:** PDF-Parsing (OCR), Tabellenerkennung, Text-Extraktion, SKR03-Klassifizierung, RAG-Suche, Feature Engineering.
  - **Datenvertrag:** Arbeitet als Blackbox. Die einzige garantierte Schnittstelle zum Backend ist das `ProcessingResult`-Schema (`package_api.py`). Es hat **keine Kenntnis** vom Backend-Code oder dessen Datenbank.
- **Workspace 3: Das E-Invoice-System (`llkjj_efaktura`)**

  - **Zweck:** Spezialist für die Erstellung und Validierung gesetzeskonformer elektronischer Rechnungen.
  - **Verantwortlichkeiten:** Generierung von XRechnung (CII) und UBL-XML, Erstellung von PDF/A-3-Hybrid-Dokumenten (Factur-X), Validierung gegen offizielle Standards (XSD, KoSIT).
- **Workspace 4: Das Export-System (`llkjj_export`)**

  - **Zweck:** Bereitstellung von Daten in verschiedenen Formaten für externe Systeme (z.B. Steuerberater).
  - **Verantwortlichkeiten:** Konvertierung von internen Buchungsdaten in DATEV-CSV, Standard-CSV oder JSON.

### 4. Detaillierte technische Blaupause (Die Implementierung)

#### 4.1. Datenfluss & Kern-Workflow (Rechnungsverarbeitung)

1. **Upload:** `api/routers/documents.py` empfängt eine PDF.
2. **Persistenz:** Der `DMSService` (`services/document_service/manager.py`) speichert die Datei, erstellt einen `Document`-Eintrag in PostgreSQL (Status `PENDING`).
3. **Hintergrund-Task:** Ein `BackgroundTasks` (`backend/tasks.py`) stößt die ML-Verarbeitung an.
4. **Orchestrierung:** Der Task ruft den `UnifiedProcessor` (`llkjj_ml/processor.py`) des ML-Plugins auf.
5. **Extraktion & Klassifizierung:** Die ML-Pipeline durchläuft ihre Phasen (Extraktion, RAG-Klassifizierung, Qualitätssicherung).
6. **Ergebnis:** Die Pipeline liefert ein `ProcessingResult`-Objekt zurück.
7. **Datenübernahme:** Der `DMSService` im Backend normalisiert das `ProcessingResult` und speichert die strukturierten Daten im `InvoiceProcessingResult`-Modell. Der Dokumentenstatus wird auf `PROCESSED` gesetzt.

#### 4.2. Datenmodelle & Validierung (Pydantic)

- **Unified Schema (`core/schemas/unified/`):** Dies ist die "Single Source of Truth" für alle API-Datenmodelle. Es stellt sicher, dass alle Systemteile eine konsistente Sprache sprechen.
- **Deutsche Validierung (`models/invoice.py`):** Spezifische Validatoren für `plz`, `ustid` und SKR03-Kontonummern sind direkt in den Pydantic-Modellen implementiert, um die Datenqualität an der Quelle zu sichern.
- **Fehlerbehandlung (`models/validation.py`):** `GermanValidationError` fängt Validierungsfehler ab und übersetzt sie in benutzerfreundliche deutsche Meldungen, was die Frontend-Integration erheblich vereinfacht.

#### 4.3. Datenbank & Persistenz

- **Primär-DB (PostgreSQL):** Dient zur Speicherung aller strukturierten Geschäftsdaten (Benutzer, Rechnungen, Buchungen, Kreditoren). Der Zugriff erfolgt ausschließlich über SQLAlchemy 2.0 mit dem asynchronen `asyncpg`-Treiber.
- **Vektor-DB (ChromaDB):** Dient als Wissensdatenbank für das RAG-System der ML-Pipeline. Sie speichert Vektor-Embeddings von Rechnungspositionen.
- **Migrationen (Alembic):** `alembic/` verwaltet alle Datenbankschema-Änderungen, um eine konsistente DB-Struktur über verschiedene Umgebungen hinweg zu gewährleisten.
- **Repository Pattern (`repository_service.py`):** Trennt die Geschäftslogik sauber vom direkten Datenbankzugriff, was Tests und Wartung erleichtert.

#### 4.4. Sicherheit & Authentifizierung

- **Authentifizierung (`core/security.py`, `api/routers/auth.py`):** Basiert auf JWT (JSON Web Tokens) mit OAuth2 Password Flow. Passwörter werden mit **Argon2** gehasht.
- **Security Auditing (`auditor.py`):** Führt statische Code-Analyse (`Bandit`), Dependency-Scanning (`Safety`) und benutzerdefinierte Suchen nach Secrets durch. Dies ist ein integraler Bestandteil der CI-Pipeline.
- **Secrets Management (`security/manager.py`):** Verwaltet sensible API-Schlüssel. Diese werden "at rest" mit AES-256 verschlüsselt, abgeleitet von einem Master-Passwort (gespeichert als Umgebungsvariable).

#### 4.5. Performance & Caching

- **Asynchronität:** Die gesamte API-Schicht ist asynchron, um blockierende I/O-Operationen zu vermeiden.
- **Connection Pooling (`database/connection_pool.py`):** Reduziert den Overhead beim Aufbau von Datenbankverbindungen, insbesondere für die `ChromaDB`.
- **Multi-Level-Caching (`caching.py`):** Ein zweistufiges Caching-System (In-Memory LRU + persistenter SQLite/File-Cache) für häufig abgerufene SKR03-Klassifizierungen und ML-Embeddings.

### 5. Anweisungen für die KI-gestützte Entwicklung

Dieses Dokument soll als primärer Kontext für jeden KI-gestützten Entwicklungs-Workspace dienen.

#### Generelle Anweisungen ("Golden Rules")

- **Sprache:** Der gesamte Code, Kommentare und die Dokumentation sind auf **Deutsch**, um die Domänensprache (deutsche Buchhaltung) konsistent abzubilden.
- **Architektur respektieren:** Halte dich strikt an die Trennung von Backend-Kern und Plugins. Keine direkten Abhängigkeiten zwischen den Workspaces, außer über die definierten Schnittstellen.
- **Sicherheit zuerst:** Jeder neue Code muss sicher sein. Vermeide hartcodierte Secrets, validiere alle Eingaben und nutze die bereitgestellten Sicherheits-Utilities.
- **Typsicherheit:** Der gesamte Code muss `MyPy --strict` konform sein.
- **Testen:** Für neue Funktionen müssen entsprechende `pytest`-Tests geschrieben werden.

---

#### Anweisungen für Workspace 1: Das Backend (Core-System)

- **Kernziel:** Die API-Schnittstelle bereitstellen, Daten verwalten und die Plugins orchestrieren.
- **Kontext:** Du bist ein Experte für FastAPI, SQLAlchemy (async) und saubere Softwarearchitektur.
- **Hauptaufgaben:** Erstellen von API-Routern, Service-Klassen für Geschäftslogik und SQLAlchemy-Modellen.
- **Interaktionen:**
  - Du rufst das **ML-Plugin** auf und erwartest ein `ProcessingResult`-Objekt zurück. Du kennst die interne Funktionsweise des Plugins nicht.
  - Du rufst das **E-Invoice-Plugin** mit einem `OutgoingEInvoiceData`-Objekt auf und erwartest XML oder PDF-Bytes zurück.
  - Du rufst das **Export-Plugin** mit internen `Booking`-Objekten auf und erwartest eine formatierte Datei (z.B. CSV) zurück.

---

#### Anweisungen für Workspace 2: Die ML-Pipeline (`llkjj_ml_plugin`)

- **Kernziel:** Eine PDF-Datei so präzise wie möglich in ein strukturiertes, SKR03-klassifiziertes `ProcessingResult`-Objekt zu verwandeln.
- **Kontext:** Du bist ein Experte für Machine Learning, NLP (spaCy, Transformers), Computer Vision (Docling, OCR) und Vektordatenbanken (ChromaDB).
- **Hauptaufgaben:** Verbesserung der Extraktions-Pipeline, Optimierung der Klassifizierungsmodelle, Implementierung von Feature Engineering.
- **Interaktionen:**
  - Deine **einzige Schnittstelle** zum Backend ist das `ProcessingResult`-Schema in `package_api.py`.
  - Du darfst **keine direkten Abhängigkeiten** zum Backend-Code oder dessen Datenbank (PostgreSQL) haben. Du bist eine eigenständige Blackbox.
  - Deine eigene Persistenz (z.B. für `ChromaDB`) verwaltest du in deinem eigenen Verzeichnis (`data/vectors`).

---

#### Anweisungen für Workspace 3: Das E-Invoice-System (`llkjj_efaktura`)

- **Kernziel:** Gesetzeskonforme elektronische Rechnungen nach deutschen und EU-Standards erstellen und validieren.
- **Kontext:** Du bist ein Experte für E-Invoicing-Standards (XRechnung, UBL, Factur-X), XML-Technologien und PDF/A-Compliance.
- **Hauptaufgaben:** Implementierung von XML-Generatoren, PDF/A-3-Kombination und Validierungslogik.
- **Interaktionen:**
  - Deine primäre Eingabe ist das `OutgoingEInvoiceData`-Schema.
  - Du bist ein eigenständiges Modul ohne Abhängigkeiten zum Backend oder zur ML-Pipeline.

---

#### Anweisungen für Workspace 4: Das Export-System (`llkjj_export`)

- **Kernziel:** Interne Buchungsdaten in verschiedene standardisierte Formate für externe Parteien (z.B. Steuerberater) zu konvertieren.
- **Kontext:** Du bist ein Experte für Datenkonvertierung und Buchhaltungs-Exportformate wie DATEV.
- **Hauptaufgaben:** Implementierung von Handlern für verschiedene Exportformate.
- **Interaktionen:**
  - Deine Eingabe sind universelle Datenmodelle wie `BookingExportData`.
  - Du bist ein eigenständiges Modul.
