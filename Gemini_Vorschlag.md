
Perfekt, das ist eine extrem klare und strategisch kluge Zielsetzung. Sie beeinflusst die Prioritäten unserer Optimierungen maßgeblich.

Wenn das Ziel lautet: **"Jetzt mit Gemini produktiv werden und später nahtlos auf eine lokale, selbstlernende Lösung umsteigen"**, dann müssen wir die Architektur auf genau diesen Übergang ausrichten.

Hier ist der optimierte Plan, der exakt auf deine Strategie zugeschnitten ist. Wir fokussieren uns auf das, was **jetzt** für die Gemini-Phase wichtig ist und was den **späteren Wechsel** zu spaCy/RAG so einfach wie möglich macht.

---

### Strategische Roadmap: Von Gemini zu lokaler Autonomie

**Phase 1: Gemini als produktives "Gehirn" (Fokus JETZT)**

* **Priorität 1: Maximale Datenqualität und -quantität für das zukünftige Training sammeln.** Jede einzelne Rechnung, die jetzt durch Gemini läuft, muss perfekte Trainingsdaten für die Zukunft liefern.
* **Priorität 2: Robuste und zuverlässige Gemini-Verarbeitung.** Der aktuelle Workflow muss stabil sein und nachvollziehbare Ergebnisse liefern.
* **Priorität 3: Effiziente Nutzung der Gemini-API.** Kosten und Geschwindigkeit im Auge behalten.

**Phase 2: Nahtloser Übergang zu spaCy & lokalem RAG (Fokus SPÄTER)**

* **Priorität 1: Die gesammelten Daten müssen 1:1 für das Training von spaCy (NER & TextCat) und den Aufbau der ChromaDB (RAG) nutzbar sein.**
* **Priorität 2: Die Architektur muss den Austausch der "Intelligenz-Engine" (Gemini -> spaCy/RAG) ermöglichen, ohne das restliche System (Backend, API) anpassen zu müssen.**

---

### Konkrete Optimierungen, sortiert nach deiner Strategie

Hier sind die wichtigsten Verbesserungen aus meiner vorherigen Liste, neu priorisiert und auf deine Strategie zugeschnitten:

#### A) Top-Priorität: Was sofort umgesetzt werden sollte (für Phase 1)

**1. Pydantic-Validierungsschicht für die Gemini-Antwort (absolut entscheidend!)**

* **Warum jetzt?** Dies ist deine **Qualitätssicherung für zukünftige Trainingsdaten**. Wenn Gemini Müll liefert (falsches Format, fehlende Felder) und du das unbemerkt speicherst, trainierst du später ein schlechtes lokales Modell.
* **Umsetzung:**
  1. Erstelle eine neue Datei `src/models/gemini_schemas.py`.
  2. Definiere darin Pydantic-Modelle, die exakt das JSON-Schema aus deinem Gemini-Prompt abbilden (`GeminiInvoiceHeader`, `GeminiLineItem`, `GeminiExtractionResult`).
  3. Im `GeminiDirectProcessor` wird die rohe JSON-Antwort von Gemini sofort in dieses Modell geparst: `validated_data = GeminiExtractionResult(**json.loads(response_text))`.
* **Ergebnis:** Du stellst sicher, dass jede einzelne verarbeitete Rechnung die korrekte Struktur hat. Das ist die Grundlage für alles Weitere.

**2. Robuste Speicherung der Trainingsdaten (der "Schatz" für Phase 2)**

* **Warum jetzt?** Du musst sicherstellen, dass die von Gemini generierten spaCy-Annotationen und RAG-Daten sauber und konsistent gespeichert werden.
* **Umsetzung:**
  1. Erweitere den `GeminiDirectProcessor`. Nach der erfolgreichen Pydantic-Validierung (Punkt 1), rufe eine neue, dedizierte Methode `_persist_training_artifacts` auf.
  2. Diese Methode speichert zwei Dinge:
     * **spaCy-Trainingsdaten:** Die korrigierten Annotationen (aus dem `SpacyAnnotationCorrector`) zusammen mit dem `raw_text` werden in eine **dedizierte JSONL-Datei** geschrieben (z.B. `data/training/gemini_spacy_annotations.jsonl`). Dieses Format ist perfekt für späteres spaCy-Training.
     * **RAG-Daten:** Jede klassifizierte Rechnungsposition (Beschreibung, Lieferant, korrektes SKR03-Konto) wird als Dokument in `ChromaDB` gespeichert, aber mit einem speziellen Metadatum, z.B. `"source": "gemini_validated"`.
* **Ergebnis:** Du baust mit jeder Rechnung systematisch zwei hochwertige Datensätze auf: einen für das NER/TextCat-Training und einen für das RAG-System.

**3. Detailliertes Audit-Logging (GoBD & Nachvollziehbarkeit)**

* **Warum jetzt?** Da du produktiv gehst, ist die Nachvollziehbarkeit (GoBD) entscheidend. Du musst jederzeit belegen können, warum ein Beleg wie verbucht wurde.
* **Umsetzung:**
  1. Erstelle einen einfachen `AuditLogger`-Service.
  2. Im `GeminiDirectProcessor`, nach jeder erfolgreichen Verarbeitung, schreibe einen unveränderlichen JSON-Eintrag in eine Log-Datei (z.B. `logs/audit.log`).
  3. Der Eintrag enthält: PDF-Hash, Timestamp, die **rohe Gemini-Antwort** und das finale `ProcessingResult`.
* **Ergebnis:** Du bist von Tag 1 an GoBD-konform und kannst jeden einzelnen Buchungsvorschlag auf die ursprüngliche KI-Antwort zurückführen.

---

#### B) Nächste Priorität: Optimierung der aktuellen Gemini-Pipeline

**4. Asynchrone Gemini-Verarbeitung**

* **Warum jetzt?** Wenn du mehrere Belege auf einmal hochlädst (z.B. deine Monatsbuchhaltung), reduziert dies die Wartezeit von Minuten auf Sekunden. Das ist ein massiver Gewinn für die User Experience.
* **Umsetzung:**
  1. Erstelle den `AsyncGeminiDirectProcessor`.
  2. Nutze `asyncio.gather` und `asyncio.Semaphore`, um mehrere PDFs parallel an die Gemini-API zu senden.
* **Ergebnis:** Deutlich schnellere Batch-Verarbeitung.

**5. PDF-Level Caching**

* **Warum jetzt?** Verhindert, dass du für die versehentlich doppelt hochgeladene Rechnung erneut API-Kosten bei Gemini verursachst.
* **Umsetzung:**
  1. Implementiere einen einfachen Cache (SQLite oder Redis).
  2. Vor dem API-Aufruf: Berechne den SHA256-Hash des PDF-Inhalts.
  3. Prüfe, ob der Hash im Cache existiert. Wenn ja, gib das gespeicherte Ergebnis zurück. Wenn nein, verarbeite normal und speichere das Ergebnis am Ende mit dem Hash im Cache.
* **Ergebnis:** Spart API-Kosten und reduziert die Verarbeitungszeit für bekannte Dokumente auf Millisekunden.

---

#### C) Wichtig für später: Vorbereitung des Wechsels (Phase 2)

**6. Strategy-Pattern für die Verarbeitung**

* **Warum das wichtig ist:** Dies ist die **architektonische Grundlage für den nahtlosen Wechsel**.
* **Umsetzung (kann jetzt schon vorbereitet werden):**
  1. Definiere eine abstrakte `ProcessingStrategy`-Klasse mit einer `execute(pdf_path)`-Methode.
  2. Benenne den `GeminiDirectProcessor` in `GeminiStrategy` um und lasse ihn von `ProcessingStrategy` erben.
  3. Der `DoclingAlternativeProcessor` wird zur `DoclingStrategy`.
  4. Erstelle einen finalen `UnifiedProcessor`, der nur noch die Strategie zur Laufzeit auswählt.

  ```python
  class UnifiedProcessor:
      def __init__(self, config):
          self.strategies = {
              "gemini": GeminiStrategy(config),
              "docling": DoclingStrategy(config),
              # Zukünftig:
              # "spacy_rag": SpacyRagStrategy(config)
          }

      def process_pdf(self, pdf_path, method="gemini"):
          return self.strategies[method].execute(pdf_path)
  ```
* **Ergebnis:** Wenn deine spaCy/RAG-Lösung fertig ist, schreibst du eine neue `SpacyRagStrategy`-Klasse. **Der einzige Code, den du ändern musst, um den Standard-Workflow umzustellen, ist eine einzige Zeile im `UnifiedProcessor`**: `method="spacy_rag"`. Das Backend und die API bleiben komplett unberührt. Das ist maximale Flexibilität.

### Zusammenfassender Action-Plan

1. **Sofort umsetzen:**

   * **Pydantic-Schema** für die Gemini-Antwort erstellen und Validierung erzwingen.
   * **Speicherung der Trainingsdaten** (JSONL für spaCy, ChromaDB für RAG) nach jeder erfolgreichen Gemini-Verarbeitung implementieren.
   * **Audit-Logging** für GoBD-Konformität einrichten.
2. **Als Nächstes für die Performance:**

   * Die Gemini-Verarbeitung **asynchron** machen.
   * Einen **PDF-Hash-Cache** implementieren, um unnötige API-Calls zu vermeiden.
3. **Architektonisch vorbereiten:**

   * Den `UnifiedProcessor` mit dem **Strategy-Pattern** umbauen, um die Weichen für den späteren, einfachen Austausch der Engine zu stellen.

Wenn du diese Schritte befolgst, hast du nicht nur eine jetzt produktiv nutzbare, robuste Gemini-Pipeline, sondern baust gleichzeitig systematisch den Datenschatz und die perfekte Architektur auf, um in Zukunft unabhängig, kostengünstig und hochpräzise mit deiner eigenen, lokalen KI zu arbeiten.
