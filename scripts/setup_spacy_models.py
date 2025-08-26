import subprocess
import sys
from pathlib import Path

# Stelle sicher, dass das 'src'-Verzeichnis im Python-Pfad ist
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

try:
    # Lade die Konfiguration, um die Liste der SpaCy-Modelle zu erhalten
    from src.settings_bridge import Config

    config = Config()
except ImportError as e:
    print(f"Fehler: Konnte das Konfigurationsmodul nicht importieren: {e}")
    print("Stelle sicher, dass das Skript vom Projekt-Root ausgeführt wird.")
    sys.exit(1)


def setup_spacy_models() -> None:
    """
    Lädt alle in der Konfiguration spezifizierten SpaCy-Modelle herunter und installiert sie.
    """
    models = [config.spacy_model_name]
    if not models:
        print("Keine SpaCy-Modelle in der Konfiguration gefunden (SPACY_MODELS).")
        return

    print("--- SpaCy-Modell-Setup ---")
    print(f"Benötigte Modelle: {', '.join(models)}")

    # sys.executable stellt sicher, dass wir den Python-Interpreter
    # aus dem aktuellen venv (z.B. von Poetry) verwenden.
    python_executable = sys.executable

    for model in models:
        print(f"\nInstalliere Modell: {model}...")
        try:
            command = [python_executable, "-m", "spacy", "download", model]
            # Führe den Befehl aus und warte auf seine Beendigung
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            print(f"'{model}' erfolgreich installiert.")
            # Zeige die letzten Zeilen der SpaCy-Ausgabe für mehr Kontext
            # (oft eine Erfolgsmeldung mit Pfad)
            for line in result.stdout.strip().split("\n")[-3:]:
                print(f"  > {line}")
        except subprocess.CalledProcessError as e:
            print(f"Fehler bei der Installation von '{model}':")
            print(e.stderr)
        except FileNotFoundError:
            print(
                f"Fehler: '{python_executable}' nicht gefunden. Falsches Environment?"
            )
            sys.exit(1)

    print("\n--- SpaCy-Setup abgeschlossen ---")


if __name__ == "__main__":
    setup_spacy_models()
