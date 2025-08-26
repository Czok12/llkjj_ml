import os
import shutil
import sys
from pathlib import Path

# Stelle sicher, dass das 'src'-Verzeichnis im Python-Pfad ist,
# damit wir das Konfigurationsmodul importieren können.
# Das macht das Skript von überall im Projekt aus aufrufbar.
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

try:
    # Lade die Konfiguration, um den ChromaDB-Pfad zu erhalten
    from src.settings_bridge import Config

    config = Config()
except ImportError as e:
    print(f"Fehler: Konnte das Konfigurationsmodul nicht importieren: {e}")
    print(
        "Stelle sicher, dass das Skript vom Projekt-Root ausgeführt wird und src/__init__.py existiert."
    )
    sys.exit(1)


def reset_chromadb() -> None:
    """
    Löscht das ChromaDB-Persistenzverzeichnis vollständig und erstellt es neu.
    """
    # Konstruiere den absoluten Pfad aus der Konfiguration
    chroma_path = PROJECT_ROOT / config.vector_db_path

    print("--- ChromaDB Reset Skript ---")
    print(f"Zielverzeichnis: {chroma_path.resolve()}")

    if not chroma_path.is_dir():
        print("Verzeichnis existiert nicht. Erstelle es...")
    else:
        print("Verzeichnis existiert. Lösche es jetzt...")
        try:
            shutil.rmtree(chroma_path)
            print("Verzeichnis erfolgreich gelöscht.")
        except OSError as e:
            print(f"Fehler beim Löschen des Verzeichnisses: {e}")
            sys.exit(1)

    try:
        os.makedirs(chroma_path)
        print("Leeres Verzeichnis erfolgreich neu erstellt.")
    except OSError as e:
        print(f"Fehler beim Erstellen des Verzeichnisses: {e}")
        sys.exit(1)

    print("--- Reset abgeschlossen ---")


if __name__ == "__main__":
    # Frage zur Sicherheit nach Bestätigung
    response = (
        input(
            "Möchtest du die ChromaDB wirklich komplett zurücksetzen? Alle Vektordaten gehen verloren. (ja/nein): "
        )
        .strip()
        .lower()
    )

    # Input validation for security
    if response in ["ja", "yes", "y"]:
        reset_chromadb()
    elif response in ["nein", "no", "n", ""]:
        print("Aktion abgebrochen.")
    else:
        print("Ungültige Eingabe. Aktion abgebrochen.")
        sys.exit(1)
