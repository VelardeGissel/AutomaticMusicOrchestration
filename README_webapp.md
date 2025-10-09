# AMO Web Application

## Automatische Musik-Orchestrierung - Web Interface

Diese Web-Anwendung ermöglicht es Benutzern, MIDI-Dateien ### In `webapp.py` ändern

```python
# Port ändern
uvicorn.run(app, host="0.0.0.0", port=8080)

# Nur lokalen Zugriff erlauben
uvicorn.run(app, host="127.0.0.1", port=8001)
```en und automatisch im Stil berühmter Komponisten orchestrieren zu lassen.

### Verfügbare Modelle

- **Beethoven** - Basierend auf Beethovens Sinfonie Nr. 7
- **Debussy** - Basierend auf Debussys "La Mer"
- **Tchaikovsky** - Basierend auf Tschaikowskys "Sugar Plum Fairy"

## Installation

### 1. Python-Umgebung vorbereiten

Stellen Sie sicher, dass Ihre bestehende Conda-Umgebung aktiv ist:

```powershell
conda activate amo
```

### 2. Web-App Abhängigkeiten installieren

```powershell
pip install -r requirements_webapp.txt
```

## Verwendung

### Server starten

```powershell
python webapp.py
```

Der Server startet auf: **http://localhost:8001**

### Web-Interface nutzen

1. Öffnen Sie einen Browser und navigieren Sie zu: `http://localhost:8001`
2. Laden Sie eine MIDI-Datei hoch (z.B. `fur-elise.mid`)
3. Wählen Sie einen Orchestrierungsstil (Beethoven, Debussy oder Tchaikovsky)
4. Klicken Sie auf "Orchestrierung starten"
5. Laden Sie die orchestrierte MIDI-Datei herunter

### Alternativer Start mit Uvicorn direkt

```powershell
uvicorn webapp:app --host 0.0.0.0 --port 8000 --reload
```

Der `--reload` Flag ermöglicht automatisches Neuladen bei Code-Änderungen (nützlich für Entwicklung).

## API Endpunkte

Die Web-App bietet folgende REST API Endpunkte:

- `GET /` - Haupt-HTML-Interface
- `GET /api/models` - Liste verfügbarer Modelle
- `POST /api/orchestrate` - MIDI-Datei orchestrieren
- `GET /api/download/{filename}` - Orchestrierte Datei herunterladen
- `GET /api/health` - Gesundheitsprüfung

### Beispiel API-Verwendung mit curl

```powershell
# Modelle abrufen
curl http://localhost:8001/api/models

# Orchestrierung mit API
curl -X POST "http://localhost:8001/api/orchestrate" `
  -F "file=@midis/fur-elise.mid" `
  -F "model=beethoven"
```

## Verzeichnisstruktur

```
AutomaticMusicOrchestration/
├── webapp.py                 # Haupt-Webanwendung
├── requirements_webapp.txt   # Web-App Abhängigkeiten
├── weights/                  # Trainierte Modelle (.joblib)
│   ├── beethoven.joblib
│   ├── debussy.joblib
│   └── tchaikovsky.joblib
├── uploads/                  # Temporäre hochgeladene Dateien (automatisch erstellt)
├── outputs/                  # Orchestrierte Ausgabedateien (automatisch erstellt)
└── amos/                     # AMO Python-Module
    └── ml_orchestration.py
```

## Funktionsweise

1. **Upload**: Benutzer lädt eine MIDI-Datei hoch
2. **Modellauswahl**: Benutzer wählt ein trainiertes Orchestrierungsmodell
3. **Verarbeitung**: Die Funktion `amo_load_and_orchestrate()` wird aufgerufen:
   - Lädt das vortrainierte Pipeline-Modell (.joblib)
   - Extrahiert musikalische Features aus der Upload-MIDI
   - Wendet das ML-Modell an, um Orchestrierung vorherzusagen
   - Generiert eine neue orchestrierte MIDI-Datei
4. **Download**: Benutzer erhält die orchestrierte MIDI-Datei

## Technologie-Stack

- **Backend**: FastAPI (Python)
- **Server**: Uvicorn (ASGI)
- **Frontend**: HTML5, Bootstrap 5, JavaScript
- **ML**: XGBoost, Scikit-learn
- **MIDI**: Mido Library

## Fehlerbehebung

### Port bereits belegt

Wenn Port 8000 bereits verwendet wird, ändern Sie den Port:

```powershell
uvicorn webapp:app --port 8080
```

### Modelle nicht gefunden

Stellen Sie sicher, dass die `.joblib` Dateien im `weights/` Verzeichnis vorhanden sind:

```powershell
ls weights/
```

### Import-Fehler

Überprüfen Sie, dass alle Abhängigkeiten installiert sind:

```powershell
pip list
```

## Erweiterte Konfiguration

### In `webapp.py` ändern:

```python
# Port ändern
uvicorn.run(app, host="0.0.0.0", port=8080)

# Nur lokalen Zugriff erlauben
uvicorn.run(app, host="127.0.0.1", port=8000)
```

## Sicherheitshinweise

Diese Anwendung ist für lokale/Entwicklungszwecke konzipiert. Für Produktionsumgebungen sollten Sie:

- HTTPS aktivieren
- Authentifizierung hinzufügen
- Datei-Upload-Größenbeschränkungen setzen
- Rate-Limiting implementieren
- Temporäre Dateien regelmäßig bereinigen

## Lizenz

Siehe Hauptprojekt-README.

## Autor

Gissel Velarde - Oktober 2025
