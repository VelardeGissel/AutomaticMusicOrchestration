# AMO Web Application

## Verzeichnisstruktur

```
AutomaticMusicOrchestration/
|-- apps/
|   `-- webapp/
|       |-- webapp.py
|       |-- requirements.txt
|       |-- start_webapp.ps1
|       |-- README.md
|       |-- QUICKSTART.md
|       |-- STATUS.md
|       `-- notebooks/
|           `-- webapp_api_examples.ipynb
|-- data/
|   `-- samples/
|       `-- midis/
|-- models/
|   `-- weights/
|-- runtime/
|   |-- uploads/
|   `-- outputs/
`-- src/
    `-- amo/
        `-- ml_orchestration.py
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

Stellen Sie sicher, dass die `.joblib` Dateien im `models/weights/` Verzeichnis vorhanden sind:

```powershell
ls models/weights/
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






