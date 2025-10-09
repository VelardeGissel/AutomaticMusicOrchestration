# 🎵 AMO Web Application - Quick Start Guide

## Was wurde erstellt?

Eine vollständige Web-Anwendung für automatische Musik-Orchestrierung mit:

### Dateien:
1. **`webapp.py`** - Haupt-Web-Anwendung (FastAPI + Uvicorn)
2. **`requirements_webapp.txt`** - Benötigte Python-Pakete
3. **`start_webapp.ps1`** - Windows PowerShell Start-Skript
4. **`README_webapp.md`** - Ausführliche Dokumentation
5. **`webapp_api_examples.ipynb`** - Jupyter Notebook mit API-Beispielen

### Features:
- ✅ Benutzerfreundliches Web-Interface mit Bootstrap 5
- ✅ MIDI-Datei Upload (Drag & Drop)
- ✅ Auswahl zwischen 3 Orchestrierungsmodellen:
  - Beethoven (Sinfonie Nr. 7)
  - Debussy (La Mer)
  - Tchaikovsky (Sugar Plum Fairy)
- ✅ Download der orchestrierten MIDI-Datei
- ✅ REST API für programmatischen Zugriff
- ✅ Responsive Design für Desktop und Mobile

## 🚀 Schnellstart

### Schritt 1: Abhängigkeiten installieren

```powershell
pip install -r requirements_webapp.txt
```

### Schritt 2: Server starten

**Option A - Mit Start-Skript (empfohlen):**
```powershell
.\start_webapp.ps1
```

**Option B - Direkt:**
```powershell
python webapp.py
```

### Schritt 3: Web-Interface öffnen

Öffnen Sie Ihren Browser und navigieren Sie zu:
```
http://localhost:8001
```

## 📝 Verwendung

### Web-Interface:
1. Klicken Sie auf das Upload-Feld oder ziehen Sie eine MIDI-Datei hinein
2. Wählen Sie einen Orchestrierungsstil (Beethoven, Debussy, Tchaikovsky)
3. Klicken Sie auf "Orchestrierung starten"
4. Warten Sie auf die Verarbeitung (ca. 10-30 Sekunden)
5. Laden Sie die orchestrierte MIDI-Datei herunter

### API-Verwendung:
Siehe `webapp_api_examples.ipynb` für detaillierte Beispiele mit Python requests.

## 🛠️ Technische Details

### Architektur:
```
Browser → FastAPI Server → amo_load_and_orchestrate() → Orchestrierte MIDI
```

### Endpunkte:
- `GET /` - Web-Interface (HTML)
- `GET /api/models` - Liste verfügbarer Modelle (JSON)
- `POST /api/orchestrate` - MIDI orchestrieren (Multipart Form)
- `GET /api/download/{filename}` - Datei herunterladen
- `GET /api/health` - Server-Status

### Dateifluss:
```
uploads/     → Temporäre Uploads (werden nach Verarbeitung gelöscht)
outputs/     → Orchestrierte MIDI-Dateien (bleiben erhalten)
weights/     → Trainierte ML-Modelle (.joblib)
```

## 🔧 Konfiguration

### Port ändern:
In `webapp.py`, letzte Zeile:
```python
uvicorn.run(app, host="0.0.0.0", port=8080)  # Statt 8000
```

### Nur lokalen Zugriff:
```python
uvicorn.run(app, host="127.0.0.1", port=8000)
```

## ⚠️ Fehlerbehebung

### "Port bereits belegt":
```powershell
# Anderen Port verwenden
uvicorn webapp:app --port 8080
```

### "Modelle nicht gefunden":
Stellen Sie sicher, dass diese Dateien existieren:
- `weights/beethoven.joblib`
- `weights/debussy.joblib`
- `weights/tchaikovsky.joblib`

### "FastAPI nicht gefunden":
```powershell
pip install -r requirements_webapp.txt
```

### "Import-Fehler aus amos/":
Stellen Sie sicher, dass Sie im richtigen Verzeichnis sind:
```powershell
cd C:\Repositories\AMO\AutomaticMusicOrchestration
```

## 📊 Test-Beispiel

```python
import requests

# Server testen
response = requests.get("http://localhost:8000/api/health")
print(response.json())  # {'status': 'healthy', 'available_models': 3}

# MIDI orchestrieren
with open('midis/fur-elise.mid', 'rb') as f:
    files = {'file': f}
    data = {'model': 'beethoven'}
    response = requests.post(
        'http://localhost:8000/api/orchestrate',
        files=files,
        data=data
    )
    print(response.json())
```

## 📚 Weitere Ressourcen

- **Ausführliche Dokumentation:** `README_webapp.md`
- **API-Beispiele:** `webapp_api_examples.ipynb`
- **Hauptprojekt:** `README.md`
- **Original-Notebook:** `save_load_orch.ipynb`

## 🎯 Nächste Schritte

1. **Testen Sie die Anwendung** mit verschiedenen MIDI-Dateien
2. **Vergleichen Sie die Stile** der verschiedenen Modelle
3. **Erweitern Sie die API** mit zusätzlichen Features
4. **Deployen Sie die App** auf einem Server (Heroku, AWS, etc.)

## 💡 Erweiterungsideen

- Audio-Preview direkt im Browser (mit MIDI.js)
- Batch-Processing für mehrere Dateien
- Modell-Vergleichs-Ansicht
- Benutzer-Authentifizierung
- Orchestrierungs-Historie
- Parameter-Tuning im Interface
- Export zu verschiedenen Formaten (MusicXML, PDF)

## 📧 Support

Bei Fragen oder Problemen, siehe `README_webapp.md` oder die Haupt-Dokumentation.

---

**Viel Erfolg mit AMO! 🎼**
