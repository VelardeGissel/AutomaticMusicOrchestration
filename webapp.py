"""
AMO Web Application
Automatic Music Orchestration Web Interface
By Gissel Velarde
Date: October 2025

A FastAPI web application that allows users to:
1. Upload a MIDI file
2. Select an ML model type (XGBoost, LSTM, etc.)
3. Select an orchestration style (Beethoven, Debussy, etc.)
4. Download the orchestrated MIDI file
"""

# Set environment variables BEFORE any other imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
from pathlib import Path
import shutil
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Add amos directory to Python path
sys.path.append('amos')

# Import AMO orchestration function
from amos.ml_orchestration import amo_load_and_orchestrate

# Initialize FastAPI app
app = FastAPI(
    title="AMO - Automatic Music Orchestration",
    description="Web interface for automatic music orchestration using machine learning",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
WEIGHTS_DIR = Path("weights")
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")

# Create directories if they don't exist
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Model type descriptions
MODEL_TYPES = {
    "xgboost": {
        "name": "XGBoost",
        "description": "Extreme Gradient Boosting - Fast and accurate ensemble method"
    },
    "random_forest": {
        "name": "Random Forest",
        "description": "Ensemble of decision trees - Robust and reliable"
    },
    "decision_tree": {
        "name": "Decision Tree",
        "description": "Single tree-based classifier - Interpretable"
    },
    "lstm": {
        "name": "LSTM",
        "description": "Long Short-Term Memory - Neural network for sequences"
    },
    "transformer": {
        "name": "Transformer",
        "description": "Attention-based architecture - State-of-the-art"
    },
    "mlp": {
        "name": "MLP",
        "description": "Multi-Layer Perceptron - Classic neural network"
    },
    "knn": {
        "name": "K-Nearest Neighbors",
        "description": "Instance-based learning - Simple and effective"
    },
    "naive_bayes": {
        "name": "Naive Bayes",
        "description": "Probabilistic classifier - Fast training"
    },
    "adaboost": {
        "name": "AdaBoost",
        "description": "Adaptive Boosting - Sequential ensemble method"
    }
}

# Style descriptions
STYLES = {
    "beethoven": {
        "name": "Beethoven",
        "description": "Orchestration style inspired by Beethoven's Symphony No. 7",
        "period": "Classical/Romantic"
    },
    "debussy": {
        "name": "Debussy",
        "description": "Orchestration style inspired by Debussy's La Mer",
        "period": "Impressionist"
    },
    "tchaikovsky": {
        "name": "Tchaikovsky",
        "description": "Orchestration style inspired by Tchaikovsky's Sugar Plum Fairy",
        "period": "Romantic"
    }
}


def scan_available_models():
    """
    Scan the weights directory for available model files.
    Returns a dictionary mapping (style, model_type) to file paths.
    
    Expected filename format: {style}_{model_type}.joblib
    Example: beethoven_xgboost.joblib
    
    Also supports legacy format: {style}.joblib (assumed to be XGBoost)
    """
    available = {}
    
    if not WEIGHTS_DIR.exists():
        return available
    
    for file_path in WEIGHTS_DIR.glob("*.joblib"):
        filename = file_path.stem  # Remove .joblib extension
        
        # Try to parse style_modeltype format
        parts = filename.split('_', 1)
        
        if len(parts) == 2:
            # New format: style_modeltype
            style, model_type = parts
            if style in STYLES and model_type in MODEL_TYPES:
                available[(style, model_type)] = str(file_path)
        elif len(parts) == 1:
            # Legacy format: style only (assume XGBoost)
            style = parts[0]
            if style in STYLES:
                available[(style, "xgboost")] = str(file_path)
    
    return available


def get_available_model_types():
    """Get list of model types that have at least one trained model."""
    available_models = scan_available_models()
    model_types = set()
    
    for (style, model_type) in available_models.keys():
        model_types.add(model_type)
    
    return sorted(list(model_types))


def get_available_styles_for_model(model_type: str):
    """Get list of styles available for a specific model type."""
    available_models = scan_available_models()
    styles = set()
    
    for (style, mtype) in available_models.keys():
        if mtype == model_type:
            styles.add(style)
    
    return sorted(list(styles))


def get_all_available_combinations():
    """
    Get all available (model_type, style) combinations.
    Returns list of dicts with model and style information.
    """
    available_models = scan_available_models()
    combinations = []
    
    for (style, model_type), file_path in available_models.items():
        combinations.append({
            "style": style,
            "style_name": STYLES[style]["name"],
            "style_description": STYLES[style]["description"],
            "model_type": model_type,
            "model_name": MODEL_TYPES[model_type]["name"],
            "model_description": MODEL_TYPES[model_type]["description"],
            "file_path": file_path
        })
    
    return combinations


@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the main HTML page."""
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AMO - Automatic Music Orchestration</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Poppins:wght@600;700&display=swap" rel="stylesheet">
    <style>
        * {
            box-sizing: border-box;
        }
        
        body {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            position: relative;
            overflow-x: hidden;
        }
        
        /* Animated musical background */
        .background-notes {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='100' height='100' viewBox='0 0 100 100'%3E%3Ctext x='10' y='50' font-size='40' fill='rgba(255,255,255,0.03)'%3E♪%3C/text%3E%3Ctext x='60' y='80' font-size='30' fill='rgba(255,255,255,0.03)'%3E♫%3C/text%3E%3C/svg%3E");
            background-repeat: repeat;
            animation: float 120s linear infinite;
            pointer-events: none;
            z-index: 0;
        }
        
        @keyframes float {
            from { background-position: 0 0; }
            to { background-position: 1000px 0; }
        }
        
        .main-container {
            max-width: 800px;
            margin: 0 auto;
            position: relative;
            z-index: 1;
        }
        
        .card {
            border-radius: 20px;
            box-shadow: 0 15px 40px rgba(0,0,0,0.4);
            margin-bottom: 24px;
            border: none;
            backdrop-filter: blur(10px);
        }
        
        .card-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #ffffff;
            border-radius: 20px 20px 0 0 !important;
            padding: 28px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
        }
        
        .card-header h1 {
            font-family: 'Poppins', sans-serif;
            font-weight: 700;
            margin-bottom: 8px;
            font-size: 2rem;
            text-shadow: 0 2px 10px rgba(0,0,0,0.2);
        }
        
        .card-header p {
            color: #f8f8f8;
            font-size: 1rem;
        }
        
        .card-body {
            background: white;
        }
        
        .card-section {
            margin-bottom: 1.5rem;
        }
        
        .form-label h5 {
            font-weight: 600;
            color: #2d3748;
            margin-bottom: 0.75rem;
        }
        
        .btn-primary {
            background: linear-gradient(135deg, #7f5eff, #9b7dff);
            border: none;
            padding: 14px 40px;
            font-size: 18px;
            font-weight: 600;
            border-radius: 12px;
            box-shadow: 0 4px 15px rgba(127,94,255,0.4);
            transition: all 0.3s ease;
            color: white;
        }
        
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(127,94,255,0.5);
            background: linear-gradient(135deg, #8b66ff, #a687ff);
        }
        
        .btn-primary:active {
            transform: translateY(0);
        }
        
        .btn-success {
            background: linear-gradient(135deg, #10b981, #059669);
            border: none;
            box-shadow: 0 4px 15px rgba(16,185,129,0.4);
            font-weight: 600;
            border-radius: 12px;
        }
        
        .btn-success:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(16,185,129,0.5);
        }
        
        .btn-secondary {
            border-radius: 12px;
            font-weight: 600;
        }
        
        .spinner-border {
            width: 3rem;
            height: 3rem;
        }
        
        #result {
            display: none;
        }
        
        .file-upload-wrapper {
            position: relative;
            overflow: hidden;
            display: inline-block;
            width: 100%;
        }
        
        .file-upload-wrapper input[type=file] {
            font-size: 100px;
            position: absolute;
            left: 0;
            top: 0;
            opacity: 0;
            cursor: pointer;
            width: 100%;
            height: 100%;
        }
        
        .file-upload-label {
            display: block;
            padding: 32px;
            background: #f8f9fa;
            border: 3px dashed #667eea;
            border-radius: 16px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .file-upload-label:hover {
            background: #e9ecef;
            border-color: #8b7aff;
            box-shadow: 0 0 15px rgba(139,122,255,0.3);
            transform: translateY(-2px);
        }
        
        .file-upload-label i {
            color: #667eea;
            animation: bounce 2s infinite;
        }
        
        @keyframes bounce {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-10px); }
        }
        
        .file-name {
            margin-top: 12px;
            font-weight: 600;
            color: #7f5eff;
            font-size: 1.05rem;
        }
        
        .form-select {
            border-radius: 12px;
            border: 2px solid #e2e8f0;
            padding: 12px 16px;
            font-size: 1rem;
            transition: all 0.3s ease;
        }
        
        .form-select:focus {
            border-color: #7f5eff;
            box-shadow: 0 0 0 3px rgba(127,94,255,0.1);
        }
        
        .text-muted {
            color: #64748b !important;
            font-size: 0.9rem;
            margin-top: 0.5rem;
            display: block;
        }
        
        .alert {
            border-radius: 16px;
            border: none;
            padding: 20px;
        }
        
        .alert-success {
            background: linear-gradient(135deg, #d1fae5, #a7f3d0);
            color: #065f46;
        }
        
        .alert-danger {
            background: linear-gradient(135deg, #fee2e2, #fecaca);
            color: #991b1b;
        }
        
        /* Info card with better contrast */
        .info-card {
            background: rgba(255,255,255,0.95);
            border-radius: 20px;
            padding: 24px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        
        .info-card h5 {
            font-weight: 600;
            color: #1e293b;
            margin-bottom: 12px;
        }
        
        .info-card p {
            color: #475569;
            line-height: 1.7;
            margin-bottom: 0;
        }
        
        /* Footer */
        .footer {
            text-align: center;
            padding: 20px;
            color: rgba(255,255,255,0.9);
            font-size: 0.9rem;
            margin-top: 20px;
        }
        
        .footer a {
            color: white;
            text-decoration: none;
            font-weight: 600;
        }
        
        .footer a:hover {
            text-decoration: underline;
        }
    </style>
</head>
<body>
    <!-- Animated background -->
    <div class="background-notes"></div>
    
    <div class="main-container">
        <!-- Header Card -->
        <div class="card">
            <div class="card-header text-center">
                <h1><i class="fas fa-music"></i> AMO - Automatic Music Orchestration</h1>
                <p class="mb-0">Upload your MIDI file and select an orchestration style</p>
            </div>
        </div>

        <!-- Upload Form Card -->
        <div class="card">
            <div class="card-body p-4">
                <form id="uploadForm" enctype="multipart/form-data">
                    <!-- File Upload Section -->
                    <div class="mb-4 card-section">
                        <label class="form-label"><h5><i class="fas fa-file-audio"></i> Upload MIDI File</h5></label>
                        <div class="file-upload-wrapper">
                            <label class="file-upload-label" for="midiFile">
                                <i class="fas fa-cloud-upload-alt fa-3x mb-3"></i>
                                <div style="font-size: 1.1rem; font-weight: 500;">Click here or drag and drop a MIDI file</div>
                                <small class="text-muted">Accepted formats: .mid, .midi</small>
                            </label>
                            <input type="file" id="midiFile" name="file" accept=".mid,.midi" required>
                        </div>
                        <div id="fileName" class="file-name"></div>
                    </div>

                    <!-- Model Type Selection -->
                    <div class="mb-4 card-section">
                        <label class="form-label"><h5><i class="fas fa-brain"></i> Select ML Model Type</h5></label>
                        <select class="form-select form-select-lg" id="modelType" name="model_type" required>
                            <option value="">Choose a model type...</option>
                        </select>
                        <small class="text-muted" id="modelTypeDesc"></small>
                    </div>

                    <!-- Style Selection -->
                    <div class="mb-4 card-section">
                        <label class="form-label"><h5><i class="fas fa-palette"></i> Select Orchestration Style</h5></label>
                        <select class="form-select form-select-lg" id="style" name="style" required disabled>
                            <option value="">First select a model type...</option>
                        </select>
                        <small class="text-muted" id="styleDesc"></small>
                    </div>

                    <!-- Submit Button -->
                    <div class="text-center mt-4">
                        <button type="submit" class="btn btn-primary btn-lg" id="submitBtn">
                            <i class="fas fa-magic"></i> Start Orchestration
                        </button>
                    </div>
                </form>

                <!-- Loading Spinner -->
                <div id="loading" class="text-center mt-4" style="display: none;">
                    <div class="spinner-border text-primary" role="status">
                        <span class="visually-hidden">Processing...</span>
                    </div>
                    <p class="mt-3">Orchestration in progress, please wait...</p>
                </div>

                <!-- Result Section -->
                <div id="result" class="mt-4">
                    <div class="alert alert-success">
                        <h5><i class="fas fa-check-circle"></i> Orchestration Complete!</h5>
                        <p class="mb-3">Your orchestrated MIDI file is ready to download.</p>
                        <a id="downloadLink" href="#" class="btn btn-success btn-lg">
                            <i class="fas fa-download"></i> Download MIDI
                        </a>
                        <button class="btn btn-secondary btn-lg ms-2" onclick="location.reload()">
                            <i class="fas fa-redo"></i> New Orchestration
                        </button>
                    </div>
                </div>

                <!-- Error Section -->
                <div id="error" class="mt-4" style="display: none;">
                    <div class="alert alert-danger">
                        <h5><i class="fas fa-exclamation-triangle"></i> Error</h5>
                        <p id="errorMessage"></p>
                        <button class="btn btn-danger" onclick="location.reload()">
                            <i class="fas fa-redo"></i> Try Again
                        </button>
                    </div>
                </div>
            </div>
        </div>

        <!-- Info Card -->
        <div class="card info-card">
            <div class="card-body">
                <h5><i class="fas fa-info-circle"></i> About AMO</h5>
                <p>This application uses machine learning models to automatically orchestrate MIDI files in the style of famous composers. The models were trained on orchestrated works by Beethoven, Debussy, and Tchaikovsky.</p>
            </div>
        </div>
        
        <!-- Footer -->
        <div class="footer">
            Made with ❤️ by Gissel Velarde & Florian Schneider | AMO v1.0 | October 2025
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // Load available model types
        async function loadModelTypes() {
            try {
                const response = await fetch('/api/models');
                const data = await response.json();
                const select = document.getElementById('modelType');
                
                data.model_types.forEach(modelType => {
                    const option = document.createElement('option');
                    option.value = modelType.id;
                    option.textContent = modelType.name;
                    option.dataset.description = modelType.description;
                    select.appendChild(option);
                });
            } catch (error) {
                console.error('Error loading model types:', error);
            }
        }

        // Load styles for selected model type
        async function loadStylesForModel(modelType) {
            const styleSelect = document.getElementById('style');
            const styleDesc = document.getElementById('styleDesc');
            
            // Reset
            styleSelect.innerHTML = '<option value="">Loading...</option>';
            styleSelect.disabled = true;
            styleDesc.textContent = '';
            
            try {
                const response = await fetch(`/api/styles/${modelType}`);
                const data = await response.json();
                
                styleSelect.innerHTML = '<option value="">Choose a style...</option>';
                
                data.styles.forEach(style => {
                    const option = document.createElement('option');
                    option.value = style.id;
                    option.textContent = `${style.name} (${style.period})`;
                    option.dataset.description = style.description;
                    option.dataset.period = style.period;
                    styleSelect.appendChild(option);
                });
                
                styleSelect.disabled = false;
            } catch (error) {
                console.error('Error loading styles:', error);
                styleSelect.innerHTML = '<option value="">Error loading styles</option>';
            }
        }

        // Model type selection handler
        document.getElementById('modelType').addEventListener('change', function(e) {
            const modelTypeDesc = document.getElementById('modelTypeDesc');
            const selected = e.target.selectedOptions[0];
            
            if (selected && selected.dataset.description) {
                modelTypeDesc.textContent = selected.dataset.description;
                loadStylesForModel(e.target.value);
            } else {
                modelTypeDesc.textContent = '';
                document.getElementById('style').disabled = true;
                document.getElementById('style').innerHTML = '<option value="">First select a model type...</option>';
                document.getElementById('styleDesc').textContent = '';
            }
        });

        // Style selection handler
        document.getElementById('style').addEventListener('change', function(e) {
            const styleDesc = document.getElementById('styleDesc');
            const selected = e.target.selectedOptions[0];
            
            if (selected && selected.dataset.description) {
                styleDesc.textContent = `${selected.dataset.description} (${selected.dataset.period} period)`;
            } else {
                styleDesc.textContent = '';
            }
        });

        // File input change handler
        document.getElementById('midiFile').addEventListener('change', function(e) {
            const fileName = e.target.files[0]?.name;
            if (fileName) {
                document.getElementById('fileName').textContent = `Selected: ${fileName}`;
            }
        });

        // Form submission
        document.getElementById('uploadForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const fileInput = document.getElementById('midiFile');
            const modelTypeInput = document.getElementById('modelType');
            const styleInput = document.getElementById('style');
            
            // Validate inputs
            if (!fileInput.files[0]) {
                alert('Please select a MIDI file.');
                return;
            }
            
            if (!modelTypeInput.value) {
                alert('Please select a model type.');
                return;
            }
            
            if (!styleInput.value) {
                alert('Please select an orchestration style.');
                return;
            }
            
            // Prepare form data
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);
            formData.append('model_type', modelTypeInput.value);
            formData.append('style', styleInput.value);
            
            // Show loading, hide other sections
            document.getElementById('loading').style.display = 'block';
            document.getElementById('submitBtn').disabled = true;
            document.getElementById('result').style.display = 'none';
            document.getElementById('error').style.display = 'none';
            
            try {
                const response = await fetch('/api/orchestrate', {
                    method: 'POST',
                    body: formData
                });
                
                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(errorData.detail || 'Orchestration failed');
                }
                
                const result = await response.json();
                
                // Show result
                document.getElementById('loading').style.display = 'none';
                document.getElementById('result').style.display = 'block';
                document.getElementById('downloadLink').href = `/api/download/${result.output_file}`;
                
            } catch (error) {
                console.error('Error:', error);
                document.getElementById('loading').style.display = 'none';
                document.getElementById('error').style.display = 'block';
                document.getElementById('errorMessage').textContent = error.message;
                document.getElementById('submitBtn').disabled = false;
            }
        });

        // Load model types on page load
        loadModelTypes();
    </script>
</body>
</html>
    """
    return html_content


@app.get("/api/models")
async def get_models():
    """Get list of available model types."""
    model_types = get_available_model_types()
    return {
        "model_types": [
            {
                "id": mt,
                "name": MODEL_TYPES[mt]["name"],
                "description": MODEL_TYPES[mt]["description"]
            }
            for mt in model_types
        ]
    }


@app.get("/api/styles/{model_type}")
async def get_styles_for_model(model_type: str):
    """Get list of available styles for a specific model type."""
    if model_type not in MODEL_TYPES:
        raise HTTPException(status_code=400, detail=f"Invalid model type: {model_type}")
    
    styles = get_available_styles_for_model(model_type)
    return {
        "styles": [
            {
                "id": style,
                "name": STYLES[style]["name"],
                "description": STYLES[style]["description"],
                "period": STYLES[style]["period"]
            }
            for style in styles
        ]
    }


@app.get("/api/combinations")
async def get_combinations():
    """Get all available model/style combinations."""
    return {
        "combinations": get_all_available_combinations(),
        "total": len(get_all_available_combinations())
    }


@app.post("/api/orchestrate")
async def orchestrate_midi(
    file: UploadFile = File(...),
    model_type: str = Form(...),
    style: str = Form(...)
):
    """
    Orchestrate a MIDI file using the selected model type and style.
    
    Parameters:
    - file: Uploaded MIDI file
    - model_type: ML model type (xgboost, lstm, random_forest, etc.)
    - style: Orchestration style (beethoven, debussy, tchaikovsky, etc.)
    """
    # Validate model type
    if model_type not in MODEL_TYPES:
        raise HTTPException(status_code=400, detail=f"Invalid model type: {model_type}")
    
    # Validate style
    if style not in STYLES:
        raise HTTPException(status_code=400, detail=f"Invalid style: {style}")
    
    # Check if this combination exists
    available_models = scan_available_models()
    if (style, model_type) not in available_models:
        raise HTTPException(
            status_code=404,
            detail=f"No trained model found for {STYLES[style]['name']} with {MODEL_TYPES[model_type]['name']}"
        )
    
    pipeline_path = Path(available_models[(style, model_type)])
    
    # Validate file type
    if not file.filename.endswith(('.mid', '.midi')):
        raise HTTPException(
            status_code=400, 
            detail="Invalid file type. Only .mid or .midi files are accepted."
        )
    
    try:
        # Save uploaded file
        upload_path = UPLOAD_DIR / file.filename
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Generate output filename
        output_filename = f"{Path(file.filename).stem}_{style}_{model_type}_orchestrated.mid"
        output_path = OUTPUT_DIR / output_filename
        
        # Perform orchestration
        print(f"[WEB APP] Starting orchestration with {STYLES[style]['name']} style using {MODEL_TYPES[model_type]['name']} model...")
        orchestrated_path = amo_load_and_orchestrate(
            pipeline_path=str(pipeline_path),
            target_midi_path=str(upload_path),
            output_midi_path=str(output_path)
        )
        
        print(f"[WEB APP] Orchestration complete: {orchestrated_path}")
        
        # Clean up uploaded file
        upload_path.unlink()
        
        return {
            "success": True,
            "message": "Orchestration completed successfully",
            "output_file": output_filename,
            "model_type": MODEL_TYPES[model_type]["name"],
            "style": STYLES[style]["name"]
        }
    
    except Exception as e:
        print(f"[WEB APP ERROR] {str(e)}")
        raise HTTPException(status_code=500, detail=f"Orchestration failed: {str(e)}")


@app.get("/api/download/{filename}")
async def download_file(filename: str):
    """Download the orchestrated MIDI file."""
    file_path = OUTPUT_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="audio/midi"
    )


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    combinations = get_all_available_combinations()
    return {
        "status": "healthy",
        "available_combinations": len(combinations),
        "model_types": len(get_available_model_types()),
        "styles": len(set(c["style"] for c in combinations))
    }


if __name__ == "__main__":
    print("=" * 60)
    print("AMO - Automatic Music Orchestration Web Application")
    print("=" * 60)
    print("Available combinations:")
    combinations = get_all_available_combinations()
    for combo in combinations:
        print(f"  ✓ {combo['style_name']} + {combo['model_name']}")
    print("=" * 60)
    print("Starting server on http://localhost:8001")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8001)
