# AMO Web Application Starter Script
# This script starts the AMO Web Application

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "AMO - Automatic Music Orchestration Web App" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Check if Python is available
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: Python not found!" -ForegroundColor Red
    Write-Host "Please install Python or activate your Conda environment." -ForegroundColor Yellow
    exit 1
}

# Show Python version
$pythonVersion = python --version
Write-Host "Python Version: $pythonVersion" -ForegroundColor Green

# Check if FastAPI is installed
$fastApiCheck = python -c "import fastapi; print('OK')" 2>$null
if (-not $fastApiCheck) {
    Write-Host ""
    Write-Host "FastAPI is not installed." -ForegroundColor Yellow
    Write-Host "Installing Web App dependencies..." -ForegroundColor Yellow
    pip install -r requirements_webapp.txt
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Installation failed!" -ForegroundColor Red
        exit 1
    }
}

# Check available models
Write-Host ""
Write-Host "Checking available models..." -ForegroundColor Yellow
$models = @("beethoven.joblib", "debussy.joblib", "tchaikovsky.joblib")
$foundModels = 0
foreach ($model in $models) {
    if (Test-Path "weights\$model") {
        Write-Host "  [OK] $model found" -ForegroundColor Green
        $foundModels++
    } else {
        Write-Host "  [!!] $model not found" -ForegroundColor Yellow
    }
}

if ($foundModels -eq 0) {
    Write-Host ""
    Write-Host "WARNING: No models found in weights/ directory!" -ForegroundColor Red
    Write-Host "Please ensure the .joblib files are present." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Starting server..." -ForegroundColor Cyan
Write-Host "URL: http://localhost:8001" -ForegroundColor Green
Write-Host "Press Ctrl+C to quit" -ForegroundColor Yellow
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Start server
python webapp.py
