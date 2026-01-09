# AMO Web Application Starter Script
# This script starts the AMO Web Application

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "AMO - Automatic Music Orchestration Web App" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..\\..")

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
    $requirementsPath = Join-Path $ScriptDir "requirements.txt"
    pip install -r $requirementsPath
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
    $modelPath = Join-Path $RepoRoot "models\\weights\\$model"
    if (Test-Path $modelPath) {
        Write-Host "  [OK] $model found" -ForegroundColor Green
        $foundModels++
    } else {
        Write-Host "  [!!] $model not found" -ForegroundColor Yellow
    }
}

if ($foundModels -eq 0) {
    Write-Host ""
    Write-Host "WARNING: No models found in models/weights/ directory!" -ForegroundColor Red
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
python (Join-Path $ScriptDir "webapp.py")
