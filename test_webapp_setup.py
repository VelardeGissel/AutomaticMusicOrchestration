"""
Test script to verify the webapp functionality
Run this before starting the webapp to ensure everything is set up correctly
"""

import sys
from pathlib import Path

print("=" * 70)
print("AMO Web Application - Pre-flight Check")
print("=" * 70)
print()

# Check 1: Python version
print("1. Python Version:")
print(f"   ✓ {sys.version}")
print()

# Check 2: Required packages
print("2. Required Packages:")
required_packages = [
    ('fastapi', 'FastAPI'),
    ('uvicorn', 'Uvicorn'),
    ('mido', 'Mido'),
    ('pandas', 'Pandas'),
    ('numpy', 'NumPy'),
    ('sklearn', 'Scikit-learn'),
    ('xgboost', 'XGBoost'),
    ('joblib', 'Joblib')
]

missing_packages = []
for package, name in required_packages:
    try:
        __import__(package)
        print(f"   ✓ {name}")
    except ImportError:
        print(f"   ✗ {name} - NOT INSTALLED")
        missing_packages.append(package)

print()

# Check 3: Model files
print("3. Model Files in weights/:")
weights_dir = Path("weights")
model_files = ['beethoven.joblib', 'debussy.joblib', 'tchaikovsky.joblib']
missing_models = []

for model_file in model_files:
    model_path = weights_dir / model_file
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"   ✓ {model_file} ({size_mb:.2f} MB)")
    else:
        print(f"   ✗ {model_file} - NOT FOUND")
        missing_models.append(model_file)

print()

# Check 4: AMO module
print("4. AMO Module:")
sys.path.append('amos')
try:
    from amos.ml_orchestration import amo_load_and_orchestrate
    print("   ✓ amo_load_and_orchestrate imported successfully")
except ImportError as e:
    print(f"   ✗ Failed to import: {e}")
    missing_packages.append('amos')

print()

# Check 5: Test MIDI files
print("5. Test MIDI Files in midis/:")
midis_dir = Path("midis")
if midis_dir.exists():
    midi_files = list(midis_dir.glob("*.mid"))
    if midi_files:
        print(f"   ✓ Found {len(midi_files)} MIDI files:")
        for midi_file in midi_files[:5]:  # Show first 5
            print(f"     - {midi_file.name}")
        if len(midi_files) > 5:
            print(f"     ... and {len(midi_files) - 5} more")
    else:
        print("   ⚠ No MIDI files found (optional)")
else:
    print("   ⚠ midis/ directory not found (optional)")

print()

# Final verdict
print("=" * 70)
if missing_packages or missing_models:
    print("❌ FAILED - Missing dependencies:")
    if missing_packages:
        print(f"   Packages: {', '.join(missing_packages)}")
        print("   Install with: pip install -r requirements_webapp.txt")
    if missing_models:
        print(f"   Models: {', '.join(missing_models)}")
        print("   Make sure model files exist in weights/ directory")
    print()
    print("Please fix the issues above before starting the webapp.")
else:
    print("✅ ALL CHECKS PASSED")
    print()
    print("You're ready to start the webapp!")
    print()
    print("To start the server, run one of:")
    print("  • powershell: .\\start_webapp.ps1")
    print("  • python: python webapp.py")
    print("  • uvicorn: uvicorn webapp:app --reload")
    print()
    print("Then open: http://localhost:8001")

print("=" * 70)
