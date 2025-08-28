# setup.ps1
# PowerShell script to set up venv, install dependencies, and build exe with PyInstaller

# Go to project directory (this script's folder)
cd $PSScriptRoot

# Step 1: Create venv if not exists
if (-not (Test-Path "venv")) {
    python -m venv venv
    Write-Host "✅ Virtual environment created."
}

# Step 2: Activate venv
& .\venv\Scripts\Activate.ps1
Write-Host "✅ Virtual environment activated."

# Step 3: Upgrade pip
python -m pip install --upgrade pip

# Step 4: Create requirements.txt if missing
if (-not (Test-Path "requirements.txt")) {
    @"
streamlit
plotly
matplotlib
tensorflow==2.15
scikit-learn
seaborn
numpy
pandas
pyinstaller
"@ | Out-File -Encoding UTF8 requirements.txt
    Write-Host "📄 requirements.txt created."
}

# Step 5: Install dependencies
python -m pip install -r requirements.txt
Write-Host "✅ Requirements installed."

# Step 6: Build exe with PyInstaller
# 👉 Change 'main.py' to your actual script filename
$mainScript = "Model.py"

if (Test-Path $mainScript) {
    python -m PyInstaller $mainScript --onefile
    Write-Host "✅ Build complete. Check the 'dist' folder."
} else {
    Write-Host "⚠ Model.py not found. Edit setup.ps1 and set the correct filename."
}
