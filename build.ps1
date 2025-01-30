# build.ps1
Write-Host "Building DocType Classifier..." -ForegroundColor Green

# Ensure we're in the correct directory
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptPath

# Activate virtual environment if it exists, create if it doesn't
$venvPath = Join-Path $scriptPath "venv"
$pythonPath = Join-Path $venvPath "Scripts\python.exe"
$pipPath = Join-Path $venvPath "Scripts\pip.exe"

if (-not (Test-Path $pythonPath)) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& "$venvPath\Scripts\Activate.ps1"

# Install required packages
Write-Host "Installing required packages..." -ForegroundColor Yellow
& $pythonPath -m pip install --upgrade pip
& $pythonPath -m pip install pyinstaller pillow python-magic-bin poppler-utils pdf2image pytesseract openai

# Clean previous builds
if (Test-Path "dist") {
    Remove-Item -Path "dist" -Recurse -Force
}
if (Test-Path "build") {
    Remove-Item -Path "build" -Recurse -Force
}

# Create assets directory if it doesn't exist
if (-not (Test-Path "assets")) {
    New-Item -ItemType Directory -Path "assets" -Force
    
    # Download a default icon if none exists
    if (-not (Test-Path "assets/icon.ico")) {
        Invoke-WebRequest -Uri "https://www.google.com/favicon.ico" -OutFile "assets/icon.ico"
    }
}

# Build the executable
Write-Host "Building executable..." -ForegroundColor Yellow
& $pythonPath -m PyInstaller DocTypeClassifier.spec --clean

# Check if executable was created
$exePath = Join-Path $scriptPath "dist\DocType Classifier.exe"
if (Test-Path $exePath) {
    # Create distribution folder
    $distFolder = "DocType_Classifier_Distribution"
    if (Test-Path $distFolder) {
        Remove-Item -Path $distFolder -Recurse -Force
    }
    New-Item -ItemType Directory -Path $distFolder

    # Copy files to distribution folder
    Copy-Item $exePath -Destination $distFolder
    Copy-Item "INSTALLATION.txt" -Destination $distFolder
    Copy-Item "README.txt" -Destination $distFolder
    
    # Create ZIP file
    Compress-Archive -Path $distFolder -DestinationPath "DocType_Classifier.zip" -Force
    
    Write-Host "Build complete! Distribution package created at: DocType_Classifier.zip" -ForegroundColor Green
} else {
    Write-Host "Build failed! Executable not found." -ForegroundColor Red
    exit 1
} 