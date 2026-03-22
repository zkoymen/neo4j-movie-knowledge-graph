$ErrorActionPreference = "Stop"

$venvPython = ".\.venv\Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    Write-Host "Creating virtual environment in .venv ..."
    python -m venv .venv
}

Write-Host "Using virtual environment: $venvPython"
& $venvPython -m pip install --upgrade pip setuptools wheel
& $venvPython -m pip install -r requirements.txt

Write-Host ""
Write-Host "Environment is ready."
Write-Host "Run the project with:"
Write-Host ".\\.venv\\Scripts\\python main.py"
