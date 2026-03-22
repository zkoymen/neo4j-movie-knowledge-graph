$ErrorActionPreference = "Stop"

$venvPython = ".\.venv\Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    throw "Virtual environment not found. Run .\setup_env.ps1 first."
}

& $venvPython main.py
