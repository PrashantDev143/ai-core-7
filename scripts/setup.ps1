# Creates backend/.venv and installs dependencies.
#
# Two things this works around:
#   - `python` on PATH is the Microsoft Store alias stub, so the interpreter is
#     located explicitly instead of trusted from PATH.
#   - pip's default torch wheel is the CUDA build (~2.5GB). This machine has no
#     GPU, so torch comes from the CPU index first and sentence-transformers
#     then finds it already satisfied.

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$venv = Join-Path $root "backend\.venv"

function Find-Python311 {
    $candidates = @(
        "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe"
    )
    foreach ($c in $candidates) {
        if (Test-Path $c) {
            $v = & $c -c "import sys; print(sys.version_info >= (3,11))" 2>$null
            if ($v -eq "True") { return $c }
        }
    }
    throw "No Python 3.11+ found. Install from python.org and re-run."
}

if (-not (Test-Path $venv)) {
    $py = Find-Python311
    Write-Host "Creating venv with $py"
    & $py -m venv $venv
}

$vpy = Join-Path $venv "Scripts\python.exe"

& $vpy -m pip install --upgrade pip setuptools wheel --quiet
& $vpy -m pip install torch --index-url https://download.pytorch.org/whl/cpu
& $vpy -m pip install -e "$root\backend[dev]"

Write-Host ""
Write-Host "Done. Activate with:"
Write-Host "  backend\.venv\Scripts\Activate.ps1"
