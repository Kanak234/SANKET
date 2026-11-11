# build_msix.ps1 — builds SANKET.exe with PyInstaller and packs it into an MSIX for the Store.
#
# WHY MSIX: the Microsoft Store signs MSIX packages itself, so no paid code-signing certificate is
#      needed; the same route Kanak used for Likhit and MEDHA.
# NEEDS: models/recognizer.onnx + models/signs.npz (from scripts/run_full_pipeline.ps1), the frozen
#      MediaPipe/Vosk models (scripts/fetch_models.py), the Windows 10/11 SDK (makeappx.exe), and
#      packaging/store_identity.json (already filled from Partner Center → Product identity, Store ID 9NH6TTQJQTGJ):
#      { "IDENTITY_NAME": "12345KanakPrabhakar.SANKET", "PUBLISHER": "CN=XXXXXXXX-....",
#        "PUBLISHER_DISPLAY_NAME": "Kanak Prabhakar", "VERSION": "1.0.0.0" }
# OUTPUT: dist\SANKET_<version>_x64.msix  → upload in Partner Center → Packages.
# Usage:  .\packaging\build_msix.ps1            (add -AllowDemo to build without trained models, for testing only)

param([switch]$AllowDemo)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

foreach ($f in @("models\recognizer.onnx", "models\signs.npz")) {
    if (-not (Test-Path $f)) {
        if ($AllowDemo) { Write-Warning "$f missing — building a DEMO package (do NOT submit to the Store)" }
        else { throw "$f missing. Run scripts\run_full_pipeline.ps1 first (or pass -AllowDemo for a test build)." }
    }
}
$idFile = "packaging\store_identity.json"
if (-not (Test-Path $idFile)) { throw "$idFile missing — copy the values from Partner Center → Product identity." }
$id = Get-Content $idFile -Raw | ConvertFrom-Json

uv run python scripts/fetch_models.py
if ($LASTEXITCODE -ne 0) { throw "fetching MediaPipe/Vosk models failed" }
uv run --with pyinstaller pyinstaller packaging/sanket.spec --noconfirm --distpath dist --workpath build
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }
& dist\SANKET\SANKET.exe --selftest | Out-Host

$stage = "build\msix"
if (Test-Path $stage) { Remove-Item $stage -Recurse -Force }
New-Item -ItemType Directory $stage | Out-Null
Copy-Item dist\SANKET "$stage\SANKET" -Recurse
New-Item -ItemType Directory "$stage\assets\store" -Force | Out-Null
Copy-Item assets\store\*.png "$stage\assets\store\"
$manifest = (Get-Content packaging\AppxManifest.xml -Raw).
    Replace("{IDENTITY_NAME}", $id.IDENTITY_NAME).Replace("{PUBLISHER}", $id.PUBLISHER).
    Replace("{PUBLISHER_DISPLAY_NAME}", $id.PUBLISHER_DISPLAY_NAME).Replace("{VERSION}", $id.VERSION)
Set-Content "$stage\AppxManifest.xml" $manifest -Encoding UTF8

$makeappx = Get-ChildItem "${env:ProgramFiles(x86)}\Windows Kits\10\bin" -Recurse -Filter makeappx.exe |
    Where-Object { $_.FullName -match "\\x64\\" } | Sort-Object FullName -Descending | Select-Object -First 1
if (-not $makeappx) { throw "makeappx.exe not found — install the Windows 10/11 SDK" }
$out = "dist\SANKET_$($id.VERSION)_x64.msix"
& $makeappx.FullName pack /d $stage /p $out /o
if ($LASTEXITCODE -ne 0) { throw "makeappx failed" }
Write-Host "`nMSIX ready: $out  → Partner Center → SANKET → Packages → upload" -ForegroundColor Green
