# run_full_pipeline.ps1 — ONE command from raw INCLUDE videos to a ready-to-ship SANKET model.
#
# WHY: Kanak runs this once on the i9 desktop (RTX 5060 Ti). Every step is resumable, so after a
#      power cut he simply runs the same command again and it continues where it stopped.
# STEPS: 1 install deps → 2 fetch frozen models (MediaPipe, Vosk) → 3 download INCLUDE (56.8 GB,
#        resumable) → 4 extract landmarks (resumable) → 5 train from scratch (resumable)
#        → 6 evaluate → 7 export ONNX + parity check → 8 build avatar sign dictionary → 9 self-test.
# The first failing step stops the script with its error message.
#
# Usage (PowerShell, in the repo folder):   .\scripts\run_full_pipeline.ps1
# Optional: $env:SANKET_DATA = "E:\INCLUDE"  to put the dataset on another drive.
#
# -FromCache: skip steps 3–4 (download + extraction) because the landmark cache was made on Google
#             Colab (notebooks/SANKET_colab_landmarks.ipynb). Put SANKET_cache.zip from Google
#             Drive next to this repo folder or pass -CacheZip <path>; it is unpacked into data\cache.

param([switch]$FromCache, [string]$CacheZip = "$env:USERPROFILE\Downloads\SANKET_cache.zip")
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

function Step($name, [scriptblock]$cmd) {
    Write-Host "`n=== $name ===" -ForegroundColor Cyan
    & $cmd
    if ($LASTEXITCODE -ne 0) { throw "Step failed: $name (exit $LASTEXITCODE)" }
}

Step "1/9 Install Python 3.12 environment + training extras" { uv sync --extra train --extra dev }
Step "2/9 Fetch MediaPipe + Vosk models (once)"           { uv run python scripts/fetch_models.py }
if ($FromCache) {
    if (Test-Path $CacheZip) {
        # Always unpack when the zip is there (-Force overwrites), so a half-finished home cache
        # can never block the complete Colab one.
        Write-Host "`n=== 3-4/9 Unpacking Colab landmark cache $CacheZip ===" -ForegroundColor Cyan
        New-Item -ItemType Directory data -Force | Out-Null
        Expand-Archive -Path $CacheZip -DestinationPath data -Force   # zip contains the 'cache' folder
    } elseif (-not (Test-Path "data\cache")) {
        throw "Cache zip not found: $CacheZip (download SANKET_cache.zip from Google Drive)"
    }
    $n = (Get-ChildItem data\cache -Recurse -Filter *.npz).Count
    Write-Host "landmark files in data\cache: $n (expected ~4,290)"
} else {
    Step "3/9 Download INCLUDE dataset (resumable)"             { uv run python -m training.download_include }
    Step "4/9 Extract landmarks from videos (resumable)"        { uv run python -m training.extract_landmarks --workers 8 }
}
Step "5/9 Train recognizer from scratch (resumable)"        { uv run python -m training.train --workers 6 }
Step "6/9 Evaluate on TEST split + calibrate threshold"     { uv run python -m training.evaluate }
Step "7/9 Export ONNX + parity check"                       { uv run python -m training.export_onnx }
Step "8/9 Build avatar sign dictionary"                     { uv run python -m training.build_sign_dictionary }
Step "9/9 App self-test"                                    { uv run python -m sanket --selftest }

Write-Host "`nAll done. Review reports\metrics.json and reports\*.png." -ForegroundColor Green
