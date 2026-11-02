# Build Windows CLI exe (CPU version) for ordinary users.
#
# Usage (from D:\minimind\learning\scratch):
#   powershell -ExecutionPolicy Bypass -File .\packaging\build_cpu.ps1
#   powershell -ExecutionPolicy Bypass -File .\packaging\build_cpu.ps1 -TextOnly
#
# Output:
#   packaging\dist\minimind\minimind.exe
#   packaging\dist\minimind\model\...
#   packaging\dist\MiniMind-CLI-Windows.zip
#
# Prerequisite: D:\minimind\.build_env with CPU torch / transformers / pillow / pyinstaller.

param(
    [string]$BuildPython = "D:\minimind\.build_env\Scripts\python.exe",
    [string]$ModelSource = "D:\minimind\learning\scratch\out\save",
    [string]$TokenizerSource = "D:\minimind\minimind\model",
    [string]$VisionSource = "D:\minimind\minimind-v\model\siglip2-base-p32-256-ve",
    [string]$OutRoot = "",
    [string]$WorkRoot = "",
    [switch]$TextOnly
)

$ErrorActionPreference = "Stop"
$Scratch = Split-Path -Parent $PSScriptRoot
if (-not $OutRoot) { $OutRoot = Join-Path $PSScriptRoot "dist" }
if (-not $WorkRoot) { $WorkRoot = Join-Path $PSScriptRoot "build_pyi" }
$Entry = Join-Path $PSScriptRoot "entry.py"

if (-not (Test-Path $BuildPython)) {
    throw "Build Python not found: $BuildPython"
}
if (-not (Test-Path $Entry)) {
    throw "Entry script not found: $Entry"
}

Write-Host "== 1/4 Clean old output =="
if (Test-Path $OutRoot) { Remove-Item $OutRoot -Recurse -Force }
if (Test-Path $WorkRoot) { Remove-Item $WorkRoot -Recurse -Force }

Write-Host "== 2/4 PyInstaller build (CPU, 3-6 min) =="
Push-Location $Scratch
try {
    & $BuildPython -m PyInstaller --noconfirm --clean `
        --onedir --console --name minimind `
        --distpath $OutRoot `
        --workpath $WorkRoot `
        --specpath $WorkRoot `
        --paths $Scratch `
        --collect-submodules mymm `
        --collect-submodules transformers `
        --collect-data transformers `
        --collect-submodules tokenizers `
        --collect-data tokenizers `
        --collect-submodules safetensors `
        $Entry
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed with exit code $LASTEXITCODE" }
} finally {
    Pop-Location
}

$AppDir = Join-Path $OutRoot "minimind"
$ModelDir = Join-Path $AppDir "model"
New-Item -ItemType Directory -Force -Path $ModelDir | Out-Null

Write-Host "== 3/4 Copy model and tokenizer =="
$llm = Join-Path $ModelSource "full_sft_768.pth"
if (-not (Test-Path $llm)) { throw "LLM weight not found: $llm" }
Copy-Item $llm (Join-Path $ModelDir "full_sft_768.pth") -Force
Write-Host "  full_sft_768.pth"

$TokenizerDir = Join-Path $ModelDir "tokenizer"
New-Item -ItemType Directory -Force -Path $TokenizerDir | Out-Null
foreach ($name in @("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")) {
    $src = Join-Path $TokenizerSource $name
    if (Test-Path $src) { Copy-Item $src (Join-Path $TokenizerDir $name) -Force }
}

if (-not $TextOnly) {
    $vlm = Join-Path $ModelSource "sft_vlm_768.pth"
    if (Test-Path $vlm) {
        Copy-Item $vlm (Join-Path $ModelDir "sft_vlm_768.pth") -Force
        Write-Host "  sft_vlm_768.pth"
    } else {
        Write-Host "  [skip] sft_vlm_768.pth not found (text chat only)"
    }
    if (Test-Path $VisionSource) {
        $VisionDir = Join-Path $ModelDir "siglip2"
        Copy-Item $VisionSource $VisionDir -Recurse -Force
        Write-Host "  siglip2/ (vision encoder)"
    } else {
        Write-Host "  [skip] vision encoder not found: $VisionSource"
    }
}

$usageSrc = Join-Path $PSScriptRoot "usage.txt"
$usageDst = Join-Path $AppDir "README.txt"
if (Test-Path $usageSrc) {
    Copy-Item $usageSrc $usageDst -Force
}

Write-Host "== 4/4 Zip =="
$zip = Join-Path $OutRoot "MiniMind-CLI-Windows.zip"
Compress-Archive -Path $AppDir -DestinationPath $zip -Force

Write-Host ""
Write-Host "Done:"
Write-Host "  folder: $AppDir"
Write-Host "  exe   : $(Join-Path $AppDir 'minimind.exe')"
Write-Host "  zip   : $zip"
