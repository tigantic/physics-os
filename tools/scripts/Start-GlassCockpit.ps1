# Quick Launch Script for Glass Cockpit with E-Core Enforcement
# 
# Usage:
#   .\Start-GlassCockpit.ps1           # Launch and enforce affinity
#   .\Start-GlassCockpit.ps1 -Release  # Use release build

param(
    [Parameter(Mandatory=$false)]
    [switch]$Release = $false,
    
    [Parameter(Mandatory=$false)]
    [switch]$Measure = $false
)

$ErrorActionPreference = "Stop"

# Navigate to glass-cockpit directory
$repoRoot = Split-Path -Parent $PSScriptRoot
$glassCockpitPath = Join-Path $repoRoot "glass-cockpit"

if (-not (Test-Path $glassCockpitPath)) {
    Write-Host "✗ Glass Cockpit directory not found: $glassCockpitPath" -ForegroundColor Red
    exit 1
}

Set-Location $glassCockpitPath

Write-Host "═══════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host " The Physics OS Glass Cockpit - Phase 0 Launcher" -ForegroundColor White
Write-Host "═══════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

# Build
if ($Release) {
    Write-Host "[1/3] Building (Release mode)..." -ForegroundColor Yellow
    cargo build --release
    $exePath = "target\release\glass-cockpit.exe"
} else {
    Write-Host "[1/3] Building (Debug mode)..." -ForegroundColor Yellow
    cargo build
    $exePath = "target\debug\glass-cockpit.exe"
}

if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Build failed" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "  ✓ Build complete" -ForegroundColor Green
Write-Host ""

# Launch in background
Write-Host "[2/3] Launching Glass Cockpit..." -ForegroundColor Yellow

$processInfo = New-Object System.Diagnostics.ProcessStartInfo
$processInfo.FileName = $exePath
$processInfo.UseShellExecute = $false
$processInfo.CreateNoWindow = $false

$process = New-Object System.Diagnostics.Process
$process.StartInfo = $processInfo

$started = $process.Start()

if (-not $started) {
    Write-Host "✗ Failed to start process" -ForegroundColor Red
    exit 1
}

Write-Host "  ✓ Process started (PID $($process.Id))" -ForegroundColor Green
Write-Host ""

# Wait for window to initialize
Start-Sleep -Seconds 2

# Apply E-core affinity
Write-Host "[3/3] Enforcing E-core affinity..." -ForegroundColor Yellow

$affinityScript = Join-Path $repoRoot "scripts\Set-ECoreAffinity.ps1"

if ($Measure) {
    & $affinityScript -ProcessName "glass-cockpit" -Measure
} else {
    & $affinityScript -ProcessName "glass-cockpit"
}

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "⚠ Affinity enforcement failed, but Glass Cockpit is running" -ForegroundColor Yellow
    Write-Host "You can manually set affinity in Task Manager" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "═══════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "Phase 0 Running" -ForegroundColor Green
Write-Host "═══════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press ESC in Glass Cockpit window to exit" -ForegroundColor Gray
Write-Host ""
