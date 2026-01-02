# Phase 4 Validation Runner (PowerShell)
# Runs validation in WSL without terminal issues

Write-Host "======================================" -ForegroundColor Cyan
Write-Host "Phase 4 Component Validation" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""

$wslPath = "/home/brad/TiganticLabz/Main_Projects/Project HyperTensor"

# Run validation in WSL
wsl bash -c "cd '$wslPath' && export GIT_PAGER=cat && python3 test_phase4_validation.py"

Write-Host ""
Write-Host "======================================" -ForegroundColor Green
Write-Host "Validation Complete" -ForegroundColor Green
Write-Host "======================================" -ForegroundColor Green
