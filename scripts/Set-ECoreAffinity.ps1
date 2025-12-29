# E-Core Affinity Enforcement Script
# 
# Doctrine 1: Computational Sovereignty
# 
# Pins Glass Cockpit process to E-cores (16-31) on i9-14900HX
# and measures stability score before/after enforcement.

param(
    [Parameter(Mandatory=$false)]
    [string]$ProcessName = "glass-cockpit",
    
    [Parameter(Mandatory=$false)]
    [switch]$Validate = $false,
    
    [Parameter(Mandatory=$false)]
    [switch]$Measure = $false
)

# E-core affinity mask (logical processors 16-31)
$ECORE_MASK = 0xFFFF0000

function Write-Banner {
    param([string]$Message)
    Write-Host ""
    Write-Host "═══════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host " $Message" -ForegroundColor White
    Write-Host "═══════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host ""
}

function Get-ProcessorInfo {
    Write-Host "Detecting CPU topology..." -ForegroundColor Yellow
    
    $processors = Get-CimInstance -ClassName Win32_Processor
    
    foreach ($proc in $processors) {
        Write-Host "  CPU: $($proc.Name)" -ForegroundColor Green
        Write-Host "  Cores: $($proc.NumberOfCores)" -ForegroundColor Green
        Write-Host "  Logical Processors: $($proc.NumberOfLogicalProcessors)" -ForegroundColor Green
    }
    
    # Validate i9-14900HX expectations
    $totalLogical = ($processors | Measure-Object -Property NumberOfLogicalProcessors -Sum).Sum
    
    if ($totalLogical -ne 32) {
        Write-Host ""
        Write-Host "⚠ WARNING: Expected 32 logical processors (i9-14900HX)" -ForegroundColor Yellow
        Write-Host "  Found: $totalLogical" -ForegroundColor Yellow
        Write-Host "  E-core mask (0xFFFF0000) assumes cores 16-31 are E-cores" -ForegroundColor Yellow
        Write-Host "  Validate with Task Manager before trusting results" -ForegroundColor Yellow
    }
    
    Write-Host ""
}

function Set-ProcessAffinityMask {
    param(
        [int]$ProcessId,
        [long]$AffinityMask
    )
    
    try {
        $process = Get-Process -Id $ProcessId -ErrorAction Stop
        
        # Set affinity using ProcessorAffinity property
        $process.ProcessorAffinity = [IntPtr]$AffinityMask
        
        return $true
    } catch {
        Write-Host "  ✗ Failed to set affinity: $_" -ForegroundColor Red
        return $false
    }
}

function Get-ProcessAffinityMask {
    param([int]$ProcessId)
    
    try {
        $process = Get-Process -Id $ProcessId -ErrorAction Stop
        return [long]$process.ProcessorAffinity
    } catch {
        return $null
    }
}

function Test-AffinityMask {
    param(
        [long]$CurrentMask,
        [long]$ExpectedMask
    )
    
    return ($CurrentMask -eq $ExpectedMask)
}

function Measure-StabilityScore {
    param([int]$ProcessId)
    
    Write-Host "Measuring stability over 10 seconds..." -ForegroundColor Yellow
    
    $samples = @()
    $iterations = 10
    
    for ($i = 0; $i -lt $iterations; $i++) {
        Start-Sleep -Seconds 1
        
        try {
            $proc = Get-Process -Id $ProcessId -ErrorAction Stop
            $cpuTime = $proc.CPU
            $samples += $cpuTime
            
            Write-Host "  Sample $($i+1)/$iterations : CPU Time = $([math]::Round($cpuTime, 2))s" -ForegroundColor Gray
        } catch {
            Write-Host "  ✗ Process terminated during measurement" -ForegroundColor Red
            return $null
        }
    }
    
    if ($samples.Count -lt 2) {
        return $null
    }
    
    # Calculate variance
    $mean = ($samples | Measure-Object -Average).Average
    $variance = ($samples | ForEach-Object { [math]::Pow($_ - $mean, 2) } | Measure-Object -Average).Average
    $stddev = [math]::Sqrt($variance)
    
    # Stability score: coefficient of variation (lower is better)
    if ($mean -gt 0) {
        $stability = $stddev / $mean
    } else {
        $stability = 0.0
    }
    
    Write-Host ""
    Write-Host "  Mean CPU Time: $([math]::Round($mean, 3))s" -ForegroundColor Green
    Write-Host "  Std Deviation: $([math]::Round($stddev, 3))s" -ForegroundColor Green
    Write-Host "  Stability Score: $([math]::Round($stability, 3))" -ForegroundColor Green
    
    return $stability
}

# ═══════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════

Write-Banner "HyperTensor E-Core Affinity Enforcement"

Get-ProcessorInfo

if ($Validate) {
    Write-Host "VALIDATION MODE: Will check current affinity only" -ForegroundColor Cyan
    Write-Host ""
}

# Find Glass Cockpit process
Write-Host "Searching for process: $ProcessName*" -ForegroundColor Yellow

$processes = Get-Process -Name "$ProcessName*" -ErrorAction SilentlyContinue

if ($processes.Count -eq 0) {
    Write-Host "  ✗ No matching processes found" -ForegroundColor Red
    Write-Host ""
    Write-Host "Expected process names:" -ForegroundColor Yellow
    Write-Host "  - glass-cockpit.exe" -ForegroundColor Gray
    Write-Host "  - glass-cockpit" -ForegroundColor Gray
    Write-Host ""
    Write-Host "Launch Glass Cockpit first, then run this script" -ForegroundColor Yellow
    exit 1
}

if ($processes.Count -gt 1) {
    Write-Host "  ⚠ Multiple processes found:" -ForegroundColor Yellow
    foreach ($p in $processes) {
        Write-Host "    PID $($p.Id): $($p.Name)" -ForegroundColor Gray
    }
    Write-Host ""
    Write-Host "Using first process: PID $($processes[0].Id)" -ForegroundColor Yellow
}

$targetProcess = $processes[0]
$pid = $targetProcess.Id

Write-Host "  ✓ Found: $($targetProcess.Name) (PID $pid)" -ForegroundColor Green
Write-Host ""

# Check current affinity
$currentMask = Get-ProcessAffinityMask -ProcessId $pid

if ($null -eq $currentMask) {
    Write-Host "  ✗ Failed to read current affinity" -ForegroundColor Red
    exit 1
}

Write-Host "Current Affinity Mask: 0x$($currentMask.ToString('X8'))" -ForegroundColor Cyan
Write-Host "Expected E-Core Mask: 0x$($ECORE_MASK.ToString('X8'))" -ForegroundColor Cyan
Write-Host ""

# Decode affinity mask
$maskBits = [Convert]::ToString($currentMask, 2).PadLeft(32, '0')
$activeCores = @()
for ($i = 0; $i -lt 32; $i++) {
    if ($maskBits[31 - $i] -eq '1') {
        $activeCores += $i
    }
}

Write-Host "Active CPU cores: $($activeCores -join ', ')" -ForegroundColor Green
Write-Host ""

if ($Validate) {
    # Validation mode: check and report
    $isCorrect = Test-AffinityMask -CurrentMask $currentMask -ExpectedMask $ECORE_MASK
    
    if ($isCorrect) {
        Write-Host "✓ PASS: Process is correctly pinned to E-cores (16-31)" -ForegroundColor Green
        exit 0
    } else {
        Write-Host "✗ FAIL: Process is NOT pinned to E-cores" -ForegroundColor Red
        Write-Host "Expected cores: 16-31" -ForegroundColor Yellow
        Write-Host "Actual cores: $($activeCores -join ', ')" -ForegroundColor Yellow
        exit 1
    }
}

# Check if already set correctly
if (Test-AffinityMask -CurrentMask $currentMask -ExpectedMask $ECORE_MASK) {
    Write-Host "✓ Process is already pinned to E-cores (16-31)" -ForegroundColor Green
    Write-Host "No action needed" -ForegroundColor Green
    
    if ($Measure) {
        Write-Host ""
        Measure-StabilityScore -ProcessId $pid
    }
    
    exit 0
}

# Measure stability before pinning (if requested)
$stabilityBefore = $null
if ($Measure) {
    Write-Host "Measuring stability BEFORE E-core pinning..." -ForegroundColor Yellow
    $stabilityBefore = Measure-StabilityScore -ProcessId $pid
    Write-Host ""
}

# Apply E-core affinity
Write-Host "Applying E-core affinity mask..." -ForegroundColor Yellow

$success = Set-ProcessAffinityMask -ProcessId $pid -AffinityMask $ECORE_MASK

if ($success) {
    Start-Sleep -Milliseconds 500  # Allow OS to apply
    
    # Verify
    $newMask = Get-ProcessAffinityMask -ProcessId $pid
    
    if ($null -ne $newMask -and (Test-AffinityMask -CurrentMask $newMask -ExpectedMask $ECORE_MASK)) {
        Write-Host "  ✓ Affinity mask applied successfully" -ForegroundColor Green
        Write-Host ""
        Write-Host "Verification:" -ForegroundColor Cyan
        Write-Host "  New Mask: 0x$($newMask.ToString('X8'))" -ForegroundColor Green
        Write-Host "  Active Cores: 16-31" -ForegroundColor Green
        Write-Host ""
        Write-Host "Verify in Task Manager:" -ForegroundColor Yellow
        Write-Host "  1. Open Task Manager → Details" -ForegroundColor Gray
        Write-Host "  2. Right-click $($targetProcess.Name)" -ForegroundColor Gray
        Write-Host "  3. Select 'Set Affinity'" -ForegroundColor Gray
        Write-Host "  4. Confirm only CPUs 16-31 are checked" -ForegroundColor Gray
        
        # Measure stability after pinning (if requested)
        if ($Measure -and $null -ne $stabilityBefore) {
            Write-Host ""
            Write-Host "Measuring stability AFTER E-core pinning..." -ForegroundColor Yellow
            $stabilityAfter = Measure-StabilityScore -ProcessId $pid
            
            if ($null -ne $stabilityAfter) {
                Write-Host ""
                Write-Host "═══════════════════════════════════════════════" -ForegroundColor Cyan
                Write-Host "STABILITY COMPARISON" -ForegroundColor White
                Write-Host "═══════════════════════════════════════════════" -ForegroundColor Cyan
                Write-Host "  Before: $([math]::Round($stabilityBefore, 3))" -ForegroundColor Yellow
                Write-Host "  After:  $([math]::Round($stabilityAfter, 3))" -ForegroundColor Green
                
                $delta = $stabilityAfter - $stabilityBefore
                $deltaPercent = ($delta / $stabilityBefore) * 100
                
                if ($delta -lt 0) {
                    Write-Host "  Delta:  $([math]::Round($delta, 3)) ($([math]::Round($deltaPercent, 1))% improvement)" -ForegroundColor Green
                } else {
                    Write-Host "  Delta:  +$([math]::Round($delta, 3)) ($([math]::Round($deltaPercent, 1))% regression)" -ForegroundColor Yellow
                }
                Write-Host "═══════════════════════════════════════════════" -ForegroundColor Cyan
            }
        }
        
        exit 0
    } else {
        Write-Host "  ✗ Verification failed: mask did not apply correctly" -ForegroundColor Red
        Write-Host "  Current mask: 0x$($newMask.ToString('X8'))" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host ""
    Write-Host "Try running PowerShell as Administrator" -ForegroundColor Yellow
    exit 1
}
