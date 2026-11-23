# Lightweight watchdog: start the resilient runner if Phase 1 is not already active.
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$RunnerScript = Join-Path $ProjectRoot "scripts\run_phase1_resilient.ps1"
$LogFile = Join-Path $ProjectRoot "outputs\logs\phase1_watchdog.log"
$LockFile = Join-Path $ProjectRoot "outputs\logs\phase1_run.lock"
$ProgressFile = Join-Path $ProjectRoot "outputs\logs\phase1_progress.json"
$CompleteFlag = Join-Path $ProjectRoot "outputs\logs\phase1_complete.flag"
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

function Write-Log([string]$Message) {
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') | $Message"
    Add-Content -Path $LogFile -Value $line -Encoding utf8
    Write-Host $line
}

function Test-ProcessRunning([string]$Pattern) {
    $procs = Get-CimInstance Win32_Process -Filter "Name='powershell.exe' OR Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -like "*$Pattern*" }
    return [bool]$procs
}

function Get-Phase1Progress {
    $done = 0
    $target = 1600
    if (Test-Path $ProgressFile) {
        try {
            $progress = Get-Content $ProgressFile -Raw -Encoding utf8 | ConvertFrom-Json
            if ($null -ne $progress.complete_gemini) {
                $done = [int]$progress.complete_gemini
            }
            if ($null -ne $progress.target_total) {
                $target = [int]$progress.target_total
            }
        } catch {
            Write-Log "Failed to read progress JSON: $_"
        }
    }
    return @{ Done = $done; Target = $target }
}

Set-Location $ProjectRoot
New-Item -ItemType Directory -Force -Path (Split-Path $LogFile) | Out-Null

if (Test-Path $CompleteFlag) {
    Write-Log "Phase 1 marked complete (flag file). Watchdog idle."
    exit 0
}

$progress = Get-Phase1Progress
$done = $progress.Done
$target = $progress.Target
if ($done -ge $target -and $target -gt 0) {
    Write-Log "Phase 1 complete ($done/$target). Watchdog idle."
    exit 0
}

if (Test-ProcessRunning "run_phase1_resilient.ps1") {
    Write-Log "Resilient runner already active ($done/$target)."
    exit 0
}

if (Test-Path $LockFile) {
    $pidText = Get-Content $LockFile -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($pidText -match '^\d+$' -and (Get-Process -Id ([int]$pidText) -ErrorAction SilentlyContinue)) {
        Write-Log "phase1_extract lock held by PID $pidText ($done/$target)."
        exit 0
    }
}

if (Test-ProcessRunning "phase1_extract.py") {
    Write-Log "phase1_extract already running ($done/$target)."
    exit 0
}

Write-Log "Starting resilient runner ($done/$target complete)."
Start-Process powershell.exe -ArgumentList @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-WindowStyle", "Hidden",
    "-File", "`"$RunnerScript`""
) -WorkingDirectory $ProjectRoot | Out-Null
