# Lightweight watchdog: start the resilient runner if Phase 1 is not already active.
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$RunnerScript = Join-Path $ProjectRoot "scripts\run_phase1_resilient.ps1"
$LogFile = Join-Path $ProjectRoot "outputs\logs\phase1_watchdog.log"
$LockFile = Join-Path $ProjectRoot "outputs\logs\phase1_run.lock"
$ProgressFile = Join-Path $ProjectRoot "outputs\logs\phase1_progress.json"
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

function Get-CompleteCount {
    if (-not (Test-Path $Python)) { return 0 }
    $code = @'
from pathlib import Path
import json
p = Path("outputs/logs/phase1_progress.json")
if p.exists():
    d = json.loads(p.read_text(encoding="utf-8"))
    print(int(d.get("complete_gemini", 0)))
else:
    n = 0
    for d in Path("data/transcripts").iterdir():
        m = d / "metadata.json"
        if m.exists():
            try:
                if json.loads(m.read_text(encoding="utf-8")).get("summary_source") == "gemini":
                    n += 1
            except Exception:
                pass
    print(n)
'@
    $out = & $Python -c $code 2>$null
    if ($out -match '^\d+$') { return [int]$out }
    return 0
}

Set-Location $ProjectRoot
New-Item -ItemType Directory -Force -Path (Split-Path $LogFile) | Out-Null

$done = Get-CompleteCount
if ($done -ge 400) {
    Write-Log "Phase 1 complete ($done/400). Watchdog idle."
    exit 0
}

if (Test-ProcessRunning "run_phase1_resilient.ps1") {
    Write-Log "Resilient runner already active ($done/400)."
    exit 0
}

if (Test-Path $LockFile) {
    $pidText = Get-Content $LockFile -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($pidText -match '^\d+$' -and (Get-Process -Id ([int]$pidText) -ErrorAction SilentlyContinue)) {
        Write-Log "phase1_extract lock held by PID $pidText ($done/400)."
        exit 0
    }
}

if (Test-ProcessRunning "phase1_extract.py") {
    Write-Log "phase1_extract already running ($done/400)."
    exit 0
}

Write-Log "Starting resilient runner ($done/400 complete)."
Start-Process powershell.exe -ArgumentList @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-WindowStyle", "Hidden",
    "-File", "`"$RunnerScript`""
) -WorkingDirectory $ProjectRoot | Out-Null
