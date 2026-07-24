# Auto-resume Phase 1 after power cuts or GPU crashes.
# - One instance only (phase1_extract.py also enforces a PID lock)
# - Restarts on non-zero exit until target is reached or daily Gemini quota hits
param(
    [int]$RetryDelaySec = 45
)

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$LockFile = Join-Path $ProjectRoot "outputs\logs\phase1_run.lock"
$ProgressFile = Join-Path $ProjectRoot "outputs\logs\phase1_progress.json"
$RunnerLog = Join-Path $ProjectRoot "outputs\logs\phase1_resilient.log"

$env:HF_HUB_DISABLE_XET = "1"
$env:HF_HOME = Join-Path $ProjectRoot ".cache\huggingface"
$env:PYTHONUNBUFFERED = "1"

function Test-Phase1ProcessRunning {
    if (Test-Path $LockFile) {
        $pidText = Get-Content $LockFile -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($pidText -match '^\d+$') {
            if (Get-Process -Id ([int]$pidText) -ErrorAction SilentlyContinue) {
                return $true
            }
        }
    }
    $procs = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -like '*phase1_extract.py*' }
    return [bool]$procs
}

function Write-Log([string]$Message) {
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') | $Message"
    Add-Content -Path $RunnerLog -Value $line -Encoding utf8
    Write-Host $line
}

function Get-Phase1Progress {
    $code = @'
from pathlib import Path
import json
import yaml
target = 1600
cfg = Path("config.yaml")
if cfg.exists():
    target = int(yaml.safe_load(cfg.read_text(encoding='utf-8')).get('phase1', {}).get('target_total', 1600))
p = Path('outputs/logs/phase1_progress.json')
if p.exists():
    d = json.loads(p.read_text(encoding='utf-8'))
    print(str(d.get('complete_gemini', 0)) + ' ' + str(d.get('target_total', target)))
else:
    n = 0
    for dpath in Path('data/transcripts').iterdir():
        m = dpath / 'metadata.json'
        if m.exists():
            try:
                if json.loads(m.read_text(encoding='utf-8')).get('summary_source') == 'gemini':
                    n += 1
            except Exception:
                pass
    print(str(n) + " " + str(target))
'@
    & $Python -c $code
}

Set-Location $ProjectRoot
New-Item -ItemType Directory -Force -Path (Split-Path $RunnerLog) | Out-Null
Write-Log "Phase 1 resilient runner started | project=$ProjectRoot"

while ($true) {
    if (Test-Phase1ProcessRunning) {
        Write-Log "phase1_extract already running; waiting 60s..."
        Start-Sleep -Seconds 60
        continue
    }

    $progress = (Get-Phase1Progress) -split ' '
    $done = [int]$progress[0]
    $target = [int]$progress[1]
    if ($done -ge $target -and $target -gt 0) {
        Write-Log "Target reached ($done/$target). Done."
        break
    }

    Write-Log "Starting phase1_extract.py --resume ($done/$target so far)..."
    & $Python -u (Join-Path $ProjectRoot "scripts\phase1_extract.py") --resume
    $code = $LASTEXITCODE

    $progress = (Get-Phase1Progress) -split ' '
    $done = [int]$progress[0]
    $target = [int]$progress[1]
    if ($code -eq 0 -and $done -ge $target) {
        Write-Log "Finished cleanly ($done/$target)."
        break
    }

    Write-Log "Run ended (exit=$code, progress=$done/$target). Retry in ${RetryDelaySec}s..."
    Start-Sleep -Seconds $RetryDelaySec
}
