# Resilient track-only Phase 3 dev eval (1940-2024, no ERA5 features).
$ErrorActionPreference = "Continue"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$LogDir = Join-Path $Here "results\sece_trackonly_1940\logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$LogFile = Join-Path $LogDir ("dev_eval_resilient_{0:yyyyMMdd_HHmmss}.log" -f (Get-Date))
$PyScript = Join-Path $Here "scripts\sece_trackonly_dev_eval.py"
$ProgressJson = Join-Path $Here "results\sece_trackonly_1940\dev_multiseed_phase3_progress.json"
$DoneFlag = Join-Path $Here "results\sece_trackonly_1940\dev_multiseed_phase3_complete.flag"
$ResumeCmd = Join-Path $Here "resume_sece_trackonly_dev.cmd"
$TaskName = "Research1_NEWWAY_TRACKONLY_DEV"

function Write-Log {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Add-Content -Path $LogFile -Value $line
    Write-Host $line
}

function Test-Complete {
    if (Test-Path $DoneFlag) { return $true }
    if (-not (Test-Path $ProgressJson)) { return $false }
    try {
        return ((Get-Content $ProgressJson -Raw | ConvertFrom-Json).status -eq "complete")
    } catch { return $false }
}

if (Test-Complete) {
    Write-Log "Track-only dev eval already complete."
    exit 0
}

$keepAwake = Join-Path $Here "keep_awake.ps1"
if (Test-Path $keepAwake) {
    Start-Process powershell -ArgumentList @("-NoProfile","-ExecutionPolicy","Bypass","-File",$keepAwake) -WindowStyle Hidden
    Write-Log "Started keep_awake.ps1"
}

schtasks /Create /TN $TaskName /SC ONLOGON /F /RL LIMITED /TR $ResumeCmd 2>$null
try {
    Set-ItemProperty -Path "HKCU:\Software\Microsoft\Windows\CurrentVersion\Run" -Name $TaskName -Value $ResumeCmd
    Write-Log "Registered post-outage resume"
} catch {
    Write-Log "Use resume_sece_trackonly_dev.cmd after power cut"
}

Write-Log "Track-only Phase 3 dev eval started. Log: $LogFile"

while ($true) {
    if (Test-Complete) { break }
    Write-Log "Running sece_trackonly_dev_eval.py --resume"
    & python $PyScript --resume
    $code = $LASTEXITCODE
    if ($code -eq 0) {
        "complete $(Get-Date -Format o)" | Set-Content -Path $DoneFlag
        schtasks /Delete /TN $TaskName /F 2>$null
        Remove-ItemProperty -Path "HKCU:\Software\Microsoft\Windows\CurrentVersion\Run" -Name $TaskName -ErrorAction SilentlyContinue
        Write-Log "Complete."
        break
    }
    $wait = if ($code -eq 2) { 60 } else { 120 }
    Write-Log "Exit $code - retry in ${wait}s"
    Start-Sleep -Seconds $wait
}
