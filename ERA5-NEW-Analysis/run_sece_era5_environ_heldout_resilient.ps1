# Resilient held-out confirm: SECE+ERA5 Phase 3 environ subset (locked architecture).
$ErrorActionPreference = "Continue"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$LogDir = Join-Path $Here "results\sece_era5_environ_heldout\logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$LogFile = Join-Path $LogDir ("heldout_eval_resilient_{0:yyyyMMdd_HHmmss}.log" -f (Get-Date))
$PyScript = Join-Path $Here "scripts\sece_era5_environ_heldout_eval.py"
$ReportScript = Join-Path $Here "scripts\write_environ_heldout_report.py"
$ProgressJson = Join-Path $Here "results\sece_era5_environ_heldout\heldout_multiseed_phase3_progress.json"
$DoneFlag = Join-Path $Here "results\sece_era5_environ_heldout\heldout_multiseed_phase3_complete.flag"
$ResumeCmd = Join-Path $Here "resume_sece_era5_environ_heldout.cmd"
$TaskName = "Research1_NEWWAY_HELDOUT_ENVIRON"

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
    Write-Log "Held-out eval already complete."
    & python $ReportScript
    exit 0
}

$keepAwake = Join-Path $Here "keep_awake.ps1"
if (Test-Path $keepAwake) {
    Start-Process powershell -ArgumentList @("-NoProfile","-ExecutionPolicy","Bypass","-File",$keepAwake) -WindowStyle Hidden
    Write-Log "Started keep_awake.ps1 - plug in laptop"
}

schtasks /Create /TN $TaskName /SC ONLOGON /F /RL LIMITED /TR $ResumeCmd 2>$null
try {
    Set-ItemProperty -Path "HKCU:\Software\Microsoft\Windows\CurrentVersion\Run" -Name $TaskName -Value $ResumeCmd
    Write-Log "Registered post-outage resume"
} catch {
    Write-Log "Use resume_sece_era5_environ_heldout.cmd after power cut"
}

Write-Log "Held-out eval started. Log: $LogFile"

while ($true) {
    if (Test-Complete) { break }
    Write-Log "Running sece_era5_environ_heldout_eval.py --resume"
    & python $PyScript --resume
    $code = $LASTEXITCODE
    if ($code -eq 0) {
        schtasks /Delete /TN $TaskName /F 2>$null
        Remove-ItemProperty -Path "HKCU:\Software\Microsoft\Windows\CurrentVersion\Run" -Name $TaskName -ErrorAction SilentlyContinue
        Write-Log "Eval complete. Writing report."
        & python $ReportScript
        Write-Log "Held-out pipeline complete."
        break
    }
    $wait = if ($code -eq 2) { 60 } else { 120 }
    Write-Log "Exit $code - retry in ${wait}s"
    Start-Sleep -Seconds $wait
}
