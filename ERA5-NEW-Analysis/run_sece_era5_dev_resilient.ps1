# Resilient SECE+ERA5 Phase 3 dev eval — survives power loss / crash.
# Checkpoints after each seed; re-run with --resume skips finished seeds.
$ErrorActionPreference = "Continue"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$LogDir = Join-Path $Here "results\sece_era5\logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$LogFile = Join-Path $LogDir ("dev_eval_resilient_{0:yyyyMMdd_HHmmss}.log" -f (Get-Date))
$PyScript = Join-Path $Here "scripts\sece_era5_dev_eval.py"
$ProgressJson = Join-Path $Here "results\sece_era5\dev_multiseed_phase3_progress.json"
$DoneFlag = Join-Path $Here "results\sece_era5\dev_multiseed_phase3_complete.flag"
$ResumeCmd = Join-Path $Here "resume_sece_era5_dev.cmd"
$TaskName = "Research1_NEWWAY_SECE_ERA5_DEV"

function Write-Log {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Add-Content -Path $LogFile -Value $line
    Write-Host $line
}

function Test-DevEvalComplete {
    if (Test-Path $DoneFlag) { return $true }
    if (-not (Test-Path $ProgressJson)) { return $false }
    try {
        $p = Get-Content $ProgressJson -Raw | ConvertFrom-Json
        return ($p.status -eq "complete")
    } catch {
        return $false
    }
}

if (Test-DevEvalComplete) {
    Write-Log "Dev eval already complete. Exiting."
    exit 0
}

# Prevent sleep + auto-resume after power cut (logon).
$keepAwake = Join-Path $Here "keep_awake.ps1"
if (Test-Path $keepAwake) {
    Start-Process -FilePath "powershell.exe" -ArgumentList @(
        "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $keepAwake
    ) -WindowStyle Hidden
    Write-Log "Started keep_awake.ps1 (plug in laptop)"
}

schtasks /Create /TN $TaskName /SC ONLOGON /F /RL LIMITED /TR $ResumeCmd 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Log "Registered logon task $TaskName -> resume_sece_era5_dev.cmd"
} else {
    Write-Log "Scheduled task not created (exit $LASTEXITCODE). Use resume_sece_era5_dev.cmd after reboot."
}

try {
    $runKey = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Run"
    Set-ItemProperty -Path $runKey -Name $TaskName -Value $ResumeCmd
    Write-Log "Registered HKCU Run key for post-outage resume"
} catch {
    Write-Log "Could not write HKCU Run key. Double-click resume_sece_era5_dev.cmd after power returns."
}

Write-Log "SECE+ERA5 Phase 3 dev eval resilient runner started"
Write-Log "Log: $LogFile"
Write-Log "Progress: $ProgressJson"

while ($true) {
    if (Test-DevEvalComplete) {
        Write-Log "Dev eval complete (flag or progress)."
        break
    }
    Write-Log "Running: python sece_era5_dev_eval.py --phase 3 --resume"
    & python $PyScript --phase 3 --resume
    $exitCode = $LASTEXITCODE
    if ($exitCode -eq 0) {
        "complete $(Get-Date -Format o)" | Set-Content -Path $DoneFlag
        Write-Log "Dev eval finished successfully."
        schtasks /Delete /TN $TaskName /F 2>$null
        Remove-ItemProperty -Path "HKCU:\Software\Microsoft\Windows\CurrentVersion\Run" -Name $TaskName -ErrorAction SilentlyContinue
        break
    }
    if ($exitCode -eq 2) {
        Write-Log "Seed failed (exit 2). Waiting 60s then resume..."
        Start-Sleep -Seconds 60
        continue
    }
    Write-Log "Exit $exitCode (likely power cut / kill). Waiting 120s then resume..."
    Start-Sleep -Seconds 120
}

Write-Log "Done."
