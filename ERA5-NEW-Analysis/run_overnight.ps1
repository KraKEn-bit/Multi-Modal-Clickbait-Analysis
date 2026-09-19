# Sequential overnight: quality (skip if done) -> ERA5 year files -> join.
# Power cut: completed year .nc files stay; logon scheduled task restarts this script.
$ErrorActionPreference = "Continue"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$LogDir = Join-Path $Here "results\logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$LogFile = Join-Path $LogDir ("overnight_{0:yyyyMMdd_HHmmss}.log" -f (Get-Date))
$DoneFlag = Join-Path $Here "results\overnight_complete.flag"
$QualityCsv = Join-Path $Here "results\ibtracs_quality_by_window.csv"
$RawDir = Join-Path $Here "datasets\era5_raw"
$YearStart = 1940
$YearEnd = 2024

function Write-Log {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Add-Content -Path $LogFile -Value $line
    Write-Host $line
}

function Test-AllYears {
    for ($y = $YearStart; $y -le $YearEnd; $y++) {
        $pl = Join-Path $RawDir ("era5_pl_{0}.nc" -f $y)
        $sl = Join-Path $RawDir ("era5_sl_{0}.nc" -f $y)
        if (-not ((Test-Path $pl) -and (Test-Path $sl))) {
            return $false
        }
    }
    return $true
}

if (Test-Path $DoneFlag) {
    Write-Log "Overnight already complete. Exiting."
    exit 0
}

$keepAwake = Join-Path $Here "keep_awake.ps1"
Start-Process -FilePath "powershell.exe" -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $keepAwake) -WindowStyle Hidden

Write-Log "NEW WAY overnight started"
Write-Log "Log: $LogFile"

$resumeCmd = Join-Path $Here "resume_overnight.cmd"
schtasks /Create /TN "Research1_NEWWAY_ERA5" /SC ONLOGON /F /RL LIMITED /TR $resumeCmd
if ($LASTEXITCODE -eq 0) {
    Write-Log "Registered logon resume task Research1_NEWWAY_ERA5"
} else {
    Write-Log "Scheduled task not created (exit $LASTEXITCODE). Using user Startup instead."
}
try {
    $runKey = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Run"
    Set-ItemProperty -Path $runKey -Name "Research1_NEWWAY_ERA5" -Value $resumeCmd
    Write-Log "Registered HKCU Run resume (starts after you log in following a power cut)"
} catch {
    Write-Log "Could not write HKCU Run key. After a power cut, double-click run_overnight.bat."
}

if (-not (Test-Path $QualityCsv)) {
    Write-Log "Step 0: IBTrACS quality"
    & python (Join-Path $Here "scripts\ibtracs_quality_by_era.py")
    if ($LASTEXITCODE -ne 0) {
        Write-Log "Quality failed; retry in 60s"
        Start-Sleep -Seconds 60
        & python (Join-Path $Here "scripts\ibtracs_quality_by_era.py")
    }
} else {
    Write-Log "Step 0: quality CSV already present; skip"
}

Write-Log "Step 1: ERA5 download (storm days only; skip files already on disk)"
while ($true) {
    & python (Join-Path $Here "scripts\era5_download.py")
    $code = $LASTEXITCODE
    if ($code -eq 0) {
        Write-Log "Download pass finished"
        Write-Log "Step 1b: join ERA5 onto tracks"
        & python (Join-Path $Here "scripts\era5_join.py")
        "complete $(Get-Date -Format o)" | Set-Content -Path $DoneFlag
        Write-Log "Overnight sequence done. Wrote $DoneFlag"
        schtasks /Delete /TN "Research1_NEWWAY_ERA5" /F 2>$null
        Remove-ItemProperty -Path "HKCU:\Software\Microsoft\Windows\CurrentVersion\Run" -Name "Research1_NEWWAY_ERA5" -ErrorAction SilentlyContinue
        break
    }
    if ($code -eq 2) {
        Write-Log "Year failed; wait 120s and resume"
        Start-Sleep -Seconds 120
        continue
    }
    Write-Log "Waiting 180s (CDS key, licence, or queue). Leave this running."
    Start-Sleep -Seconds 180
}
