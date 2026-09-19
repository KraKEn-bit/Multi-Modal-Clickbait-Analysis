# Full spatial controlled pipeline: join -> build -> dev eval (5deg + 3deg) -> compare
# Outage-safe: resume flags on join and each dev eval.
$ErrorActionPreference = "Continue"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$LogDir = Join-Path $Here "results\spatial_pipeline\logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$LogFile = Join-Path $LogDir ("spatial_pipeline_{0:yyyyMMdd_HHmmss}.log" -f (Get-Date))
$ResumeCmd = Join-Path $Here "resume_spatial_controlled_pipeline.cmd"
$TaskName = "Research1_NEWWAY_SPATIAL_PIPELINE"

function Write-Log { param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Add-Content -Path $LogFile -Value $line
    Write-Host $line
}

$keepAwake = Join-Path $Here "keep_awake.ps1"
if (Test-Path $keepAwake) {
    Start-Process powershell -ArgumentList @("-NoProfile","-ExecutionPolicy","Bypass","-File",$keepAwake) -WindowStyle Hidden
    Write-Log "Started keep_awake.ps1 - plug in laptop"
}
schtasks /Create /TN $TaskName /SC ONLOGON /F /RL LIMITED /TR $ResumeCmd 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Log "Registered scheduled task $TaskName (on logon)"
} else {
    Write-Log "Scheduled task not created (exit $LASTEXITCODE) - HKCU Run key is fallback"
}
try {
    Set-ItemProperty -Path "HKCU:\Software\Microsoft\Windows\CurrentVersion\Run" -Name $TaskName -Value $ResumeCmd
    Write-Log "Registered post-outage resume: $ResumeCmd"
} catch {
    Write-Log "HKCU Run key failed - double-click resume_spatial_controlled_pipeline.cmd after power cut"
}

Write-Log "Spatial controlled pipeline started. Log: $LogFile"

function Test-JoinComplete {
    param([string]$JoinProg)
    if (-not (Test-Path $JoinProg)) { return $false }
    try {
        return ((Get-Content $JoinProg -Raw | ConvertFrom-Json).status -eq "complete")
    } catch { return $false }
}

# 1) Spatial join (resumes via SID+ISO_TIME keys in sidecar CSVs)
$JoinProg = Join-Path $Here "results\era5_spatial_join_progress.json"
while (-not (Test-JoinComplete $JoinProg)) {
    Write-Log "Step 1: era5_spatial_join.py (resume-safe)"
    & python (Join-Path $Here "scripts\era5_spatial_join.py")
    if ($LASTEXITCODE -eq 0) { break }
    Write-Log "Join interrupted exit $LASTEXITCODE - retry in 120s"
    Start-Sleep -Seconds 120
}
Write-Log "Step 1: spatial join complete"

foreach ($patch in @("5deg", "3deg")) {
    $BuildProg = Join-Path $Here "results\build_modeling_dataset_spatial_${patch}_progress.json"
    $ModelCsv = Join-Path $Here "datasets\bangladesh_nextstep_dataset_era5_spatial_${patch}.csv"
    if (-not (Test-Path $ModelCsv)) {
        Write-Log "Step 2: build modeling dataset patch=$patch"
        & python (Join-Path $Here "scripts\build_modeling_dataset_spatial.py") --patch $patch
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
    $DoneFlag = Join-Path $Here "results\sece_era5_spatial_${patch}\dev_multiseed_phase3_complete.flag"
    if (-not (Test-Path $DoneFlag)) {
        Write-Log "Step 3: dev eval patch=$patch (resilient loop)"
        while (-not (Test-Path $DoneFlag)) {
            & python (Join-Path $Here "scripts\sece_era5_spatial_dev_eval.py") --patch $patch --resume
            if ($LASTEXITCODE -eq 0) { break }
            Start-Sleep -Seconds 120
        }
    } else {
        Write-Log "Step 3: dev eval patch=$patch already complete"
    }
}

Write-Log "Step 4: compare_era5_spatial_final_dev.py"
& python (Join-Path $Here "scripts\compare_era5_spatial_final_dev.py")
schtasks /Delete /TN $TaskName /F 2>$null
Remove-ItemProperty -Path "HKCU:\Software\Microsoft\Windows\CurrentVersion\Run" -Name $TaskName -ErrorAction SilentlyContinue
Write-Log "Spatial pipeline complete."
