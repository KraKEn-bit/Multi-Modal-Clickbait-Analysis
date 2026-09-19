# Held-out Task A: export test preds (9 seeds) + Wilcoxon/CI report. Resumable via cached CSVs per seed.
$ErrorActionPreference = "Continue"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Py = Join-Path $Here "scripts\heldout_task_a_significance.py"
$keepAwake = Join-Path $Here "keep_awake.ps1"
if (Test-Path $keepAwake) {
    Start-Process powershell -ArgumentList @("-NoProfile","-ExecutionPolicy","Bypass","-File",$keepAwake) -WindowStyle Hidden
    Write-Host "keep_awake started - plug in laptop"
}
while ($true) {
    & python $Py
    if ($LASTEXITCODE -eq 0) { break }
    Write-Host "Retry in 120s (exit $LASTEXITCODE)"
    Start-Sleep -Seconds 120
}
