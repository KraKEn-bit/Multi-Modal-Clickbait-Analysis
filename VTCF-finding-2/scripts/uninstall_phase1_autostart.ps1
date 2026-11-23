# Remove the Phase 1 auto-resume scheduled task (run when extraction is complete).
$TaskName = "VTCF-Phase1-AutoResume"
Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
Write-Host "Removed scheduled task: $TaskName (if it existed)."

Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -like '*run_phase1_resilient*' -or $_.CommandLine -like '*ensure_phase1*' } |
    ForEach-Object {
        Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
        Write-Host "Stopped watchdog runner PID $($_.ProcessId)"
    }

Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -like '*phase1_extract*' } |
    ForEach-Object {
        Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
        Write-Host "Stopped phase1_extract PID $($_.ProcessId)"
    }

$LockFile = Join-Path (Split-Path -Parent $PSScriptRoot) "outputs\logs\phase1_run.lock"
$CompleteFlag = Join-Path (Split-Path -Parent $PSScriptRoot) "outputs\logs\phase1_complete.flag"
if (Test-Path $LockFile) {
    Remove-Item $LockFile -Force
    Write-Host "Removed stale lock file."
}
if (-not (Test-Path $CompleteFlag)) {
    New-Item -ItemType File -Path $CompleteFlag -Force | Out-Null
    Set-Content -Path $CompleteFlag -Value "Phase 1 marked complete on $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')." -Encoding utf8
    Write-Host "Created phase1_complete.flag"
}

Write-Host "Phase 1 autostart disabled."
