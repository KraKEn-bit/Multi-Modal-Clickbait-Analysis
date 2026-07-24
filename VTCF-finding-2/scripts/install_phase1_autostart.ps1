# Register a Windows Scheduled Task to auto-resume Phase 1 after power cuts / reboots.
# Run once: powershell -ExecutionPolicy Bypass -File scripts\install_phase1_autostart.ps1
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$EnsureScript = Join-Path $ProjectRoot "scripts\ensure_phase1_running.ps1"
$TaskName = "VTCF-Phase1-AutoResume"

$action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$EnsureScript`"" `
    -WorkingDirectory $ProjectRoot

$logonTrigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$repeatTrigger = New-ScheduledTaskTrigger -Once -At (Get-Date).Date `
    -RepetitionInterval (New-TimeSpan -Minutes 15) `
    -RepetitionDuration (New-TimeSpan -Days 7)

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 5)

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger @($logonTrigger, $repeatTrigger) `
    -Settings $settings `
    -Force | Out-Null

Write-Host "Installed scheduled task: $TaskName"
Write-Host "  - Runs at logon"
Write-Host "  - Re-checks every 15 minutes for 7 days"
Write-Host "  - Starts scripts\run_phase1_resilient.ps1 when Phase 1 is not running"
Write-Host ""
Write-Host "Starting watchdog now..."
& $EnsureScript
