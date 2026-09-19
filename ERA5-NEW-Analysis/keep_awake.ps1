# Keeps system awake while overnight ERA5 runs (plug in laptop).
Add-Type @"
using System.Runtime.InteropServices;
public static class PowerKeepAwakeNW {
    [DllImport("kernel32.dll", CharSet = CharSet.Auto, SetLastError = true)]
    public static extern uint SetThreadExecutionState(uint esFlags);
    public static void PreventSleep() {
        SetThreadExecutionState(0x80000041U);
    }
}
"@

Write-Host "Keeping system awake. Close this window when overnight is done."
while ($true) {
    [PowerKeepAwakeNW]::PreventSleep()
    Start-Sleep -Seconds 60
}
