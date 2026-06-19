param(
    [string]$TaskName = "OpenSCAD Copilot Public Link"
)

$ErrorActionPreference = "Stop"

$task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($task) {
    Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "[startup] Removed scheduled task: $TaskName"
} else {
    Write-Host "[startup] Scheduled task was not installed: $TaskName"
}

$StartupDir = [Environment]::GetFolderPath("Startup")
$ShortcutPath = Join-Path $StartupDir "$TaskName.lnk"
if (Test-Path $ShortcutPath) {
    Remove-Item -LiteralPath $ShortcutPath -Force
    Write-Host "[startup] Removed Startup shortcut:"
    Write-Host "  $ShortcutPath"
}
