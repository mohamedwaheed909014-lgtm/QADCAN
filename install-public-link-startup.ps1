param(
    [string]$TaskName = "OpenSCAD Copilot Public Link",
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$KeepAliveScript = Join-Path $ProjectRoot "keep-cloudflare-quick-tunnel-online.ps1"

if (-not (Test-Path $KeepAliveScript)) {
    throw "Missing keepalive script: $KeepAliveScript"
}

$PowerShell = Join-Path $env:SystemRoot "System32\WindowsPowerShell\v1.0\powershell.exe"
$Argument = "-NoProfile -ExecutionPolicy Bypass -File `"$KeepAliveScript`" -Port $Port"

$installed = $false

try {
    $Action = New-ScheduledTaskAction -Execute $PowerShell -Argument $Argument -WorkingDirectory $ProjectRoot
    $Trigger = New-ScheduledTaskTrigger -AtLogOn
    $Settings = New-ScheduledTaskSettingsSet `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -MultipleInstances IgnoreNew `
        -RestartCount 999 `
        -RestartInterval (New-TimeSpan -Minutes 1)

    Register-ScheduledTask `
        -TaskName $TaskName `
        -Action $Action `
        -Trigger $Trigger `
        -Settings $Settings `
        -Description "Keeps the OpenSCAD Copilot backend and Cloudflare quick tunnel running after Codex is closed." `
        -Force | Out-Null

    Start-ScheduledTask -TaskName $TaskName
    Write-Host "[startup] Installed and started scheduled task: $TaskName"
    $installed = $true
} catch {
    Write-Host "[startup] Scheduled task install failed: $($_.Exception.Message)"
    Write-Host "[startup] Falling back to the current user's Startup folder."
}

if (-not $installed) {
    $StartupDir = [Environment]::GetFolderPath("Startup")
    $ShortcutPath = Join-Path $StartupDir "$TaskName.lnk"
    $Shell = New-Object -ComObject WScript.Shell
    $Shortcut = $Shell.CreateShortcut($ShortcutPath)
    $Shortcut.TargetPath = $PowerShell
    $Shortcut.Arguments = $Argument
    $Shortcut.WorkingDirectory = $ProjectRoot
    $Shortcut.WindowStyle = 7
    $Shortcut.Description = "Keeps the OpenSCAD Copilot backend and Cloudflare quick tunnel running."
    $Shortcut.Save()

    Start-Process -FilePath $PowerShell `
        -ArgumentList $Argument `
        -WorkingDirectory $ProjectRoot `
        -WindowStyle Hidden

    Write-Host "[startup] Installed Startup shortcut:"
    Write-Host "  $ShortcutPath"
    Write-Host "[startup] Started keepalive process."
}

Write-Host "[startup] Current public URL will be written to:"
Write-Host "  $ProjectRoot\public-url.txt"
