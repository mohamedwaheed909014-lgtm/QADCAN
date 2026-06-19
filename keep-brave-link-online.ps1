$ErrorActionPreference = "Continue"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$BackendDir = Join-Path $ProjectRoot "backend"
$Subdomain = "brave-impala-39"
$Port = 8000
$LocalStatusUrl = "http://127.0.0.1:$Port/api/status"
$PublicStatusUrl = "https://$Subdomain.loca.lt/api/status"
$CheckSeconds = 30

$ServerOut = Join-Path $ProjectRoot "server-keepalive.out.log"
$ServerErr = Join-Path $ProjectRoot "server-keepalive.err.log"
$TunnelOut = Join-Path $ProjectRoot "tunnel-brave.out.log"
$TunnelErr = Join-Path $ProjectRoot "tunnel-brave.err.log"
$WatchLog = Join-Path $ProjectRoot "keep-brave-link-online.log"

function Write-WatchLog {
    param([string]$Message)
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "$stamp $Message" | Tee-Object -FilePath $WatchLog -Append
}

function Test-UrlOk {
    param([string]$Url)
    try {
        $response = Invoke-WebRequest -UseBasicParsing $Url -TimeoutSec 15
        return ($response.StatusCode -eq 200 -and $response.Content -match '"status"\s*:\s*"ok"')
    } catch {
        return $false
    }
}

function Load-DotEnv {
    $envPath = Join-Path $ProjectRoot ".env"
    if (-not (Test-Path $envPath)) { return }

    Get-Content $envPath | ForEach-Object {
        $line = $_.Trim()
        if ($line -and -not $line.StartsWith("#") -and $line.Contains("=")) {
            $name, $value = $line.Split("=", 2)
            [Environment]::SetEnvironmentVariable($name.Trim(), $value.Trim(), "Process")
        }
    }
}

function Start-Backend {
    Load-DotEnv
    Write-WatchLog "[backend] starting on port $Port"
    Start-Process -FilePath "python" `
        -ArgumentList "-m uvicorn app:app --host 0.0.0.0 --port $Port" `
        -WorkingDirectory $BackendDir `
        -WindowStyle Hidden `
        -RedirectStandardOutput $ServerOut `
        -RedirectStandardError $ServerErr
}

function Stop-LocalTunnel {
    Get-CimInstance Win32_Process |
        Where-Object {
            ($_.Name -match "cmd.exe|node.exe") -and
            ($_.CommandLine -match "localtunnel|npx") -and
            ($_.CommandLine -match $Subdomain)
        } |
        ForEach-Object {
            Write-WatchLog "[tunnel] stopping stale process $($_.ProcessId)"
            Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
        }
}

function Start-LocalTunnel {
    Stop-LocalTunnel
    Start-Sleep -Seconds 2
    Write-WatchLog "[tunnel] starting https://$Subdomain.loca.lt"
    Start-Process -FilePath "cmd.exe" `
        -ArgumentList "/c npx --yes localtunnel --subdomain $Subdomain --port $Port --local-host localhost" `
        -WorkingDirectory $ProjectRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput $TunnelOut `
        -RedirectStandardError $TunnelErr
}

Write-WatchLog "[watcher] started for https://$Subdomain.loca.lt"

while ($true) {
    if (-not (Test-UrlOk $LocalStatusUrl)) {
        Start-Backend
        Start-Sleep -Seconds 12
    }

    if (-not (Test-UrlOk $PublicStatusUrl)) {
        Start-LocalTunnel
        Start-Sleep -Seconds 15
    }

    Start-Sleep -Seconds $CheckSeconds
}
