param(
    [Parameter(Mandatory = $true)]
    [string]$Hostname,

    [string]$TunnelName = "openscad-copilot",
    [int]$Port = 8000,
    [int]$CheckSeconds = 30
)

$ErrorActionPreference = "Continue"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$BackendDir = Join-Path $ProjectRoot "backend"
$Cloudflared = Join-Path $ProjectRoot "cloudflared.exe"
$ConfigPath = Join-Path $ProjectRoot ".cloudflared\config.yml"

$LocalStatusUrl = "http://127.0.0.1:$Port/api/status"
$PublicStatusUrl = "https://$Hostname/api/status"

$ServerOut = Join-Path $ProjectRoot "server-cloudflare.out.log"
$ServerErr = Join-Path $ProjectRoot "server-cloudflare.err.log"
$TunnelOut = Join-Path $ProjectRoot "cloudflare-named-tunnel.out.log"
$TunnelErr = Join-Path $ProjectRoot "cloudflare-named-tunnel.err.log"
$WatchLog = Join-Path $ProjectRoot "keep-cloudflare-named-tunnel-online.log"

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

function Stop-CloudflareTunnel {
    Get-CimInstance Win32_Process |
        Where-Object {
            ($_.Name -eq "cloudflared.exe") -and
            ($_.CommandLine -match [regex]::Escape($ProjectRoot))
        } |
        ForEach-Object {
            Write-WatchLog "[tunnel] stopping stale cloudflared process $($_.ProcessId)"
            Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
        }
}

function Start-CloudflareTunnel {
    if (-not (Test-Path $ConfigPath)) {
        Write-WatchLog "[tunnel] missing config: $ConfigPath"
        Write-WatchLog "[tunnel] run setup-cloudflare-named-tunnel.ps1 first"
        return
    }

    Stop-CloudflareTunnel
    Start-Sleep -Seconds 2
    Write-WatchLog "[tunnel] starting Cloudflare named tunnel '$TunnelName' for https://$Hostname"
    Start-Process -FilePath $Cloudflared `
        -ArgumentList "tunnel --config `"$ConfigPath`" run $TunnelName" `
        -WorkingDirectory $ProjectRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput $TunnelOut `
        -RedirectStandardError $TunnelErr
}

Write-WatchLog "[watcher] started for https://$Hostname"

while ($true) {
    if (-not (Test-UrlOk $LocalStatusUrl)) {
        Start-Backend
        Start-Sleep -Seconds 12
    }

    if (-not (Test-UrlOk $PublicStatusUrl)) {
        Start-CloudflareTunnel
        Start-Sleep -Seconds 15
    }

    Start-Sleep -Seconds $CheckSeconds
}
