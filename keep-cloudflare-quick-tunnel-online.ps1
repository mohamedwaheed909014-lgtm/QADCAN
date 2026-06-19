param(
    [int]$Port = 8000,
    [int]$CheckSeconds = 30
)

$ErrorActionPreference = "Continue"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$BackendDir = Join-Path $ProjectRoot "backend"
$Cloudflared = Join-Path $ProjectRoot "cloudflared.exe"

$LocalStatusUrl = "http://127.0.0.1:$Port/api/status"
$ServerOut = Join-Path $ProjectRoot "server-quick-tunnel.out.log"
$ServerErr = Join-Path $ProjectRoot "server-quick-tunnel.err.log"
$TunnelOut = Join-Path $ProjectRoot "cloudflare-quick-tunnel.out.log"
$TunnelErr = Join-Path $ProjectRoot "cloudflare-quick-tunnel.err.log"
$WatchLog = Join-Path $ProjectRoot "keep-cloudflare-quick-tunnel-online.log"
$PublicUrlFile = Join-Path $ProjectRoot "public-url.txt"

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

function Get-QuickTunnelProcess {
    Get-CimInstance Win32_Process |
        Where-Object {
            ($_.Name -eq "cloudflared.exe") -and
            ($_.CommandLine -match [regex]::Escape($ProjectRoot)) -and
            ($_.CommandLine -match "tunnel") -and
            ($_.CommandLine -match "--url")
        }
}

function Stop-QuickTunnel {
    Get-QuickTunnelProcess | ForEach-Object {
        Write-WatchLog "[tunnel] stopping stale cloudflared process $($_.ProcessId)"
        Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
    }
}

function Get-CurrentPublicUrl {
    if (Test-Path $PublicUrlFile) {
        $fromFile = (Get-Content -LiteralPath $PublicUrlFile -ErrorAction SilentlyContinue | Select-Object -First 1).Trim()
        if ($fromFile -match '^https://[-a-z0-9]+\.trycloudflare\.com$') {
            return $fromFile
        }
    }

    if (Test-Path $TunnelErr) {
        $matches = Select-String -LiteralPath $TunnelErr -Pattern 'https://[-a-z0-9]+\.trycloudflare\.com' -AllMatches
        $last = $matches | Select-Object -Last 1
        if ($last) {
            return $last.Matches[$last.Matches.Count - 1].Value
        }
    }

    return $null
}

function Start-QuickTunnel {
    if (-not (Test-Path $Cloudflared)) {
        Write-WatchLog "[tunnel] missing cloudflared.exe at $Cloudflared"
        return
    }

    Stop-QuickTunnel
    Start-Sleep -Seconds 2

    Remove-Item -LiteralPath $TunnelOut, $TunnelErr -Force -ErrorAction SilentlyContinue
    Write-WatchLog "[tunnel] starting Cloudflare quick tunnel for $LocalStatusUrl"
    Start-Process -FilePath $Cloudflared `
        -ArgumentList "tunnel --url http://127.0.0.1:$Port" `
        -WorkingDirectory $ProjectRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput $TunnelOut `
        -RedirectStandardError $TunnelErr

    for ($i = 0; $i -lt 20; $i++) {
        Start-Sleep -Seconds 2
        $url = Get-CurrentPublicUrl
        if ($url) {
            Set-Content -LiteralPath $PublicUrlFile -Value $url -Encoding ascii
            Write-WatchLog "[tunnel] public URL: $url"
            return
        }
    }

    Write-WatchLog "[tunnel] started, but no public URL was found yet"
}

Write-WatchLog "[watcher] started for Cloudflare quick tunnel on port $Port"

while ($true) {
    if (-not (Test-UrlOk $LocalStatusUrl)) {
        Start-Backend
        Start-Sleep -Seconds 12
    }

    $publicUrl = Get-CurrentPublicUrl
    $publicOk = $false
    if ($publicUrl) {
        $publicOk = Test-UrlOk "$publicUrl/api/status"
    }

    if (-not $publicOk) {
        Start-QuickTunnel
        Start-Sleep -Seconds 15
    }

    Start-Sleep -Seconds $CheckSeconds
}
