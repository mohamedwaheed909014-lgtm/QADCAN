param(
    [Parameter(Mandatory = $true)]
    [string]$Hostname,

    [string]$TunnelName = "openscad-copilot",
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Cloudflared = Join-Path $ProjectRoot "cloudflared.exe"
$ConfigDir = Join-Path $ProjectRoot ".cloudflared"

if (-not (Test-Path $Cloudflared)) {
    throw "cloudflared.exe was not found at $Cloudflared"
}

New-Item -ItemType Directory -Path $ConfigDir -Force | Out-Null

$UserCloudflaredDir = Join-Path $env:USERPROFILE ".cloudflared"
$CertPath = Join-Path $UserCloudflaredDir "cert.pem"

if (-not (Test-Path $CertPath)) {
    Write-Host "[cloudflare] No Cloudflare origin cert found."
    Write-Host "[cloudflare] A browser login will open. Choose the Cloudflare account/zone that owns $Hostname."
    & $Cloudflared tunnel login
}

$TunnelJson = & $Cloudflared tunnel list --output json | ConvertFrom-Json
$Tunnel = $TunnelJson | Where-Object { $_.name -eq $TunnelName } | Select-Object -First 1

if (-not $Tunnel) {
    Write-Host "[cloudflare] Creating named tunnel '$TunnelName'"
    & $Cloudflared tunnel create $TunnelName
    $TunnelJson = & $Cloudflared tunnel list --output json | ConvertFrom-Json
    $Tunnel = $TunnelJson | Where-Object { $_.name -eq $TunnelName } | Select-Object -First 1
}

if (-not $Tunnel) {
    throw "Tunnel '$TunnelName' was not found after creation."
}

$CredentialSource = Join-Path $UserCloudflaredDir "$($Tunnel.id).json"
$CredentialTarget = Join-Path $ConfigDir "$($Tunnel.id).json"

if (-not (Test-Path $CredentialSource)) {
    throw "Tunnel credentials were not found at $CredentialSource"
}

Copy-Item -LiteralPath $CredentialSource -Destination $CredentialTarget -Force

$ConfigPath = Join-Path $ConfigDir "config.yml"
$Config = @"
tunnel: $($Tunnel.id)
credentials-file: $CredentialTarget

ingress:
  - hostname: $Hostname
    service: http://127.0.0.1:$Port
  - service: http_status:404
"@

Set-Content -LiteralPath $ConfigPath -Value $Config -Encoding ascii

Write-Host "[cloudflare] Routing https://$Hostname to tunnel '$TunnelName'"
& $Cloudflared tunnel route dns $TunnelName $Hostname

Write-Host ""
Write-Host "[cloudflare] Named tunnel is ready."
Write-Host "[cloudflare] Config: $ConfigPath"
Write-Host "[cloudflare] Start it with:"
Write-Host "  powershell -ExecutionPolicy Bypass -File .\keep-cloudflare-named-tunnel-online.ps1 -Hostname $Hostname"
