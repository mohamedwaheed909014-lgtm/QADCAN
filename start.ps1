$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (Test-Path ".env") {
    Get-Content ".env" | ForEach-Object {
        $line = $_.Trim()
        if ($line -and -not $line.StartsWith("#") -and $line.Contains("=")) {
            $name, $value = $line.Split("=", 2)
            [Environment]::SetEnvironmentVariable($name.Trim(), $value.Trim(), "Process")
        }
    }
    Write-Host "[start] Loaded .env"
}

Write-Host "[start] Installing Python dependencies..."
python -m pip install -r requirements.txt --quiet

Set-Location "backend"
Write-Host "[start] Starting server at http://localhost:8000"
Write-Host "[start] Press Ctrl+C to stop."
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
