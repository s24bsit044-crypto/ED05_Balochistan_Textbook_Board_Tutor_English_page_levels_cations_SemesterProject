$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$HostAddress = "127.0.0.1"
$Port = 8000
$Url = "http://${HostAddress}:${Port}/health"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Virtual environment not found at $Python. Create it with: py -3.11 -m venv .venv"
}

$existing = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($existing) {
    try {
        $health = Invoke-RestMethod -Uri $Url -TimeoutSec 5
        Write-Host "ED-05 server is already running: http://${HostAddress}:${Port}/"
        exit 0
    } catch {
        Write-Warning "Port $Port is occupied but the ED-05 health check failed. Stop the process using that port and try again."
        exit 1
    }
}

Write-Host "Starting ED-05 server..."
& $Python -m uvicorn backend.main:app --host $HostAddress --port $Port
