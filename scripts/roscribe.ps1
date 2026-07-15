# ROScribe — native Windows control script (PowerShell). Mirrors roscribe_smart.sh:
#   .\scripts\roscribe.ps1 start
#   .\scripts\roscribe.ps1 stop
#   .\scripts\roscribe.ps1 status
#
# Run from the repo root, or it will cd there itself.

param(
    [Parameter(Position=0)][ValidateSet("start","stop","status")]
    [string]$Command = "status"
)

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$PidFile = Join-Path $RepoRoot ".roscribe_app.pid"
$LogFile = Join-Path $RepoRoot "data\roscribe_app.log"
$VenvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"

function Load-EnvFile {
    $envFile = Join-Path $RepoRoot ".env_smart"
    if (-not (Test-Path $envFile)) { $envFile = Join-Path $RepoRoot ".env" }
    if (-not (Test-Path $envFile)) { return }
    Get-Content $envFile | ForEach-Object {
        if ($_ -match '^\s*#' -or $_ -notmatch '=') { return }
        $parts = $_ -split '=', 2
        $key = $parts[0].Trim()
        $val = $parts[1].Trim().Trim('"').Trim("'")
        [System.Environment]::SetEnvironmentVariable($key, $val, "Process")
    }
}

function Get-RunningPid {
    if (-not (Test-Path $PidFile)) { return $null }
    $p = Get-Content $PidFile -ErrorAction SilentlyContinue
    if ($p -and (Get-Process -Id $p -ErrorAction SilentlyContinue)) { return $p }
    return $null
}

switch ($Command) {
    "start" {
        if (-not (Test-Path $VenvPython)) {
            Write-Host "Virtual environment not found. Run:" -ForegroundColor Red
            Write-Host "  python -m venv .venv; .venv\Scripts\pip install -r requirements.txt"
            exit 1
        }
        $existing = Get-RunningPid
        if ($existing) {
            Write-Host "ROScribe already running (PID $existing)." -ForegroundColor Yellow
            exit 0
        }
        Load-EnvFile
        Write-Host "Starting ROScribe Smart server..." -ForegroundColor Green
        $proc = Start-Process -FilePath $VenvPython `
            -ArgumentList "-u", "app\workspace_smart.py" `
            -WorkingDirectory $RepoRoot `
            -RedirectStandardOutput $LogFile `
            -RedirectStandardError "$LogFile.err" `
            -WindowStyle Hidden -PassThru
        Set-Content -Path $PidFile -Value $proc.Id

        Write-Host "Waiting for app to start" -NoNewline
        for ($i = 0; $i -lt 30; $i++) {
            try {
                $r = Invoke-WebRequest -Uri "http://127.0.0.1:8081/login" -UseBasicParsing -TimeoutSec 2
                if ($r.StatusCode -eq 200) { Write-Host " Online!" -ForegroundColor Green; break }
            } catch {}
            Write-Host "." -NoNewline
            Start-Sleep -Seconds 1
        }
        Write-Host "Local URL: http://127.0.0.1:8081"
    }
    "stop" {
        $p = Get-RunningPid
        if ($p) {
            Stop-Process -Id $p -Force
            Remove-Item $PidFile -ErrorAction SilentlyContinue
            Write-Host "Stopped ROScribe (PID $p)." -ForegroundColor Green
        } else {
            Write-Host "ROScribe is not running." -ForegroundColor Yellow
        }
    }
    "status" {
        $p = Get-RunningPid
        if ($p) {
            Write-Host "App Server : RUNNING (PID $p)" -ForegroundColor Green
            Write-Host "Local URL  : http://127.0.0.1:8081"
        } else {
            Write-Host "App Server : STOPPED" -ForegroundColor Red
        }
        $funnel = & tailscale funnel status 2>$null
        if ($funnel -match "https://\S+\.ts\.net") {
            Write-Host "Public URL : $($Matches[0])"
        }
    }
}
