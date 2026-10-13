<#
.SYNOPSIS
    GodhomeRL -- one-command setup for Windows PowerShell.

.DESCRIPTION
    Installs the Python dependencies, builds the patched game from your copy of
    the Godhome standalone download, and runs the end-to-end doctor check.

.PARAMETER Venv
    Create and use a virtual environment at this path.

.PARAMETER Install
    Path to the Godhome standalone folder (the one containing Godhome.zip).
    Defaults to $env:GODHOME_RL_INSTALL.

.EXAMPLE
    .\setup.ps1 -Install "C:\Games\godhome-standalone_e1c66"

.EXAMPLE
    .\setup.ps1 -Venv .venv
#>
[CmdletBinding()]
param(
    [string]$Venv = "",
    [string]$Install = ""
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

function Find-Python {
    foreach ($candidate in @("python3", "python", "py")) {
        $cmd = Get-Command $candidate -ErrorAction SilentlyContinue
        if (-not $cmd) { continue }
        try {
            $version = & $candidate -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
            if ($LASTEXITCODE -eq 0) {
                $parts = $version.Split('.')
                if ([int]$parts[0] -gt 3 -or ([int]$parts[0] -eq 3 -and [int]$parts[1] -ge 10)) {
                    return $candidate
                }
            }
        } catch { }
    }
    return $null
}

Write-Host "[setup] locating Python ..." -ForegroundColor Cyan
$py = Find-Python
if (-not $py) {
    Write-Host "[setup] Python 3.10+ is required but was not found." -ForegroundColor Red
    Write-Host "[setup] Install it from https://python.org and tick 'Add python.exe to PATH'." -ForegroundColor Red
    exit 1
}
Write-Host "[setup] using $py ($(& $py --version))"

if ($Venv) {
    $pythonExe = Join-Path $Venv "Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $pythonExe)) {
        Write-Host "[setup] creating a virtual environment at $Venv" -ForegroundColor Cyan
        & $py -m venv $Venv
        if ($LASTEXITCODE -ne 0) { throw "venv creation failed" }
    }
    $py = $pythonExe
}

Write-Host "[setup] installing dependencies ..." -ForegroundColor Cyan
& $py -m pip install --upgrade pip --quiet
& $py -m pip install -e . --quiet
if ($LASTEXITCODE -ne 0) { throw "dependency installation failed" }

$buildArgs = @()
if ($Install) { $buildArgs = @("--install", $Install) }
elseif ($env:GODHOME_RL_INSTALL) { $buildArgs = @("--install", $env:GODHOME_RL_INSTALL) }

Write-Host "[setup] building the patched game (first run unpacks ~4000 assets) ..." -ForegroundColor Cyan
& $py -m godhome_rl build @buildArgs
if ($LASTEXITCODE -ne 0) { throw "build failed" }

Write-Host "[setup] verifying ..." -ForegroundColor Cyan
& $py -m godhome_rl doctor
if ($LASTEXITCODE -ne 0) { Write-Host "[setup] doctor reported problems" -ForegroundColor Yellow }

Write-Host ""
Write-Host "[setup] done." -ForegroundColor Green
if ($Venv) { Write-Host "[setup] activate with:  .\$Venv\Scripts\Activate.ps1" }
Write-Host "[setup] then run:  python -m godhome_rl train --name godhome --bosses all --n-envs 3"
Write-Host "[setup]           python -m godhome_rl play  --model runs/godhome/checkpoints/latest.pt"