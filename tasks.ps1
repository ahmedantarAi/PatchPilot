<#
.SYNOPSIS  Windows equivalent of the Makefile. Usage: .\tasks.ps1 <task>
#>
param(
    [Parameter(Position = 0)]
    [ValidateSet('help', 'api', 'setup', 'lint', 'typecheck', 'test', 'check', 'up', 'down',
                 'obs-up', 'obs-down', 'config', 'sandbox-build', 'clean')]
    [string]$Task = 'help'
)

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

function Write-PthFile {
    if (Test-Path '.venv\Lib\site-packages') {
        Set-Content -Path '.venv\Lib\site-packages\patchpilot.pth' -Value (Resolve-Path 'src').Path -Encoding ascii
    }
}
Write-PthFile

function Invoke-Step {
    param([string]$Exe, [string[]]$Arguments)
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "'$Exe $($Arguments -join ' ')' failed with exit code $LASTEXITCODE" }
}

function Assert-EnvReady {
    if (-not (Test-Path .env)) { throw ".env missing: copy .env.example and replace CHANGEME" }
    if (Select-String -Path .env -Pattern 'CHANGEME' -Quiet) { throw "Replace CHANGEME values in .env" }
}

switch ($Task) {
    'help'      { 'Tasks: api setup lint typecheck test check up down obs-up obs-down config sandbox-build clean' }
    'api'       { Invoke-Step uv @('run', '--frozen', 'uvicorn', 'patchpilot.api:app', '--host', '127.0.0.1', '--port', '8000') }
    'setup'     { Invoke-Step uv @('sync', '--frozen'); Write-PthFile }
    'lint'      {
        Invoke-Step uv @('run', '--frozen', 'ruff', 'check', '.')
        Invoke-Step uv @('run', '--frozen', 'ruff', 'format', '--check', '.')
    }
    'typecheck' { Invoke-Step uv @('run', '--frozen', 'mypy', 'src') }
    'test'      { Invoke-Step uv @('run', '--frozen', 'pytest', '-q') }
    'check'     { & $PSCommandPath lint; & $PSCommandPath typecheck; & $PSCommandPath test }
    'up'        { Invoke-Step docker @('compose', 'up', '-d', '--wait', 'qdrant') }
    'down'      { Invoke-Step docker @('compose', 'down') }
    'obs-up'    { Assert-EnvReady; Invoke-Step docker @('compose', '--profile', 'observability', 'up', '-d') }
    'obs-down'  { Invoke-Step docker @('compose', '--profile', 'observability', 'down') }
    'config'    {
        Invoke-Step docker @('compose', 'config', '-q')
        Invoke-Step docker @('compose', '--profile', 'observability', 'config', '-q')
    }
    'sandbox-build' { Write-PthFile; Invoke-Step docker @('build', '-t', 'patchpilot-sandbox:0.0.0', 'sandbox') }
    'clean'     { Invoke-Step docker @('compose', '--profile', 'observability', 'down', '-v', '--remove-orphans') }
}
