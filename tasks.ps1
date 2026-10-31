<#
.SYNOPSIS  Windows equivalent of the Makefile. Usage: .\tasks.ps1 <task>
#>
param(
    [Parameter(Position = 0)]
    [ValidateSet('help', 'setup', 'lint', 'typecheck', 'test', 'check', 'up', 'down',
                 'obs-up', 'obs-down', 'config', 'sandbox-build', 'clean')]
    [string]$Task = 'help'
)

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

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
    'help'      { 'Tasks: setup lint typecheck test check up down obs-up obs-down config sandbox-build clean' }
    'setup'     { Invoke-Step uv @('sync', '--frozen') }
    'lint'      {
        Invoke-Step uv @('run', '--frozen', 'ruff', 'check', '.')
        Invoke-Step uv @('run', '--frozen', 'ruff', 'format', '--check', '.')
    }
    'typecheck' { Invoke-Step uv @('run', '--frozen', 'mypy', 'src') }
    'test'      { Invoke-Step uv @('run', '--frozen', 'pytest', '-q') }
    'check'     { & $PSCommandPath lint; & $PSCommandPath typecheck; & $PSCommandPath test }
    'up'        { Invoke-Step docker @('compose', 'up', '-d', '--build', 'api') }
    'down'      { Invoke-Step docker @('compose', 'down') }
    'obs-up'    { Assert-EnvReady; Invoke-Step docker @('compose', '--profile', 'observability', 'up', '-d') }
    'obs-down'  { Invoke-Step docker @('compose', '--profile', 'observability', 'down') }
    'config'    {
        Invoke-Step docker @('compose', 'config', '-q')
        Invoke-Step docker @('compose', '--profile', 'observability', 'config', '-q')
    }
    'sandbox-build' { Invoke-Step docker @('compose', 'build', 'sandbox') }
    'clean'     { Invoke-Step docker @('compose', '--profile', 'observability', 'down', '-v', '--remove-orphans') }
}
