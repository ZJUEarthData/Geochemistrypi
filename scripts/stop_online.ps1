[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$stateFile = Join-Path $projectRoot 'runtime\online-processes.json'

if (-not (Test-Path -LiteralPath $stateFile)) {
    Write-Host 'No Online processes were recorded.'
    exit 0
}

$state = Get-Content -LiteralPath $stateFile -Raw | ConvertFrom-Json

function Stop-RecordedProcess($record, [string]$label) {
    if (-not $record) {
        Write-Host "$label was not started by the one-click launcher."
        return
    }

    $process = Get-Process -Id ([int]$record.id) -ErrorAction SilentlyContinue
    if (-not $process) {
        Write-Host "$label is already stopped."
        return
    }

    $actualPath = $process.Path
    $recordedPath = [IO.Path]::GetFullPath([string]$record.path)
    if (-not $actualPath -or -not [string]::Equals([IO.Path]::GetFullPath($actualPath), $recordedPath, [StringComparison]::OrdinalIgnoreCase)) {
        Write-Warning "$label PID now belongs to another process; it will not be stopped."
        return
    }

    Stop-Process -Id $process.Id
    Write-Host "$label stopped."
}

Stop-RecordedProcess $state.frontend 'Vue development server'
Stop-RecordedProcess $state.backend 'Online API'
Remove-Item -LiteralPath $stateFile -Force
