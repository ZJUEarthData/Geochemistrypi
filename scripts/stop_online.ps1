[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$stateFile = Join-Path $projectRoot 'runtime\online-processes.json'

function Get-ShortSha256([string]$value) {
    $sha256 = [Security.Cryptography.SHA256]::Create()
    try {
        $bytes = [Text.Encoding]::UTF8.GetBytes($value)
        $hash = $sha256.ComputeHash($bytes)
        return ([BitConverter]::ToString($hash).Replace('-', '').ToLowerInvariant()).Substring(0, 16)
    }
    finally {
        $sha256.Dispose()
    }
}

$normalizedProjectRoot = [IO.Path]::GetFullPath($projectRoot).Replace('\', '/').TrimEnd('/').ToLowerInvariant()
$instanceId = Get-ShortSha256 $normalizedProjectRoot

if (-not (Test-Path -LiteralPath $stateFile)) {
    Write-Host 'No Online processes were recorded.'
    exit 0
}

$state = Get-Content -LiteralPath $stateFile -Raw | ConvertFrom-Json

if (-not $state.instanceId -or $state.instanceId -ne $instanceId) {
    Write-Warning 'The process-state file belongs to a different or legacy project instance. No process was stopped.'
    exit 1
}

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

    $processInfo = Get-CimInstance Win32_Process -Filter "ProcessId=$($process.Id)" -ErrorAction SilentlyContinue
    if (-not $record.commandLine -or -not $processInfo -or $processInfo.CommandLine -ne $record.commandLine) {
        Write-Warning "$label PID command line no longer matches the recorded process; it will not be stopped."
        return
    }

    Stop-Process -Id $process.Id
    Write-Host "$label stopped."
}

Stop-RecordedProcess $state.frontend 'Vue development server'
Stop-RecordedProcess $state.backend 'Online API'
Remove-Item -LiteralPath $stateFile -Force
