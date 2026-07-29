[CmdletBinding()]
param(
    [switch]$NoBrowser,
    [switch]$SkipInstall
)

$ErrorActionPreference = 'Stop'

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$frontendRoot = Join-Path $projectRoot 'geochemistrypi\frontend'
$runtimeRoot = Join-Path $projectRoot 'runtime'
$logsRoot = Join-Path $runtimeRoot 'logs'
$stateFile = Join-Path $runtimeRoot 'online-processes.json'
$backendUrl = 'http://127.0.0.1:8000'
$frontendUrl = 'http://127.0.0.1:5173/online'

New-Item -ItemType Directory -Force -Path $logsRoot | Out-Null

function Refresh-OnlineProcessPath {
    $machinePath = [Environment]::GetEnvironmentVariable('Path', 'Machine')
    $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
    $pathParts = @($machinePath, $userPath, $env:Path) | Where-Object { $_ }
    $env:Path = $pathParts -join ';'
}

function Install-OnlineTool([string]$packageId, [string]$displayName) {
    $wingetCommand = Get-Command winget -ErrorAction SilentlyContinue
    if (-not $wingetCommand) {
        throw (
            "$displayName is required, but WinGet is unavailable. " +
            'Install Microsoft App Installer/WinGet, or install the tool manually, then run this script again.'
        )
    }

    Write-Host "$displayName was not found. Installing it with WinGet..." -ForegroundColor Yellow
    $arguments = @(
        'install',
        '--id', $packageId,
        '--exact',
        '--source', 'winget',
        '--silent',
        '--accept-package-agreements',
        '--accept-source-agreements',
        '--disable-interactivity'
    )
    & $wingetCommand.Source @arguments | Out-Host
    $installExitCode = $LASTEXITCODE
    if ($installExitCode -ne 0) {
        throw (
            "$displayName installation failed with WinGet exit code $installExitCode. " +
            'Windows may require administrator approval or a manual installation.'
        )
    }

    Refresh-OnlineProcessPath
}

function Get-CompatiblePythonPath {
    $candidates = @()
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCommand) {
        $candidates += $pythonCommand.Source
    }
    $candidates += @(
        (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe'),
        (Join-Path $env:ProgramFiles 'Python312\python.exe')
    )

    foreach ($candidate in ($candidates | Select-Object -Unique)) {
        if (-not $candidate -or -not (Test-Path -LiteralPath $candidate)) {
            continue
        }
        $details = & $candidate -c 'import sys; print(f"{sys.executable}|{sys.version_info.major}|{sys.version_info.minor}")' 2>$null
        if ($LASTEXITCODE -ne 0 -or -not $details) {
            continue
        }
        $parts = "$details".Trim().Split('|')
        if ($parts.Count -eq 3 -and ([int]$parts[1] -gt 3 -or ([int]$parts[1] -eq 3 -and [int]$parts[2] -ge 11))) {
            return $parts[0]
        }
    }

    $launcherCommand = Get-Command py -ErrorAction SilentlyContinue
    if ($launcherCommand) {
        $details = & $launcherCommand.Source -3 -c 'import sys; print(f"{sys.executable}|{sys.version_info.major}|{sys.version_info.minor}")' 2>$null
        if ($LASTEXITCODE -eq 0 -and $details) {
            $parts = "$details".Trim().Split('|')
            if ($parts.Count -eq 3 -and ([int]$parts[1] -gt 3 -or ([int]$parts[1] -eq 3 -and [int]$parts[2] -ge 11))) {
                return $parts[0]
            }
        }
    }

    return $null
}

function Resolve-BootstrapPython {
    $pythonPath = Get-CompatiblePythonPath
    if (-not $pythonPath) {
        if ($SkipInstall) {
            throw 'Python 3.11 or newer was not found and -SkipInstall prevents automatic installation.'
        }
        Install-OnlineTool 'Python.Python.3.12' 'Python 3.12'
        $pythonPath = Get-CompatiblePythonPath
    }

    if (-not $pythonPath) {
        throw 'Python installation completed, but Python 3.11 or newer could not be detected. Restart Windows and try again.'
    }

    $pythonVersion = & $pythonPath -c 'import platform; print(platform.python_version())'
    Write-Host "Python $pythonVersion detected: $pythonPath"
    return $pythonPath
}

function Get-CompatibleNodePath {
    $candidates = @()
    $nodeCommand = Get-Command node -ErrorAction SilentlyContinue
    if ($nodeCommand) {
        $candidates += $nodeCommand.Source
    }
    $userProfilePath = [Environment]::GetFolderPath('UserProfile')
    $candidates += @(
        (Join-Path $userProfilePath '.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe'),
        (Join-Path $env:ProgramFiles 'nodejs\node.exe'),
        (Join-Path $env:LOCALAPPDATA 'Programs\nodejs\node.exe')
    )

    foreach ($candidate in ($candidates | Select-Object -Unique)) {
        if (-not $candidate -or -not (Test-Path -LiteralPath $candidate)) {
            continue
        }
        $versionText = & $candidate --version 2>$null
        if ($LASTEXITCODE -ne 0 -or -not $versionText) {
            continue
        }
        $majorVersion = 0
        if ([int]::TryParse("$versionText".Trim().TrimStart('v').Split('.')[0], [ref]$majorVersion) -and $majorVersion -ge 20) {
            return $candidate
        }
    }

    return $null
}

function Resolve-OnlineNode {
    $nodePath = Get-CompatibleNodePath
    if (-not $nodePath) {
        if ($SkipInstall) {
            throw 'Node.js 20 or newer was not found and -SkipInstall prevents automatic installation.'
        }
        Install-OnlineTool 'OpenJS.NodeJS.LTS' 'Node.js LTS'
        $nodePath = Get-CompatibleNodePath
    }

    if (-not $nodePath) {
        throw 'Node.js installation completed, but Node.js 20 or newer could not be detected. Restart Windows and try again.'
    }

    $nodeVersion = & $nodePath --version
    Write-Host "Node.js $nodeVersion detected: $nodePath"
    return $nodePath
}

function Install-FrontendDependencies([string]$nodeExecutable) {
    $nodeDirectory = Split-Path -Parent $nodeExecutable
    $originalProcessPath = $env:Path
    try {
        $env:Path = "$nodeDirectory;$originalProcessPath"
        $pnpmCommand = Get-Command pnpm -ErrorAction SilentlyContinue
        $userProfilePath = [Environment]::GetFolderPath('UserProfile')
        $bundledPnpm = Join-Path $userProfilePath '.cache\codex-runtimes\codex-primary-runtime\dependencies\bin\fallback\pnpm.cmd'
        $npmCommand = Get-Command npm -ErrorAction SilentlyContinue
        if ($pnpmCommand) {
            & $pnpmCommand.Source install
        }
        elseif (Test-Path -LiteralPath $bundledPnpm) {
            & $bundledPnpm install
        }
        elseif ($npmCommand) {
            & $npmCommand.Source install
        }
        else {
            throw 'Neither pnpm nor npm was found. Install pnpm and run this script again.'
        }
        if ($LASTEXITCODE -ne 0) {
            throw "Frontend dependency installation failed with exit code $LASTEXITCODE."
        }
    }
    finally {
        $env:Path = $originalProcessPath
    }
}

function Test-BackendReady {
    try {
        $response = Invoke-RestMethod -Uri "$backendUrl/api/health" -TimeoutSec 2
        return $response.status -eq 'ok' -and $response.service -eq 'geochemistrypi-online'
    }
    catch {
        return $false
    }
}

function Test-FrontendReady {
    try {
        $response = Invoke-WebRequest -UseBasicParsing -Uri $frontendUrl -TimeoutSec 2
        return $response.StatusCode -eq 200
    }
    catch {
        return $false
    }
}

function Wait-OnlineService([scriptblock]$readyCheck, [int]$timeoutSeconds = 30) {
    $deadline = (Get-Date).AddSeconds($timeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        if (& $readyCheck) {
            return $true
        }
        Start-Sleep -Milliseconds 500
    }
    return $false
}

function Get-ListenerRecord([int]$port, $fallbackProcess, [string]$fallbackPath) {
    $connection = Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($connection) {
        $listenerProcess = Get-Process -Id $connection.OwningProcess -ErrorAction SilentlyContinue
        if ($listenerProcess -and $listenerProcess.Path) {
            return [ordered]@{ id = $listenerProcess.Id; path = $listenerProcess.Path }
        }
    }

    return [ordered]@{ id = $fallbackProcess.Id; path = $fallbackPath }
}

$venvPython = Join-Path $projectRoot '.venv-online\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $venvPython)) {
    if ($SkipInstall) {
        throw "Online Python environment not found: $venvPython"
    }

    $bootstrapPython = Resolve-BootstrapPython
    Write-Host 'Creating the Online Python environment...'
    & $bootstrapPython -m venv $venvPython.Replace('\Scripts\python.exe', '')
    if ($LASTEXITCODE -ne 0) {
        throw "Python environment creation failed with exit code $LASTEXITCODE."
    }
}
else {
    $onlinePythonVersion = & $venvPython -c 'import platform; print(platform.python_version())'
    Write-Host "Existing Online Python environment detected: $onlinePythonVersion"
}

& $venvPython -c 'import fastapi, uvicorn, pandas, openpyxl, multipart, rich, scipy, sklearn, joblib' 2>$null
$runtimeImportsReady = $LASTEXITCODE -eq 0
if (-not $runtimeImportsReady) {
    if ($SkipInstall) {
        throw 'Online Python dependencies are missing. Run without -SkipInstall to install them.'
    }
    Write-Host 'Installing the minimal Online Python dependencies...'
    & $venvPython -m pip install -r (Join-Path $projectRoot 'requirements-online.txt')
    if ($LASTEXITCODE -ne 0) {
        throw "Python dependency installation failed with exit code $LASTEXITCODE."
    }
}

$nodeExecutable = Resolve-OnlineNode
$viteEntry = Join-Path $frontendRoot 'node_modules\vite\bin\vite.js'
if (-not (Test-Path -LiteralPath $viteEntry)) {
    if ($SkipInstall) {
        throw 'Frontend dependencies are missing. Run without -SkipInstall to install them.'
    }
    Write-Host 'Installing frontend dependencies...'
    Push-Location $frontendRoot
    try {
        Install-FrontendDependencies $nodeExecutable
    }
    finally {
        Pop-Location
    }
}

$state = [ordered]@{
    startedAt = (Get-Date).ToString('o')
    backend = $null
    frontend = $null
}
if (Test-Path -LiteralPath $stateFile) {
    try {
        $previousState = Get-Content -LiteralPath $stateFile -Raw | ConvertFrom-Json
        $state.backend = $previousState.backend
        $state.frontend = $previousState.frontend
    }
    catch {
        Write-Warning 'The previous process-state file is invalid and will be replaced.'
    }
}

$startedBackend = $null
$startedFrontend = $null

if (-not (Test-BackendReady)) {
    Write-Host 'Starting the Online API...'
    $startedBackend = Start-Process `
        -FilePath $venvPython `
        -ArgumentList @('-m', 'uvicorn', 'geochemistrypi.online.app:app', '--host', '127.0.0.1', '--port', '8000') `
        -WorkingDirectory $projectRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $logsRoot 'backend.out.log') `
        -RedirectStandardError (Join-Path $logsRoot 'backend.err.log') `
        -PassThru

    if (-not (Wait-OnlineService ${function:Test-BackendReady})) {
        if (-not $startedBackend.HasExited) {
            Stop-Process -Id $startedBackend.Id -ErrorAction SilentlyContinue
        }
        Remove-Item -LiteralPath $stateFile -Force -ErrorAction SilentlyContinue
        throw "Online API startup failed. Check backend logs in $logsRoot"
    }
    $state.backend = Get-ListenerRecord 8000 $startedBackend $venvPython
}
else {
    Write-Host 'Online API is already running.'
}

if (-not (Test-FrontendReady)) {
    Write-Host 'Starting the Vue development server...'
    $startedFrontend = Start-Process `
        -FilePath $nodeExecutable `
        -ArgumentList @($viteEntry, '--host', '127.0.0.1', '--port', '5173') `
        -WorkingDirectory $frontendRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $logsRoot 'frontend.out.log') `
        -RedirectStandardError (Join-Path $logsRoot 'frontend.err.log') `
        -PassThru

    if (-not (Wait-OnlineService ${function:Test-FrontendReady})) {
        if (-not $startedFrontend.HasExited) {
            Stop-Process -Id $startedFrontend.Id -ErrorAction SilentlyContinue
        }
        Remove-Item -LiteralPath $stateFile -Force -ErrorAction SilentlyContinue
        throw "Vue startup failed. Check frontend logs in $logsRoot"
    }
    $state.frontend = Get-ListenerRecord 5173 $startedFrontend $nodeExecutable
}
else {
    Write-Host 'Vue development server is already running.'
}

$state | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $stateFile -Encoding UTF8

Write-Host ''
Write-Host 'Geochemistry Pi Online is ready.' -ForegroundColor Green
Write-Host "Online page: $frontendUrl"
Write-Host "API docs:    $backendUrl/docs"
Write-Host "Logs:        $logsRoot"

if (-not $NoBrowser) {
    Start-Process $frontendUrl
}
