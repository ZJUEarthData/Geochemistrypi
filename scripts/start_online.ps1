[CmdletBinding()]
param(
    [switch]$NoBrowser,
    [switch]$SkipInstall
)

$ErrorActionPreference = 'Stop'

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$frontendRoot = Join-Path $projectRoot 'geochemistrypi\frontend'
$frontendManifest = Join-Path $frontendRoot 'package.json'
$frontendLockFile = Join-Path $frontendRoot 'pnpm-lock.yaml'
$frontendDependencyStamp = Join-Path $frontendRoot 'node_modules\.online-dependencies.sha256'
$runtimeRoot = Join-Path $projectRoot 'runtime'
$logsRoot = Join-Path $runtimeRoot 'logs'
$stateFile = Join-Path $runtimeRoot 'online-processes.json'
$backendUrl = 'http://127.0.0.1:8000'
$frontendUrl = 'http://127.0.0.1:5173/online'
$frontendIdentityUrl = 'http://127.0.0.1:5173/__geochemistrypi_instance'

New-Item -ItemType Directory -Force -Path $logsRoot | Out-Null

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

function Get-OnlineBuildId {
    $sourcePaths = @(
        (Join-Path $projectRoot 'geochemistrypi\online'),
        (Join-Path $projectRoot 'geochemistrypi\frontend\src'),
        (Join-Path $projectRoot 'geochemistrypi\frontend\vite.config.ts'),
        (Join-Path $projectRoot 'geochemistrypi\_version.py'),
        (Join-Path $projectRoot 'scripts\start_online.ps1')
    )
    $files = foreach ($sourcePath in $sourcePaths) {
        if (Test-Path -LiteralPath $sourcePath -PathType Leaf) {
            Get-Item -LiteralPath $sourcePath
        }
        elseif (Test-Path -LiteralPath $sourcePath -PathType Container) {
            Get-ChildItem -LiteralPath $sourcePath -Recurse -File |
                Where-Object { $_.FullName -notmatch '[\\/](__pycache__|node_modules|dist)[\\/]' }
        }
    }
    $fingerprintLines = foreach ($file in ($files | Sort-Object FullName -Unique)) {
        $relativePath = $file.FullName.Substring($projectRoot.Length).TrimStart('\', '/').Replace('\', '/')
        $fileHash = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        "$relativePath`:$fileHash"
    }
    return Get-ShortSha256 ($fingerprintLines -join "`n")
}

$normalizedProjectRoot = [IO.Path]::GetFullPath($projectRoot).Replace('\', '/').TrimEnd('/').ToLowerInvariant()
$instanceId = Get-ShortSha256 $normalizedProjectRoot
$sourceRevision = 'unknown'
$gitCommand = Get-Command git -ErrorAction SilentlyContinue
if ($gitCommand) {
    $revisionOutput = & $gitCommand.Source -C $projectRoot rev-parse --short=12 HEAD 2>$null
    if ($LASTEXITCODE -eq 0 -and $revisionOutput) {
        $sourceRevision = "$revisionOutput".Trim()
    }
}
$buildId = Get-OnlineBuildId
$env:GEOCHEMISTRYPI_ONLINE_INSTANCE_ID = $instanceId
$env:GEOCHEMISTRYPI_SOURCE_REVISION = $sourceRevision
$env:GEOCHEMISTRYPI_BUILD_ID = $buildId

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
            & $pnpmCommand.Source install --frozen-lockfile
        }
        elseif (Test-Path -LiteralPath $bundledPnpm) {
            & $bundledPnpm install --frozen-lockfile
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

function Get-FrontendDependencyFingerprint {
    $manifestHash = (Get-FileHash -LiteralPath $frontendManifest -Algorithm SHA256).Hash
    $lockHash = if (Test-Path -LiteralPath $frontendLockFile) {
        (Get-FileHash -LiteralPath $frontendLockFile -Algorithm SHA256).Hash
    }
    else {
        'NO-LOCKFILE'
    }
    return "$manifestHash`:$lockHash"
}

function Stop-FrontendForDependencySync([string]$expectedViteEntry) {
    $connection = Get-NetTCPConnection -State Listen -LocalPort 5173 -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if (-not $connection) {
        return
    }

    $processInfo = Get-CimInstance Win32_Process -Filter "ProcessId=$($connection.OwningProcess)"
    $isCurrentViteProcess =
        $processInfo -and
        $processInfo.CommandLine -and
        $processInfo.CommandLine.IndexOf($expectedViteEntry, [StringComparison]::OrdinalIgnoreCase) -ge 0
    $isGeochemistryPiViteProcess =
        $processInfo -and
        $processInfo.CommandLine -and
        $processInfo.CommandLine -match 'geochemistrypi[\\/]+frontend[\\/]+node_modules[\\/]+vite[\\/]+bin[\\/]+vite\.js'
    if (-not $isCurrentViteProcess -and -not $isGeochemistryPiViteProcess) {
        throw 'Frontend dependencies changed, but port 5173 is occupied by an unrelated process. Stop it and try again.'
    }

    Write-Host 'Frontend dependencies changed. Restarting the Vue server safely...' -ForegroundColor Yellow
    Stop-Process -Id $connection.OwningProcess
    Wait-Process -Id $connection.OwningProcess -Timeout 5 -ErrorAction SilentlyContinue
}

function Test-BackendReady {
    try {
        $response = Invoke-RestMethod -Uri "$backendUrl/api/health" -TimeoutSec 2
        return (
            $response.status -eq 'ok' -and
            $response.service -eq 'geochemistrypi-online' -and
            $response.instance_id -eq $instanceId -and
            $response.build_id -eq $buildId
        )
    }
    catch {
        return $false
    }
}

function Test-FrontendReady {
    try {
        $response = Invoke-RestMethod -Uri $frontendIdentityUrl -TimeoutSec 2
        return (
            $response.service -eq 'geochemistrypi-online-frontend' -and
            $response.instance_id -eq $instanceId -and
            $response.build_id -eq $buildId
        )
    }
    catch {
        return $false
    }
}

function Get-ListenerInfo([int]$port) {
    $connection = Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if (-not $connection) {
        return $null
    }

    $processInfo = Get-CimInstance Win32_Process -Filter "ProcessId=$($connection.OwningProcess)" -ErrorAction SilentlyContinue
    $commandLines = @()
    $currentProcessInfo = $processInfo
    for ($depth = 0; $currentProcessInfo -and $depth -lt 4; $depth++) {
        if ($currentProcessInfo.CommandLine) {
            $commandLines += $currentProcessInfo.CommandLine
        }
        if (-not $currentProcessInfo.ParentProcessId) {
            break
        }
        $currentProcessInfo = Get-CimInstance Win32_Process -Filter "ProcessId=$($currentProcessInfo.ParentProcessId)" -ErrorAction SilentlyContinue
    }

    return [pscustomobject]@{
        port = $port
        id = [int]$connection.OwningProcess
        processInfo = $processInfo
        lineageCommandLine = $commandLines -join "`n"
    }
}

function Stop-VerifiedOnlineListener($listener, [string]$label) {
    Write-Host "A different Geochemistry Pi $label instance is using port $($listener.port). Replacing it..." -ForegroundColor Yellow
    Stop-Process -Id $listener.id
    $deadline = (Get-Date).AddSeconds(5)
    while ((Get-Date) -lt $deadline) {
        if (-not (Get-NetTCPConnection -State Listen -LocalPort $listener.port -ErrorAction SilentlyContinue)) {
            return
        }
        Start-Sleep -Milliseconds 200
    }
    throw "The previous Geochemistry Pi $label instance on port $($listener.port) could not be stopped."
}

function Resolve-BackendPort {
    if (Test-BackendReady) {
        return $true
    }

    $listener = Get-ListenerInfo 8000
    if (-not $listener) {
        return $false
    }

    $health = $null
    try {
        $health = Invoke-RestMethod -Uri "$backendUrl/api/health" -TimeoutSec 2
    }
    catch {
        # The process is still classified by its command line below.
    }
    $isGeochemistryPiBackend =
        $health -and
        $health.service -eq 'geochemistrypi-online' -and
        $listener.lineageCommandLine -match 'geochemistrypi\.online\.app:app'
    if (-not $isGeochemistryPiBackend) {
        throw "Port 8000 is occupied by an unrelated process (PID $($listener.id)). It was not stopped."
    }

    Stop-VerifiedOnlineListener $listener 'backend'
    return $false
}

function Resolve-FrontendPort {
    if (Test-FrontendReady) {
        return $true
    }

    $listener = Get-ListenerInfo 5173
    if (-not $listener) {
        return $false
    }

    $isGeochemistryPiFrontend =
        $listener.lineageCommandLine -match 'geochemistrypi[\\/]+frontend[\\/]+node_modules[\\/]+vite[\\/]+bin[\\/]+vite\.js'
    if (-not $isGeochemistryPiFrontend) {
        throw "Port 5173 is occupied by an unrelated process (PID $($listener.id)). It was not stopped."
    }

    Stop-VerifiedOnlineListener $listener 'frontend'
    return $false
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
            $processInfo = Get-CimInstance Win32_Process -Filter "ProcessId=$($listenerProcess.Id)" -ErrorAction SilentlyContinue
            return [ordered]@{
                id = $listenerProcess.Id
                path = $listenerProcess.Path
                commandLine = $processInfo.CommandLine
                instanceId = $instanceId
                buildId = $buildId
            }
        }
    }

    return [ordered]@{
        id = $fallbackProcess.Id
        path = $fallbackPath
        commandLine = $null
        instanceId = $instanceId
        buildId = $buildId
    }
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
$currentDependencyFingerprint = Get-FrontendDependencyFingerprint
$installedDependencyFingerprint = if (Test-Path -LiteralPath $frontendDependencyStamp) {
    (Get-Content -LiteralPath $frontendDependencyStamp -Raw).Trim()
}
else {
    ''
}
$frontendDependenciesNeedInstall =
    -not (Test-Path -LiteralPath $viteEntry) -or
    $installedDependencyFingerprint -ne $currentDependencyFingerprint

if ($frontendDependenciesNeedInstall) {
    if ($SkipInstall) {
        throw 'Frontend dependencies are missing or outdated. Run without -SkipInstall to synchronize them.'
    }

    Stop-FrontendForDependencySync $viteEntry
    Write-Host 'Synchronizing frontend dependencies...'
    Push-Location $frontendRoot
    try {
        Install-FrontendDependencies $nodeExecutable
    }
    finally {
        Pop-Location
    }

    $viteCache = Join-Path $frontendRoot 'node_modules\.vite'
    if (Test-Path -LiteralPath $viteCache) {
        Remove-Item -LiteralPath $viteCache -Recurse -Force
    }
    Get-FrontendDependencyFingerprint |
        Set-Content -LiteralPath $frontendDependencyStamp -Encoding UTF8
}

$state = [ordered]@{
    startedAt = (Get-Date).ToString('o')
    instanceId = $instanceId
    sourceRevision = $sourceRevision
    buildId = $buildId
    backend = $null
    frontend = $null
}

$startedBackend = $null
$startedFrontend = $null
$backendAlreadyRunning = Resolve-BackendPort

if (-not $backendAlreadyRunning) {
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
    Write-Host 'The matching Online API is already running.'
    $state.backend = Get-ListenerRecord 8000 $null $venvPython
}

$frontendAlreadyRunning = Resolve-FrontendPort
if (-not $frontendAlreadyRunning) {
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
    Write-Host 'The matching Vue development server is already running.'
    $state.frontend = Get-ListenerRecord 5173 $null $nodeExecutable
}

$state | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $stateFile -Encoding UTF8

Write-Host ''
Write-Host 'Geochemistry Pi Online is ready.' -ForegroundColor Green
Write-Host "Online page: $frontendUrl"
Write-Host "API docs:    $backendUrl/docs"
Write-Host "Instance:    $instanceId"
Write-Host "Build:       $buildId ($sourceRevision)"
Write-Host "Logs:        $logsRoot"

if (-not $NoBrowser) {
    Start-Process $frontendUrl
}
