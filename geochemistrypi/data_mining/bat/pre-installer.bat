@echo off

chcp 65001 >nul

setlocal

REM 检查系统中是否安装了 conda
where conda >nul 2>&1
if %errorlevel% equ 0 (
    echo Conda is already installed.
) else (
    echo Conda is not installed. Installing now...
    
    REM 检查是否提供了下载 URL 和安装路径参数
    if "%~1"=="" (
        echo Usage: %~nx0 https://repo.anaconda.com/archive/Anaconda3-2023.07-1-Windows-x86_64.exe %UserProfile%/geochemistrypi/Anaconda3-2023.07-1-Windows-x86_64.exe
        goto end
    )
    
    REM 设置下载 URL 和安装路径
    set url=%~1
    set install_path=%~2
    
    REM 提取文件名
    for /f "tokens=*" %%i in ("%url%") do set installer=%%~nxi
    
    REM 检查并创建 %ProgramFiles(x86)%\geochemistrypi 文件夹
    if not exist "%ProgramFiles(x86)%\geochemistrypi" (
        echo Creating directory: "%ProgramFiles(x86)%\geochemistrypi"
        mkdir "%ProgramFiles(x86)%\geochemistrypi"
    )
    
    echo Downloading Anaconda installer...
    powershell -Command "Invoke-WebRequest -Uri %url% -OutFile %install_path%
/%installer%"
    
    REM 安装 Anaconda
    echo Installing Anaconda...
    start /wait "%installer%" /S /D=%install_path%
    
    REM 删除安装程序
    del %installer%
    
    echo Anaconda installation complete.
)

:end
endlocal
pause