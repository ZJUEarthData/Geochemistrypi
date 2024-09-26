@echo off

chcp 65001 > nul

setlocal enabledelayedexpansion

set ENVIRONMENT_NAME=GeoUI

:: 检查 activate 命令是否存在
where activate >nul 2>&1
if %errorlevel% neq 0 (
    echo activate is not installed or not in the PATH.
    exit /b 1
) else (
    echo Activate is available.
    for /f "tokens=*" %%i in ('where activate') do set ActivatePATH=%%i
    echo ActivatePATH is !ActivatePATH!
)

:: 检查环境是否已经存在
call conda info --envs | findstr /C:"%ENVIRONMENT_NAME% " >nul
if %errorlevel% equ 0 (
    echo Environment %ENVIRONMENT_NAME% already exists.
    goto :eof
) else (
    echo Environment %ENVIRONMENT_NAME% does not exist. Creating...
    call conda create -n %ENVIRONMENT_NAME% python=3.9 -y
    if %errorlevel% neq 0 (
        echo Error: Failed to create conda environment.
        pause
        exit /b %errorlevel%
    )
)


:: 激活新创建的 conda 虚拟环境
call conda activate %ENVIRONMENT_NAME%
if %errorlevel% equ 0 (
    echo Environment %ENVIRONMENT_NAME% activated successfully.
) else (
    echo Error: Failed to activate conda environment.
    pause
    exit /b %errorlevel%
)

echo Choice: 【1】Online installation  【2】Local installation
choice /C:12 /N
set _erl=%errorlevel%

:: 检查用户选择
if %_erl%==1 goto PipGeo
if %_erl%==2 goto InstallDependencies

endlocal

:: 在激活的环境中安装 geochemistrypi
:PipGeo
echo Online installation is underway...
pip install geochemistrypi==0.5.0
if %errorlevel% equ 0 (
    echo geochemistrypi installed successfully.
) else (
    echo Error: Failed to install geochemistrypi.
    pause
    exit /b %errorlevel%
)

pip install tkinterdnd2
if %errorlevel% equ 0 (
    echo tkinterdnd2 installed successfully.
) else (
    echo Error: Failed to install tkinterdnd2.
    pause
    exit /b %errorlevel%
)

:: 检查 geochemistrypi 和 tkinterdnd2 是否安装成功
python -c "import geochemistrypi"
if %errorlevel% equ 0 (
    echo geochemistrypi is installed.
) else (
    echo Error: geochemistrypi is not installed.
    pause
    exit /b %errorlevel%
)

python -c "import tkinterdnd2"
if %errorlevel% equ 0 (
    echo tkinterdnd2 is installed.
) else (
    echo Error: tkinterdnd2 is not installed.
    pause
    exit /b %errorlevel%
)
pause
goto :eof

:InstallDependencies
echo Local installation is underway...
:: 获取当前路径地址
set "CurrentFolder=%~dp0"
echo The current folder path: %cd%

:: 安装库
cd %cd%
echo Dependency package is downloading...

call pip install --no-index --find-links=wheels -r %CurrentFolder%requirements.txt
if %errorlevel% equ 0 (
    echo Dependencies installed successfully.
) else (
    echo Error: Failed to install dependencies.
    pause
    exit /b %errorlevel%
)

call pip install geochemistrypi
pause
goto :eof



pause