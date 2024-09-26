@echo off
setlocal enabledelayedexpansion

chcp 65001 >nul

:: 获取参数
set param=%1
if "%param%"=="" (
    echo Usage: %~nx0 [version]
    exit /b 1
)

:: 设置下载链接和文件名
set download_url="https://github.com/ZJUEarthData/Geochemistrypi/releases/download/%param%/geochemistrypi.exe"
set output_file="%UserProfile%\Downloads\geochemistrypi.exe"

echo Getting the download file: %download_url%
echo Saved address: %output_file%

:: 检查文件是否已经存在
if exist "%output_file%" (
    echo File "%output_file%" already exists. Skipping download...
) else (
    echo Downloading software...
    curl -L -o "%output_file%" "%download_url%"

    :: 检查下载是否成功
    if %errorlevel% neq 0 (
        echo Download failed. Exiting...
        exit /b 1
    )

    :: 检查下载的文件是否存在
    if not exist "%output_file%" (
        echo Downloaded file not found. Exiting...
        exit /b 1
    )
)


:: 安装更新的文件
echo Installing software...
set destination="%ProgramFiles(x86)%\geochemistrypi"

:: 检查目标目录是否存在，如果不存在则创建
if not exist "%destination%\" (
    echo Creating directory: %destination%
    mkdir %destination%
) else (
    echo Directory already exists: %destination%
)
rem start "" "%output_file%"

start "" /wait %output_file% /D %destination%

if exist "%destination%\geochemistrypi.exe" (
    echo File installed successfully.
) else (
    echo Failed to install geochemistrypi.exe
)

echo Installation completed successfully.

endlocal
exit /b 0