@echo off
setlocal enabledelayedexpansion

REM Setup Visual Studio environment if not already set
if not defined INCLUDE (
    echo Setting up Visual Studio environment...
    call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
)

echo Building C++ LLM Inference Engine...
echo.

REM Create build directory if it doesn't exist
if not exist build mkdir build
cd build

REM Clean old exe if it exists
if exist llm_engine.exe del llm_engine.exe

REM Compile sources with MSVC
echo Compiling source files with MSVC...
cl /std:c++17 /O2 /EHsc /I..\include ..\src\*.cpp /Fe:llm_engine.exe

if %ERRORLEVEL% EQU 0 (
    echo.
    echo Build successful! Running the program...
    echo.
    timeout /t 1 /nobreak
    llm_engine.exe
) else (
    echo Build failed with error code %ERRORLEVEL%
    pause
)

cd ..
