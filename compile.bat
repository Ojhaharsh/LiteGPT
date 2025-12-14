@echo off
REM Simple build script with VS environment setup
setlocal

REM Load Visual Studio environment
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1

REM Change to project root directory
cd /d "%~dp0"
if %ERRORLEVEL% NEQ 0 (
    echo Failed to change to project directory
    exit /b 1
)

echo Compiling Phase 1, 2, 3 ^& 4...
REM Compile directly to current directory, not build subdirectory
cl.exe /std:c++17 /O2 /EHsc /I.\include .\src\tensor.cpp .\src\matmul.cpp .\src\model.cpp .\src\layers.cpp .\src\embedding.cpp .\src\forward_pass.cpp .\src\tokenizer.cpp .\src\inference_engine.cpp .\src\main.cpp /Fe:llm_engine.exe

if %ERRORLEVEL% NEQ 0 (
    echo Compilation failed!
    pause
    exit /b 1
)

echo.
echo ==========================================
echo Compilation successful! Running Phase 1, 2, 3 ^& 4...
echo ==========================================
echo.

llm_engine.exe

echo.
echo Press any key to continue...
pause
endlocal