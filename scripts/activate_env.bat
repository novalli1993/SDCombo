@echo off
REM ============================================================
REM  Activate the SDCombo training/inference environment (cmd).
REM  Usage:  scripts\activate_env.bat
REM
REM  Note: this only activates the conda env. Rebuilding the DCNv3 CUDA
REM  extension additionally needs the MSVC environment -- use
REM  scripts\with_msvc.bat for that.
REM ============================================================
set "CONDA_ROOT=D:\Programs\miniforge3"
set "ENV_NAME=SDCombo"

call "%CONDA_ROOT%\Scripts\activate.bat" "%CONDA_ROOT%"
call conda activate %ENV_NAME%
if errorlevel 1 (
    echo [ERROR] conda activate %ENV_NAME% failed
    exit /b 1
)

REM Help torch.utils.cpp_extension locate nvcc (conda-forge cuda-nvcc).
set "CUDA_HOME=%CONDA_PREFIX%\Library"
set "CUDA_PATH=%CONDA_PREFIX%\Library"
set "CUDA_PATH_V12_8=%CONDA_PREFIX%\Library"
REM Target Blackwell / RTX 50 series when JIT-compiling CUDA extensions.
set "TORCH_CUDA_ARCH_LIST=12.0"

cd /d "%~dp0.."
echo.
echo SDCombo environment activated.
echo   python    : %CONDA_PREFIX%\python.exe
echo   CUDA_HOME : %CUDA_HOME%
echo   repo root : %CD%
echo.
