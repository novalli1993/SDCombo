@echo off
REM ============================================================================
REM  Run any command inside the MSVC (x64) + CUDA (conda-forge nvcc) environment.
REM  Usage:
REM      scripts\with_msvc.bat python tools\verify_env.py
REM      scripts\with_msvc.bat cmd
REM ============================================================================
setlocal enableextensions

set "CONDA_ROOT=D:\Programs\miniforge3"
set "ENV_NAME=SDCombo"
set "CONDA_PREFIX=%CONDA_ROOT%\envs\%ENV_NAME%"

REM ---- Drop the variables injected by the conda-forge `vc` metapackage;
REM ---- they point at VS2019/16.0 and break vcvars64.
set "VS_VERSION="
set "VS_MAJOR="
set "VS_YEAR="
set "INCLUDE="
set "LIB="
set "LIBPATH="
set "DISTUTILS_USE_SDK="
set "MSSdk="
set "WindowsSdkDir="
set "WindowsSDKVersion="
set "VCINSTALLDIR="
set "VCToolsInstallDir="
set "VCToolsVersion="

REM ---- Enter the MSVC x64 toolchain environment ----
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] vcvars64.bat failed
    exit /b 1
)

REM ---- Put the conda env first on PATH (python / ninja / nvcc) ----
set "PATH=%CONDA_PREFIX%;%CONDA_PREFIX%\Scripts;%CONDA_PREFIX%\Library\bin;%CONDA_PREFIX%\Library\bin\x64;%PATH%"

REM ---- CUDA toolkit location: conda-forge cuda-nvcc lives in env\Library ----
set "CUDA_HOME=%CONDA_PREFIX%\Library"
set "CUDA_PATH=%CONDA_PREFIX%\Library"
set "CUDA_PATH_V12_8=%CONDA_PREFIX%\Library"
REM cudart.lib / cudart_static.lib live under env\Library\lib; add them to the
REM linker search path (nvcc's own -L does not cover the final MSVC link step).
set "LIB=%CONDA_PREFIX%\Library\lib;%LIB%"

REM ---- Target Blackwell (RTX 50 series) ----
if not defined TORCH_CUDA_ARCH_LIST set "TORCH_CUDA_ARCH_LIST=12.0"

REM ---- Force English tool output.
REM ---- torch.utils.cpp_extension parses `cl` version output; on a zh-CN Windows
REM ---- the localized bytes break its decoder ('cp1' codec errors).
set "VSLANG=1033"
set "DOTNET_CLI_UI_LANGUAGE=en"
REM ---- UTF-8 mode: `cl` prints in the ANSI code page (cp936 here) while
REM ---- torch decodes subprocess pipes with a hardcoded codec list.
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

REM ---- Let nvcc accept a newer MSVC than officially listed (VS2022 17.14 / 19.44) ----
if not defined NVCC_PREPEND_FLAGS set "NVCC_PREPEND_FLAGS=-allow-unsupported-compiler"
set "CFLAGS=%CFLAGS% /D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH"

cd /d "%~dp0.."

endlocal & (
    set "PATH=%PATH%"
    set "CUDA_HOME=%CUDA_HOME%"
    set "CUDA_PATH=%CUDA_PATH%"
    set "CUDA_PATH_V12_8=%CUDA_PATH_V12_8%"
    set "TORCH_CUDA_ARCH_LIST=%TORCH_CUDA_ARCH_LIST%"
    set "NVCC_PREPEND_FLAGS=%NVCC_PREPEND_FLAGS%"
    set "VSLANG=%VSLANG%"
    set "PYTHONIOENCODING=%PYTHONIOENCODING%"
    set "PYTHONUTF8=%PYTHONUTF8%"
    set "CFLAGS=%CFLAGS%"
    set "INCLUDE=%INCLUDE%"
    set "LIB=%LIB%"
    set "LIBPATH=%LIBPATH%"
    set "VCToolsInstallDir=%VCToolsInstallDir%"
    set "VCINSTALLDIR=%VCINSTALLDIR%"
    set "WindowsSdkDir=%WindowsSdkDir%"
    set "WindowsSDKVersion=%WindowsSDKVersion%"
)
%*
exit /b %errorlevel%
