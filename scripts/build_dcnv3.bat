@echo off
REM ============================================================================
REM  编译 InternImage 的 DCNv3 CUDA 扩展（inplace）。
REM
REM  必须先进入 MSVC + conda-forge nvcc 环境，标准用法：
REM      scripts\with_msvc.bat scripts\build_dcnv3.bat
REM
REM  产物：Backbone\InternImage\ops_dcnv3\DCNv3.cp310-win_amd64.pyd
REM  （配合 <conda env>\Lib\site-packages\sitecustomize.py 把该目录加入 sys.path，
REM    模型里的 `import DCNv3` 才可用；详见 ReadMe_训练环境.md）
REM ============================================================================
setlocal
cd /d "%~dp0..\Backbone\InternImage\ops_dcnv3"
if errorlevel 1 (
    echo [ERROR] 找不到 Backbone\InternImage\ops_dcnv3
    exit /b 1
)
echo [build_dcnv3] cwd=%CD%
echo [build_dcnv3] TORCH_CUDA_ARCH_LIST=%TORCH_CUDA_ARCH_LIST%
python setup.py build_ext --inplace
set RC=%errorlevel%
echo [build_dcnv3] exit=%RC%
exit /b %RC%
