# ============================================================
#  Activate the SDCombo training/inference environment (Git Bash).
#  Usage:  source scripts/activate_env.sh
#
#  Note: this only activates the conda env. Rebuilding the DCNv3 CUDA
#  extension additionally needs the MSVC environment -- use
#  scripts/with_msvc.bat for that.
# ============================================================
CONDA_ROOT="/d/Programs/miniforge3"
ENV_NAME="SDCombo"

# shellcheck disable=SC1091
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}" || { echo "[ERROR] conda activate ${ENV_NAME} failed"; return 1; }

# Help torch.utils.cpp_extension locate nvcc (conda-forge cuda-nvcc).
export CUDA_HOME="${CONDA_PREFIX}/Library"
export CUDA_PATH="${CONDA_PREFIX}/Library"
# Target Blackwell / RTX 50 series when JIT-compiling CUDA extensions.
export TORCH_CUDA_ARCH_LIST="12.0"

cd "$(dirname "${BASH_SOURCE[0]}")/.." || return 1

echo
echo "SDCombo environment activated."
echo "  python    : ${CONDA_PREFIX}/python.exe"
echo "  CUDA_HOME : ${CUDA_HOME}"
echo "  repo root : $(pwd)"
echo
