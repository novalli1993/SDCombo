# SDCombo — Semantic Segmentation with Depth Information

论文代码仓（Han Li, University of Birmingham, MSc Computer Science, 2023-09）。
论文原文见仓库根目录的 `SDCombo Semantic Segmentation with Depth Information.pdf`。

---

## ⚠️ 先读这两个文件

| 文档 | 内容 |
| --- | --- |
| **[ReadMe_训练环境.md](ReadMe_训练环境.md)** | 本机（Windows + RTX 5090 / sm_120）**已验证可用**的环境配置、数据准备、训练与评估命令、已知陷阱 |
| **[论文符合性检视报告.md](论文符合性检视报告.md)** | 代码与论文的逐条符合性核对：哪些一致、哪些是论文未写的实现细节、哪些是缺陷 |

下面「原始安装说明」是仓库自带的历史指令（**Linux + CUDA 11.8**），
在 RTX 50 系（Blackwell / sm_120）上**不可用**：CUDA 11.8 不含 sm_120 的 kernel，
运行时会报 `no kernel image is available for execution on the device`。

| 项目 | 原始 ReadMe | 本机可用配置 |
| --- | --- | --- |
| PyTorch | conda `pytorch-cuda=11.8` | `torch==2.9.1+cu128`（pip，cu128 起原生支持 sm_120） |
| CUDA 编译器 | 系统 CUDA Toolkit | conda-forge `cuda-nvcc=12.8.93`（免管理员） |
| C++ 编译器 | Linux gcc | VS 2022 Build Tools（MSVC 19.44） |
| 编译命令 | `sh make.sh` | `scripts\with_msvc.bat scripts\build_dcnv3.bat` |
| DCNv3 目录 | `Segmentation\Models\InternImage\ops_dcnv3`（**本仓库中已不存在**） | **`Backbone\InternImage\ops_dcnv3`** |
| 数据路径 | 硬编码 `J:/Dataset/...` | 用 `--data-path` 指定 |

## 快速开始（本机）

```bat
conda activate SDCombo

REM 环境自检（Python/torch/GPU/DCNv3 数值一致性）
python tools\verify_env.py

REM 全链路冒烟（合成数据，不需要真实数据集）
python tools\smoke_test.py --batch-size 4 --amp

REM 在现有数据集 VKITTI 2 上训练论文主线模型（Joint/model_DL4sDL.py，约 20.0 M 参数）
python train_VKITTI.py --data-path datasets\VKITTI_II ^
  --batch-size 16 --epochs 10 --lr 5e-5 --max-grad-norm 1.0 ^
  --class-weight median --num-workers 4

REM 评估
python evaluation_VKITTI.py --weight work_dir\model\model_<mark>_9.pth
```

> 论文 §6 报告的成绩来自 **Stanford2D3D + HHA**（`train_HHA.py` / `train_depth.py` / `evaluation_HHA.py`），
> 本机目前只有 VKITTI 2，因此 `train_VKITTI.py` 的数值**不能**直接与论文表格比较；
> Stanford2D3D 的脚本保持原样，数据到位后加 `--data-path` 即可使用。

---

## 原始安装说明（历史，已不适用于本机）

Evormentation install
```
conda create -n SDCombo python=3.10 # dcnv3 can not be install if python newer than 3.10
conda activate SDCombo
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

cd Segmentation\Models\InternImage\ops_dcnv3
sh make.sh
py test.py
```

Tensorboard Login
```shell
tensorboard --logdir=work_dir/board --port 11451
```
