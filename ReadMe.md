Evormentation install
```
conda create -n SDCombo python=3.10.0 # dcnv3 can not be install if python newer than 3.10
conda activate SDCombo
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

cd Segmentation\Models\InternImage\ops_dcnv3
sh make.sh
py test.py
```

---

## ⚠️ Windows + RTX 50 系（Blackwell / sm_120）配置请阅读本文档

上面的原始步骤是 **Linux + CUDA 11.8** 的写法，在 Windows + RTX 5090 等 Blackwell
显卡上**无法使用**（CUDA 11.8 不含 sm_120 的 kernel）。本机上已验证可用的完整配置见：

**➡️ [ReadMe_环境配置.md](ReadMe_环境配置.md)**

关键差异速览：

| 项目 | 原 ReadMe | 本机可用配置 |
| --- | --- | --- |
| PyTorch | `pytorch-cuda=11.8`（conda） | `torch==2.9.1+cu128`（pip，cu128 及以上） |
| CUDA 编译器 | 系统 CUDA Toolkit | conda-forge `cuda-nvcc=12.8.93`（免管理员） |
| C++ 编译器 | Linux gcc | VS 2022 Build Tools（MSVC 19.44） |
| 编译命令 | `sh make.sh` | `python setup.py build_ext --inplace` |
| 数据路径 | `J:/Dataset/...` | 用 `--data-path` 指定本机路径 |

配置完成后的标准用法：

```bat
conda activate SDCombo
cd /d E:\SDCombo

REM 环境自检
python tools\verify_env.py

REM 训练
python train.py --data-path E:\SDCombo\datasets\VKITTI_II --batch-size 4 --epochs 10

REM 推理/评估
python evaluation.py --data-path E:\SDCombo\datasets\VKITTI_II ^
                     --pretrained work_dir\model\model_<mark>_<epoch>.pth
```
