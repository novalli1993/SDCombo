# SDCombo 训练与推理环境配置文档

> 本文档记录了在 **本机实际环境**（Windows + RTX 5090 D v2）上，从零配置 SDCombo
> 训练/推理环境的完整过程、已验证的版本矩阵、数据集准备方式，以及运行命令。
>
> 配置完成日期：2026-10-01　|　配置主机：`KEPHAN`
> 所有结论均经实机验证，验证脚本见 `tools/`。

---

## 目录

1. [仓库结构与数据流](#1-仓库结构与数据流)
2. [环境总览与版本矩阵](#2-环境总览与版本矩阵)
3. [从零配置步骤](#3-从零配置步骤)
4. [编译 DCNv3 CUDA 扩展](#4-编译-dcnv3-cuda-扩展)
5. [数据集准备](#5-数据集准备)
6. [运行训练](#6-运行训练)
7. [运行推理/评估](#7-运行推理评估)
8. [环境自检与冒烟测试](#8-环境自检与冒烟测试)
9. [显存占用与 batch size 建议](#9-显存占用与-batch-size-建议)
10. [为适配 PyTorch 2.9 所做的代码改动](#10-为适配-pytorch-29-所做的代码改动)
11. [已知问题与注意事项](#11-已知问题与注意事项)
12. [故障排查速查表](#12-故障排查速查表)

---

## 1. 仓库结构与数据流

### 1.1 目录结构

```
SDCombo/
├── train.py                    # 训练入口
├── evaluation.py               # 推理/评估入口
├── ReadMe.md                   # 原始说明（Linux 路线，本机不适用）
├── ReadMe_环境配置.md           # 本文档
├── .gitignore
├── Dataset/                    # 数据加载与数据增强
│   ├── dataset_VKITTI.py       #   VKITTI 数据集（train.py / evaluation.py 实际使用）
│   ├── dataset_Stanford2D3D.py #   Stanford2D3D 数据集（备用，train.py 中被注释）
│   ├── dataset_Stanford2D3D_original.py
│   ├── transforms.py           #   RandomResize / RandomCrop / ToTensor / Normalize
│   └── semantic_labels.json
├── Joint/                      # 联合语义分割 + 深度分支
│   ├── model.py                #   SDCombo 主体 = InternImage + UPerNet + SDCHead
│   └── SDCHead.py              #   Semantic-Depth Combination Head
├── Segmentation/Models/        # 骨干与分割头
│   ├── UPerHead.py             #   UPerNet（PPM + FPN）
│   └── InternImage/
│       ├── intern_image.py     #   InternImage 主干
│       ├── drop.py             #   DropPath
│       └── ops_dcnv3/          #   DCNv3 可变形卷积（含 CUDA 内核）
│           ├── src/            #     CUDA/C++ 源码
│           ├── functions/      #     autograd Function + 纯 PyTorch 参考实现
│           ├── modules/        #     DCNv3 nn.Module
│           ├── setup.py        #     编译脚本
│           └── DCNv3.cp310-win_amd64.pyd   # 编译产物（本地生成）
├── Utils/
│   ├── train_val.py            # train_one_epoch / evaluate / criterion / 学习率调度
│   ├── distributed_utils.py    # ConfusionMatrix / MetricLogger / SmoothedValue
│   ├── DataPreparation/        # 数据集制作脚本（见 §5.3）
│   ├── Depth/                  # 深度图辅助工具（依赖 opencv，非主流程）
│   └── Joint/                  # 早期调试脚本（依赖已不存在的 Backbone 包，勿用）
├── tools/                      # ★ 本次新增：验证脚本
│   ├── verify_env.py           #   环境自检（GPU / 依赖 / DCNv3 数值一致性）
│   ├── verify_dcnv3.py         #   单独验证 DCNv3 CUDA 扩展
│   └── smoke_test.py           #   合成数据全链路冒烟测试 + 显存标定
├── scripts/                    # ★ 本次新增：环境脚本
│   ├── activate_env.bat        #   激活 conda 环境（cmd）
│   ├── activate_env.sh         #   激活 conda 环境（Git Bash）
│   └── with_msvc.bat           #   在 MSVC 环境下运行命令（编译 CUDA 扩展用）
├── datasets/VKITTI_II/         # ★ 本次新增：数据目录骨架（需自行放入数据）
└── work_dir/                   # 训练输出（logger / evaluation / model）
```

### 1.2 模型数据流

```
image (B,3,H,W)  ──► InternImage(DCNv3 主干, 输出 4 级特征)
                          │  (64, 128, 256, 512) @ stride 4/8/16/32
                          ▼
                     UPerNet(PPM + FPN → fpn_dim=512)
                          │  → seg_0 (B,15,H/4,W/4)，经 log_softmax
                          ▼
depth (B,H,W)    ──►  SDCHead  ──► 输出 (B,15,H,W)
```

`SDCHead` 把深度图与分割 logits 拼接后经 4 层卷积融合，最后在 `SDCombo.forward`
中双线性插值回原始分辨率。**注意 `SDCHead.forward` 内部先经 `out_conv` 把通道压回
`num_classes`，再与深度图拼接。**

### 1.3 关键超参数（`train.py` 中的默认值）

| 项目 | 值 | 位置 |
| --- | --- | --- |
| 类别数 | 15 | `--num-classes` |
| `base_size` / `crop_size` | 375 / 256 | `get_transform()` |
| 归一化均值 | (33.6045, 33.9644, 27.2941) | `get_transform()` |
| 归一化标准差 | (19.3824, 19.3147, 20.1879) | `get_transform()` |
| batch size | 4 | `-b` |
| epochs | 10 | `--epochs` |
| 初始学习率 | 1e-2 | `--lr` |
| 优化器 | AdamW(betas=(0.9,0.999), wd=0.05) | `main()` |
| 学习率调度 | CosineAnnealingLR(half_life=10, eta_min=1e-5) | `--cos` |
| 损失 | CrossEntropyLoss(`ignore_index=0`) | `Utils/train_val.py` |
| 混合精度 | 默认开启 | `--amp` |
| 可训练模块 | internimage, upernet, SDHead | `--module_trained` |

> **重要**：`criterion` 使用 `ignore_index=0`，即**类别 0 不参与损失**。语义标签应为
> `0..14`，其中 0 为忽略类（通常是 `<UNK>`/背景）。
>
> **重要**：`base_size=375` 时 `RandomResize` 的范围是 `int(0.75*375)=281` 到
> `int(2.0*375)=750`（最小边长缩放）。而 `RandomCrop(256)` 会先 `pad_if_smaller`
> 到至少 256，再随机裁剪 256×256。因此**训练时的实际输入固定为 256×256**（batch 内
> 尺寸一致，`collate_fn` 几乎不需要 padding）。推理时更是固定为 256×256。

---

## 2. 环境总览与版本矩阵

### 2.1 硬件

| 项目 | 值 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5090 D v2（24 GB，**算力 sm_120 / Blackwell**） |
| 驱动 | 616.92（CUDA UMD 13.4） |
| CPU 逻辑核 | 见 `os.cpu_count()`，`DataLoader` 的 `num_workers` 取 `min(cpu, bs, 8)` |

> ⚠️ **本机 GPU 是 Blackwell（sm_120）**，这是整个配置过程中最关键的一条约束。
> 原始 `ReadMe.md` 给出的 `pytorch-cuda=11.8` **无法驱动 sm_120**，
> 必须使用 CUDA 12.8 及以上的 PyTorch 构建。

### 2.2 软件版本矩阵（已实机验证）

| 组件 | 版本 | 说明 |
| --- | --- | --- |
| OS | Windows 11 (10.0.26300) | ANSI 代码页 **936 / GBK** |
| conda | 26.3.2（miniforge3） | 路径 `D:\Programs\miniforge3` |
| conda 环境名 | **`SDCombo`** | 路径 `D:\Programs\miniforge3\envs\SDCombo` |
| Python | 3.10.21 | 与 ReadMe 要求的 3.10 一致 |
| PyTorch | **2.9.1+cu128** | 预编译已含 `sm_120` |
| torchvision | 0.24.1+cu128 | |
| CUDA Toolkit(nvcc) | **12.8.93**（conda-forge `cuda-nvcc`） | 装在 `envs\SDCombo\Library` |
| cudart | 12.8.90（conda-forge） | |
| MSVC | **19.44.35207**（VS 2022 Build Tools 17.14） | |
| Windows SDK | 10.0.26100.0 | |
| ninja | 1.13.2（pip） | 编译扩展必需 |
| numpy | 2.2.6 | |
| Pillow | 12.3.0 | |
| DCNv3 | 1.0（本地编译） | `DCNv3.cp310-win_amd64.pyd`，1.09 MB |

`torch.cuda.get_arch_list()` = `['sm_70','sm_75','sm_80','sm_86','sm_90','sm_100','sm_120']`
—— **原生支持 sm_120，无需 PTX JIT**。

### 2.3 第三方依赖

主流程（`train.py` / `evaluation.py`）实际只需要：

```
torch  torchvision  numpy  Pillow
```

**不需要** mmcv / mmseg / mmdet / timm。`Utils/DataPreparation/utils.py`
中的 `OpenEXR` / `Imath` / `scipy.ndimage.imread` 属于早期数据制作脚本，不在训练/推理链路上。
`Utils/DataProcessing`、`Utils/Depth`、`Utils/Joint` 下的脚本依赖 `cv2` 或不存在的
`Backbone` 包，属于历史遗留，**请勿在环境检查时把它们当作必需依赖**。

---

## 3. 从零配置步骤

> 本机已完成全部步骤。以下命令供**重建环境**或**在其它机器复现**时使用。

### 3.1 创建 conda 环境

```bat
conda create -n SDCombo python=3.10 -y
conda activate SDCombo
```

> Python 版本建议严格用 **3.10**：`dcnv3` 的 CUDA 扩展在 3.11+ 上编译/导入更容易踩坑。

### 3.2 安装 PyTorch（必须 cu128 或更高）

```bat
python -m pip install torch==2.9.1 torchvision==0.24.1 ^
    --index-url https://download.pytorch.org/whl/cu128
```

> ❌ **不要**使用 `conda install pytorch pytorch-cuda=11.8`（原 ReadMe 的写法）。
> CUDA 11.8 不包含 sm_120 的 kernel，运行时会报
> `no kernel image is available for execution on the device`。

验证：

```bat
python -c "import torch;print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
# 期望：2.9.1+cu128  NVIDIA GeForce RTX 5090 D v2  (12, 0)
```

### 3.3 安装编译工具链（编译 DCNv3 必需）

需要两样东西：**CUDA 编译器 nvcc** 和 **MSVC C++ 编译器**。

**(a) nvcc —— 用 conda-forge 的 `cuda-nvcc`**（免管理员权限，装在环境内）：

```bat
conda install -n SDCombo -c conda-forge cuda-nvcc=12.8.93 -y
```

安装后 nvcc 位于 `%CONDA_PREFIX%\Library\bin\nvcc.exe`。
（同时会带入 `cuda-cudart-dev`、`cuda-crt-dev` 等，其中 `cudart.lib` 在
`%CONDA_PREFIX%\Library\lib`。）

**(b) MSVC —— 安装 VS 2022 生成工具**（需要管理员权限，约 3.4 GB）：

```bat
winget install --id Microsoft.VisualStudio.2022.BuildTools ^
    --accept-package-agreements --accept-source-agreements ^
    --override "--quiet --wait --norestart --nocache ^
                --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
```

安装后 `cl.exe` 位于
`C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\<ver>\bin\Hostx64\x64\cl.exe`。

**(c) ninja**：

```bat
python -m pip install ninja
```

### 3.4 应用本机必需的两处环境补丁

这两步是**本机（中文 Windows + conda nvcc）特有的坑**，已写入环境，
重建环境时必须重做：

**补丁 1 — `sitecustomize.py`（解决 3 个问题）**

文件：`D:\Programs\miniforge3\envs\SDCombo\Lib\site-packages\sitecustomize.py`

| # | 解决的问题 |
| --- | --- |
| 1 | 把 `Segmentation/Models/InternImage/ops_dcnv3` 加入 `sys.path`，使 `import DCNv3` 可用（源码是绝对导入） |
| 2 | 注册 `<env>\Library\bin` 为 DLL 目录，使 DCNv3 能找到 `cudart64_12.dll`；并补设 `CUDA_HOME`/`CUDA_PATH` |
| 3 | 把 `torch.utils.cpp_extension.SUBPROCESS_DECODE_ARGS` 从 `('cp1252',)` 改为 `('utf-8','replace')` |

> 第 3 点的原因：本机 ANSI 代码页是 **cp936**，`cl.exe` 输出中文，而 torch 在 Windows
> 上硬编码用 cp1252 解码子进程输出，会抛 `UnicodeDecodeError`，导致 cl 被误判为
> 不兼容（`Error checking compiler version for cl: 'cp1' codec can't decode ...`），
> 从而使 CUDA 扩展编译失败。注意 **`VSLANG=1033` 无法让本机的 cl 输出英文**，实测无效。

**补丁 2 — 编译期环境变量：见 `scripts/with_msvc.bat`**

该脚本做四件事：

1. `call vcvars64.bat` 进入 MSVC x64 环境
   （前先清掉 conda-forge `vc` 包注入的 `VS_VERSION=16.0`/`VS_YEAR=2019`/`INCLUDE`/`LIB`
   等变量，否则会指向不存在的 VS2019）；
2. 把 conda 环境的 `Library\lib` 追加到 `LIB`，让链接器找到 `cudart.lib`；
3. 设置 `CUDA_HOME`/`CUDA_PATH`、`TORCH_CUDA_ARCH_LIST=12.0`（目标 sm_120）；
4. 设置 `NVCC_PREPEND_FLAGS=-allow-unsupported-compiler`
   —— 见下一节的说明。

**为什么会需要第 4 点**：CUDA 12.8 官方只声明支持 MSVC 19.3x，而本机装的是
**MSVC 19.44**，`crt/host_config.h` 里 `_MSC_VER >= 1950` 才报错（1950 = MSVC 19.50），
19.44 < 19.50，因此**本机其实并未触发版本拒绝**，编译直接通过。
该标志作为保险保留；若你的机器装了更新的 MSVC（≥ 19.50）则需要它。

---

## 4. 编译 DCNv3 CUDA 扩展

### 4.1 编译命令

```bat
scripts\with_msvc.bat cmd /c "cd /d Segmentation\Models\InternImage\ops_dcnv3 && python setup.py build_ext --inplace"
```

或先激活环境再执行：

```bat
scripts\activate_env.bat
cd Segmentation\Models\InternImage\ops_dcnv3
python setup.py build_ext --inplace
```

成功后在 `ops_dcnv3\` 下生成 `DCNv3.cp310-win_amd64.pyd`。

> 原 ReadMe 写的是 `sh make.sh`（内部执行 `python setup.py build install`）。
> **Windows 上建议用 `build_ext --inplace`**：`install` 会把 `.pyd` 复制到
> `sysconfig.get_paths()["purelib"]`，本机实测该步骤会复制到一个无法被 import 的位置，
> 反而导致 `ModuleNotFoundError: No module named 'DCNv3'`。
> `--inplace` 让产物留在源码树，再配合上面补丁 1 的 `sys.path` 注入即可直接使用。

### 4.2 验证

```bat
python tools\verify_dcnv3.py
```

实测输出：

```
import DCNv3  -> OK: ...\ops_dcnv3\DCNv3.cp310-win_amd64.pyd
forward  max_abs_err = 1.746e-09  allclose = True
backward d(input)  max_abs_err = 3.427e-07  allclose = True
backward d(offset) max_abs_err = 1.141e-08  allclose = True
backward d(mask)   max_abs_err = 8.941e-08  allclose = True
DCNv3 module on NVIDIA GeForce RTX 5090 D v2: out=(2, 32, 32, 64) grad_ok=True
DCNv3 VERIFY OK
```

即 CUDA 内核与仓库内纯 PyTorch 参考实现（`dcnv3_core_pytorch`）在
forward 与三路 backward 上数值一致（误差 ~1e-7，属浮点累加差异）。

性能：同一测试用例下 CUDA 算子 **0.03 ms/次**，纯 PyTorch 参考实现 0.65 ms/次，
**约 22 倍加速**——这也是必须编译 CUDA 扩展而不能直接用 `DCNv3_pytorch` 的原因。

---

## 5. 数据集准备

### 5.1 需要的目录结构

`train.py` / `evaluation.py` 使用 `Dataset/dataset_VKITTI.py`，它要求
`--data-path` 指向如下结构：

```
<data-path>/
├── images/
│   ├── training/      <- rgb_*.png / *.jpg（RGB 图像）
│   └── validation/
├── annotations/
│   ├── training/      <- 同名文件，单通道灰度语义标签（取值 0..14，0 为忽略类）
│   └── validation/
└── depth/
    ├── training/      <- 同名文件，单通道灰度深度图
    └── validation/
```

> **三个子目录的文件名必须一一对应。** `dataset_VKITTI.py` 用
> `os.walk` 分别收集三个列表，只校验**数量相等**，不校验文件名是否配对。
> 如果三边排序不一致，样本会静默错配，训练结果无意义。
> 建议构建数据后用 `Utils/DataPreparation/SameName.py` 之类脚本核对文件名集合。

本仓库已预建好该骨架（空目录）：`datasets/VKITTI_II/`。

### 5.2 数据来源

数据集为 **Virtual KITTI 2（VKITTI 2）**，官方页面：
<https://europe.naverlabs.com/proxy-virtual-worlds-vkitti-2/>

本机 `J:` 盘不存在，`--data-path` 的默认值 `J:/Dataset/VKITTI_II` 是原作者的机器路径，
**必须显式传入**本机实际路径。

社区镜像可参考（非官方，请自行确认完整性与许可）：
<https://huggingface.co/datasets/ZhengGuangze/VKITTI2>

> ⚠️ **本次配置未包含数据集下载与制作**：VKITTI 2 体积较大，且官网需要填写表单获取
> 下载链接。环境侧已全部就绪，补齐数据后即可直接训练。

### 5.3 从原始 VKITTI 2 制作训练数据

仓库提供了制作脚本，其目标结构就是 §5.1（脚本里的 `T:/Dataset/VKITTI_II` 是原作者的盘符）：

| 脚本 | 作用 |
| --- | --- |
| `Utils/DataPreparation/Allocation.py` | 从原始 `VKITTI` 挑出 `training`/`validation` 两个 split（S01/S06/S18/S20 → training，其余 → validation），并复制 `depth` |
| `Utils/DataPreparation/3to1_S01.py` 等（S01/S02/S06/S18/S20） | 把 RGB 彩色语义图按调色板映射成单通道灰度标签 |
| `Utils/DataPreparation/ClassAndPalette.py` | 类别与调色板定义 |
| `Utils/DataPreparation/FileRename.py` / `SameName.py` | 文件名规整 / 一致性检查 |

**官方调色板（`3to1_S0x.py` 中的 `classes` 字典）**：

| 类别号 | RGB | 类别号 | RGB |
| --- | --- | --- | --- |
| 0 | (210, 0, 200) | 8 | (200, 200, 0) |
| 1 | (90, 200, 255) | 9 | (255, 130, 0) |
| 2 | (0, 199, 0) | 10 | (80, 80, 80) |
| 3 | (90, 240, 0) | 11 | (160, 60, 60) |
| 4 | (140, 140, 140) | 12 | (255, 127, 80) |
| 5 | (100, 60, 100) | 13 | (0, 139, 139) |
| 6 | (250, 100, 255) | 14 | (0, 0, 0) |
| 7 | (255, 255, 0) | | |

> 制作脚本用的是 Python 逐像素双重循环（`for h ... for w ...`），**非常慢**；
> 若数据量大建议先向量化改写，或直接用官方提供的
> `vkitti_2.0.3_classSegmentation` 版本。

---

## 6. 运行训练

```bat
conda activate SDCombo
cd /d E:\SDCombo

python train.py ^
  --data-path E:\SDCombo\datasets\VKITTI_II ^
  --batch-size 4 ^
  --epochs 10 ^
  --num-classes 15 ^
  --device cuda
```

### 参数字段说明

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--data-path` | `J:/Dataset/VKITTI_II` | **必须改**为本机数据根目录 |
| `-b/--batch-size` | 4 | 见 §9 显存建议 |
| `--epochs` | 10 | |
| `--lr` | 1e-2 | |
| `--cos` | `10 1e-5 -1` | 三个数：half-life / 最小 lr / last_epoch |
| `--wd` | 0.05 | weight decay |
| `--print-freq` | 50 | 日志打印间隔（iters） |
| `--amp` | True | `--amp False` 关闭混合精度 |
| `--module_trained` | `internimage,upernet,SDHead` | 逗号分隔；只训练指定顶层模块 |
| `--pretrained` | None | 预训练权重 `.pth`（见下） |
| `--start-epoch` | 0 | |
| `--fold-num` / `--aux` / `--resume` | — | 当前代码未实际使用 |

### 关于预训练权重

- `--pretrained` 期望的 `.pth` 内层结构是 `{'model': state_dict}`。
- 官方 InternImage 分类权重（如
  [internimage_s_1k_224.pth](https://huggingface.co/OpenGVLab/InternImage/resolve/main/internimage_s_1k_224.pth)）
  用的是 `channels=80, depths=[4,4,21,4], groups=[5,10,20,40]`，
  而本项目 `Joint/model.py` 里是
  **`channels=64, depths=[4,4,18,4], groups=[4,8,16,32]`**，
  两者结构不同，**不能直接加载**。
- 因此**本机采用从头训练**（不传 `--pretrained`）。此时
  `missing_keys` 为空、`module_trained` 覆盖全部三个模块，所有参数 `requires_grad=True`。

### 输出位置

```
work_dir/
├── logger/loggers<mark>.txt        # 每个 epoch 的进度/耗时/显存日志
├── evaluation/evaluation<mark>.txt # 每个 epoch 的 loss / lr / 混淆矩阵指标
└── model/model_<mark>_<epoch>.pth  # checkpoint（含 model/optimizer/lr_scheduler/epoch/args）
```

`<mark>` 为启动时刻 `%Y%m%d_%H%M%S`。三个子目录由 `train.py` 启动时自动创建。

---

## 7. 运行推理/评估

```bat
conda activate SDCombo
cd /d E:\SDCombo

python evaluation.py ^
  --data-path E:\SDCombo\datasets\VKITTI_II ^
  --pretrained work_dir\model\model_20261001_164738_0.pth ^
  --num-classes 15 ^
  --device cuda
```

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--data-path` | `J:/Dataset/VKITTIS` | **必须改**；指向含 `validation/` 的数据根 |
| `--pretrained` | `work_dir/model/model_20230727_230404_9.pth` | 权重路径（默认值不存在，需显式指定） |
| `--num-classes` | 15 | |
| `--device` | cuda | |
| `-b/--batch-size` | 8 | **注意：该参数在 `evaluation.py` 中未被使用**，验证集 `DataLoader` 硬编码 `batch_size=1` |

评估指标由 `Utils/distributed_utils.py` 的 `ConfusionMatrix` 输出：
`global correct` / 每类 `average row correct` / 每类 `IoU` / `mean IoU`。

---

## 8. 环境自检与冒烟测试

### 8.1 环境自检（推荐每次重建环境后先跑）

```bat
conda activate SDCombo
python tools\verify_env.py
```

检查 Python/依赖版本 → GPU 与 sm_120 → `import DCNv3` → DCNv3 forward/backward
数值一致性 → DCNv3 与参考实现的性能对比。全部通过时以
`[ OK ] 环境配置完成，可以进行训练与推理。` 结束。

### 8.2 全链路冒烟测试（不需要真实数据集）

`tools/smoke_test.py` 会自动生成符合 §5.1 结构的**合成数据**，然后跑通
数据集 → 模型 → 损失 → 反向 → 优化器 → 评估的完整链路，并报告峰值显存：

```bat
python tools\smoke_test.py --batch-size 4
```

实测输出（batch=4）：

```
train=4 样本, val=1 样本
参数量: 61.96 M
image=(4, 3, 256, 256) annotation=(4, 256, 256) depth=(4, 256, 256)
output=(4, 15, 256, 256)
train_loss=2.8379  lr=1.00e-02
训练峰值显存: allocated=7.45 GiB, reserved=10.65 GiB
SMOKE_SUMMARY batch=4 crop=256 amp=False params_M=61.96 peak_allocated_GiB=7.45 ...
冒烟测试通过：数据集 -> 模型 -> 损失 -> 反向 -> 评估 全链路可用。
```

`tools/smoke_test.py` 支持 `--crop-size` 用于探测更大输入下的显存上限。

---

## 9. 显存占用与 batch size 建议

本机实测（RTX 5090 D v2，24 GB，FP32，`amp=False`）：

| batch | crop | 峰值 allocated | 峰值 reserved | 结论 |
| --- | --- | --- | --- | --- |
| 1 | 256 | 1.88 GiB | 2.40 GiB | 富余 |
| 4 | 256 | **7.45 GiB** | **10.65 GiB** | ✅ **推荐（原代码默认）** |
| 8 | 256 | 7.45 GiB | 10.65 GiB | 与 batch=4 相同，见下方说明 |
| 2 | 512 | 9.84 GiB | 11.92 GiB | 富余 |
| 4 | 512 | 18.46 GiB | 23.26 GiB | ⚠️ 接近上限 |
| 2 | 768 | 21.27 GiB | 26.18 GiB | ❌ 超出显存（reserved > 24 GB） |

> **为什么 batch=8 与 batch=4 的峰值显存完全相同？**
> 因为默认 `base_size=375` 使 `RandomResize(281, 750)` 后每个样本恰好被缩放到
> **256×256**，`RandomCrop(256)` 直接整图返回。DCNv3 的 CUDA 内核按
> `(N, C, H, W)` 展平，中间张量的显存主要由 `C·H·W`（即 crop 面积）主导，
> 因此在本设置下**显存几乎与 batch size 无关，而与 crop 面积近似平方相关**。
> `collate_fn` 也只在 batch 内尺寸不一致时才 padding，此处不会触发。

**建议**：

- 默认用 `--batch-size 4`（与原代码一致，显存占用约 10.7 GiB，留出充分余量）。
- 想提速可直接加到 `--batch-size 8`，显存在本设置下不会增加；
  若之后再调大 `crop_size`，请按上表的平方关系重新估算。
- 24 GB 卡上 `crop_size` 建议不超过 **512**（batch=4 时已用 23.26 GiB reserved）。
- 显存紧张时用 `--amp`（默认已开启）可明显降低占用。

---

## 10. 为适配 PyTorch 2.9 所做的代码改动

原始代码基于 PyTorch 1.x 编写。为在本机（PyTorch 2.9.1 + MSVC 19.44 + conda nvcc）
跑通，做了以下**最小必要改动**。所有改动均以 `[SDCombo-patch]` 注释标注。

### 10.1 DCNv3 编译相关（必须）

文件：`Segmentation/Models/InternImage/ops_dcnv3/src/`

| 文件 | 原内容 | 改为 | 原因 |
| --- | --- | --- | --- |
| `cuda/dcnv3_cuda.cu` | `#include <ATen/cuda/CUDAContext.h>`<br>`#include <torch/torch.h>` | `#include <c10/cuda/CUDAException.h>`<br>`#include <c10/cuda/CUDAStream.h>` | 前者链式拉入 `cusparse.h`（conda 工具链无此头）；后者在 nvcc 下与 conda 自带 libcu++ 的 `std` 命名空间冲突（`error C2872: "std": 不明确的符号`） |
| `cuda/dcnv3_cuda.cu` | `input.type()`（2 处） | `input.scalar_type()` | PyTorch 2.x 的 `AT_DISPATCH_*` 需要 `c10::ScalarType`，而 `.type()` 返回 `at::DeprecatedTypeProperties` |
| `cuda/dcnv3_cuda.cu` | `torch::kHalf`（3 处） | `at::kHalf` | 不再依赖 `torch/torch.h` |
| `cuda/dcnv3_im2col_cuda.cuh` | `#include <ATen/cuda/CUDAContext.h>`<br>`#include <THC/THCAtomics.cuh>` | `#include <c10/cuda/CUDAException.h>`<br>`#include <c10/cuda/CUDAStream.h>`<br>`#include <cuda_runtime.h>`<br>`#include <cuda_fp16.h>` | 前者同上；`THC/THCAtomics.cuh` **在 PyTorch 2.x 已被删除**。该文件实际只用 CUDA 内建 `atomicAdd`（由 `cuda_runtime.h` 提供），故无需 THC |
| `cpu/dcnv3_cpu.cpp` | `#include <ATen/cuda/CUDAContext.h>` | （删除） | CPU 实现不需要 CUDA 上下文，该头会拉入 `cusparse.h` |

> **设计要点**：`.cu` 文件里**不包含任何 torch/C++ 标准库头**（保持原来的写法），
> 只有 `src/vision.cpp` 包含 `<torch/extension.h>`。
> 这样可绕开 conda nvcc 的 `nvcc.profile` 注入 `-I<...>/include/targets/x64`
> 导致 libcu++ 的 `std/` 头文件遮蔽 MSVC 标准库的问题。
> 如果将来需要往 `.cu` 里加 `torch/extension.h`，会重新触发
> `compiled_autograd.h(1134): error C2872: "std"`，请避免。

### 10.2 PyTorch 2.6+ `torch.load` 行为变更（必须）

PyTorch 2.6 起 `torch.load` 的 `weights_only` 默认为 `True`。本项目 checkpoint
除 `model` 外还存了 `optimizer` / `lr_scheduler` / **`args`（`argparse.Namespace`）**，
反序列化会失败：

```
_pickle.UnpicklingError: Weights only load failed. ...
WeightsUnpickler error: Unsupported global: GLOBAL argparse.Namespace
```

已在 `train.py` 与 `evaluation.py` 的 `create_model()` 中显式加 `weights_only=False`
（checkpoint 由本仓库自己产生，可信）。

### 10.3 混合精度 API 弃用（消除告警，行为不变）

| 文件 | 原 | 改为 |
| --- | --- | --- |
| `train.py` | `torch.cuda.amp.GradScaler()` | `torch.amp.GradScaler('cuda')` |
| `Utils/train_val.py` | `torch.cuda.amp.autocast(...)` | `torch.amp.autocast('cuda', ...)` |
| `.../functions/dcnv3_func.py` | `from torch.cuda.amp import custom_bwd, custom_fwd` | `from torch.amp import custom_bwd, custom_fwd`，并加 `device_type='cuda'` |
| `.../functions/dcnv3_func.py` | `torch.meshgrid(...)` | 补 `indexing='ij'`（锁定行为，消除告警） |

### 10.4 修复的既有 Bug

**(a) `argparse` 的 `type=list`（`train.py`）**

原代码 `--cos` 用 `type=list`、`--module_trained` 默认值是 `list`。
`argparse` 对 `type=list` 会把命令行字符串**拆成单字符列表**，
所以任何命令行传参都必然得到错误结果（`--cos 10 1e-5 -1` 直接报错；
`--module_trained upernet` 会变成 `['u','p','e','r','n','e','t']`，静默失效）。

已改为：

- `--cos`：`type=float, nargs=3`（默认值不变）
- `--module_trained`：逗号分隔字符串 + `lambda` 解析（默认值语义不变）

**(b) `work_dir` 子目录未创建（`train.py` / `evaluation.py`）**

原代码只创建 `work_dir`，但 `MetricLogger`、结果记录、checkpoint 保存分别直接写
`work_dir/logger`、`work_dir/evaluation`、`work_dir/model`，首次运行会
`FileNotFoundError`。已改为在启动时 `os.makedirs(..., exist_ok=True)` 建三个子目录。

### 10.5 未改动但需注意的地方

- `Utils/DataPreparation/parameters.py` 引用了不存在的
  `Segmentation.Models.model.IIP`；`Utils/Joint/TestTensor.py` 引用了不存在的
  `Backbone` 包。**均为历史遗留，不在训练/推理链路上，不要执行。**
- `evaluation.py` 的 `-b/--batch-size` 参数未被使用（`DataLoader` 硬编码为 1）。

---

## 11. 已知问题与注意事项

| # | 问题 | 状态 / 处理 |
| --- | --- | --- |
| 1 | 原 ReadMe 的 `pytorch-cuda=11.8` 不支持 sm_120 | 已改为 **PyTorch 2.9.1+cu128** |
| 2 | 原 ReadMe 的 `sh make.sh` 在 Windows 不可用 | 已改为 `setup.py build_ext --inplace`（见 §4.1） |
| 3 | 数据路径默认值是原作者的 `J:` / `T:` / `M:` 盘 | **必须**用 `--data-path` 指定本机路径 |
| 4 | VKITTI 2 数据集体积大且需申请下载 | **本次未下载**；骨架目录已建好，见 §5 |
| 5 | 官方 InternImage 预训练权重与本项目结构不匹配 | 采用从头训练，见 §6 |
| 6 | 数据集三目录仅校验数量、不校验文件名配对 | 制作数据后务必核对文件名集合 |
| 7 | 本机是中文 Windows（cp936） | 依赖环境内的 `sitecustomize.py` 补丁，见 §3.4 |
| 8 | `cuda-cusparse` / `cuda-cublas` 在 conda-forge 无 win-64 包 | 已通过移除对它们的头文件依赖解决，无需安装 |
| 9 | `tools/smoke_test.py` 的合成数据集目录 | 运行时会创建 `datasets/VKITTI_II_smoke/`，已在 `.gitignore` 中 |
| 10 | 终端显示中文可能乱码 | 本机控制台代码页为 936，属显示问题，不影响运行；可 `chcp 65001` |

---

## 12. 故障排查速查表

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| `no kernel image is available for execution on the device` | PyTorch 的 CUDA 版本不含 sm_120 | 重装 `torch==2.9.1+cu128` 或更高；不要用 CUDA 11.8 |
| `ModuleNotFoundError: No module named 'DCNv3'` | 扩展未编译，或 `.pyd` 不在可导入位置 | 编译（§4.1）；确认 `sitecustomize.py` 已注入 `ops_dcnv3` 路径 |
| `ImportError: DLL load failed ... cudart64_12.dll` | CUDA 运行时 DLL 目录未注册 | 确认 `sitecustomize.py` 中 `os.add_dll_directory(<env>\Library\bin)` 生效，或在 `scripts/with_msvc.bat` 下运行 |
| `Error checking compiler version for cl: 'cp1' codec can't decode ...` | cp936 本地化输出 vs torch 硬编码 cp1252 | 应用 `sitecustomize.py` 补丁 3（§3.4） |
| `fatal error C1083: 无法打开包括文件: "cusparse.h"` | 误引入了 `ATen/cuda/CUDAContext.h` | 改回 `c10/cuda/CUDAStream.h`（§10.1） |
| `error C2872: "std": 不明确的符号` | `.cu` 文件里包含了 `torch/extension.h` 等 C++ 标准库头 | torch 相关代码放 `.cpp`，`.cu` 只留纯 CUDA（§10.1） |
| `Error checking compiler version for cl` + 编译器被判为不兼容 | nvcc 的 MSVC 版本门限 | 设 `NVCC_PREPEND_FLAGS=-allow-unsupported-compiler`，或安装 MSVC ≤ 19.44 |
| `LINK : fatal error LNK1181: 无法打开输入文件"cudart.lib"` | 链接器找不到 CUDA 导入库 | 把 `<env>\Library\lib` 加入 `LIB`（`with_msvc.bat` 已做） |
| `_pickle.UnpicklingError: Unsupported global: GLOBAL argparse.Namespace` | PyTorch 2.6+ `weights_only=True` | `torch.load(..., weights_only=False)`（§10.2） |
| `FileNotFoundError: work_dir/logger/loggers*.txt` | 输出子目录不存在 | 使用本仓库已修复的 `train.py`/`evaluation.py` |
| `NVIDIA-SMI has failed` / 显存被占满 | 有别的进程占用 GPU | `nvidia-smi` 查看并释放；本机桌面/浏览器也占少量显存 |
| 训练 loss 不下降 / mIoU 恒为 0 | 标签与深度图文件名错配，或标签未映射成 0..14 | 核对 §5.1 的文件名一致性；确认标签为单通道灰度且最大值 ≤ 14 |

---

## 附：本机一键复现命令序列

```bat
REM ---------- 1. 创建环境 ----------
conda create -n SDCombo python=3.10 -y
conda activate SDCombo

REM ---------- 2. PyTorch (Blackwell 必须 cu128) ----------
python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install ninja

REM ---------- 3. CUDA 编译器 (免管理员) ----------
conda install -n SDCombo -c conda-forge cuda-nvcc=12.8.93 -y

REM ---------- 4. MSVC (需管理员, ~3.4GB) ----------
winget install --id Microsoft.VisualStudio.2022.BuildTools ^
  --accept-package-agreements --accept-source-agreements ^
  --override "--quiet --wait --norestart --nocache --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"

REM ---------- 5. 环境补丁: sitecustomize.py (见 §3.4 补丁1) ----------

REM ---------- 6. 编译 DCNv3 ----------
cd /d E:\SDCombo
scripts\with_msvc.bat cmd /c "cd /d Segmentation\Models\InternImage\ops_dcnv3 && python setup.py build_ext --inplace"

REM ---------- 7. 验证 ----------
python tools\verify_env.py
python tools\smoke_test.py --batch-size 4

REM ---------- 8. 训练 / 推理 ----------
python train.py --data-path E:\SDCombo\datasets\VKITTI_II --batch-size 4 --epochs 10
python evaluation.py --data-path E:\SDCombo\datasets\VKITTI_II ^
                     --pretrained work_dir\model\model_<mark>_<epoch>.pth
```
