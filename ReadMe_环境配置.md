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
5. [数据集准备](#5-数据集准备)（含 §5.4 原脚本问题清单）
6. [运行训练](#6-运行训练)
7. [运行推理/评估](#7-运行推理评估)
8. [环境自检与冒烟测试](#8-环境自检与冒烟测试)
9. [显存占用与 batch size 建议](#9-显存占用与-batch-size-建议)（含 §9b 2060 妥协项评估、§9c 数据准备替代方案）
10. [为适配 PyTorch 2.9 所做的代码改动](#10-为适配-pytorch-29-所做的代码改动)
11. [已知问题与注意事项](#11-已知问题与注意事项)
12. [故障排查速查表](#12-故障排查速查表)
13. [真实数据训练验证结果](#13-真实数据训练验证结果rtx-5090)

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

### 5.4 原数据准备脚本的问题清单

`Utils/DataPreparation/` 下的制作链路存在多处**会静默产生错误数据**的问题，
如果不用 `tools/prepare_vkitti.py` 而坚持走原脚本，请逐条核对：

| # | 位置 | 问题 | 后果 |
| --- | --- | --- | --- |
| 1 | `3to1_S01.py:48`、`3to1_S06.py:47`、`3to1_S20.py:47` | `if i.count("Sxx") != 1 or a <= N:` 的续跑计数 `a` 在 if/else **两个分支都自增**，且跨 `training`/`validation` 目录**不复位** | 每个 split 会**静默丢弃**前 N 张标签（S01 丢 5278、S06 丢 14020、S20 丢 26200）。S02/S18 无此逻辑，可见这是中断后临时加的续跑代码被遗留 |
| 2 | 全部 `3to1_S0x.py:29-32` | 逐像素 `classes[tuple(arr[h][w])]`，1242×375 = 46.6 万次字典查询/图 | 2 万帧约 **1.8 小时**（实测 316 ms/帧）；且遇到调色板外颜色直接 `KeyError` 中断 |
| 3 | `3to1_S0x.py:32` | `arr = arr[:, :, 1]` 取绿通道当作类别号 | 因为查找发生在取通道**之前**，结果恰好正确；但极易被误改（15 个类别里 0 与 14 的绿通道同为 0，靠绿通道无法区分） |
| 4 | `Allocation.py:45-69` | `rgb`/`cls`/`ins` 的 `shutil.copy2` **全被注释**，只有 `depth` 真的复制 | 以为在搬数据，实际只搬了深度图 |
| 5 | `Allocation.py:70-73` | `for i in a: s = sum(a[...])` 在循环里重复求和 | 只是低效，无正确性影响 |
| 6 | `norm_para.py:21-24` | 计数器 `a` 写在 `for d in range(3)` 通道循环**内部**，每张图加 3 次 | `mean /= a`、`std /= a` 中 `a = 3N`，**均值方差整体偏小 3 倍**。`train.py` 里那组 `(33.60, 33.96, 27.29)/(19.38, 19.31, 20.19)` 正是该脚本的产物；VKITTI 的 8-bit RGB 正确均值应在 **百量级**，所以现有归一化几乎只是把像素缩小了 3 倍，并没有真正零均值化 |
| 7 | `FileRename.py` / `FilesCount.py` | 硬编码 `T:`/`M:` 盘符；`FileRename.py` 不检查源路径是否存在 | 换机器必须改路径；路径不存在时静默产出空目录 |
| 8 | `SameName.py` | 只把 `depth` 的前缀改成 `rgb`，靠 `n[3:]` 字符串拼接 | 依赖"短名恰好 3 字符"这一巧合（`15l`/`fog`/`mor`/`ove`/`rai`/`sun`/`clo` 都正好 3 字符），可用但脆弱 |

**关于第 6 条的建议**：归一化常数应基于**训练集**重新计算。修正后的写法是
每张图累加一次（而不是每通道一次），且用`mean of means` 或直接统计全体像素：

```python
# 正确：累加每张图的通道均值，最后除以图片数 N
for img_path in train_images:
    arr = np.asarray(Image.open(img_path), dtype=np.float32)
    mean += arr.reshape(-1, 3).mean(axis=0)   # 每张图只加 1 次
    std  += arr.reshape(-1, 3).std(axis=0)
N = len(train_images)
mean, std = mean / N, std / N
```

现有常数偏小 3 倍**不会让训练跑不起来**（归一化是仿射变换，后续 LayerNorm/
BatchNorm 会吸收尺度差异），但会改变有效学习率与梯度条件数，重新计算通常
能带来更稳的收敛。是否重算属于实验选择，建议**先按原常数复现基线，再对照重算版本**。

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
train=20 样本, val=1 样本, batch_size=4
参数量: 61.96 M
image=(4, 3, 256, 256) annotation=(4, 256, 256) depth=(4, 256, 256)
output=(4, 15, 256, 256)
train_loss=2.7799  lr=1.00e-02
训练峰值显存: allocated=7.93 GiB, reserved=10.84 GiB
SMOKE_SUMMARY batch=4 crop=256 amp=False params_M=61.96 peak_allocated_GiB=7.93 ...
冒烟测试通过：数据集 -> 模型 -> 损失 -> 反向 -> 评估 全链路可用。
```

> 样本数会自动补足到 `max(--samples, batch_size)`，否则一个 batch 装不满、
> 测出的显存会偏小（详见 §9 的告警说明）。

`tools/smoke_test.py` 支持 `--crop-size` 用于探测更大输入下的显存上限。

---

## 9. 显存占用与 batch size 建议

本机实测（RTX 5090 D v2，24 GB，FP32，`amp=False`，`tools/smoke_test.py`，
样本数 ≥ batch_size 以保证 batch 装满）：

| batch | crop | 峰值 allocated | 峰值 reserved | 结论 |
| --- | --- | --- | --- | --- |
| 1 | 256 | 1.88 GiB | 2.40 GiB | 富余 |
| **4** | **256** | **7.93 GiB** | **10.84 GiB** | ✅ **稳妥推荐（原代码默认）** |
| 8 | 256 | 10.63 GiB | 16.59 GiB | ✅ 可用 |
| 16 | 256 | 18.28 GiB | 29.03 GiB | ⚠️ reserved 超显存，靠分配器复用才没 OOM |
| 2 | 384 | 5.84 GiB | 6.90 GiB | ✅ 很富余 |
| 4 | 384 | 16.95 GiB | 23.53 GiB | ⚠️ 接近上限 |
| 8 | 384 | 23.03 GiB | 36.11 GiB | ❌ 无余量 |
| 2 | 512 | 9.84 GiB | 11.92 GiB | ✅ **想放大 crop 时推荐** |
| 4 | 512 | 18.93 GiB | 23.26 GiB | ⚠️ 激进，需先关掉占显存的桌面程序 |
| 2 | 768 | 21.27 GiB | 26.18 GiB | ❌ 超出显存 |

> **显存近似正比于 `batch_size × crop_size²`。**
> 实测 `batch=4, crop=512` 是 `batch=4, crop=256` 的 2.4 倍（18.93 vs 7.93 GiB），
> 与面积比 4 倍同量级（非线性来自固定开销）。`batch=8, crop=384` 单看 allocated
> 只有 23.03 GiB，但 reserved 已达 36.11 GiB —— **判断能否跑起来要看 reserved，
> 而不是 allocated**（reserved 是 torch 向驱动申请的总量）。
>
> ⚠️ **测显存时样本数必须 ≥ batch_size**：早期版本的 `tools/smoke_test.py`
> 只生成 4 个样本，`--batch-size 8/16` 实际仍只跑 4 张，测出的显存与 batch=4
> 完全相同，并因此得出"显存与 batch 无关"的错误结论。现已修正并会打印告警。

**建议**：

- **稳妥起手**：`--batch-size 4`（crop 256 默认），约 10.8 GiB reserved，余量充足。
- **想提升精度**：把 crop 从 256 提到 **512**，配 `--batch-size 2`（9.84 GiB）。
  这是"看得更多"性价比最高的一档 —— 原图 1242×375，256 裁切只覆盖约 14% 视野。
- **想更激进**：显式打开梯度检查点（`Joint/model.py` 里把 `with_cp=False` 改成
  `True`，仅 `model.train()` 时生效），可大幅降低激活显存。
- `--amp` 默认已开启，能进一步降低占用。

---

## 9b. 原 RTX 2060 时代的妥协与现阶段可放宽项

原代码是按 6 GB 显存的 RTX 2060 调的，在 24 GB 卡上照搬会浪费算力。逐项评估：

| 项 | 原值（2060 妥协） | 现建议 | 依据 |
| --- | --- | --- | --- |
| `base_size` / `crop_size` | 375 / 256 | 512 / 512（配 `-b 2`） | 256 只覆盖原图约 14% 视野；InternImage 是 stride-32 结构，256 输入的最深层特征仅 8×8，PPM 的 `pool_scales=(1,2,3,6)` 在 8×8 上几乎退化 |
| `-b` batch size | 4 | 8（crop 256）/ 2（crop 512） | 见 §9 实测 |
| `epochs` | 10 | 建议 ≥ 30，并让 `--cos` 的 half-life 等于总轮数 | 10 轮对 62 M 参数模型偏少；`CosineAnnealingLR` 的 `T_max` 取的是 `args.cos[0]`，与 `--epochs` 无关联，轮数一变学习率就会过早触底或提前收官 |
| 评估用 `RandomCrop` | 随机裁 256 | 改为整图或 `CenterCrop` | 现评估每轮只随机看 256×256（约 14%），且每轮裁的位置都不同，指标不可复现。整图推理实测仅 **1.61 GiB / 0.70 s**，完全放得下 |
| `norm_para.py` 的 mean/std | (33.60, 33.96, 27.29) / (19.38, 19.31, 20.19) | **需重算** | 该脚本计数器 `a` 写在通道循环内，结果被除以 3；正确均值应在百量级。见 §5.4 |
| `with_cp` 梯度检查点 | False | 需要更大 crop 时改 True | 2060 上关掉是为省时间，现在若能用显存换精度则可打开 |
| `num_workers` | `min(cpu, bs, 8)` | 可固定 8~16 | 该表达式被 batch size 压制；与显存无关，JPEG 解码是 CPU 活，可自由调大 |

**不建议改**的：模型结构（`channels=64, depths=[4,4,18,4], groups=[4,8,16,32]`
就是官方 InternImage-S 配置，没有缩水）、数据增强管线、优化器与损失。

---

## 9c. 数据处理脚本的可用替代方案

原 `Utils/DataPreparation/` 的制作流程需手工依次执行 5 类脚本，且存在多处会
静默出错的问题（详见 §5.3 / §5.4）。已提供**一条命令**的替代实现：

```bat
python tools\prepare_vkitti.py ^
  --rgb      D:\dl\vkitti_2.0.3_rgb.tar ^
  --depth    D:\dl\vkitti_2.0.3_depth.tar ^
  --classseg D:\dl\vkitti_2.0.3_classSegmentation.tar ^
  --out      datasets\VKITTI_II ^
  --report   datasets\vkitti_report.json
```

它直接从 3 个 tar 流式读取到最终目录结构，并做完整校验：

- **向量化标签转换**：24-bit RGB → 类别号的 16 MiB LUT。实测
  **3.23 ms/帧 vs 原逐像素实现 316 ms/帧（98×）**，2 万帧从约 1.8 小时降到
  约 1.1 分钟，且输出逐元素一致。
- **三目录主文件名严格一致性校验**（`dataset_VKITTI.py` 本身只校验数量，
  错配会静默训练到错误标签/深度）。
- **未收录颜色检测**：调色板外的像素会被逐个列出，而不是 `KeyError` 崩溃
  或静默当背景。
- **深度保持 16 bit**，并在读到 8 bit 时告警（说明下错了 tar）。
- 输出类别分布、忽略类占比、深度值域报告。

回归测试：`python tools\_test_prepare.py`（25 项断言，含 LUT 往返、
与逐像素参考实现逐元素比对、合成 tar 端到端全流程）。

---

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

## 13. 真实数据训练验证结果（RTX 5090）

数据：VKITTI 2，`datasets/VKITTI_II`，42,520 帧（training 37,860 / validation 4,660），
三维目录主文件名严格一致，标签全在 15 色调色板内，深度为 16 bit。
训练环境：crop 256、batch 4、AMP 开、单卡 5090。

### 13.1 结论速览

| 项 | 结果 |
| --- | --- |
| 数据准备 | ✅ 42,520 帧全部就绪，对齐与取值均校验通过 |
| 训练链路 | ✅ 可正常运行、保存 checkpoint、完成评估 |
| **原始配置 (lr=1e-2) 能否收敛** | ❌ **发散**：epoch 1 中途 loss→nan，global correct 掉到 4.7%，模型坍缩到只预测 4 类 |
| 修正后 (lr=6e-5 + `ignore_index=255`) | ✅ 稳定收敛，3 epoch 内 mIoU 7.0 → 10.1 → **24.6**，学到 11 类 |
| 单 epoch 耗时 | ≈ 16 分钟（9,465 iters，0.10 s/iter） |
| 显存占用 | ≈ 3.2 GiB（crop 256 / batch 4） |

**训练链路本身没有问题；原仓库的超参数配置在本数据上会发散。**

### 13.2 发现 1：`lr=1e-2` + AdamW 导致发散（致命）

原 `train.py` 用 `AdamW(lr=1e-2)`。实测：

```
Epoch: [1]  [7500/9465]  loss: 0.9978 (1.1933)     <- 仍然正常
Epoch: [1]  [8000/9465]  loss: nan (nan)           <- 9,465 步内爆掉
[epoch: 1]  train_loss: nan   global correct: 4.7%
```

对照官方 InternImage 的 ADE20K UPerNet 配置
（[`upernet_internimage_s_512_160k_ade20k.py`](https://github.com/OpenGVLab/InternImage/blob/master/segmentation/configs/ade20k/upernet_internimage_s_512_160k_ade20k.py)）：

```python
optimizer = dict(type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.05,
                 constructor='CustomLayerDecayOptimizerConstructor', ...)
lr_config = dict(policy='poly', warmup='linear',
                 warmup_iters=1500, warmup_ratio=1e-6, power=1.0)
```

| 项 | 本仓库 | 官方 InternImage-S | 差异 |
| --- | --- | --- | --- |
| 优化器 / wd | AdamW / 0.05 | AdamW / 0.05 | 一致 |
| **学习率** | **1e-2** | **6e-5** | **本仓库高 166×** |
| warmup | 无 | linear 1500 iter | 缺失 |
| 调度 | CosineAnnealing | poly | 不同 |
| 梯度裁剪 | 无 | 无 | — |

即 warmup 与层级 lr 衰减都被去掉后，学习率却沿用了 SGD 量级。实测最大梯度范数
约 **59**，配合 1e-2 的 lr，9,465 步内必然溢出。

> 用 lr=1e-2 短跑 150 步并不会出现 nan（1.5 分钟内全部有限），
> 说明这是**随机触发的延迟发散**，不能靠短跑验证稳定性。

**修正**：`--lr 6e-5`（并建议补 warmup，见 §13.5）。

### 13.3 发现 2：`ignore_index=0` 丢弃了 17.7% 的监督信号

`Utils/train_val.py` 原为 `cross_entropy(..., ignore_index=0)`。但 VKITTI 2 的
**类别 0 是 Terrain（地面）**，是真实类别，不是忽略标签：

| 数据集 | 类别 0 (Terrain) 占比 |
| --- | --- |
| training | **17.66%**（3,114,775,050 像素） |
| validation | 3.84% |

后果：Terrain 永远学不会，IoU 恒为 0.0，且这部分像素完全不贡献梯度。
`dataset_VKITTI.py` 的 `collate_fn` 用 **255** 做 padding，且标签只取 0..14，
所以正确的忽略值就是 **255**。

**已改为 `IGNORE_INDEX = 255`**（`Utils/train_val.py`），并让
`ConfusionMatrix` 与 `evaluate()` 使用同一个忽略值。

**A/B 实测对比**（其余配置完全相同：lr=6e-5、3 epochs、crop 256、batch 4）：

| epoch | ignore_index=0（对照） | ignore_index=255（修正） |
| --- | --- | --- |
| 0 | mIoU 10.3 / acc 21.6% | mIoU 7.0 / acc 14.6% |
| 1 | mIoU 12.3 / acc 34.7% | mIoU 10.1 / acc 39.9% |
| 2 | （训练已结束） | **mIoU 24.6 / acc 66.3%** |
| 预测到的类别数（epoch 0） | 4 | 6（且 Terrain 在 epoch 1 起 IoU 7.8% → 22.4%） |
| Terrain IoU | 恒为 0.0 | 0.0 → **7.8 → 22.4** |

结论：`ignore_index=0` 在早期 epoch 因为少学一个难类而指标"看着更好"，
但**轨迹已经停滞**（12.3），而修正组的 mIoU 仍在快速上升（24.6），
并且真正学会了 Terrain。**修正版才是正确且更有潜力的配置。**

### 13.4 发现 3：类别极端不平衡导致预测坍缩

数据分布极不平衡（训练集）：

| 类别 | 占比 | 类别 | 占比 |
| --- | --- | --- | --- |
| Vegetation | 26.73% | Misc | 0.77% |
| GuardRail | 23.10% | TrafficLight | 0.81% |
| Terrain | 17.66% | Car | 0.84% |
| Tree | 15.21% | Truck | 0.88% |
| Van | 6.55% | Building | 1.33% |
| Road | 3.85% | TrafficSign | 1.36% |
| | | **Pole** | **0.18%** |

最稀有类（Pole 0.18%）与最多类（Vegetation 26.73%）相差 **145×**。
实测模型在早期会坍缩到 1~6 个类，例如某一 epoch 把 **87% 的像素预测成 Van**。

这是 Transformer 分割在长尾数据上的典型问题，需要**损失重加权**才能根治，
不是调 lr 能解决的。建议按中位数频率加权（`sklearn` 风格）或 focal loss：

```python
# 用训练集频率的倒数/中位数频率做权重
freq = torch.tensor([...15 个类的像素频率...])
w = freq.median() / freq.clamp(min=1e-6)
w = w / w.mean()
criterion = lambda x, y: F.cross_entropy(x, y, weight=w.cuda(), ignore_index=255)
```

### 13.5 下一步建议（按优先级）

1. **延长训练**：3 epoch 远未收敛，mIoU 仍在快速上升。官方配方是 160k iter，
   本机 0.10 s/iter ⇒ 单 epoch 16 分钟，**100 epoch ≈ 27 小时**。
   建议至少跑到 30~50 epoch 看曲线是否进入平台。
2. **补 warmup 与层级 lr 衰减**：官方用 `warmup_iters=1500, warmup_ratio=1e-6`
   和 `layer_decay_rate`。当前 `train.py` 没有 warmup，低 lr 下早期收敛偏慢。
3. **类别重加权 / focal loss**：解决 §13.4 的坍缩，这是提升 mIoU 的关键。
4. **重新计算归一化常数**：见 §5.4 第 6 条，现有 mean≈33 比真实值小 3 倍。
5. **评估改整图**（可选）：现评估每轮随机裁 256（约 14% 视野），指标不可复现。
   整图推理实测仅 1.61 GiB / 0.70 s。

### 13.6 相关脚本

| 脚本 | 用途 |
| --- | --- |
| `tools/ingest_vkitti.py` | 校验 MD5 并把 tar 搬到指定目录 |
| `tools/prepare_vkitti.py` | 从 tar 一站式生成数据集（含一致性校验） |
| `tools/check_dataset.py` | 抽查三目录对齐、标签取值、深度位深 |
| `tools/diag_metrics.py` | 解释 `mean IoU: nan` 的来源 |
| `tools/diag_collapse.py` | 诊断预测坍缩：预测分布 vs 真实分布 |
| `tools/exp_lr_stability.py` | 不同 lr / 裁剪下的稳定性受控实验 |
| `tools/_test_metrics.py` | `ConfusionMatrix` 修复的回归测试（14 项断言） |

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
