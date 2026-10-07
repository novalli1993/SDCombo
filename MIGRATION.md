# 环境与训练配置迁移说明

> **用途**：在新代码基础上重建工作区时，把本机已验证的**硬件适配、环境配置、训练参数**
> 一次性迁移过去。本文件自包含，不依赖任何特定代码结构 —— 换代码后仍可直接照做。
>
> **记录时间**：2026-10-07　**记录主机**：KEPHAN（Windows 11，10.0.26300）
> **来源**：原工作区 `E:\SDCombo`（git 提交 `51d076a`，工作区干净）

---

## 0. 迁移清单（先看这里）

新工作区建好后，按序完成：

| # | 事项 | 是否与代码有关 | 章节 |
| --- | --- | --- | --- |
| 1 | 复用 `SDCombo` conda 环境（**无需重建**） | 无关 | §2、§3 |
| 2 | 更新 `sitecustomize.py` 里的仓库根路径 | **有关**（路径改了） | §3.4 |
| 3 | 更新 `scripts/with_msvc.bat` 里的仓库路径 | 无关（脚本自带相对定位） | §4 |
| 4 | 编译移植过来的 DCNv3 扩展 | **有关** | §4 |
| 5 | 数据目录（VKITTI_II）可直接复用或重放 | 无关 | §5 |
| 6 | 套用训练参数（§6 表） | 无关 | §6 |
| 7 | 避开 §8 的已知缺陷清单 | **有关** | §8 |
| 8 | 跑 §9 的验收清单 | 两者都有 | §9 |

**一句话总结**：**conda 环境不用重建**（它就是 Python + CUDA 工具链，与代码无关）；
真正要动的只有"仓库根路径"这类硬编码，以及重新编译 CUDA 扩展。

---

## 1. 硬件与系统（固定事实）

| 项 | 值 | 影响 |
| --- | --- | --- |
| GPU | NVIDIA GeForce RTX 5090 D v2 | 24 GB 显存 |
| **算力** | **sm_120（Blackwell）** | **最关键约束**：必须 CUDA 12.8+ 的 PyTorch |
| 驱动 | 616.92（CUDA UMD 13.4） | 满足 12.8 运行时要求 |
| CPU | 16 逻辑核 | 决定 DataLoader worker 数 |
| 内存 | 61.6 GB | 充足 |
| **Windows 分页文件** | **仅 3.9 GB** | ⚠️ 见 §7.2，会限制 worker 数量 |
| 系统区域 | 中文（ANSI 代码页 **936/GBK**） | ⚠️ 见 §7.1，会导致 CUDA 编译失败 |
| 磁盘 | C: 38.6 GB / D: 76.5 GB / E: 336.1 GB 可用 | 数据/产物建议放 E: |

---

## 2. 软件版本矩阵（已实机验证，照抄即可）

| 组件 | 版本 | 位置 / 说明 |
| --- | --- | --- |
| conda | 26.3.2（miniforge3） | `D:\Programs\miniforge3` |
| **conda 环境名** | **`SDCombo`** | `D:\Programs\miniforge3\envs\SDCombo` |
| Python | **3.10.21** | DCNv3 要求 ≤3.10，不要用 3.11+ |
| PyTorch | **2.9.1+cu128** | 预编译已含 sm_120，无需 PTX JIT |
| torchvision | 0.24.1+cu128 | |
| CUDA 编译器 nvcc | **12.8.93**（conda-forge `cuda-nvcc`） | 装在环境内，`%CONDA_PREFIX%\Library\bin\nvcc.exe` |
| cudart | 12.8.90（conda-forge） | `cudart.lib` 在 `%CONDA_PREFIX%\Library\lib` |
| MSVC | **19.44.35207**（VS 2022 Build Tools 17.14） | `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools` |
| Windows SDK | 10.0.26100.0 | vcvars64 自动带入 |
| ninja | 1.13.2（pip） | 编译扩展必需 |
| numpy / Pillow | 2.2.6 / 12.3.0 | |
| pypdf | 6.19.0 | 可选，用于读 PDF |

`torch.cuda.get_arch_list()` = `sm_70,sm_75,sm_80,sm_86,sm_90,sm_100,sm_120` → **原生支持 sm_120**。

---

## 3. 从零重建环境（仅在环境丢失时需要）

### 3.1 创建环境 + PyTorch

```bat
conda create -n SDCombo python=3.10 -y
conda activate SDCombo
python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install ninja
```

> ❌ **绝对不要用** `conda install pytorch pytorch-cuda=11.8`。
> CUDA 11.8 不含 sm_120 的 kernel，运行时报
> `no kernel image is available for execution on the device`。

### 3.2 安装 CUDA 编译器（免管理员）

```bat
conda install -n SDCombo -c conda-forge cuda-nvcc=12.8.93 -y
```

会同时带入 `cuda-cudart-dev`（提供 `cudart.lib` 与 `cuda_runtime.h`）。

### 3.3 安装 MSVC（需要管理员，约 3.4 GB）

```bat
winget install --id Microsoft.VisualStudio.2022.BuildTools ^
  --accept-package-agreements --accept-source-agreements ^
  --override "--quiet --wait --norestart --nocache ^
              --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
```

### 3.4 ⚠️ 必须重建的环境补丁：`sitecustomize.py`

**这是本机最容易漏、也最难自己排查出来的一环。**
文件位置：`D:\Programs\miniforge3\envs\SDCombo\Lib\site-packages\sitecustomize.py`

它随解释器启动自动加载，解决三件事：

```python
"""
sitecustomize.py —— 随解释器启动自动加载的环境补丁。

1) 让 `import DCNv3` 可见：DCNv3 扩展编译产物在仓库内某个目录下，
   而模型代码用的是绝对导入 `import DCNv3`。
   >>> 换新工作区后，必须把下面的 _OPS_DCNV3 改成新路径 <<<

2) 注册 CUDA 运行时 DLL 目录：DCNv3.pyd 依赖 cudart64_12.dll（在
   <env>\Library\bin）。torch 只会从 CUDA_HOME/CUDA_PATH 推导该目录，
   未设置时不会自动注册，导致 ImportError: DLL load failed。
   必须在 import torch 之前完成。

3) 修正 torch 的子进程解码参数：本机 ANSI 代码页是 cp936，cl.exe 输出
   中文，而 torch.utils.cpp_extension 在 Windows 上把
   SUBPROCESS_DECODE_ARGS 硬编码为 ('cp1252',)，解码中文抛
   UnicodeDecodeError，使 cl 被误判为不兼容，CUDA 扩展编译失败。
"""
import os
import sys

_ENV_ROOT = r"D:\Programs\miniforge3\envs\SDCombo"
_REPO_ROOT = r"E:\SDCombo"                      # <<< 换工作区时改这里
_OPS_DCNV3 = os.path.join(_REPO_ROOT, "Segmentation", "Models",
                          "InternImage", "ops_dcnv3")   # <<< 按新代码结构改
_CUDA_BIN = os.path.join(_ENV_ROOT, "Library", "bin")

# 1) 让 DCNv3 可导入
if os.path.isdir(_OPS_DCNV3) and _OPS_DCNV3 not in sys.path:
    sys.path.append(_OPS_DCNV3)

# 2) 注册 CUDA DLL 目录（必须在 torch 导入前）
if sys.platform == "win32" and os.path.isdir(_CUDA_BIN):
    try:
        os.add_dll_directory(_CUDA_BIN)
    except (AttributeError, OSError):
        pass
    os.environ.setdefault("CUDA_HOME", os.path.join(_ENV_ROOT, "Library"))
    os.environ.setdefault("CUDA_PATH", os.path.join(_ENV_ROOT, "Library"))

# 便于任意工作目录下导入仓库内模块
if os.path.isdir(_REPO_ROOT) and _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)

# 3) 修正 torch 子进程解码
if sys.platform == "win32":
    try:
        from torch.utils import cpp_extension as _ce
        _ce.SUBPROCESS_DECODE_ARGS = ("utf-8", "replace")
    except Exception:
        pass
```

> 实测 **`VSLANG=1033` 无法让本机的 cl.exe 输出英文**，所以只能从解码端修。

---

## 4. 编译 CUDA 扩展（DCNv3）

### 4.1 需要一份 MSVC 环境封装脚本 `scripts/with_msvc.bat`

作用与要点（脚本本身与代码无关，可直接搬）：

1. 先清掉 conda-forge `vc` 包注入的 `VS_VERSION=16.0` / `VS_YEAR=2019` /
   `INCLUDE` / `LIB` 等变量，否则 vcvars64 会指向不存在的 VS2019；
2. `call "...\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"`；
3. 把 conda 环境放到 PATH 最前：`%CONDA_PREFIX%;%CONDA_PREFIX%\Scripts;%CONDA_PREFIX%\Library\bin`；
4. `set CUDA_HOME=%CONDA_PREFIX%\Library`、`CUDA_PATH` 同理；
5. **`set LIB=%CONDA_PREFIX%\Library\lib;%LIB%`** —— 让链接器找到 `cudart.lib`
   （nvcc 自己的 `-L` 不覆盖最后一步 MSVC 链接）；
6. `set TORCH_CUDA_ARCH_LIST=12.0`（目标 sm_120）；
7. `set PYTHONUTF8=1`、`set PYTHONIOENCODING=utf-8`（配合 §3.4 第 3 点）；
8. `set NVCC_PREPEND_FLAGS=-allow-unsupported-compiler`（保险，见下）；
9. `cd /d "%~dp0.."` 后执行 `%*`。

**关于 MSVC 版本**：CUDA 12.8 官方只声明支持 MSVC 19.3x，本机是 **19.44**。
`crt/host_config.h` 的判据是 `_MSC_VER >= 1950` 才报错，19.44 < 19.50，
所以**本机实际并未触发版本拒绝，能直接编译**。`-allow-unsupported-compiler`
作为保险保留（若换成 ≥19.50 的 MSVC 就需要它）。

### 4.2 编译命令

```bat
cd /d <新工作区>
scripts\with_msvc.bat cmd /c "cd /d <DCNv3 所在目录> && python setup.py build_ext --inplace"
```

> ⚠️ **用 `build_ext --inplace`，不要用 `python setup.py install`。**
> 实测 `install` 会把 `.pyd` 复制到 `sysconfig.get_paths()["purelib"]`，
> 该步骤在本机复制到一个**无法被 import 的位置**，反而导致
> `ModuleNotFoundError: No module named 'DCNv3'`。
> `--inplace` 让产物留在源码树，再配合 §3.4 的 `sys.path` 注入即可。

### 4.3 若新代码的 DCNv3 仍是旧版，需要改这 5 处才能编译

原始 InternImage 的 DCNv3 基于 PyTorch 1.x，在 PyTorch 2.9 下必然编译失败：

| 文件 | 原 | 改为 | 原因 |
| --- | --- | --- | --- |
| `cuda/*.cu`、`cuda/*.cuh` | `#include <ATen/cuda/CUDAContext.h>` | `#include <c10/cuda/CUDAStream.h>`<br>`#include <c10/cuda/CUDAException.h>` | 前者链式拉入 `cusparse.h`，而 **conda-forge 没有 win-64 的 cusparse 包** |
| `cuda/*.cu` | `#include <torch/torch.h>` | 删除 | 与 conda 自带 libcu++ 的 `std` 命名空间冲突 → `error C2872: "std": 不明确的符号` |
| `cuda/*.cuh` | `#include <THC/THCAtomics.cuh>` | 删除（改用 `cuda_runtime.h` 的内建 `atomicAdd`） | 该头在 PyTorch 2.x 已删除 |
| `cuda/*.cu` | `input.type()` | `input.scalar_type()` | `AT_DISPATCH_*` 需要 `c10::ScalarType` |
| `cuda/*.cu` | `torch::kHalf` | `at::kHalf` | 不再依赖 `torch/torch.h` |

> **关键设计**：`.cu` 文件里**不要包含任何 torch / C++ 标准库头**，
> torch 绑定放在单独的 `.cpp`。因为 conda 的 nvcc.profile 会注入
> `-I<...>/include/targets/x64`，让 libcu++ 的 `std/` 遮蔽 MSVC 标准库。
> 实测：`.cpp` 里 `#include <torch/extension.h>` 没问题，`.cu` 里就会报 C2872。
> 同理 `.cu`/`.cpp` 里不要 `#include <ATen/cuda/CUDAContext.h>`
> （注意 `ATen/cuda/CUDAContextLight.h` **也会** include `cusparse.h`，
> 要全局流就用 `c10/cuda/CUDAStream.h` 的 `at::cuda::getCurrentCUDAStream()`）。

### 4.4 验证

```bat
<新工作区>\scripts\with_msvc.bat python -c "import DCNv3; print(DCNv3.__file__)"
```

期望：forward 与纯 PyTorch 参考实现误差 ~1e-9，backward ~1e-7，加速 22~31 倍。

---

## 5. 数据

### 5.1 已有数据可直接复用

现有 VKITTI 2 数据在 `E:\SDCombo\datasets\VKITTI_II\`（15.2 GiB）：

```
VKITTI_II/
├── images/{training,validation}/        37860 / 4660   (jpg)
├── annotations/{training,validation}/   37860 / 4660   (png, 单通道 0..14)
└── depth/{training,validation}/         37860 / 4660   (png, 16bit)
```

三目录**主文件名**（去掉扩展名）严格一致 —— 已校验。
原始 tar 在 `datasets\_raw\`（15.5 GiB），md5 已核对：
`rgb 1e00a143…`、`depth 1d34a96f…`、`classSegmentation e8658ab4…`。

### 5.2 若需重新下载

只用 3 个归档（其余 6 个与本任务无关，共约 117 GiB 不要下）：

```
https://download.europe.naverlabs.com/virtual_kitti_2.0.3/vkitti_2.0.3_rgb.tar
https://download.europe.naverlabs.com/virtual_kitti_2.0.3/vkitti_2.0.3_depth.tar
https://download.europe.naverlabs.com/virtual_kitti_2.0.3/vkitti_2.0.3_classSegmentation.tar
```

校验和见官方 `vkitti_2.0.3_md5_checksums.txt`。

### 5.3 数据格式要点（踩过的坑）

- **VKITTI 2 的语义标签是 8-bit RGB 真彩 PNG**（实测 IHDR color type = 2），
  **不是** 0..14 索引图，**必须做颜色→类别映射**后才能训练。
  15 色调色板见原仓库 `Utils/DataPreparation/3to1_S0x.py`。
- **深度是 16-bit PNG（`I;16`），1 单位 = 1cm，远平面 655.35m 被裁剪**。
  实测**抽样 120 帧，每帧最大值都恰好是 65535**。
- 原图为 **1242×375**。
- 原图 `vkitti_2.0.3_*` 解压后目录形如
  `Scene01/15-deg-left/frames/rgb/Camera_0/rgb_00000.jpg`，
  本仓库需要的扁平命名由准备脚本生成。
- ⚠️ 准备脚本原版有 3 个会**静默出错**的地方（见 §8.4）。

---

## 6. 训练配置（可直接套用）

### 6.1 已验证可用的参数

| 参数 | 值 | 依据 |
| --- | --- | --- |
| **学习率** | **6e-5**（余弦退火到 1e-6） | 论文 §4.6.1 明确写 "lr above 1e-4 → loss will shock and cannot converge"，作者用 5e-5 |
| **梯度裁剪** | **max_norm = 1.0** | 实测最大梯度范数 ≈59，无裁剪时会在数千步后 nan |
| warmup | 1 epoch（LinearLR → CosineAnnealingLR） | 官方 InternImage 用 1500 iter linear warmup |
| lr 调度 | **T_max = 总 epoch 数** | 原实现把 `--cos[0]` 当 T_max，与 `--epochs` 解耦，轮数一变就中途触底 |
| **类别权重** | **median 频率法** | 实测最稀有类 0.18% vs 最多类 26.7%，**相差 145 倍**；不加权会坍缩到少数大类 |
| **ignore_index** | **255**（不是 0） | 见 §8.1 |
| **batch size** | **16**（crop 256） | 见 §6.2 显存实测 |
| crop / base size | 256 / 256 | 见 §6.3 |
| optimizer | AdamW, betas=(0.9,0.999), wd=0.05, `fused=True` | ⚠️ 论文 §6.2 用的是 **wd=0.01**，若对齐论文应改 |
| epochs | 10（起步）→ 建议 30~50 | 官方配方是 160k iter |
| AMP | 开 | |
| DataLoader | `persistent_workers=True`、`prefetch_factor=4`、`pin_memory=True`、`zero_grad(set_to_none=True)` | |

### 6.2 显存 / 吞吐实测（RTX 5090 24 GB，AMP 开，crop 256）

| batch | crop | img/s | 峰值 reserved | 占比 | 备注 |
| --- | --- | --- | --- | --- | --- |
| 4 | 256 | 56.2 | 4.44 GB | 18.6% | 原配置，**严重浪费** |
| 8 | 256 | 77.6 | 7.79 GB | 32.7% | |
| **16** | **256** | **93.2** | **14.90 GB** | **62.5%** | ✅ **推荐**，比 batch4 快 **1.66×** |
| 24 | 256 | 98.9 | 22.04 GB | 92.4% | 略快但太满 |
| 32 | 256 | **16.1** | 29.18 GB | 122% | ❌ 显存抖动，吞吐崩塌 |
| 8 | 384 | 44.0 | 16.76 GB | 70.2% | |
| 16 | 384 | 1.0 | 32.95 GB | 138% | ❌ 不可用 |
| 16 | 256 | 66.4 | 16.20 GB | 67.9% | `channels_last=True` |
| 8 | 512 | 18.2 | 24.70 GB | 103% | 梯度检查点 |

**结论**：
- 显存近似正比于 `batch × crop²`。
- **不要超过约 22 GB reserved**，否则吞吐崩塌。
- ⚠️ **`channels_last` 在本模型上是有害的**（93.2 → 66.4 img/s）。
  原因是主干以 LayerNorm/Linear 为主，格式转换开销大于收益 ——
  与"卷积网络常用它提速"的直觉相反，**不要开**。
- 梯度检查点在 crop 512 下反而更慢，不如直接用 crop 256。
- 应用后实测：**GPU 利用率 100%、功耗 434 W、显存 15.8 GB**
  （原配置只有 61% / 260 W / 3.2 GB）。

### 6.3 crop size 的取法

- 原实现 `base_size=375`（`RandomResize` 范围 `[281, 750]`）但 `crop_size=256`，
  实际输入被固定成 256×256，只覆盖原图 1242×375 的 **14%**。
- 论文 §6.2 用的是**缩放 [256, 1080]、裁剪 256×256**。
- 若想"看得更多"，应**同时**放大 `base_size` 与 `crop_size`（建议两者相当），
  例如 512/512 配 batch 2（9.84 GB）。实测 `crop=512` 使 mIoU 从 43.7 提到 **47.9**（论文 §4.5 表 4）。
- 数据增强的缩放范围（相对 base）应约 `[0.5, 2.0]`。

### 6.4 评估协议

- 原实现**评估时也用 `RandomCrop`** → 指标不可复现，且只覆盖 14.1%。
- 论文 §6.4.2 说明**正式评估用整图**，§6.4.1 的
  `batch = max(218/crop², 1)` 只是"RTX 2060 6GB 限制下的快速评估"。
- **建议**：训练中快速评估用确定性 `CenterCrop`（可复现），
  最终评估用**整图**。整图推理实测仅 **1.62 GB / 1.07 s**（1242×375），完全放得下。
- 覆盖率对照：`RandomCrop(256)` 14.1% → `CenterCrop(375)` 30.2% → 整图 **100%**。
- ⚠️ `CenterCrop` 实现里要先 `pad_if_smaller`，否则 crop > 图高（如 384 > 375）会直接抛异常。

---

## 7. 本机特有的两个环境陷阱

### 7.1 中文 Windows（cp936）导致 CUDA 扩展**编译失败**

现象：`Error checking compiler version for cl: 'cp1' codec can't decode bytes…`，
随后编译器被误判为不兼容。
根因：`cl.exe` 用 cp936 输出中文，而 torch 在 Windows 上硬编码用 cp1252 解码子进程输出。
**解决**：§3.4 的 `sitecustomize.py` 第 3 点（`SUBPROCESS_DECODE_ARGS = ("utf-8","replace")`）。
> 实测 `VSLANG=1033` 与 `DOTNET_CLI_UI_LANGUAGE=en` **都无效**，不要在这上面浪费时间。

### 7.2 Windows 分页文件仅 3.9 GB → DataLoader **静默卡死**

现象：训练跑到验证阶段后**完全卡死**（GPU 0%、所有 worker 的 CPU 时间停止增长），
日志出现
`RuntimeError: Couldn't open shared file mapping: <torch_...>, error code: <1455>`。
`1455 = ERROR_COMMITMENT_LIMIT`（提交限制已达上限）。

根因：分页文件只有 **3.9 GB**，而 `pin_memory=True` 下每个 worker 都要分配固定内存
缓冲区（batch16/crop256 约 640 MB），**12 个 worker** 叠加 24 GB 显存占用后超限。

**解决**：
- 用 **`--num-workers 4`**（或 ≤8）。实测数据加载耗时仅 **0.0001 s/iter**，
  **数据管线根本不是瓶颈**（纯 GPU-bound），worker 不需要多。
- 若确需更多，应调大 Windows 分页文件（需管理员：系统属性 → 高级 → 性能 → 虚拟内存）。
- ⚠️ 这个故障是**静默卡死**（不报错退出）。排查方法：
  看「日志文件最后写入时间」与「进程 CPU 时间是否还在增长」，
  不要只看 GPU 利用率。

---

## 8. 原仓库的已知缺陷清单（移植新代码时逐条排除）

以下是本机在原代码上**实测确认**的问题，新代码若继承同样写法需一并改掉。

### 8.1 `ignore_index` 用错（影响最大）

原 `criterion` 硬编码 `ignore_index=0`，但 VKITTI 2 的**类别 0 是 Terrain（地面）**，
是真实类别、训练集占 **17.66%**。后果：该类别永远学不会（IoU 恒为 0），
17.7% 的监督信号被丢弃。

**正确做法**：用 **255**。依据：`collate_fn` 的 padding 值就是 255，且标签只取 0..14。
（若是 Stanford2D3D 数据集，需确认该数据集的忽略约定。）

### 8.2 `mean IoU` 会变成 `nan`

原 `ConfusionMatrix.compute()` 里 `iu = diag / (row + col - diag)`，
对"GT 与预测都没出现过"的类别为 `0/0 = nan`，`iu.mean()` 随之变 nan。

实测验证集里有 3 个类别从未出现（TrafficSign / Truck / 另一类），mean IoU 恒为 nan。

**正确做法**：分母 `clamp(min=1e-9)`；mean IoU 只在"GT 中出现过"的类别上求平均，
并排除 `ignore_index` 对应的类别。

### 8.3 logits 被做了两次归一化

若分类头（UPerNet 等）在 `forward` 末尾做了 `log_softmax` / `softmax`，
而损失用 `F.cross_entropy`（**内部还会再做一次 log_softmax**），
等于双重归一化，梯度被严重压缩。

**正确做法**：分类头**只返回原始 logits**，归一化交给损失函数。
需要概率时用 `softmax`，需要类别图时用 `argmax`。
（官方 mmseg 的 `UPerHead.forward` 就是只返回 `cls_seg` 输出。）

### 8.4 深度分支可能实际未生效

常见错误写法：
```python
depth_limit = torch.max(depth).item()
depth = depth / depth_limit * torch.max(seg_0).item()
```
因为 VKITTI 深度**每帧最大值恒为 65535**（远平面裁剪），
真实深度中位数 20.2m（=2019）被压到 **0.031**，四分位跨度仅 0.09，
**深度分支等同于常数输入，模型实际只用了 RGB**。

**正确做法**（论文 §6.3.1 / 作者多数模型文件用的是后者）：
```python
depth = F.normalize(depth.type(torch.float)) * 256      # 作者统一写法
# 或按物理尺度 + log 压缩：
depth = log1p(min(depth, 10000cm)) / log1p(10000)       # 中位值 0.031 -> 0.826
```
⚠️ 另外 `.item()` 每次都会强制 **GPU→CPU 同步**，在小 batch 下是白白的性能损失。
任何在 forward 里调 `.item()` 的写法都应去掉。

### 8.5 其它

| 项 | 问题 |
| --- | --- |
| `argparse` 的 `type=list` | 会把命令行字符串拆成**单字符列表**，`--cos 10 1e-5 -1` 直接报错、`--module_trained upernet` 静默失效。改用 `nargs=3` / 逗号分隔 + `lambda` |
| `torch.load` | PyTorch 2.6+ 默认 `weights_only=True`；若 checkpoint 里存了 `argparse.Namespace` 会反序列化失败。需显式 `weights_only=False` |
| `torch.cuda.amp.*` | 已弃用，改 `torch.amp.GradScaler('cuda')` / `torch.amp.autocast('cuda', ...)` / `from torch.amp import custom_fwd, custom_bwd` 并加 `device_type='cuda'` |
| `work_dir` 子目录 | 日志/结果/权重会直接写入 `work_dir/{logger,evaluation,model}`，启动时必须 `os.makedirs(..., exist_ok=True)` |
| Conv2d 无 padding | `kernel_size=3` 不补零会让特征图逐层收缩，与回插值配合产生边界错位 |
| BatchNorm 用于小 batch | batch 只有 4~8 且输入含量纲差异大的通道时统计量不稳，可考虑 GroupNorm |
| `torch.meshgrid` | 补 `indexing='ij'` 锁定行为 |
| 未实现的参数 | 只声明不使用的 CLI 参数（如 `--aux`、`--resume`）会误导，要么实现要么删 |

---

## 9. 验收清单（迁移完成后逐项跑）

```bat
REM 1) 环境
python -c "import torch;print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
REM    期望: 2.9.1+cu128  NVIDIA GeForce RTX 5090 D v2  (12, 0)

REM 2) CUDA 扩展
scripts\with_msvc.bat python -c "import DCNv3;print(DCNv3.__file__)"

REM 3) 数据三目录对齐（数量 + 主文件名一致）
REM    期望: training 37860 / validation 4660，三边一致

REM 4) 小规模冒烟（合成数据跑通 数据集->模型->损失->反向->评估）

REM 5) 真实数据单 epoch（观察 loss 单调下降、无 nan、类别数不坍缩）
```

**关键验收指标**：

| 指标 | 期望 |
| --- | --- |
| 训练是否发散 | **不发散**（loss 单调下降，无 nan） |
| 预测到的类别数 | 接近全部类别（原实现在早期只有 4 个） |
| GPU 利用率 / 功耗 | ~100% / ~430 W |
| 显存占用 | ~15.8 GB（batch 16 / crop 256） |
| 吞吐 | ~93 img/s |
| 单 epoch 耗时 | ~10 分钟（37,860 帧 / batch 16 / 0.25 s per iter） |
| mean IoU | 正常数值（不是 nan） |

**参考基线**（10 epoch，VKITTI 2，修正版）：
`mIoU 7.4 → 14.9 → 13.6 → 13.7 → 14.8 → 22.9 → 24.1 → 17.7 → 27.4 → 32.6`，
末轮 acc 77.5%。同协议复核 epoch9 得 **mIoU 29.1 / acc 75.0%**。

> 注：逐 epoch 报的是**当轮**指标而非历史最优，所以中途回落属正常；
> 若要求平稳，可降低 class-weight 强度或改用 focal loss。

---

## 10. 需要在新工作区重建的脚本（与代码无关，可直接照搬）

| 脚本 | 作用 |
| --- | --- |
| `scripts/with_msvc.bat` | MSVC + CUDA 编译环境封装（见 §4.1） |
| `scripts/activate_env.bat/.sh` | 激活 conda 环境并设 `CUDA_HOME`/`TORCH_CUDA_ARCH_LIST` |
| `tools/verify_env.py` | 环境自检：依赖版本、GPU、`import DCNv3`、数值一致性 |
| `tools/verify_dcnv3.py` | 单独验证 DCNv3 forward/backward 与参考实现一致 |
| `tools/bench_train.py` | batch×crop×channels_last×梯度检查点 的吞吐/显存基准 |
| `tools/compare_runs.py` | 汇总多次运行的逐 epoch 指标（含稳定性 std） |
| `tools/eval_ckpts.py` | **同一评估协议**下对比多个 checkpoint（消除口径差异） |
| `tools/diag_depth.py` | 深度分布与归一化尺度分析 |
| `tools/diag_pred_dist.py` | 预测分布诊断（判断是否坍缩） |
| `tools/diag_metrics.py` | 解释 `mean IoU: nan` 的来源 |

> 这些脚本都写在 `<仓库根>/tools/` 下，通过 `os.path.dirname(os.path.dirname(
> os.path.abspath(__file__)))` 推导仓库根，**换目录后无需修改**。

---

## 11. 遗留未解决项（供新工作区接手）

1. **验证 mIoU 仍有振荡**：即使 lr=6e-5，逐 epoch 仍在 13~32 间波动，
   原因是模型的**类别偏好在 epoch 间摆动**（实测 GuardRail 16.4%→1.1%→18.7%，
   Van 17.3%→30.2%→9.2%），且始终把 Tree 预测偏多（真实 1.4%）、
   Building 偏少（真实 25.1%）。降学习率显著缓解但未根除。
2. **绝对精度偏低**：修正版 10 epoch 到 mIoU 29.1，远未收敛。
   官方配方是 160k iteration + 层级 lr 衰减。
3. **未对齐论文的关键设置**：Focal Loss (α=0.5, γ=2)、HHA 深度编码、
   `weight_decay=0.01`（原代码是 0.05）。论文实测 HHA 比原始深度
   提升巨大（17.6 → 36.5 mIoU）。
4. **原工作区只是作者工程的中间快照**：作者完整仓库在
   `E:\OneDrive\Study\MyModel\hxl172`，其中 `Joint/model_DL4sDL.py` /
   `Backup/SDCBottleneck.py` / `Dataset/dataset_HHA.py` /
   `Utils/DataPreparation/Depth2HHA-python/` 才是论文的主线模型；
   原工作区对应的是论文 §4.3 被否定的末端融合路线。

---

## 12. 换工作区时的操作顺序与数据处置

### 12.1 先确认代码已备份（重要）

原工作区 `E:\SDCombo` 有 8 个提交是本项目新增的，**相对作者快照 `8f34619`**：

```
6f70905 upload paper
ee89649 配置 SDCombo 训练/推理环境（Windows + RTX 5090 / sm_120）
dc571ef 接入 VKITTI 2 真实数据并修正训练/评估缺陷
2595cf9 审计并修正模型/训练/推理设计，充分利用 RTX 5090
4b0f8d8 记录 Windows 分页文件过小导致 DataLoader 卡死的问题与对策
4630dca 修正文档中对训练结果的表述，如实记录 mIoU 振荡这一未解决问题
42a9ca7 补齐 10-epoch 学习率对比结果，确认 lr=6e-5 为推荐配置
51d076a 对照论文审视文件组织与模型结构
```

**换目录前先确认这些提交已在远端**（`origin` = `git@github.com:novalli1993/SDCombo.git`）：

```bat
git -C E:\SDCombo log --oneline -1                 REM 看本地最新提交
git -C E:\SDCombo ls-remote --heads origin         REM 与远端对比哈希
git -C E:\SDCombo push origin master               REM 若远端落后则推送
```

> 记录本文件时远端 `refs/heads/master` = `51d076a`，与本地一致，**已备份**。
> 但若你在那之后又提交过东西，请重新确认。

### 12.2 `MIGRATION.md` 放在哪里

本文件是**纯文本、不依赖代码**，建议：

1. 复制一份留 **工作区之外**（例如 `E:\SDCombo_MIGRATION.md`），
   这样重命名/删除工作区都不会丢；
2. 它随 git 提交后也会进远端，但**别只依赖 git** —— 若新代码是另一个仓库，
   历史就不在一起了。

### 12.3 数据不要跟着"旧代码仓库"一起被冷落

工作区里有两块**与代码无关、可直接复用**的大体积数据，重命名工作区时要注意它们的去向：

| 目录 | 体积 | 建议 |
| --- | --- | --- |
| `datasets\VKITTI_II` | 15.2 GiB | **保留并复用**。新工作区要么把它移/复制过去，要么用 `--data-path` 指向旧位置 |
| `datasets\_raw` | 15.5 GiB | 三个原始 tar，已完整解压到 `VKITTI_II`。**确认不再需要重放准备流程后可以删**，能省 15.5 GiB |
| `work_dir` | 20.8 GiB | 历史运行的日志/指标/checkpoint（含 `_baseline10ep*`、`_run10ep_lr6e-5*` 等归档）。**建议先挑出要保留的 checkpoint 单独备份，其余可删** |

**推荐做法**：不要移动 15 GiB 的数据，而是让新工作区**直接引用旧路径**。例如：

- 训练时 `--data-path E:\<重命名后的旧工作区>\datasets\VKITTI_II`；
- 或者在新工作区建一个目录联接（不需要管理员）：
  ```bat
  mklink /J "E:\新工作区\datasets" "E:\旧工作区\datasets"
  ```
  这样新工作区看起来有 `datasets\`，实际数据只有一份，不占额外空间。

### 12.4 建议的执行顺序

```bat
REM ---- 1. 确认提交已推送（见 §12.1）----
git -C E:\SDCombo push origin master

REM ---- 2. 挑出要保留的训练产物，单独备份 ----
REM     例如把 best checkpoint 复制到工作区之外

REM ---- 3. 复制 MIGRATION.md 到工作区之外留底 ----

REM ---- 4. 重命名旧工作区 ----
ren E:\SDCombo  SDCombo_old_20261007

REM ---- 5. 新建同名目录，放入新代码 ----
mkdir E:\SDCombo
REM     把新代码 checkout/复制到 E:\SDCombo

REM ---- 6. 按 §12.3 让新工作区能访问数据（mklink /J 或 --data-path）----

REM ---- 7. 更新 sitecustomize.py 的 _REPO_ROOT / _OPS_DCNV3（见 §3.4）----
REM     >>> 这一步最容易漏，漏了就会 ModuleNotFoundError: No module named 'DCNv3'

REM ---- 8. 编译新代码里的 CUDA 扩展（见 §4.2）----

REM ---- 9. 跑 §9 验收清单 ----
```

### 12.5 重命名后需要改的硬编码路径（清单）

| 位置 | 内容 | 是否必改 |
| --- | --- | --- |
| `<env>\Lib\site-packages\sitecustomize.py` | `_REPO_ROOT`、`_OPS_DCNV3` | ✅ **必改**，否则 DCNv3 导入失败 |
| `scripts\with_msvc.bat` | 只用 `%~dp0..` 相对定位；**若新代码目录结构与旧的不同**（例如 DCNv3 换了位置），编译命令里的 `cd` 目标要改 | ⚠️ 看情况 |
| `tools\*.py` | 均用 `__file__` 推导仓库根，**无需修改** | ❌ |
| 训练/评估命令 | `--data-path` 指向数据实际位置 | ✅ 看 §12.3 |

> `MD` 里提到的所有绝对路径（`D:\Programs\miniforge3\...`、`E:\SDCombo\...`）
> 都是**本机当前值**，换工作区时按上表替换。

---

## 附：一键复现命令序列

```bat
REM ---------- 环境（若已存在则跳过）----------
conda create -n SDCombo python=3.10 -y
conda activate SDCombo
python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install ninja
conda install -n SDCombo -c conda-forge cuda-nvcc=12.8.93 -y
winget install --id Microsoft.VisualStudio.2022.BuildTools ^
  --accept-package-agreements --accept-source-agreements ^
  --override "--quiet --wait --norestart --nocache --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"

REM ---------- 环境补丁 ----------
REM 写 sitecustomize.py，并把 _REPO_ROOT / _OPS_DCNV3 改为新工作区路径（见 §3.4）

REM ---------- 编译 CUDA 扩展 ----------
cd /d <新工作区>
scripts\with_msvc.bat cmd /c "cd /d <DCNv3 目录> && python setup.py build_ext --inplace"

REM ---------- 验证 ----------
scripts\with_msvc.bat python -c "import DCNv3;print(DCNv3.__file__)"

REM ---------- 训练（推荐配置）----------
python train.py ^
  --data-path datasets\VKITTI_II ^
  --batch-size 16 --base-size 256 --crop-size 256 --eval-batch-size 8 ^
  --epochs 10 --lr 6e-5 --lr-min 1e-6 --warmup-epochs 1 ^
  --max-grad-norm 1.0 --class-weight median --num-workers 4

REM ---------- 整图评估 ----------
python evaluation.py --data-path datasets\VKITTI_II ^
  --pretrained work_dir\model\model_<mark>_9.pth --full-res -b 4
```
