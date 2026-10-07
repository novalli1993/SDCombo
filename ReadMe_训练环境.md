# SDCombo 训练环境与使用说明（Windows + RTX 5090 / sm_120）

> 本文件是**本机（KEPHAN）可用的实际配置**，不是论文时代的 Linux + CUDA 11.8 写法。
> 仓库原先的 [ReadMe.md](ReadMe.md) 里的 `pytorch-cuda=11.8` + `sh make.sh` 在本机**不可用**
> （CUDA 11.8 不含 Blackwell / sm_120 的 kernel）。
>
> 配套文档：代码与论文的逐条符合性检视见 [论文符合性检视报告.md](论文符合性检视报告.md)。

---

## 1. 本机环境（已实机验证）

| 组件 | 版本 / 值 | 位置或说明 |
| --- | --- | --- |
| GPU | NVIDIA GeForce RTX 5090 D v2（24 GB，**sm_120**） | 驱动 617.42 |
| conda | miniforge3 | `D:\Programs\miniforge3` |
| conda 环境 | `SDCombo` | `D:\Programs\miniforge3\envs\SDCombo` |
| Python | 3.10.21 | DCNv3 要求 ≤ 3.10 |
| PyTorch | 2.9.1+cu128（含 sm_120 kernel） | `torch.cuda.get_arch_list()` 含 `sm_120` |
| torchvision | 0.24.1+cu128 | |
| nvcc | 12.8.93（conda-forge `cuda-nvcc`） | `%CONDA_PREFIX%\Library\bin\nvcc.exe` |
| MSVC | 19.44（VS 2022 Build Tools） | `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools` |
| tensorboard | 2.21.0（用于记录曲线；缺失时训练仍可跑，只跳过 TensorBoard） | |

环境相关的三个“坑”都已在 `sitecustomize.py` 里处理（见 §2.3）。

---

## 2. 一次性配置

### 2.1 激活环境

```bat
conda activate SDCombo
```

或直接用封装脚本（不依赖 `conda activate`，会设好 `CUDA_HOME`/`PATH`）：

```bat
scripts\activate_env.bat
```

### 2.2 安装依赖（仅新环境需要）

```bat
conda create -n SDCombo python=3.10 -y
conda activate SDCombo
python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install ninja tensorboard
conda install -n SDCombo -c conda-forge cuda-nvcc=12.8.93 -y
```

### 2.3 `sitecustomize.py`（本机必做，且换目录后必改）

位置：`D:\Programs\miniforge3\envs\SDCombo\Lib\site-packages\sitecustomize.py`

它随解释器启动自动加载，解决三件事：

1. 把 **`Backbone\InternImage\ops_dcnv3`** 加进 `sys.path`，让模型里的 `import DCNv3` 可见
   （**换工作区后必须改这个路径**，否则 `ModuleNotFoundError: No module named 'DCNv3'`）；
2. 注册 `cudart64_12.dll` 所在目录（`<env>\Library\bin`），否则 `DCNv3.pyd` 会 `ImportError: DLL load failed`；
3. 把 `torch.utils.cpp_extension.SUBPROCESS_DECODE_ARGS` 改成 utf-8：本机是中文 Windows
   （ANSI 代码页 cp936），`cl.exe` 输出中文而 torch 硬编码用 cp1252 解码，会让 cl 被误判为不可用。

### 2.4 编译 DCNv3 CUDA 扩展

**只有 InternImage 相关模型（`Joint/model_II4sII.py`、`Joint/model_II4sIsRes.py`）需要；
VKITTI 训练用的主线模型（MobileNetV3 双分支）不需要。**

```bat
scripts\with_msvc.bat scripts\build_dcnv3.bat
```

产物：`Backbone\InternImage\ops_dcnv3\DCNv3.cp310-win_amd64.pyd`

要点：
* 必须用 `build_ext --inplace`，不要用 `setup.py install`；
* `scripts\with_msvc.bat` 会清掉 conda-forge `vc` 包注入的 `VS_VERSION/INCLUDE/LIB`，
  再 `vcvars64.bat`，并把 `%CONDA_PREFIX%\Library\lib` 加到 `LIB`（否则链接不到 `cudart.lib`）；
* `TORCH_CUDA_ARCH_LIST=12.0`（Blackwell）；
* ⚠️ **`src` 里的 `.cu/.cuh/.cpp` 不能含非 ASCII 注释**：本机 cp936 代码页会让 nvcc 的行结构错乱，
  表现为 `identifier "scalar_t" is undefined`（宏根本没展开）。仓库里这三个文件已按此要求
  改成 ASCII 注释，并补上了 `#include <ATen/ATen.h>`（PyTorch 2.x 下 `AT_DISPATCH_*` 宏的来源）；
* ⚠️ **`.bat` 文件必须是 CRLF 行尾**，LF-only 的批处理会被 cmd 吃掉字符而报
  `'orlevel' is not recognized`。`.gitattributes` 已强制 `*.bat text eol=crlf`。

### 2.5 环境自检

```bat
python tools\verify_env.py        REM 版本/GPU/DCNv3 数值一致性/速度
python tools\verify_dcnv3.py      REM 只验证 DCNv3 forward/backward
python tools\smoke_test.py --batch-size 4 --amp    REM 合成数据全链路冒烟
python tools\bench_train.py       REM batch x crop 吞吐与显存扫描
python tools\check_dataset.py     REM 数据集三目录一致性与标签范围
```

`verify_env.py` 期望输出：`2.9.1+cu128` / `NVIDIA GeForce RTX 5090 D v2` / `sm_120` /
`import DCNv3 -> ...ops_dcnv3\DCNv3.cp310-win_amd64.pyd` / forward 误差 ~1e-9、backward ~1e-7。

---

## 3. 数据集

### 3.1 当前工作区里的数据：VKITTI 2

```
datasets\VKITTI_II\
├── images\{training,validation}\        37860 / 4660   (jpg/png, 1242x375)
├── annotations\{training,validation}\   37860 / 4660   (png, 单通道 0..13)
└── depth\{training,validation}\         37860 / 4660   (png, 16bit, 1 单位 = 1cm)
```

* 类别顺序（14 类，由 `tools/prepare_vkitti.py` 的 `PALETTE` 决定）：
  `Terrain, Tree, Vegetation, Building, Road, GuardRail, TrafficSign, TrafficLight, Pole, Misc, Truck, Car, Van, Undefined`
* 标签是**单通道 0..13**，没有 `<UNK>`；**`ignore_index` 必须是 255**（数据集 `collate_fn` 的填充值）。
  若沿用仓库 Stanford 脚本里的 `ignore_index=0`，等于把 Terrain（约 16% 像素）整个丢掉。
* 深度是 16bit 原始值（最大 65535，远平面被裁剪）；模型侧按
  `depth = F.normalize(depth.float()) * 256` 后过 `depth_conv(1→3)`（与论文 §6.3.1 一致）。
* 类别极不平衡：Pole 0.26% vs Vegetation 26.6%（相差约 100 倍）——`train_VKITTI.py`
  默认用 median frequency 类别权重来缓解。
* 逐类像素直方图在 `datasets\vkitti_report.json`（`class_histogram`），类别权重直接读它。

### 3.2 数据从哪来 / 怎么重建

原始 tar（3 个，共约 15.5 GB）与重建脚本：

```bat
python tools\ingest_vkitti.py --dest datasets\_raw          REM 校验/收纳官方 tar
python tools\prepare_vkitti.py ^
  --rgb datasets\_raw\vkitti_2.0.3_rgb.tar ^
  --depth datasets\_raw\vkitti_2.0.3_depth.tar ^
  --classseg datasets\_raw\vkitti_2.0.3_classSegmentation.tar ^
  --out datasets\VKITTI_II --report datasets\vkitti_report.json
python tools\check_dataset.py
```

官方下载地址（只下这 3 个，其余 6 个与本任务无关）：

```
https://download.europe.naverlabs.com/virtual_kitti_2.0.3/vkitti_2.0.3_rgb.tar
https://download.europe.naverlabs.com/virtual_kitti_2.0.3/vkitti_2.0.3_depth.tar
https://download.europe.naverlabs.com/virtual_kitti_2.0.3/vkitti_2.0.3_classSegmentation.tar
```

> 注意：VKITTI 的语义标签是**真彩 PNG**，必须按 `PALETTE` 做颜色→类别映射，
> `prepare_vkitti.py` 用向量化 LUT 完成（比作者原脚本的逐像素循环快约三个数量级）。

### 3.3 论文主线数据集：Stanford2D3D

论文 §6 的实验用的是 Stanford2D3D（`Dataset\dataset_HHA.py` + HHA 编码，
预处理脚本在 `Utils\DataPreparation\Depth2HHA-python\`）。
本机**尚无该数据集**，相应脚本（`train_HHA.py` / `train_depth.py` / `evaluation_HHA.py` 等）
保持原样未改，数据到位后加 `--data-path` 即可用。注意其 `ignore_index=0` 是数据集约定
（0 = `<UNK>`），与 VKITTI 不同。

---

## 4. 训练

### 4.1 VKITTI 2 上训练论文主线模型

```bat
conda activate SDCombo
python train_VKITTI.py --data-path datasets\VKITTI_II ^
  --batch-size 16 --base_size 375 --crop_size 256 --eval-batch-size 8 ^
  --epochs 10 --lr 5e-5 --lr-min 1e-6 --warmup-epochs 1 ^
  --max-grad-norm 1.0 --class-weight median --num-workers 4
```

快速自检（只取 256 张图、1 个 epoch）：

```bat
python train_VKITTI.py --limit 256 --epochs 1 --batch-size 8
```

主要参数：

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--data-path` | `datasets/VKITTI_II` | 数据根目录 |
| `--batch-size` | 16 | 24 GB 卡上很宽裕（`tools\bench_train.py` 可实测推荐值） |
| `--base_size` / `--crop_size` | 375 / 256 | 缩放范围 `[crop_size, base_size]`；**crop 不要超过 375**（超了 RandomCrop 会补零污染标签） |
| `--epochs` | 10 | 论文 §6 主线用 10 |
| `--lr` / `--lr-min` | 5e-5 / 1e-6 | 论文 §6.2 是 5e-5、最小 1e-7；论文 §4.6.1 明确 lr > 1e-4 会 loss shock |
| `--warmup-epochs` | 1 | 大 batch + AdamW 更稳；设 0 即完全对齐论文（无 warmup） |
| `--wd` | 0.01 | 论文 §6.2 |
| `--max-grad-norm` | 1.0 | 梯度裁剪；不裁剪时梯度范数可达数十，训练数千步后出 nan |
| `--class-weight` | `median` | `none` / `median` / `inverse`；论文只用 Focal Loss，本机数据极不平衡故默认加权 |
| `--focal-alpha` / `--focal-gamma` | 0.5 / 2.0 | 论文 §6.2 |
| `--num-workers` | 4 | 本机分页文件仅约 4 GB，worker 太多会在 `pin_memory` 下**静默卡死**（见 §6.1） |
| `--eval-random-crop` | 关 | 默认中心裁剪（结果可复现）；打开则用论文 §6.4.1 的随机裁剪快速评估 |
| `--no-amp` | 关 | 默认开 AMP |
| `--resume` | 空 | 从 `work_dir\model\model_<mark>_<epoch>.pth` 续训 |

模型固定为 `Joint/model_DL4sDL.py` 的 `_SDCombo`（论文 §5 主线：MobileNetV3-Large 双分支 +
6 级 Fusion + DeepLabV3/ASPP 分类头 + aux 头），参数量约 **20.0 M**。

### 4.2 输出约定（`work_dir/`）

```
work_dir\
├── logger\loggers<mark>.txt          逐 step 日志（含 loss / lr / 梯度范数 / 显存）
├── evaluation\evaluation<mark>.txt   训练参数 + 逐 epoch 的 acc / 逐类 IoU / mIoU
├── model\model_<mark>_<epoch>.pth    每个 epoch 的权重（model/optimizer/scheduler/scaler/args）
└── board\<mark>\                     TensorBoard 事件
```

`<mark>` 是启动时间戳 `YYYYmmdd_HHMMSS`。

**归档惯例**（便于多次实验对比，来自本机既有实践）：一次运行结束后，把
`logger/evaluation/model` 里的文件整体移进 `work_dir\_<实验名>_<mark>\`，例如：

```powershell
$mark = (Get-ChildItem work_dir\logger -File | Select-Object -First 1).BaseName -replace '^loggers',''
foreach ($k in 'logger','evaluation','model') {
  New-Item -ItemType Directory -Force -Path "work_dir\_vkitti10ep_$mark" | Out-Null
  Move-Item "work_dir\$k\*" "work_dir\_vkitti10ep_$mark\" -Force -ErrorAction SilentlyContinue
}
python tools\compare_runs.py        # 汇总各次运行的逐 epoch 指标
```

---

## 5. 评估

```bat
REM VKITTI：默认整图评估（1242x375 整图，显存很小，不需要论文时代的裁剪折衷）
python evaluation_VKITTI.py --weight work_dir\model\model_<mark>_9.pth

REM 也可用裁剪协议（与论文 §6.4 的快速评估对齐）
python evaluation_VKITTI.py --weight work_dir\model\model_<mark>_9.pth --crop_size 256 --eval-random-crop

REM Stanford2D3D（数据到位后）
python evaluation_HHA.py --data-path <Stanford2D3D> --weight <checkpoint>
```

`evaluation_VKITTI.py` 会打印并保存 14 类的 acc / IoU / mIoU 以及全局准确率。

---

## 6. 已知陷阱（都在这台机器上踩过）

### 6.1 Windows 分页文件过小 → DataLoader 静默卡死

`--num-workers` 开大（如 12）时，训练在进入验证阶段后**完全卡死**（GPU 0%、worker 的 CPU 时间
停止增长），日志出现：

```
RuntimeError: Couldn't open shared file mapping: <torch_...>, error code: <1455>
```

`1455 = ERROR_COMMITMENT_LIMIT`：本机分页文件约 4 GB，而 `pin_memory=True` 下每个 worker
都要分配固定内存缓冲区，worker 多时叠加超出提交限制。

对策：**`--num-workers 4`**（实测数据加载 ~0.0001 s/iter，瓶颈在 GPU，worker 不需要多）。
这是静默卡死、不会报错退出，排查要看“日志最后写入时间 + 进程 CPU 时间是否还在增长”。

### 6.2 `torch.load` 在 PyTorch 2.6+ 需要 `weights_only=False`

本仓库的 checkpoint 里存了 `argparse.Namespace`（`train_HHA.py:173` 等），
PyTorch 2.6 起 `torch.load` 默认 `weights_only=True`，会直接抛
`UnpicklingError: Weights only load failed`。

* 本仓库新增的 `train_VKITTI.py` / `evaluation_VKITTI.py` 已显式传 `weights_only=False`；
* 原有的 `evaluation_HHA.py` / `evaluation_depth.py` / `demo_*.py` **尚未修改**，
  在 PyTorch 2.9 上需要自己补这个参数（属于 `论文符合性检视报告.md` §10 列的待办）。

### 6.3 `work_dir` 子目录

训练脚本会自己建 `work_dir/{logger,evaluation,model,board}`；
但 `evaluation_*.py` 只建了 `evaluation/`，干净环境里直接跑评估前先确保
`work_dir\logger` 存在（新增的 `evaluation_VKITTI.py` 已补上）。

### 6.4 中文 Windows（cp936）与源码编码

* `.cu/.cuh/.cpp` 里不要写中文注释（会让 nvcc 编译失败，见 §2.4）；
* `.bat` 必须 CRLF（见 §2.4）；
* 控制台输出中文乱码时用 `set PYTHONIOENCODING=utf-8`。

### 6.5 `channels_last` 在本模型上有害

历史上实机测过：同一配置下 `channels_last=True` 反而更慢（主干以 LayerNorm/Linear 为主，
格式转换开销大于收益）。`tools\bench_train.py --channels-last` 可复现对比。

---

## 7. 与论文的差异（复现前必读）

本仓库是论文作者 2023-09 的完整代码快照；逐条符合性核对（含“哪些与论文一致、哪些是
论文未写的实现细节、哪些是缺陷”）见 [论文符合性检视报告.md](论文符合性检视报告.md)。
对**训练**影响最大的三点：

1. **数据集不同**：论文 §6 报的是 Stanford2D3D 的成绩（HHA 36.5 mIoU / 原始深度 17.6 mIoU），
   本机当前只有 VKITTI 2，因此 `train_VKITTI.py` 的结果**不能**直接与论文表格比较；
2. **深度编码**：论文证明 HHA 明显优于原始深度，VKITTI 路线目前只用原始深度
   （`Utils\DataPreparation\Depth2HHA-python\` 是 Stanford2D3D 的 HHA 脚本，未做 VKITTI 适配）；
3. **评估口径**：论文 §6.4.1 训练中的“快速评估”用随机裁剪；本仓库的 VKITTI 脚本默认改成
   中心裁剪以保证可复现（`--eval-random-crop` 可切回论文口径）。

---

## 8. 文件组织

| 路径 | 内容 |
| --- | --- |
| `Joint/` | 模型：`model_DL4sDL.py`（论文主线）、`model_II4s*.py`、`model_Res4sRes.py`、`model.py` |
| `Backbone/` | `MobileNetV3.py`、`ResNet.py`、`InternImage/`（含 DCNv3 扩展源码） |
| `Segmentation/` | `DeepLabV3.py`、`UPerHead.py`（分类头 / 对照组） |
| `Dataset/` | `dataset_HHA.py`、`dataset_Stanford2D3D.py`、`dataset_VKITTI.py`、`transforms.py` |
| `Utils/` | `train_val.py`（Stanford）、`train_val_vkitti.py`（VKITTI）、`distributed_utils.py`（指标/日志）、`DataPreparation/`（数据预处理脚本） |
| `tools/` | 环境自检、数据准备与检查、冒烟、基准、日志汇总等脚本 |
| `scripts/` | `with_msvc.bat`（MSVC+CUDA 环境）、`build_dcnv3.bat`、`activate_env.bat/.sh` |
| `work_dir/` | 训练产物（仓库里保留了作者的 2023 年历史记录作为实验证据） |
| `datasets/` | 数据集（不入库；当前为 `VKITTI_II`） |
| 根目录 | `train_HHA.py` / `train_depth.py` / `train_DL.py` / `train_VKITTI.py`、`evaluation_*.py`、`demo_*.py` |
