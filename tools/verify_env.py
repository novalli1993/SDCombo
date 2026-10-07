"""
环境自检脚本 —— 确认 SDCombo 训练/推理环境是否正确配置。

用法（在仓库根目录 E:\\SDCombo 下执行）:
    conda activate SDCombo
    python tools\\verify_env.py

检查项:
  1. Python / 依赖版本
  2. CUDA / GPU（含算力型号、是否匹配 PyTorch）
  3. DCNv3 CUDA 扩展编译产物能否导入
  4. DCNv3 forward / backward 与纯 PyTorch 参考实现的一致性
  5. DCNv3 forward 计时（与参考实现对比加速比）
"""
import os
import sys
import time
import traceback

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

OK = "[ OK ]"
FAIL = "[FAIL]"
WARN = "[WARN]"


def section(title):
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


def step(label, fn):
    try:
        result = fn()
        print(f"{OK} {label}" + (f" -> {result}" if result is not None else ""))
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"{FAIL} {label} -> {type(exc).__name__}: {exc}")
        traceback.print_exc(limit=3)
        return False


def check_python():
    print(f"Python      : {sys.version.split()[0]}  ({sys.executable})")
    if sys.version_info[:2] != (3, 10):
        print(f"{WARN} 官方 ReadMe 要求 python 3.10（DCNv3 需要 <=3.10），当前为 {sys.version.split()[0]}")
    return True


def check_deps():
    import torch
    import torchvision
    import PIL
    import numpy

    print(f"torch       : {torch.__version__}")
    print(f"torchvision : {torchvision.__version__}")
    print(f"pillow      : {PIL.__version__}")
    print(f"numpy       : {numpy.__version__}")
    print(f"torch CUDA  : {torch.version.cuda}   (cuDNN {torch.backends.cudnn.version()})")
    return True


def check_gpu():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "torch.cuda.is_available() == False。请确认已安装 CUDA 版 PyTorch（cu128 及以上）"
        )
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    arch_list = torch.cuda.get_arch_list()
    print(f"GPU         : {name}")
    print(f"算力 (CC)   : sm_{cap[0]}{cap[1]}")
    print(f"torch 支持的架构: {', '.join(arch_list)}")
    sm = f"sm_{cap[0]}{cap[1]}"
    if sm not in arch_list and "compute_" + str(cap[0]) + str(cap[1]) not in arch_list:
        print(f"{WARN} torch 预编译架构列表中没有 {sm}，将依赖 PTX JIT，首次运行会较慢")
    x = torch.randn(1000, 1000, device="cuda")
    y = (x @ x).sum().item()
    torch.cuda.synchronize()
    print(f"矩阵乘法自检 : OK (sum={y:.2f})")
    print(f"显存        : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GiB")
    return True


def check_dcnv3_import():
    import DCNv3  # noqa: F401

    return getattr(DCNv3, "__file__", "built-in")


def check_dcnv3_math():
    import torch
    from Backbone.InternImage.ops_dcnv3.functions.dcnv3_func import (
        DCNv3Function,
        dcnv3_core_pytorch,
    )

    H_in, W_in = 8, 8
    N, M, D = 2, 4, 16
    Kh, Kw = 3, 3
    P = Kh * Kw
    offset_scale = 2.0
    pad = 1
    dilation = 1
    stride = 1
    H_out = (H_in + 2 * pad - (dilation * (Kh - 1) + 1)) // stride + 1
    W_out = (W_in + 2 * pad - (dilation * (Kw - 1) + 1)) // stride + 1

    torch.manual_seed(3)
    results = {}

    # ---- forward 一致性（float32） ----
    with torch.no_grad():
        inp = torch.rand(N, H_in, W_in, M * D).cuda() * 0.01
        off = torch.rand(N, H_out, W_out, M * P * 2).cuda() * 10
        msk = torch.rand(N, H_out, W_out, M, P).cuda() + 1e-5
        msk /= msk.sum(-1, keepdim=True)
        msk = msk.reshape(N, H_out, W_out, M * P)

        ref = dcnv3_core_pytorch(inp, off, msk, Kh, Kw, stride, stride,
                                 Kh // 2, Kw // 2, dilation, dilation, M, D, offset_scale)
        cuda = DCNv3Function.apply(inp, off, msk, Kh, Kw, stride, stride,
                                   Kh // 2, Kw // 2, dilation, dilation, M, D,
                                   offset_scale, 2)
        fwd_ok = torch.allclose(cuda, ref, rtol=1e-2, atol=1e-3)
        max_err = (cuda - ref).abs().max().item()
    print(f"  forward  一致性: {'一致' if fwd_ok else '不一致'} (max_abs_err={max_err:.2e})")
    results["forward"] = fwd_ok
    if not fwd_ok:
        raise AssertionError("DCNv3 CUDA forward 与参考实现不一致")

    # ---- backward 一致性（float32） ----
    def _grads(use_cuda):
        i = (torch.rand(N, H_in, W_in, M * D).cuda() * 0.01).detach().requires_grad_(True)
        o = (torch.rand(N, H_out, W_out, M * P * 2).cuda() * 10).detach().requires_grad_(True)
        m = (torch.rand(N, H_out, W_out, M, P).cuda() + 1e-5)
        m = (m / m.sum(-1, keepdim=True)).reshape(N, H_out, W_out, M * P).detach().requires_grad_(True)
        if use_cuda:
            out = DCNv3Function.apply(i, o, m, Kh, Kw, stride, stride, Kh // 2, Kw // 2,
                                      dilation, dilation, M, D, offset_scale, 2)
        else:
            out = dcnv3_core_pytorch(i, o, m, Kh, Kw, stride, stride, Kh // 2, Kw // 2,
                                     dilation, dilation, M, D, offset_scale)
        out.sum().backward()
        return i.grad, o.grad, m.grad

    torch.manual_seed(3)
    gi_r, go_r, gm_r = _grads(False)
    torch.manual_seed(3)
    gi_c, go_c, gm_c = _grads(True)
    for label, a, b in (("d(input)", gi_r, gi_c), ("d(offset)", go_r, go_c), ("d(mask)", gm_r, gm_c)):
        same = torch.allclose(a, b, rtol=1e-2, atol=1e-3)
        print(f"  backward {label:<10}: {'一致' if same else '不一致'} "
              f"(max_abs_err={(a - b).abs().max().item():.2e})")
        results[label] = same
    if not all(results.values()):
        raise AssertionError("DCNv3 CUDA backward 与参考实现不一致")
    return "forward+backward 数值一致"


def check_speed():
    import torch
    from Backbone.InternImage.ops_dcnv3.functions.dcnv3_func import (
        DCNv3Function,
        dcnv3_core_pytorch,
    )

    N, H_in, W_in, M, D = 2, 32, 32, 8, 8
    Kh = Kw = 3
    P = Kh * Kw
    off = torch.rand(N, H_in, W_in, M * P * 2, device="cuda") * 10
    msk = torch.rand(N, H_in, W_in, M, P, device="cuda") + 1e-5
    msk = (msk / msk.sum(-1, keepdim=True)).reshape(N, H_in, W_in, M * P)
    inp = torch.rand(N, H_in, W_in, M * D, device="cuda") * 0.01

    with torch.no_grad():
        t0 = time.perf_counter()
        for _ in range(3):
            dcnv3_core_pytorch(inp, off, msk, Kh, Kw, 1, 1, 1, 1, 1, 1, M, D, 1.0)
        torch.cuda.synchronize()
        t_ref = (time.perf_counter() - t0) / 3

        t0 = time.perf_counter()
        for _ in range(10):
            DCNv3Function.apply(inp, off, msk, Kh, Kw, 1, 1, 1, 1, 1, 1, M, D, 1.0, 128)
        torch.cuda.synchronize()
        t_cuda = (time.perf_counter() - t0) / 10
    print(f"  纯 PyTorch 参考: {t_ref * 1000:.2f} ms/次")
    print(f"  CUDA 算子      : {t_cuda * 1000:.2f} ms/次   (加速 {t_ref / t_cuda:.1f}x)")
    return f"{t_ref / t_cuda:.1f}x"


def main():
    section("SDCombo 环境自检")
    print(f"仓库根目录: {REPO_ROOT}")

    section("1. 基础信息")
    ok = step("Python 版本", check_python)
    ok &= step("第三方依赖", check_deps)

    section("2. GPU / CUDA")
    ok &= step("CUDA 可用性与自检", check_gpu)

    section("3. DCNv3 扩展")
    ok &= step("import DCNv3", check_dcnv3_import)
    if ok:
        ok &= step("DCNv3 数值一致性", check_dcnv3_math)
        step("DCNv3 性能对比", check_speed)

    section("结论")
    if ok:
        print(f"{OK} 环境配置完成，可以进行训练与推理。")
    else:
        print(f"{FAIL} 环境存在未通过项，请根据上面日志修复后重跑本脚本。")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
