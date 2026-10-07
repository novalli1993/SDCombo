"""验证 DCNv3 CUDA 扩展：导入 + forward/backward 与纯 PyTorch 参考实现一致性。"""
import sys
import os

import torch

print("torch:", torch.__version__, "| cuda:", torch.cuda.is_available())
print("cwd:", os.getcwd())

try:
    import DCNv3
    print("import DCNv3  -> OK:", DCNv3.__file__)
except ImportError as exc:
    print("import DCNv3  -> FAILED:", exc)
    here = os.path.dirname(os.path.abspath(__file__))
    ops = os.path.join(here, "..", "Segmentation", "Models", "InternImage", "ops_dcnv3")
    print("ops_dcnv3 on sys.path:", os.path.abspath(ops) in sys.path)
    raise SystemExit(1)

# ---- 数值一致性 ----
H_in, W_in = 8, 8
N, M, D = 2, 4, 16
Kh = Kw = 3
P = Kh * Kw
pad = dilation = stride = 1
offset_scale = 2.0
H_out = (H_in + 2 * pad - (dilation * (Kh - 1) + 1)) // stride + 1
W_out = (W_in + 2 * pad - (dilation * (Kw - 1) + 1)) // stride + 1

from Segmentation.Models.InternImage.ops_dcnv3.functions.dcnv3_func import (  # noqa: E402
    DCNv3Function,
    dcnv3_core_pytorch,
)

torch.manual_seed(3)
with torch.no_grad():
    inp = torch.rand(N, H_in, W_in, M * D).cuda() * 0.01
    off = torch.rand(N, H_out, W_out, M * P * 2).cuda() * 10
    msk = torch.rand(N, H_out, W_out, M, P).cuda() + 1e-5
    msk = (msk / msk.sum(-1, keepdim=True)).reshape(N, H_out, W_out, M * P)

    ref = dcnv3_core_pytorch(inp, off, msk, Kh, Kw, stride, stride,
                             Kh // 2, Kw // 2, dilation, dilation, M, D, offset_scale)
    out = DCNv3Function.apply(inp, off, msk, Kh, Kw, stride, stride,
                              Kh // 2, Kw // 2, dilation, dilation, M, D, offset_scale, 2)
print(f"forward  max_abs_err = {(out - ref).abs().max().item():.3e}  "
      f"allclose = {torch.allclose(out, ref, rtol=1e-2, atol=1e-3)}")


def grads(use_cuda):
    torch.manual_seed(11)
    i = (torch.rand(N, H_in, W_in, M * D).cuda() * 0.01).requires_grad_(True)
    o = (torch.rand(N, H_out, W_out, M * P * 2).cuda() * 10).requires_grad_(True)
    m = torch.rand(N, H_out, W_out, M, P).cuda() + 1e-5
    m = (m / m.sum(-1, keepdim=True)).reshape(N, H_out, W_out, M * P).requires_grad_(True)
    if use_cuda:
        y = DCNv3Function.apply(i, o, m, Kh, Kw, stride, stride, Kh // 2, Kw // 2,
                                dilation, dilation, M, D, offset_scale, 2)
    else:
        y = dcnv3_core_pytorch(i, o, m, Kh, Kw, stride, stride, Kh // 2, Kw // 2,
                               dilation, dilation, M, D, offset_scale)
    y.sum().backward()
    return i.grad, o.grad, m.grad


gr, go, gm = grads(False)
gc, goc, gmc = grads(True)
for name, a, b in (("d(input) ", gr, gc), ("d(offset)", go, goc), ("d(mask)  ", gm, gmc)):
    print(f"backward {name} max_abs_err = {(a - b).abs().max().item():.3e}  "
          f"allclose = {torch.allclose(a, b, rtol=1e-2, atol=1e-3)}")

# ---- 端到端：真机跑一次 DCNv3 模块（含 dw_conv / offset / mask 线性层） ----
from Segmentation.Models.InternImage.ops_dcnv3.modules import DCNv3  # noqa: E402

mod = DCNv3(channels=64, kernel_size=3, group=4).cuda()
x = torch.randn(2, 32, 32, 64, device="cuda", requires_grad=True)
y = mod(x)
y.sum().backward()
print(f"DCNv3 module on {torch.cuda.get_device_name(0)}: out={tuple(y.shape)} "
      f"grad_ok={x.grad is not None and torch.isfinite(x.grad).all().item()}")
print("DCNv3 VERIFY OK")
