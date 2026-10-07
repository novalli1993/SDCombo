"""提取论文 PDF 的文本，便于核对模型设计意图。"""
import os
import sys

from pypdf import PdfReader

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pdf = os.path.join(ROOT, "SDCombo Semantic Segmentation with Depth Information.pdf")
out = os.path.join(ROOT, "work_dir", "paper.txt")
os.makedirs(os.path.dirname(out), exist_ok=True)

r = PdfReader(pdf)
print(f"页数: {len(r.pages)}")
parts = []
for i, pg in enumerate(r.pages):
    t = pg.extract_text() or ""
    parts.append(f"\n\n========== PAGE {i + 1} ==========\n{t}")
    print(f"  第 {i+1} 页: {len(t)} 字符")
text = "".join(parts)
with open(out, "w", encoding="utf-8") as f:
    f.write(text)
print(f"\n已写入 {out}  ({len(text)} 字符)")
