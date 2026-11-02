"""Compare DDColor ONNX outputs across execution configs on ONE image.
   python diag.py ../ddcolor_model/ddcolor-tiny-fp16.onnx foto.jpg [outdir]
Prints per-config ab stats, GPU repeatability, and difference vs the CPU reference,
and saves each result as PNG so you can look at them.
"""
import sys
from pathlib import Path
import numpy as np
import onnxruntime as ort
from PIL import Image
import colorize as C

model, photo = sys.argv[1], sys.argv[2]
out = Path(sys.argv[3] if len(sys.argv) > 3 else "diag_out"); out.mkdir(exist_ok=True)
img = Image.open(photo).convert("L").convert("RGB")


def make(providers, opt=None):
    so = ort.SessionOptions(); so.log_severity_level = 3
    if opt == "off":
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(model, so, providers=providers)


def raw_ab(sess, size=512):
    """Same network input as colorize(), returns the raw (2,S,S) ab."""
    small = np.asarray(img.resize((size, size), Image.BICUBIC), dtype=np.float64) / 255.0
    lab = C.rgb2lab(small); lab[..., 1:] = 0
    x = C.lab2rgb(lab).astype(np.float32).transpose(2, 0, 1)[None]
    inp = sess.get_inputs()[0]
    dt = np.float16 if "float16" in inp.type else np.float32
    return sess.run(None, {inp.name: x.astype(dt)})[0][0].astype(np.float32)


def stats(name, ab):
    print(f"{name:28s} mean a={ab[0].mean():7.2f} b={ab[1].mean():7.2f}  "
          f"|ab|={np.abs(ab).mean():6.2f}  std a={ab[0].std():6.2f} b={ab[1].std():6.2f}  "
          f"nan={np.isnan(ab).any()}")


configs = {
    "cpu": make(["CPUExecutionProvider"]),
    "cuda": make(["CUDAExecutionProvider", "CPUExecutionProvider"]),
    "cuda_noopt": make(["CUDAExecutionProvider", "CPUExecutionProvider"], "off"),
    "cpu_noopt": make(["CPUExecutionProvider"], "off"),
}
res = {}
for n, s in configs.items():
    res[n] = raw_ab(s); stats(n, res[n])
    C.colorize(img, s).save(out / f"{Path(photo).stem}_{n}.png")

print("\nGPU repeatability (same session, 3 runs):")
base = raw_ab(configs["cuda"])
for i in range(3):
    r = raw_ab(configs["cuda"]); print(f"  run {i}: max|diff vs first| = {np.abs(r - base).max():.3f}")

print("\nDifference vs cpu reference (max |dab|, correlation):")
for n in ("cuda", "cuda_noopt", "cpu_noopt"):
    d = np.abs(res[n] - res["cpu"]).max()
    c = np.corrcoef(res[n].ravel(), res["cpu"].ravel())[0, 1]
    print(f"  {n:12s} max|d|={d:7.2f}  corr={c:.4f}")
print("\nSaved PNGs in", out)
