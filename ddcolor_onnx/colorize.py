"""Prototype DDColor inference with ONNX Runtime (numpy + Pillow only).

Pipeline (mirrors the official DDColor inference):
  RGB -> Lab; keep L at full resolution.
  Build a 512x512 grayscale RGB input from L (a=b=0), run the model,
  get predicted ab (1,2,512,512), resize ab to the original size,
  recombine with the ORIGINAL L, convert back to RGB.
Because L is never touched, structure/detail of the source is preserved.

ASSUMPTIONS (verify against the real ONNX file):
  * input  (1,3,S,S) float32, RGB in [0,1], S = 512
  * output (1,2,S,S) float32, Lab a/b channels in Lab units (~[-128,127])
"""
import argparse, time
import numpy as np
from PIL import Image

_M = np.array([[0.4124564, 0.3575761, 0.1804375],
               [0.2126729, 0.7151522, 0.0721750],
               [0.0193339, 0.1191920, 0.9503041]])
_WHITE = np.array([0.95047, 1.0, 1.08883])  # D65


def rgb2lab(rgb):
    c = np.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    xyz = (c @ _M.T) / _WHITE
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    return np.stack([116 * f[..., 1] - 16,
                     500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])], -1)


def lab2rgb(lab):
    fy = (lab[..., 0] + 16) / 116
    fx, fz = fy + lab[..., 1] / 500, fy - lab[..., 2] / 200
    f = np.stack([fx, fy, fz], -1)
    xyz = np.where(f ** 3 > 216 / 24389, f ** 3, (116 * f - 16) / (24389 / 27)) * _WHITE
    c = np.clip(xyz @ np.linalg.inv(_M).T, 0, 1)
    return np.where(c > 0.0031308, 1.055 * c ** (1 / 2.4) - 0.055, 12.92 * c)


def _resize(arr2d, size):  # float32 HxW -> size=(w,h)
    return np.asarray(Image.fromarray(arr2d.astype(np.float32), "F")
                      .resize(size, Image.BICUBIC))


# --- fast full-resolution path: LUTs + float32 (the network only sees 512x512) ---
_DEC = np.arange(256) / 255.0
_DEC = np.where(_DEC > 0.04045, ((_DEC + 0.055) / 1.055) ** 2.4, _DEC / 12.92).astype(np.float32)
_NENC = 16384
_x = np.linspace(0, 1, _NENC)
_ENC = np.round(np.where(_x > 0.0031308, 1.055 * _x ** (1 / 2.4) - 0.055, 12.92 * _x) * 255).astype(np.uint8)
_YROW = _M[1].astype(np.float32)
_MINV = np.linalg.inv(_M).T.astype(np.float32)
_WH = _WHITE.astype(np.float32)
_EPS, _KAP = 216 / 24389, 24389 / 27


def _full_L(rgb_u8):
    """L channel (0..100) of an HxWx3 uint8 image, float32, without the full Lab transform."""
    y = _DEC[rgb_u8] @ _YROW
    f = np.where(y > _EPS, np.cbrt(y), (_KAP * y + 16) / 116).astype(np.float32)
    return 116 * f - 16


def _lab_to_u8(L, a, b):
    fy = (L + 16) / 116
    f = np.stack([fy + a / 500, fy, fy - b / 200], -1)
    f3 = f * f * f
    xyz = np.where(f3 > _EPS, f3, (116 * f - 16) / _KAP) * _WH
    lin = np.clip(xyz @ _MINV, 0, 1)
    return _ENC[(lin * (_NENC - 1) + 0.5).astype(np.int32)]


def colorize(img: Image.Image, session, size=512, timings=None) -> Image.Image:
    t0 = time.perf_counter()
    img = img.convert("RGB")
    w, h = img.size
    L = _full_L(np.asarray(img))                       # full-res luminance (kept untouched)

    # network input: L of the downscaled image, rebuilt as grayscale RGB (a=b=0)
    small = np.asarray(img.resize((size, size), Image.BICUBIC), dtype=np.float64) / 255.0
    lab_s = rgb2lab(small)
    lab_s[..., 1:] = 0
    gray = lab2rgb(lab_s).astype(np.float32).transpose(2, 0, 1)[None]

    t1 = time.perf_counter()
    inp = session.get_inputs()[0]
    dtype = np.float16 if "float16" in inp.type else np.float32  # fp16 exports need fp16 input
    ab = session.run(None, {inp.name: gray.astype(dtype)})[0][0].astype(np.float32)  # (2,S,S)

    t2 = time.perf_counter()
    a = _resize(ab[0], (w, h))
    b = _resize(ab[1], (w, h))
    res = Image.fromarray(_lab_to_u8(L, a, b))
    if timings is not None:
        timings.update(pre=t1 - t0, infer=t2 - t1, post=time.perf_counter() - t2,
                       ab_abs=float(np.abs(ab).mean()), nan=bool(np.isnan(ab).any()))
    return res


def _providers(gpu):
    import onnxruntime as ort
    return ort.get_available_providers() if gpu else ["CPUExecutionProvider"]


def main():
    import onnxruntime as ort
    p = argparse.ArgumentParser()
    p.add_argument("model"), p.add_argument("input"), p.add_argument("output")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--gpu", action="store_true",
                   help="use CUDA if available. OFF by default: on GB10/Blackwell the fp16 model "
                        "gave wrong colours on CUDA (greenish, red patches) but correct on CPU")
    a = p.parse_args()
    so = ort.SessionOptions()
    so.log_severity_level = 3  # errors only: fp16 weights trigger harmless constant-folding warnings on CPU
    sess = ort.InferenceSession(a.model, so, providers=_providers(a.gpu))
    colorize(Image.open(a.input), sess, a.size).save(a.output)


if __name__ == "__main__":
    main()
