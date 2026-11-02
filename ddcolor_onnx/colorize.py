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
import argparse
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


def colorize(img: Image.Image, session, size=512) -> Image.Image:
    rgb = np.asarray(img.convert("RGB"), dtype=np.float64) / 255.0
    h, w = rgb.shape[:2]
    L = rgb2lab(rgb)[..., 0]

    small = np.stack([_resize(L, (size, size)),
                      np.zeros((size, size)), np.zeros((size, size))], -1)
    gray = lab2rgb(small).astype(np.float32).transpose(2, 0, 1)[None]

    name = session.get_inputs()[0].name
    ab = session.run(None, {name: gray})[0][0]  # (2,S,S)

    ab_full = np.stack([_resize(ab[0], (w, h)), _resize(ab[1], (w, h))], -1)
    out = lab2rgb(np.concatenate([L[..., None], ab_full], -1))
    return Image.fromarray((out * 255).round().astype(np.uint8))


def main():
    import onnxruntime as ort
    p = argparse.ArgumentParser()
    p.add_argument("model"), p.add_argument("input"), p.add_argument("output")
    p.add_argument("--size", type=int, default=512)
    a = p.parse_args()
    sess = ort.InferenceSession(a.model, providers=ort.get_available_providers())
    colorize(Image.open(a.input), sess, a.size).save(a.output)


if __name__ == "__main__":
    main()
