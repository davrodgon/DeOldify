"""Verifies the pre/post-processing with a stand-in ONNX model (NOT DDColor).
The stand-in outputs constant ab = (+30, -20), so the result must keep the
original L exactly and show a uniform chroma shift."""
import numpy as np, onnx, onnxruntime as ort
from onnx import helper, TensorProto as T
from PIL import Image
from colorize import colorize, rgb2lab, lab2rgb

S = 512
ab = np.stack([np.full((S, S), 30.), np.full((S, S), -20.)])[None].astype(np.float32)
init = [onnx.numpy_helper.from_array(ab, "ab"),
        onnx.numpy_helper.from_array(np.float32(0), "zero")] + [
    helper.make_tensor(n, T.INT64, [1], v) for n, v in (("s", [0]), ("e", [2]), ("ax", [1]))]
nodes = [helper.make_node("Mul", ["x", "zero"], ["xz"]),          # keeps the input used
         helper.make_node("Slice", ["xz", "s", "e", "ax"], ["z"]),  # (1,2,S,S) zeros
         helper.make_node("Add", ["z", "ab"], ["out"])]
g = helper.make_graph(nodes, "stub",
                      [helper.make_tensor_value_info("x", T.FLOAT, [1, 3, S, S])],
                      [helper.make_tensor_value_info("out", T.FLOAT, [1, 2, S, S])], init)
m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)], ir_version=10)
onnx.checker.check_model(m)
sess = ort.InferenceSession(m.SerializeToString())

# round-trip colour conversion
rng = np.random.default_rng(0)
x = rng.random((64, 64, 3))
assert np.abs(lab2rgb(rgb2lab(x)) - x).max() < 1e-6

# non-square gray gradient image
g_img = np.tile(np.linspace(40, 215, 300, dtype=np.uint8), (200, 1))
src = Image.fromarray(g_img).convert("RGB")
out = colorize(src, sess)
assert out.size == src.size
L_in = rgb2lab(np.asarray(src) / 255.)[..., 0]
lab_out = rgb2lab(np.asarray(out) / 255.)
print("max |dL|:", np.abs(lab_out[..., 0] - L_in).max(),
      "mean ab:", lab_out[..., 1].mean(), lab_out[..., 2].mean())
assert np.abs(lab_out[..., 0] - L_in).max() < 1.5   # luminance preserved (gamut clipping slack)
assert lab_out[..., 1].mean() > 10 and lab_out[..., 2].mean() < -5

# fast path vs float64 reference on a random colour image (reference: full Lab pipeline)
import colorize as C
rgbimg = Image.fromarray((rng.random((180, 260, 3)) * 255).astype(np.uint8))
fast = np.asarray(colorize(rgbimg, sess)).astype(int)
arr = np.asarray(rgbimg) / 255.
Lref = rgb2lab(arr)[..., 0]
abf = np.stack([np.full(Lref.shape, 30.), np.full(Lref.shape, -20.)], -1)
ref = (lab2rgb(np.concatenate([Lref[..., None], abf], -1)) * 255).round().astype(int)
print("fast vs reference max diff (0-255):", np.abs(fast - ref).max())
assert np.abs(fast - ref).max() <= 2
print("OK")
