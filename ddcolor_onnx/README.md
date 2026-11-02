# DDColor + ONNX Runtime (prototype)

`colorize.py` runs a DDColor ONNX model with only numpy, Pillow and onnxruntime.
It predicts chroma (ab) only and recombines it with the original, full-resolution
luminance, so source detail is preserved.

    python colorize.py ddcolor.onnx old_photo.jpg out.jpg

## Status
- Verified: Lab conversion round-trip, resizing, luminance preservation, non-square
  inputs (`python test_pipeline.py`, using a stand-in ONNX model).
- NOT verified: real DDColor weights. They could not be downloaded in the sandbox
  (huggingface.co, github.com and modelscope.cn were blocked by policy).
- Assumed model contract (check against your file): input `(1,3,512,512)` float32
  RGB in [0,1]; output `(1,2,512,512)` ab in Lab units. If the model expects ImageNet
  normalisation or outputs a different range, adjust `colorize()`.
- `export_onnx.py` is untested; alternatively use a prebuilt export such as
  https://huggingface.co/edgetools/ddcolor (confirm its license).

## Next steps
Run on real weights vs. DeOldify on your own photo set; tile or sliding-window for
large images; port to C# with Microsoft.ML.OnnxRuntime if desired.
