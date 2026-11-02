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

## Hints GUI (prototype)
Steer the colours of a photo by hand. Luminance is never altered; only chroma (ab) is edited.

    python gui.py --model ../ddcolor_model/ddcolor-tiny-fp16.onnx --inputs fotos/ --out hints_out/
    # open http://127.0.0.1:8765   (remote host: ssh -L 8765:localhost:8765 user@host)

Click on the photo to drop a hint, choose the colour, then tune tolerance (luminance
similarity), radius, softness and strength; "Mostrar región" highlights the affected area.
Global saturation / a / b sliders correct overall casts. "Guardar" writes `<name>.png` at
full resolution and `<name>.hints.json`; reproduce later with
`python colorize.py MODEL in.jpg out.png --hints name.hints.json`.

Status: flow tested in a headless browser with a stand-in model (region selection, recolour,
save, CLI reproduces the saved result exactly). NOT tested with real DDColor output or real
photos. Regions are computed on a <=1000px preview and upscaled; without scipy the region
search falls back to a slower numpy loop (install scipy for big regions). CPU by default;
`--gpu` is opt-in because CUDA gave wrong colours with the fp16 model on GB10.
