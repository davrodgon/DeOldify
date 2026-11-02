"""Export official DDColor weights to ONNX. UNTESTED here (no network access to
the weights in the authoring sandbox). Run inside a clone of
https://github.com/piddnad/DDColor with its requirements installed.

    python export_onnx.py --weights modelscope/damo/cv_ddcolor_image-colorization/pytorch_model.pt \
        --out ddcolor.onnx

The upstream repo also documents its own ONNX export; prefer it if this
script's model construction drifts from upstream.
"""
import argparse, torch
from basicsr.archs.ddcolor_arch import DDColor  # from the DDColor repo

p = argparse.ArgumentParser()
p.add_argument("--weights", required=True)
p.add_argument("--out", default="ddcolor.onnx")
p.add_argument("--size", type=int, default=512)
a = p.parse_args()

# Constructor args follow upstream's inference code for the paper_tiny model; check there.
model = DDColor(encoder_name="convnext-t", decoder_name="MultiScaleColorDecoder",
                input_size=[a.size, a.size], num_output_channels=2, last_norm="Spectral",
                do_normalize=False, num_queries=100, num_scales=3,
                dec_layers=9).eval()
sd = torch.load(a.weights, map_location="cpu")
model.load_state_dict(sd.get("params", sd), strict=False)
torch.onnx.export(model, torch.zeros(1, 3, a.size, a.size), a.out,
                  input_names=["x"], output_names=["ab"], opset_version=17)
print("wrote", a.out)
