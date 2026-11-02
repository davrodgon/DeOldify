"""Batch comparison: DDColor-ONNX vs DeOldify (artistic / stable).

The engines usually need DIFFERENT Python environments (DeOldify: fastai 1.0.60,
PyTorch 1.11; DDColor: onnxruntime only), so each runs as its own stage and writes
to a shared output dir. The montage stage only needs Pillow.

  python compare.py ddcolor --model ../ddcolor_model/ddcolor-tiny-fp16.onnx --inputs fotos/ --out cmp/
  python compare.py deoldify --inputs fotos/ --out cmp/ --root ..   # --root = folder with models/*.pth
  python compare.py montage  --inputs fotos/ --out cmp/

Fairness: every input is first converted to grayscale RGB, so colour photos
can be used as ground truth. Timing excludes the first image (warm-up/cudnn).
Results: cmp/<engine>/<name>.png, cmp/timings_<engine>.csv, cmp/montage/<name>.jpg
"""
import argparse, csv, sys, time
from pathlib import Path
from PIL import Image, ImageDraw

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(d):
    return sorted(p for p in Path(d).iterdir() if p.suffix.lower() in EXTS)


def to_gray(path):
    return Image.open(path).convert("L").convert("RGB")


def write_timings(out, engine, rows):
    with open(Path(out) / f"timings_{engine}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "width", "height", "seconds"])
        w.writerows(rows)
    warm = [r[3] for r in rows[1:]]  # skip first (warm-up)
    if warm:
        print(f"{engine}: {len(rows)} images, mean {sum(warm)/len(warm):.2f}s/img "
              f"(excluding first)")


def run_ddcolor(a):
    import onnxruntime as ort
    sys.path.insert(0, str(Path(__file__).parent))
    from colorize import colorize
    so = ort.SessionOptions()
    so.log_severity_level = 3
    prov = ort.get_available_providers() if a.gpu else ["CPUExecutionProvider"]
    sess = ort.InferenceSession(a.model, so, providers=prov)
    print("providers:", sess.get_providers())
    out = Path(a.out) / "ddcolor"; out.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in list_images(a.inputs):
        img = to_gray(p)
        t = time.perf_counter()
        tm = {}
        res = colorize(img, sess, timings=tm)
        dt = time.perf_counter() - t
        res.save(out / f"{p.stem}.png")
        rows.append([p.name, *img.size, round(dt, 3)])
        print(f"{p.name} {img.width}x{img.height}: {dt:.2f}s "
              f"(pre {tm['pre']:.2f} infer {tm['infer']:.2f} post {tm['post']:.2f}) "
              f"mean|ab|={tm['ab_abs']:.2f}{' NaN!' if tm['nan'] else ''}")
    write_timings(a.out, "ddcolor", rows)


def run_deoldify(a):
    from deoldify import device                      # import FIRST (see _device.py)
    from deoldify.device_id import DeviceId
    device.set(DeviceId.CPU if a.cpu else DeviceId.GPU0)
    from deoldify.visualize import get_image_colorizer
    name = "deoldify_" + a.variant
    vis = get_image_colorizer(root_folder=Path(a.root), artistic=a.variant == "artistic",
                              render_factor=a.render_factor)
    out = Path(a.out) / name; out.mkdir(parents=True, exist_ok=True)
    tmp = Path(a.out) / "_gray_tmp"; tmp.mkdir(exist_ok=True)
    rows = []
    for p in list_images(a.inputs):
        g = tmp / f"{p.stem}.png"
        img = to_gray(p); img.save(g)
        t = time.perf_counter()
        res = vis.get_transformed_image(g, render_factor=a.render_factor, watermarked=False)
        dt = time.perf_counter() - t
        res.save(out / f"{p.stem}.png")
        rows.append([p.name, *img.size, round(dt, 3)])
        print(f"{p.name}: {dt:.2f}s")
    write_timings(a.out, name, rows)


def run_montage(a):
    out = Path(a.out)
    engines = [d.name for d in sorted(out.iterdir())
               if d.is_dir() and d.name not in ("montage", "_gray_tmp")]
    (out / "montage").mkdir(exist_ok=True)
    H = a.height
    for p in list_images(a.inputs):
        panels = [("original", Image.open(p).convert("RGB")), ("gray", to_gray(p))]
        for e in engines:
            f = out / e / f"{p.stem}.png"
            if f.exists():
                panels.append((e, Image.open(f).convert("RGB")))
        tiles = []
        for label, im in panels:
            im = im.resize((max(1, round(im.width * H / im.height)), H), Image.LANCZOS)
            ImageDraw.Draw(im).text((8, 8), label, fill=(255, 255, 0))
            tiles.append(im)
        sheet = Image.new("RGB", (sum(t.width for t in tiles), H))
        x = 0
        for t in tiles:
            sheet.paste(t, (x, 0)); x += t.width
        sheet.save(out / "montage" / f"{p.stem}.jpg", quality=92)
    print("montages in", out / "montage")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("ddcolor", "deoldify", "montage"):
        s = sub.add_parser(n)
        s.add_argument("--inputs", required=True)
        s.add_argument("--out", required=True)
    sub.choices["ddcolor"].add_argument("--model", required=True)
    sub.choices["ddcolor"].add_argument("--gpu", action="store_true",
                 help="use CUDA. Off by default: wrong colours seen on GB10 with the fp16 model")
    d = sub.choices["deoldify"]
    d.add_argument("--root", default=".", help="folder containing models/*.pth")
    d.add_argument("--variant", choices=["artistic", "stable"], default="artistic")
    d.add_argument("--render-factor", type=int, default=35)
    d.add_argument("--cpu", action="store_true")
    sub.choices["montage"].add_argument("--height", type=int, default=512)
    a = ap.parse_args()
    {"ddcolor": run_ddcolor, "deoldify": run_deoldify, "montage": run_montage}[a.cmd](a)


if __name__ == "__main__":
    main()
