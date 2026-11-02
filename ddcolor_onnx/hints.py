"""User hints for DDColor output: recolour regions / adjust global chroma.

Luminance is never touched; hints only edit the predicted chroma (ab).
A hint document (JSON) looks like:
  {"hints": [{"x": 0.42, "y": 0.31,          # click position, normalised 0..1
              "tol": 12,                     # luminance tolerance (L units, 0..100)
              "radius": 0.0,                 # max distance from the click, fraction of long side (0 = unlimited)
              "feather": 0.004,              # edge softness, fraction of long side
              "rgb": [40, 90, 200],          # target colour
              "strength": 0.9}],             # 0..1 mix with the model's colour
   "global": {"saturation": 1.0, "a_bias": 0, "b_bias": 0}}
Regions are computed on a <=1000px preview and upscaled to the output size.
"""
import numpy as np
from PIL import Image, ImageFilter
import colorize as C

PREVIEW_MAX = 1000
DEFAULT_GLOBAL = {"saturation": 1.0, "a_bias": 0.0, "b_bias": 0.0}


def preview_size(w, h, m=PREVIEW_MAX):
    s = min(1.0, m / max(w, h))
    return max(1, round(w * s)), max(1, round(h * s))


def clean_doc(doc):
    """Validate/clamp untrusted input (comes from the browser)."""
    f = lambda v, lo, hi, d: float(min(hi, max(lo, v))) if isinstance(v, (int, float)) else d
    hints = []
    for h in (doc or {}).get("hints", [])[:200]:
        rgb = h.get("rgb", [128, 128, 128])
        rgb = [int(min(255, max(0, c))) if isinstance(c, (int, float)) else 128 for c in rgb[:3]]
        hints.append({"x": f(h.get("x"), 0, 1, 0.5), "y": f(h.get("y"), 0, 1, 0.5),
                      "tol": f(h.get("tol"), 0, 100, 12), "radius": f(h.get("radius"), 0, 2, 0),
                      "feather": f(h.get("feather"), 0, 0.1, 0.004), "rgb": rgb,
                      "strength": f(h.get("strength"), 0, 1, 1)})
    g = (doc or {}).get("global", {})
    glob = {"saturation": f(g.get("saturation"), 0, 4, 1.0), "a_bias": f(g.get("a_bias"), -60, 60, 0.0),
            "b_bias": f(g.get("b_bias"), -60, 60, 0.0)}
    return {"hints": hints, "global": glob}


def _component(cand, seed):
    try:
        from scipy.ndimage import label
        lab, _ = label(cand)
        return lab == lab[seed]
    except ImportError:  # numpy fallback: slower on big regions
        m = np.zeros_like(cand); m[seed] = True
        n = 1
        while True:
            d = m.copy()
            d[1:] |= m[:-1]; d[:-1] |= m[1:]; d[:, 1:] |= m[:, :-1]; d[:, :-1] |= m[:, 1:]
            d &= cand
            k = int(d.sum())
            if k == n:
                return d
            m, n = d, k


def region_mask(Lp, hint):
    """Soft mask (float32, preview res): connected area around the click with similar luminance."""
    h, w = Lp.shape
    xi = int(np.clip(round(hint["x"] * (w - 1)), 0, w - 1))
    yi = int(np.clip(round(hint["y"] * (h - 1)), 0, h - 1))
    seed_L = float(np.median(Lp[max(0, yi - 2):yi + 3, max(0, xi - 2):xi + 3]))
    cand = np.abs(Lp - seed_L) <= hint["tol"]
    if hint["radius"] > 0:
        r = hint["radius"] * max(w, h)
        yy, xx = np.ogrid[:h, :w]
        cand &= (yy - yi) ** 2 + (xx - xi) ** 2 <= r * r
    cand[yi, xi] = True
    m = _component(cand, (yi, xi)).astype(np.uint8) * 255
    im = Image.fromarray(m)
    fe = hint["feather"] * max(w, h)
    if fe > 0.5:
        im = im.filter(ImageFilter.GaussianBlur(fe))
    return np.asarray(im, dtype=np.float32) / 255.0


def compute_masks(Lp, hints):
    return [region_mask(Lp, h) for h in hints]


def resize_mask(m, size):
    return np.asarray(Image.fromarray(m, "F").resize(size, Image.BILINEAR))


def edit_ab(ab, masks, hints, glob):
    """ab (h,w,2) float32. Global tweak on the model's chroma first, then exact local hints."""
    ab = ab * glob["saturation"] + np.array([glob["a_bias"], glob["b_bias"]], np.float32)
    for m, h in zip(masks, hints):
        tgt = C.rgb2lab(np.array([[h["rgb"]]], dtype=np.float64) / 255.0)[0, 0, 1:].astype(np.float32)
        k = (m * h["strength"])[..., None]
        ab = ab * (1 - k) + tgt * k
    return ab


def compose(L, ab_net, masks, hints, glob, size):
    """L float32 (h,w) at `size`=(w,h); ab_net raw (2,S,S); masks already at `size`. Returns uint8 RGB."""
    ab = np.stack([C._resize(ab_net[0], size), C._resize(ab_net[1], size)], -1).astype(np.float32)
    ab = edit_ab(ab, masks, hints, glob)
    return C._lab_to_u8(L, ab[..., 0], ab[..., 1])


def preview_L(img):
    w, h = img.size
    pw, ph = preview_size(w, h)
    return C._full_L(np.asarray(img.convert("RGB").resize((pw, ph), Image.LANCZOS)))


def colorize_with_hints(img, session, doc, size=512, ab_net=None):
    """Full-resolution result with hints applied (used by the CLI and the GUI export)."""
    doc = clean_doc(doc)
    img = img.convert("RGB")
    w, h = img.size
    if ab_net is None:
        ab_net = C.predict_ab(img, session, size)
    L = C._full_L(np.asarray(img))
    masks = [resize_mask(m, (w, h)) for m in compute_masks(preview_L(img), doc["hints"])]
    return Image.fromarray(compose(L, ab_net, masks, doc["hints"], doc["global"], (w, h)))
