"""Local web GUI to colourize with DDColor and steer the result with hints.

    python gui.py --model ../ddcolor_model/ddcolor-tiny-fp16.onnx --inputs fotos/ --out hints_out/
    # open http://127.0.0.1:8765   (remote machine: ssh -L 8765:localhost:8765 user@host)

Click on the photo to place a hint, pick the right colour, then tune tolerance /
radius / strength. Luminance is never altered; only chroma is edited.
Saving writes <out>/<name>.png (full resolution) and <out>/<name>.hints.json
(re-apply later with: python colorize.py MODEL IN OUT --hints FILE.json).
Stdlib server; binds to localhost only.
"""
import argparse, io, json, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import numpy as np
from PIL import Image
import colorize as C
import hints as H

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
LOCK = threading.Lock()


class State:
    session = None
    inputs = out = None
    names = []
    cur = None  # dict for the open image


def jpeg(arr_or_img, q=88):
    im = arr_or_img if isinstance(arr_or_img, Image.Image) else Image.fromarray(arr_or_img)
    b = io.BytesIO(); im.save(b, "JPEG", quality=q); return b.getvalue()


def open_image(name):
    if name not in State.names:
        raise ValueError("unknown image")
    img = Image.open(State.inputs / name).convert("L").convert("RGB")  # same gray input as compare.py
    w, h = img.size
    pw, ph = H.preview_size(w, h)
    prev = img.resize((pw, ph), Image.LANCZOS)
    State.cur = dict(name=name, img=img, size=(w, h), psize=(pw, ph), prev=prev,
                     Lp=C._full_L(np.asarray(prev)), ab=C.predict_ab(img, State.session))
    saved = State.out / f"{Path(name).stem}.hints.json"
    doc = json.loads(saved.read_text()) if saved.exists() else {"hints": [], "global": H.DEFAULT_GLOBAL}
    return dict(w=w, h=h, pw=pw, ph=ph, doc=H.clean_doc(doc))


def render(doc, view, show_mask):
    c = State.cur
    if view == "original":
        return jpeg(c["prev"])
    doc = H.clean_doc(doc)
    masks = H.compute_masks(c["Lp"], doc["hints"])
    out = H.compose(c["Lp"], c["ab"], masks, doc["hints"], doc["global"], c["psize"])
    if show_mask and masks:
        u = np.max(np.stack(masks), 0)[..., None]
        out = (out * (1 - 0.55 * u) + np.array([255, 0, 255]) * 0.55 * u).astype(np.uint8)
    return jpeg(out)


def export(doc):
    c = State.cur
    doc = H.clean_doc(doc)
    res = H.colorize_with_hints(c["img"], State.session, doc, ab_net=c["ab"])
    stem = Path(c["name"]).stem
    State.out.mkdir(parents=True, exist_ok=True)
    res.save(State.out / f"{stem}.png")
    (State.out / f"{stem}.hints.json").write_text(json.dumps(doc, indent=1))
    return str(State.out / f"{stem}.png")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def _send(self, code, body, ctype):
        self.send_response(code); self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        elif self.path == "/api/list":
            self._send(200, json.dumps(State.names).encode(), "application/json")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            if n > 1_000_000: raise ValueError("body too large")
            req = json.loads(self.rfile.read(n) or b"{}")
            with LOCK:
                if self.path == "/api/open":
                    self._send(200, json.dumps(open_image(req["name"])).encode(), "application/json")
                elif State.cur is None:
                    raise ValueError("no image open")
                elif self.path == "/api/render":
                    self._send(200, render(req.get("doc"), req.get("view"), bool(req.get("mask"))), "image/jpeg")
                elif self.path == "/api/export":
                    self._send(200, json.dumps({"path": export(req.get("doc"))}).encode(), "application/json")
                else:
                    self._send(404, b"not found", "text/plain")
        except Exception as e:  # prototype: report to the UI
            self._send(400, json.dumps({"error": f"{type(e).__name__}: {e}"}).encode(), "application/json")


PAGE = r"""<!doctype html><html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DDColor hints</title><style>
:root{--bg:#f4f4f2;--fg:#1c1c1c;--card:#fff;--line:#d8d8d4;--acc:#2b6cb0}
@media(prefers-color-scheme:dark){:root{--bg:#1b1b1c;--fg:#e8e8e6;--card:#262628;--line:#3a3a3d;--acc:#6aa6e8}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px system-ui,sans-serif;display:flex;height:100vh}
#main{flex:1;min-width:0;display:flex;align-items:center;justify-content:center;padding:12px;overflow:auto}
#wrap{position:relative;line-height:0}#img{max-width:100%;max-height:calc(100vh - 24px);display:block}
#ov{position:absolute;inset:0;width:100%;height:100%;cursor:crosshair}
#side{width:300px;padding:14px 16px;background:var(--card);border-left:1px solid var(--line);overflow:auto;display:flex;flex-direction:column;gap:10px}
h3{margin:6px 0 0;font-size:12px;text-transform:uppercase;letter-spacing:.06em;opacity:.7}
label{display:flex;justify-content:space-between;gap:8px;align-items:center}label span{min-width:44px;text-align:right;opacity:.7}
input[type=range]{flex:1}select,button{font:inherit;padding:6px 8px;border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--fg)}
button{cursor:pointer}button.p{background:var(--acc);color:#fff;border-color:var(--acc)}button:focus-visible,select:focus-visible,input:focus-visible{outline:2px solid var(--acc)}
#list{display:flex;flex-direction:column;gap:4px;max-height:160px;overflow:auto}
.it{display:flex;align-items:center;gap:8px;padding:4px 6px;border:1px solid var(--line);border-radius:6px;cursor:pointer}.it.sel{border-color:var(--acc);outline:1px solid var(--acc)}
.sw{width:16px;height:16px;border-radius:4px;border:1px solid var(--line)}#st{min-height:2.6em;opacity:.8;font-size:13px}
@media(max-width:700px){body{flex-direction:column;height:auto}#side{width:auto;border-left:0}}
</style></head><body>
<div id="main"><div id="wrap"><img id="img" alt="foto"><canvas id="ov"></canvas></div></div>
<div id="side">
<select id="pick" aria-label="Foto"></select>
<label><input type="checkbox" id="orig"> Ver original (gris)</label>
<label><input type="checkbox" id="mask"> Mostrar región</label>
<h3>Pistas</h3><div id="list"></div>
<label>Color <input type="color" id="col" value="#2d5fbf"></label>
<label>Tolerancia <input type="range" id="tol" min="1" max="60" value="12"><span id="tolv"></span></label>
<label>Radio <input type="range" id="rad" min="0" max="100" value="0"><span id="radv"></span></label>
<label>Suavizado <input type="range" id="fea" min="0" max="40" value="4"><span id="feav"></span></label>
<label>Fuerza <input type="range" id="str" min="0" max="100" value="90"><span id="strv"></span></label>
<div style="display:flex;gap:6px"><button id="del">Borrar pista</button><button id="clr">Borrar todas</button></div>
<h3>Global</h3>
<label>Saturación <input type="range" id="sat" min="0" max="200" value="100"><span id="satv"></span></label>
<label title="verde ↔ rojo">Sesgo a <input type="range" id="ab_a" min="-30" max="30" value="0"><span id="abav"></span></label>
<label title="azul ↔ amarillo">Sesgo b <input type="range" id="ab_b" min="-30" max="30" value="0"><span id="abbv"></span></label>
<button class="p" id="save">Guardar a resolución completa</button>
<div id="st">Elige una foto. Haz clic sobre ella para colocar una pista.</div></div>
<script>
const $=id=>document.getElementById(id);
let hints=[],sel=-1,glob={saturation:1,a_bias:0,b_bias:0},busy=false,again=false,timer=null,cur=null;
const hex2rgb=h=>[1,3,5].map(i=>parseInt(h.slice(i,i+2),16)),rgb2hex=r=>'#'+r.map(c=>c.toString(16).padStart(2,'0')).join('');
const say=t=>$('st').textContent=t;
async function post(u,b){const r=await fetch(u,{method:'POST',body:JSON.stringify(b)});if(!r.ok){let e;try{e=(await r.json()).error}catch(_){e=r.status}throw new Error(e)}return r}
function ctl(){return{rgb:hex2rgb($('col').value),tol:+$('tol').value,radius:+$('rad').value/100,feather:+$('fea').value/1000,strength:+$('str').value/100}}
function labels(){$('tolv').textContent=$('tol').value;$('radv').textContent=+$('rad').value?$('rad').value+'%':'∞';$('feav').textContent=$('fea').value;
$('strv').textContent=$('str').value+'%';$('satv').textContent=($('sat').value/100).toFixed(2);$('abav').textContent=$('ab_a').value;$('abbv').textContent=$('ab_b').value}
function loadCtl(h){$('col').value=rgb2hex(h.rgb);$('tol').value=h.tol;$('rad').value=Math.round(h.radius*100);$('fea').value=Math.round(h.feather*1000);$('str').value=Math.round(h.strength*100);labels()}
function schedule(){clearTimeout(timer);timer=setTimeout(render,120)}
async function render(){if(!cur)return;if(busy){again=true;return}busy=true;
 try{const r=await post('/api/render',{doc:{hints,global:glob},view:$('orig').checked?'original':'result',mask:$('mask').checked});
  const u=URL.createObjectURL(await r.blob());const old=$('img').src;$('img').src=u;if(old.startsWith('blob:'))setTimeout(()=>URL.revokeObjectURL(old),500)}
 catch(e){say('Error: '+e.message)}finally{busy=false;if(again){again=false;render()}}}
function overlay(){const o=$('ov'),im=$('img');o.width=im.clientWidth;o.height=im.clientHeight;const c=o.getContext('2d');c.clearRect(0,0,o.width,o.height);
 hints.forEach((h,i)=>{const x=h.x*o.width,y=h.y*o.height;c.beginPath();c.arc(x,y,7,0,7);c.fillStyle=rgb2hex(h.rgb);c.fill();c.lineWidth=i==sel?3:1.5;c.strokeStyle=i==sel?'#fff':'#000';c.stroke();c.fillStyle='#000';c.font='11px sans-serif';c.fillText(i+1,x+10,y-8)})}
function list(){const l=$('list');l.innerHTML='';hints.forEach((h,i)=>{const d=document.createElement('div');d.className='it'+(i==sel?' sel':'');
 d.innerHTML=`<div class="sw" style="background:${rgb2hex(h.rgb)}"></div>#${i+1} (${Math.round(h.x*100)}%, ${Math.round(h.y*100)}%)`;d.onclick=()=>{sel=i;loadCtl(h);list();overlay()};l.appendChild(d)});overlay()}
$('ov').onclick=e=>{if(!cur)return;const r=$('ov').getBoundingClientRect();hints.push({x:(e.clientX-r.left)/r.width,y:(e.clientY-r.top)/r.height,...ctl()});sel=hints.length-1;list();schedule()};
['col','tol','rad','fea','str'].forEach(i=>$(i).oninput=()=>{labels();if(sel>=0){Object.assign(hints[sel],ctl());list();schedule()}});
$('sat').oninput=$('ab_a').oninput=$('ab_b').oninput=()=>{glob={saturation:$('sat').value/100,a_bias:+$('ab_a').value,b_bias:+$('ab_b').value};labels();schedule()};
$('orig').onchange=$('mask').onchange=render;
$('del').onclick=()=>{if(sel>=0){hints.splice(sel,1);sel=hints.length-1;list();schedule()}};
$('clr').onclick=()=>{hints=[];sel=-1;list();schedule()};
$('img').onload=overlay;window.onresize=overlay;
$('save').onclick=async()=>{if(!cur)return;say('Guardando (resolución completa)…');try{const r=await(await post('/api/export',{doc:{hints,global:glob}})).json();say('Guardado: '+r.path)}catch(e){say('Error: '+e.message)}};
async function openImg(n){say('Procesando '+n+'…');try{cur=await(await post('/api/open',{name:n})).json();hints=cur.doc.hints;glob=cur.doc.global;sel=hints.length-1;
 $('sat').value=glob.saturation*100;$('ab_a').value=glob.a_bias;$('ab_b').value=glob.b_bias;labels();list();await render();say(`${n} (${cur.w}×${cur.h}). Clic en la foto para añadir una pista.`)}catch(e){say('Error: '+e.message)}}
$('pick').onchange=()=>openImg($('pick').value);
fetch('/api/list').then(r=>r.json()).then(ns=>{ns.forEach(n=>{const o=document.createElement('option');o.textContent=n;$('pick').appendChild(o)});labels();if(ns.length)openImg(ns[0]);else say('No hay imágenes en la carpeta.')});
</script></body></html>"""


def main():
    import onnxruntime as ort
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--inputs", required=True)
    ap.add_argument("--out", default="hints_out"); ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--gpu", action="store_true",
                    help="CUDA (off by default: wrong colours seen on GB10 with the fp16 model)")
    a = ap.parse_args()
    so = ort.SessionOptions(); so.log_severity_level = 3
    State.session = ort.InferenceSession(a.model, so, providers=C._providers(a.gpu))
    State.inputs, State.out = Path(a.inputs), Path(a.out)
    State.names = sorted(p.name for p in State.inputs.iterdir() if p.suffix.lower() in EXTS)
    print("providers:", State.session.get_providers(), "| images:", len(State.names))
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"http://127.0.0.1:{a.port}  (Ctrl+C to stop)")
    srv.serve_forever()


if __name__ == "__main__":
    main()
