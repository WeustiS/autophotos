"""Static website export — the "galaxy" view.

Single self-contained gallery.html built from cached embeddings:
  - 2D layout from PCA (no model at runtime)
  - per-photo aesthetic via the numpy LAION head
  - named semantic axes (cold->warm, nature->urban, ...) when a CLIP text tower
    is available; PCA axes always present
  - precomputed k-NN for click-to-similar; thumbnails inlined as base64

    python -m autophotos.export "C:/path/to/library"
"""
from __future__ import annotations
import base64, glob, io, json, os

import numpy as np
from PIL import Image

from . import config
from .hashing import content_hash
from .score import AestheticHead

NAMED_AXES = {
    "cold->warm": (["a warm photo", "sunset", "orange tones"],
                   ["a cold photo", "blue tones", "overcast"]),
    "nature->urban": (["city", "buildings", "concrete"],
                      ["nature", "landscape", "wilderness"]),
    "calm->busy": (["busy", "crowded", "cluttered scene"],
                   ["minimal", "calm", "empty scene"]),
    "wide->close": (["extreme close-up", "macro detail"],
                    ["wide landscape", "distant view"]),
}


def _n01(v):
    lo, hi = float(v.min()), float(v.max())
    return (v - lo) / (hi - lo + 1e-8)


def _unit(v):
    return v / (np.linalg.norm(v) + 1e-8)


def build(lib_path: str, thumb_px: int = 96, knn: int = 10) -> tuple[str, int]:
    lib = config.Library(lib_path)
    ids = json.load(open(lib.ids_path))
    emb = np.load(lib.emb_path).astype(np.float32)
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    meta = {}
    try:
        if os.path.exists(lib.model_path):
            meta = json.load(open(lib.model_path))
    except Exception:
        meta = {}

    fn = {}
    for f in glob.glob(os.path.join(lib.root, "*")):
        if os.path.splitext(f)[1].lower() in config.RAW_EXTS:
            try:
                fn[content_hash(f)] = os.path.basename(f)
            except OSError:
                pass

    x = emb - emb.mean(0)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    pcs = np.stack([_n01((x @ vt[i]).astype(np.float32)) for i in range(3)], 1)

    head = AestheticHead.load(lib.cache_dir)
    aes = aesn = None
    if head is not None and head.dim == emb.shape[1]:
        aes = np.array([head(emb[i]) for i in range(len(ids))], np.float32)
        aesn = _n01(aes)

    # named text axes (need a CLIP text tower)
    axes = {}
    if meta.get("semantic"):
        try:
            from .embed import get_embedder
            e = get_embedder(prefer_clip=True)
            if getattr(e, "semantic", False):
                for name, (pos, neg) in NAMED_AXES.items():
                    d = _unit(e.embed_texts(pos).mean(0) - e.embed_texts(neg).mean(0))
                    axes[name] = _n01(emb @ d)
        except Exception as ex:
            print(f"[export] named axes skipped: {type(ex).__name__}: {ex}")

    def thumb_b64(h):
        p = os.path.join(lib.thumb_dir, h, "256.jpg")
        if not os.path.exists(p):
            return None
        im = Image.open(p).convert("RGB"); im.thumbnail((thumb_px, thumb_px))
        b = io.BytesIO(); im.save(b, "JPEG", quality=70)
        return base64.b64encode(b.getvalue()).decode()

    sims = emb @ emb.T
    np.fill_diagonal(sims, -1.0)
    nn = np.argsort(-sims, axis=1)[:, :knn].tolist()

    items = []
    for i, h in enumerate(ids):
        it = {"f": fn.get(h, h[:8]), "x": round(float(pcs[i, 0]), 4),
              "y": round(float(pcs[i, 1]), 4), "z": round(float(pcs[i, 2]), 4),
              "a": round(float(aesn[i]), 4) if aes is not None else None,
              "av": round(float(aes[i]), 2) if aes is not None else None,
              "n": nn[i], "t": thumb_b64(h),
              "ax": {k: round(float(axes[k][i]), 4) for k in axes}}
        items.append(it)

    data = {"items": items, "count": len(items),
            "library": os.path.basename(lib.root),
            "axes": list(axes.keys())}
    html = _HTML.replace("/*__DATA__*/", json.dumps(data))
    out = os.path.join(lib.state_dir, "gallery.html")
    open(out, "w", encoding="utf-8").write(html)
    return out, len(items)


_HTML = r"""<!doctype html><html><head><meta charset="utf-8"><title>autophotos galaxy</title>
<style>
 :root{--bg:#0d0f12;--fg:#e8e8ea;--mut:#8a909a}
 *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--fg);
   font:13px/1.4 system-ui,sans-serif;overflow:hidden}
 #bar{position:fixed;top:0;left:0;right:0;height:46px;display:flex;gap:12px;
   align-items:center;padding:0 14px;background:#15181d;border-bottom:1px solid #23272e;z-index:10}
 #bar b{color:#fff} select,input{background:#23272e;color:var(--fg);border:1px solid #30353d;
   border-radius:6px;padding:4px 6px} label{color:var(--mut)}
 #stage{position:absolute;top:46px;left:0;right:0;bottom:0;overflow:hidden}
 .tile{position:absolute;width:46px;height:46px;transform:translate(-50%,-50%);
   border-radius:3px;cursor:pointer;transition:left .6s cubic-bezier(.4,0,.2,1),
   top .6s cubic-bezier(.4,0,.2,1),width .15s,height .15s,opacity .2s,box-shadow .15s;
   background-size:cover;background-position:center;outline:1px solid #000}
 .tile:hover{width:150px;height:150px;z-index:50;box-shadow:0 6px 24px #000}
 .dim{opacity:.12} .hot{box-shadow:0 0 0 2px #ffcf33,0 6px 20px #000;z-index:40}
 #tip{position:fixed;pointer-events:none;background:#000c;border:1px solid #333;
   padding:3px 7px;border-radius:5px;font-size:12px;display:none;z-index:99}
 .leg{margin-left:auto;color:var(--mut)} .swatch{display:inline-block;width:90px;height:8px;
   border-radius:4px;vertical-align:middle;background:linear-gradient(90deg,#2b6cff,#27c4a4,#ffd23f,#ff6b3d,#ff2d55)}
</style></head><body>
<div id="bar"><b>autophotos galaxy</b> <span id="cnt" class="mut"></span>
 <label>X</label><select id="ax"></select> <label>Y</label><select id="ay"></select>
 <label>size</label><input id="sz" type="range" min="20" max="80" value="46">
 <span class="leg">cool <span class="swatch"></span> high aesthetic · hover zoom, click=similar</span></div>
<div id="stage"></div><div id="tip"></div>
<script>
const DATA=/*__DATA__*/;
const stage=document.getElementById('stage'),tip=document.getElementById('tip');
const base={'PCA 1':'x','PCA 2':'y','PCA 3':'z'}; if(DATA.items[0]&&DATA.items[0].a!=null)base['Aesthetic']='a';
const named={}; (DATA.axes||[]).forEach(n=>named[n]='ax:'+n);
const OPTS=Object.assign({},base,named);
const ax=document.getElementById('ax'),ay=document.getElementById('ay');
for(const k in OPTS){ax.add(new Option(k,OPTS[k]));ay.add(new Option(k,OPTS[k]));}
ax.value='x';ay.value='y';
document.getElementById('cnt').textContent=DATA.count+' · '+DATA.library;
function val(it,key){return key.startsWith('ax:')?(it.ax?it.ax[key.slice(3)]:.5):(it[key]==null?.5:it[key]);}
function color(a){if(a==null)return '#888';const s=[[43,108,255],[39,196,164],[255,210,63],[255,107,61],[255,45,85]];
 const t=a*(s.length-1),i=Math.floor(t),f=t-i,c=s[Math.min(i,4)],d=s[Math.min(i+1,4)];
 return`rgb(${c[0]+(d[0]-c[0])*f|0},${c[1]+(d[1]-c[1])*f|0},${c[2]+(d[2]-c[2])*f|0})`;}
const tiles=DATA.items.map((it,i)=>{const e=document.createElement('div');e.className='tile';
 if(it.t)e.style.backgroundImage=`url(data:image/jpeg;base64,${it.t})`;
 e.style.borderBottom='3px solid '+color(it.a);
 e.onmouseenter=()=>{tip.style.display='block';tip.textContent=it.f+(it.av!=null?'  · aes '+it.av:'');};
 e.onmousemove=ev=>{tip.style.left=(ev.clientX+12)+'px';tip.style.top=(ev.clientY+12)+'px';};
 e.onmouseleave=()=>tip.style.display='none'; e.onclick=()=>hl(i); stage.appendChild(e);return e;});
function layout(){const kx=ax.value,ky=ay.value,W=stage.clientWidth,H=stage.clientHeight,m=80;
 DATA.items.forEach((it,i)=>{tiles[i].style.left=(m+val(it,kx)*(W-2*m))+'px';
  tiles[i].style.top=(H-m-val(it,ky)*(H-2*m))+'px';});}
function size(){const s=document.getElementById('sz').value;tiles.forEach(t=>{t.style.width=s+'px';t.style.height=s+'px';});}
let sel=null;function hl(i){if(sel===i){tiles.forEach(t=>t.classList.remove('dim','hot'));sel=null;return;}
 sel=i;const near=new Set(DATA.items[i].n);near.add(i);
 tiles.forEach((t,j)=>{t.classList.toggle('dim',!near.has(j));t.classList.toggle('hot',near.has(j)&&j!==i);});}
ax.onchange=ay.onchange=layout;document.getElementById('sz').oninput=size;
addEventListener('resize',layout);size();layout();
</script></body></html>"""
