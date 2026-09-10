"""Build a self-contained interactive viewer for trained routing evidence."""

from __future__ import annotations

import json
from pathlib import Path


def build_trained_demo(output: Path) -> Path:
    captures = json.loads((output / "captures.json").read_text(encoding="utf-8"))
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    cards = []
    for row in captures:
        if row["transform"] != "identity" or row["sample_index"] != 0:
            continue
        cards.append({
            "family": row["family"],
            "module": row["module"],
            "overlay": f"overlay-{row['family']}-{row['module'].replace('.', '-')}.png",
            "active": row["active_dominant_experts"],
            "counts": row["dominant_token_count"],
            "fractions": row["dominant_token_fraction"],
            "probability": row["mean_expert_probability"],
            "entropy": row["normalized_entropy"]["mean"],
            "margin": row["top1_margin"]["mean"],
            "spatial_variation": row["neighbor_probability_l1_mean"],
        })
    payload = {
        "cards": cards,
        "appearance": summary["appearance"],
        "figures": {
            "appearance": "appearance-sensitivity.png",
            "attribution": "router-attribution.png",
            "scatter": "sensitivity-scatter.png",
            "usage": "expert-usage-bars.png",
        },
    }
    encoded = json.dumps(payload, separators=(",", ":"))
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>E3 trained routing evidence</title>
<style>
:root{{--bg:#050c1b;--panel:#09172d;--line:#21678d;--cyan:#00e5ff;--violet:#a56cff;--green:#42e58c;--text:#eaf4ff;--muted:#9bb8d6}}
*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(circle at top,#10264a,var(--bg) 55%);color:var(--text);font:15px/1.45 Inter,Segoe UI,sans-serif}}
.shell{{max-width:1500px;margin:auto;padding:24px}}.hero,.panel{{background:rgba(9,23,45,.96);border:1px solid var(--line);border-radius:18px;padding:22px;box-shadow:0 18px 50px #0008}}
.eyebrow{{color:var(--cyan);letter-spacing:.14em;font-weight:800}}h1{{font-size:34px;margin:7px 0}}.hero p{{color:var(--muted);max-width:1100px}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:18px}}
.card{{background:#0c2342;border:1px solid #285474;border-radius:12px;padding:13px}}.card b{{display:block;color:var(--green);font-size:22px}}.grid{{display:grid;grid-template-columns:300px 1fr;gap:16px;margin-top:16px}}
label{{display:block;color:var(--muted);margin:8px 0 4px}}select,button{{width:100%;border:1px solid #285474;background:#0d203b;color:var(--text);padding:10px;border-radius:9px}}
.tabs{{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:14px}}button{{cursor:pointer}}button.active{{background:linear-gradient(90deg,#006f8c,#53359a);font-weight:700}}
.compare{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}figure{{margin:0}}figcaption{{color:var(--muted);font-weight:700;margin-bottom:8px}}img{{width:100%;height:500px;object-fit:contain;background:#050c1b;border-radius:12px}}
.stats{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:12px}}.stat{{background:#0c2342;border:1px solid #285474;border-radius:10px;padding:10px}}.stat b{{display:block;color:var(--cyan);font-size:19px}}
.note{{margin-top:12px;border-left:3px solid var(--violet);padding:10px 13px;background:#25183b;color:#e5d8ff}}#analysis{{display:none}}#analysis img{{height:620px}}pre{{max-height:220px;overflow:auto;white-space:pre-wrap;color:#a9c4de;font-size:11px}}
@media(max-width:900px){{.grid,.compare{{grid-template-columns:1fr}}.cards,.stats{{grid-template-columns:1fr 1fr}}}}
</style></head><body><main class="shell">
<section class="hero"><div class="eyebrow">E3://TRAINED ROUTING EVIDENCE</div><h1>From constant routing to measurable spatial specialization</h1><p>True router probabilities captured from Tencent YOLO-Master after compatible weight transfer and 10 COCO8 epochs. Colors are per-token argmax expert IDs; every panel keeps counts, probability, entropy and margin beside the image.</p>
<div class="cards"><div class="card"><b>192</b>trained captures</div><div class="card"><b>160</b>aligned comparisons</div><div class="card"><b>4 + 4</b>MOT/MOA spatial layers</div><div class="card"><b>0</b>core forward edits</div></div></section>
<section class="grid"><aside class="panel"><label>Family</label><select id="family"><option value="mot">MOT</option><option value="moa">MOA</option></select><label>Spatial router layer</label><select id="layer"></select>
<div class="tabs"><button data-tab="routing" class="active">Routing map</button><button data-tab="appearance">Appearance</button><button data-tab="attribution">Attribution</button><button data-tab="scatter">Scatter</button><button data-tab="usage">Expert usage</button></div>
<div class="note">A layer is not required to show all three colors. Active expert counts are measured, not cosmetically forced.</div><pre id="detail"></pre></aside>
<section class="panel"><div id="routing"><div class="compare"><figure><figcaption>ORIGINAL / COCO8 GROUND TRUTH</figcaption><img src="../p2-v2-five-family-final/inputs/sample-0--ground-truth.png"></figure><figure><figcaption>TRAINED DOMINANT EXPERT MAP</figcaption><img id="overlay"></figure></div><div class="stats" id="stats"></div></div>
<div id="analysis"><figure><figcaption id="analysisTitle"></figcaption><img id="analysisImage"></figure></div></section></section></main>
<script>const DATA={encoded};const q=id=>document.getElementById(id);let tab='routing';
function fmt(x,n=4){{return Number(x).toFixed(n)}}
function familyCards(){{return DATA.cards.filter(x=>x.family===q('family').value)}}
function fillLayers(){{q('layer').innerHTML=familyCards().map(x=>`<option value="${{x.module}}">${{x.module}}</option>`).join('');render()}}
function render(){{const card=familyCards().find(x=>x.module===q('layer').value)||familyCards()[0];if(!card)return;q('overlay').src=card.overlay;q('stats').innerHTML=[['ACTIVE EXPERTS',`${{card.active}} / 3`],['TOKEN COUNTS',card.counts.join(' / ')],['MEAN PROBABILITY',card.probability.map(x=>fmt(x,3)).join(' / ')],['ENTROPY',fmt(card.entropy)],['TOP-1 MARGIN',fmt(card.margin,6)],['SPATIAL L1',fmt(card.spatial_variation,6)],['FAMILY',card.family.toUpperCase()],['LAYER',card.module.split('.')[1]]].map(x=>`<div class="stat"><b>${{x[1]}}</b>${{x[0]}}</div>`).join('');q('detail').textContent=JSON.stringify(card,null,2)}}
function showTab(name){{tab=name;document.querySelectorAll('button[data-tab]').forEach(b=>b.classList.toggle('active',b.dataset.tab===name));q('routing').style.display=name==='routing'?'block':'none';q('analysis').style.display=name==='routing'?'none':'block';if(name!=='routing'){{q('analysisImage').src=DATA.figures[name];q('analysisTitle').textContent=name.toUpperCase()+' / TRAINED CHECKPOINT EVIDENCE'}}}}
q('family').onchange=fillLayers;q('layer').onchange=render;document.querySelectorAll('button[data-tab]').forEach(b=>b.onclick=()=>showTab(b.dataset.tab));fillLayers();document.body.dataset.ready='true';</script></body></html>"""
    path = output / "trained-demo.html"
    path.write_text(html, encoding="utf-8")
    (output / "trained-demo-index.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path

