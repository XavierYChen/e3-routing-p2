"""Render spatial evidence, a static WebUI, and the timed demo."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from .spatial import normalized_entropy, project_to_original


def _overlay(image: np.ndarray, heat: np.ndarray, title: str, note: str, out: Path, *, label: str = "routing probability") -> None:
    fig, ax = plt.subplots(figsize=(8, 5.2), dpi=150)
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    shown = ax.imshow(heat, cmap="turbo", alpha=0.52, vmin=0, vmax=max(float(heat.max()), 1e-6))
    ax.set_title(title, fontsize=13, weight="bold")
    ax.text(0.01, 0.02, note, transform=ax.transAxes, color="white", fontsize=9,
            bbox={"facecolor": "black", "alpha": .72, "pad": 4})
    ax.axis("off")
    fig.colorbar(shown, ax=ax, fraction=.035, pad=.02, label=label)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def render_family(image: np.ndarray, transform: dict, family: str, capture: dict, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    maps = capture["maps"]
    projected = np.stack([project_to_original(item, transform) for item in maps])
    variance = projected.reshape(projected.shape[0], -1).var(1)
    expert = int(np.argmax(variance if capture["spatial_granularity"] == "token" else projected.mean((1, 2))))
    display = family.upper()
    qualifier = "true token-level spatial routing" if capture["spatial_granularity"] == "token" else "image-level routing broadcast over pixels"
    files = {}
    for idx, heat in enumerate(projected):
        name = f"{family}_expert_{idx}.png"
        _overlay(image, heat, f"{display} · Expert {idx}", qualifier, out_dir / name)
        files[f"expert_{idx}"] = name
    selected = projected[expert]
    raw_note = f"{qualifier} · p min={selected.min():.6f}, mean={selected.mean():.6f}, max={selected.max():.6f}"
    primary = f"{family}_primary.png"
    _overlay(image, selected, f"{display} · expert {expert} absolute probability", raw_note, out_dir / primary)
    files["primary"] = primary
    span = float(selected.max() - selected.min())
    contrast = (selected - selected.min()) / span if span > 1e-12 else np.zeros_like(selected)
    contrast_name = f"{family}_relative_contrast.png"
    _overlay(image, contrast, f"{display} · expert {expert} within-map relative contrast", raw_note, out_dir / contrast_name, label="relative contrast [0,1]")
    files["relative_contrast"] = contrast_name
    entropy = normalized_entropy(projected)
    entropy_name = f"{family}_entropy.png"
    _overlay(image, entropy, f"{display} · normalized routing entropy", qualifier, out_dir / entropy_name)
    files["entropy"] = entropy_name
    dominant = projected.argmax(0).astype(np.float32)
    dom_name = f"{family}_dominant.png"
    _overlay(image, dominant, f"{display} · dominant expert id", qualifier, out_dir / dom_name)
    files["dominant"] = dom_name
    return {"files": files, "representative_expert": expert, "projected_maps": projected, "display_heat": contrast,
            "selected_stats": {"min": float(selected.min()), "mean": float(selected.mean()), "max": float(selected.max()), "range": span}}


def render_comparison(image: np.ndarray, rendered: dict[str, dict], out: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2), dpi=150)
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)); axes[0].set_title("COCO8 source")
    for ax, family in zip(axes[1:], ("moe", "mot", "latent")):
        data = rendered[family]
        idx = data["representative_expert"]
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax.imshow(data["display_heat"], cmap="turbo", alpha=.52, vmin=0, vmax=1)
        scope = "TOKEN" if family == "mot" else "IMAGE"
        ax.set_title(f"{family.upper()} · {scope}-LEVEL")
        stats = data["selected_stats"]
        ax.text(.02, .02, f"p={stats['mean']:.4f} · range={stats['range']:.6f}", transform=ax.transAxes,
                color="white", fontsize=8, bbox={"facecolor":"black", "alpha":.7, "pad":3})
    for ax in axes: ax.axis("off")
    fig.suptitle("E3 P2 routing-to-image alignment · one forward · seed 0", weight="bold")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def write_viewer(records: list[dict], out_dir: Path) -> None:
    options = []
    for record in records:
        for label, filename in record["artifacts"].items():
            options.append({"family": record["family_display"], "metric": label, "file": f"artifacts/{filename}",
                            "scope": record["spatial_granularity"], "layer": record["layer"]})
    payload = json.dumps(options, ensure_ascii=False)
    html = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>E3 P2 Routing Viewer</title><style>body{{margin:0;background:#08111f;color:#e8f0ff;font:16px system-ui}}main{{max-width:1100px;margin:auto;padding:28px}}nav a{{color:#70b7ff;margin-right:18px}}.panel{{background:#101d31;border:1px solid #263b59;border-radius:16px;padding:20px;margin-top:20px}}select{{padding:10px;background:#14253c;color:white;border:1px solid #466181;border-radius:8px}}img{{width:100%;margin-top:18px;border-radius:10px;background:white}}code{{color:#85dcff}}.pill{{background:#173b55;padding:5px 9px;border-radius:12px}}</style></head>
<body><main><nav><a href="https://github.com/XavierYChen/e3-routing-smoke">Smoke</a><a href="https://github.com/XavierYChen/e3-routing-p0">P0</a><a href="https://github.com/XavierYChen/e3-routing-p1">P1</a><b>P2</b></nav>
<h1>E3 P2 · Token Routing Spatial Viewer</h1><p>面向 B1、D1、A3 与 WebUI 的只读路由证据。<span class="pill">MOT = token</span> <span class="pill">MOE/LATENT = image</span></p>
<div class="panel"><label>视图 <select id="pick"></select></label><p id="meta"></p><img id="plot" alt="routing overlay"></div>
<script>const data={payload};const pick=document.querySelector('#pick'),plot=document.querySelector('#plot'),meta=document.querySelector('#meta');data.forEach((x,i)=>{{let o=document.createElement('option');o.value=i;o.textContent=`${{x.family}} · ${{x.metric}}`;pick.appendChild(o)}});function show(){{const x=data[+pick.value];plot.src=x.file;meta.innerHTML=`<b>${{x.family}}</b> · <code>${{x.layer}}</code> · ${{x.scope}}-level`;}}pick.onchange=show;show();</script></main></body></html>'''
    (out_dir / "viewer.html").write_text(html, encoding="utf-8")


def make_demo(comparison: Path, records: list[dict], out: Path, fps: int = 10, seconds: int = 120) -> None:
    by_family = {record["family"]: record for record in records}
    def canvas(path: Path) -> np.ndarray:
        source = cv2.imread(str(path))
        if source is None: raise FileNotFoundError(path)
        scale = min(1280 / source.shape[1], 545 / source.shape[0])
        resized = cv2.resize(source, (int(source.shape[1] * scale), int(source.shape[0] * scale)))
        result = np.full((720, 1280, 3), (18, 13, 7), dtype=np.uint8)
        x, y = (1280 - resized.shape[1]) // 2, 105 + (545 - resized.shape[0]) // 2
        result[y:y+resized.shape[0], x:x+resized.shape[1]] = resized
        return result
    artifact_root = comparison.parent / "artifacts"
    slides = {
        "overview": canvas(comparison),
        "moe": canvas(artifact_root / by_family["moe"]["artifacts"]["primary"]),
        "mot": canvas(artifact_root / by_family["mot"]["artifacts"]["relative_contrast"]),
        "latent": canvas(artifact_root / by_family["latent"]["artifacts"]["primary"]),
    }
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1280, 720))
    segments = [
        (0, 15, "overview", "E3 P2 ROUTING-TO-IMAGE VIEWER", "One real COCO8 forward · external hooks · no core forward edits"),
        (15, 38, "moe", "MOE", "Image-level Top-K routing · broadcast overlay is labelled, not token localization"),
        (38, 66, "mot", "MOT", "True token routing · relative contrast shown with absolute probability range"),
        (66, 88, "latent", "LATENT", "Image-level scale mixture · uniform initialization is preserved as evidence"),
        (88, 104, "overview", "UNIFIED SCHEMA + WEBUI", "Machine-readable records serve B1 · D1 · A3 · new WebUI"),
        (104, 120, "overview", "BOUNDARIES", "3/5 families: MOA and MoLoRA are explicit follow-up adapters"),
    ]
    for frame_idx in range(fps * seconds):
        t = frame_idx / fps
        slide, title, subtitle = next((key, a, b) for start, end, key, a, b in segments if start <= t < end)
        frame = slides[slide].copy()
        cv2.rectangle(frame, (0, 0), (1280, 105), (5, 14, 28), -1)
        cv2.rectangle(frame, (0, 650), (1280, 720), (5, 14, 28), -1)
        cv2.putText(frame, title, (38, 45), cv2.FONT_HERSHEY_SIMPLEX, .92, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, subtitle, (38, 81), cv2.FONT_HERSHEY_SIMPLEX, .55, (165, 213, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"{int(t)//60:02d}:{int(t)%60:02d} / 02:00", (1045, 693), cv2.FONT_HERSHEY_SIMPLEX, .58, (255,255,255), 1, cv2.LINE_AA)
        writer.write(frame)
    writer.release()
