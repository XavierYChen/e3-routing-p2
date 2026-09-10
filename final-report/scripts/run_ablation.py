"""Run the preregistered E3 routing-specialization ablation on MOT and MOA."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import random
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
P2_SRC = ROOT.parent / "E3-Routing-P2" / "src"
YOLO_ROOT = ROOT.parent / "YOLO-Master"
sys.path[:0] = [str(P2_SRC), str(YOLO_ROOT)]

from e3_p2.capture import SpatialRouterCollector, routing_diagnostics  # noqa: E402
from e3_p2.geometry import letterbox  # noqa: E402
from e3_p2.plotting import save_dominant_overlay  # noqa: E402
from e3_p2.trained_analysis import apply_transform, comparison  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mean_std(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {"mean": float(array.mean()), "sd": float(array.std(ddof=1)) if len(array) > 1 else 0.0}


def safe_relative(after: float, before: float) -> float | None:
    if abs(before) < 1e-12:
        return None
    return (after - before) / abs(before)


def evaluate_decision(aggregate: dict, rule: dict, family: str) -> dict:
    base = aggregate[family]["pretrained_init"]
    trained = aggregate[family]["trained_10ep"]
    gains = {
        "top1_margin_relative_gain": safe_relative(trained["top1_margin"]["mean"], base["top1_margin"]["mean"]),
        "spatial_variation_relative_gain": safe_relative(trained["spatial_variation"]["mean"], base["spatial_variation"]["mean"]),
        "all_experts_active_absolute_gain": trained["all_experts_active_rate"]["mean"] - base["all_experts_active_rate"]["mean"],
    }
    criteria = {}
    for name, threshold in rule["criteria"].items():
        if gains[name] is None:
            metric = "spatial_variation" if name.startswith("spatial_variation") else "top1_margin"
            criteria[name] = trained[metric]["mean"] > base[metric]["mean"] + 1e-12
        else:
            criteria[name] = gains[name] >= float(threshold)
    appearance_drop = base["appearance_agreement"]["mean"] - trained["appearance_agreement"]["mean"]
    stability_ok = appearance_drop <= float(rule["maximum_appearance_agreement_drop"])
    return {
        "gains": gains,
        "null_gain_note": "A null relative gain means the baseline was exactly zero; the criterion then uses a strict positive-change check.",
        "criteria_met": criteria,
        "criteria_met_count": sum(criteria.values()),
        "appearance_agreement_drop": appearance_drop,
        "appearance_guardrail_met": stability_ok,
        "supports_clearer_specialization": sum(criteria.values()) >= int(rule["minimum_criteria_met"]) and stability_ok,
    }


def make_model(YOLO, torch, family_cfg: dict, condition: str, condition_cfg: dict, seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if condition == "trained_10ep":
        path = Path(condition_cfg["checkpoints"][family_cfg["name"]][seed])
        return YOLO(path), path
    model = YOLO(Path(family_cfg["model_yaml"]))
    if condition == "pretrained_init":
        path = Path(condition_cfg["weights"])
        model.load(path)
        return model, path
    return model, Path(family_cfg["model_yaml"])


def save_figures(summary: dict, output: Path) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": "#050c1b", "axes.facecolor": "#09172d", "savefig.facecolor": "#050c1b",
        "text.color": "#eaf4ff", "axes.labelcolor": "#b9cce0", "xtick.color": "#9bb8d6",
        "ytick.color": "#9bb8d6", "axes.edgecolor": "#21678d", "font.size": 10,
    })
    colors = {"random_init": "#53657d", "pretrained_init": "#8f65db", "trained_10ep": "#00c8d9"}
    conditions = list(summary["conditions"])
    labels = ["Random\n0 epoch", "Transfer\n0 epoch", "Transfer\n10 epochs"]
    x = np.arange(len(conditions))

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    specs = [
        ("mAP50_95", "Detection pipeline check", "mAP50–95"),
        ("all_experts_active_rate", "All experts active", "Capture fraction"),
        ("top1_margin", "Routing confidence", "Mean top-1 margin"),
        ("spatial_variation", "Spatial routing variation", "Neighbor probability L1"),
    ]
    for axis, (metric, title, ylabel) in zip(axes.flat, specs):
        for offset, family in zip((-0.18, 0.18), ("mot", "moa")):
            vals = [summary["aggregate"][family][c][metric]["mean"] for c in conditions]
            errs = [summary["aggregate"][family][c][metric]["sd"] for c in conditions]
            bars = axis.bar(x + offset, vals, 0.34, yerr=errs, capsize=3, label=family.upper(),
                            color="#20c7d9" if family == "mot" else "#a56cff")
            axis.bar_label(bars, labels=[f"{v:.3f}" for v in vals], fontsize=7, padding=2)
        axis.set(title=title, ylabel=ylabel, xticks=x, xticklabels=labels)
        axis.grid(axis="y", alpha=0.15)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("E3 FINAL / ROUTING SPECIALIZATION ABLATION\n3 seeds · same COCO8 budget · mean ± sample SD", color="#00e5ff", fontsize=18)
    fig.savefig(output / "ablation-dashboard.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.7), constrained_layout=True)
    for axis, family in zip(axes, ("mot", "moa")):
        agreement = [summary["aggregate"][family][c]["appearance_agreement"]["mean"] * 100 for c in conditions]
        mae = [summary["aggregate"][family][c]["appearance_probability_mae"]["mean"] for c in conditions]
        axis2 = axis.twinx()
        bars = axis.bar(x - 0.17, agreement, 0.34, color=[colors[c] for c in conditions], alpha=0.9)
        axis2.bar(x + 0.17, mae, 0.34, color="#ffcf4a", alpha=0.75)
        axis.bar_label(bars, fmt="%.1f", fontsize=8, padding=2)
        axis.set(title=f"{family.upper()} appearance stability", ylabel="Dominant agreement (%)", xticks=x, xticklabels=labels, ylim=(0, 105))
        axis2.set_ylabel("Probability MAE", color="#ffcf4a")
        axis.grid(axis="y", alpha=0.15)
    fig.suptitle("E3 FINAL / APPEARANCE GUARDRAIL\nBrightness, contrast and blur against identity", color="#00e5ff", fontsize=18)
    fig.savefig(output / "appearance-guardrail.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 6), constrained_layout=True)
    markers = {"random_init": "o", "pretrained_init": "s", "trained_10ep": "^"}
    for family, color in (("mot", "#20c7d9"), ("moa", "#a56cff")):
        for condition in conditions:
            rows = [r for r in summary["seed_results"] if r["family"] == family and r["condition"] == condition]
            axis.scatter([r["top1_margin"] for r in rows], [r["appearance_agreement"] * 100 for r in rows],
                         marker=markers[condition], s=90, c=color, edgecolors="#eaf4ff", linewidths=0.5,
                         label=f"{family.upper()} / {condition.replace('_', ' ')}")
    axis.set(title="Confidence–stability trade-off", xlabel="Mean top-1 margin", ylabel="Appearance agreement (%)")
    axis.grid(alpha=0.15)
    axis.legend(frameon=False, fontsize=8, ncols=2)
    fig.savefig(output / "confidence-stability-scatter.png", dpi=180)
    plt.close(fig)

    from PIL import ImageDraw, ImageFont
    conditions = ["random_init", "pretrained_init", "trained_10ep"]
    labels = ["Random / 0 epoch", "Tencent transfer / 0 epoch", "Tencent transfer / 10 epochs"]
    cards = []
    for family in ("mot", "moa"):
        for condition, label in zip(conditions, labels):
            cards.append((family.upper(), label, Image.open(output / f"overlay-{family}-{condition}.png").convert("RGB")))
    width = max(card.width for _, _, card in cards)
    height = max(card.height for _, _, card in cards)
    header, gap = 54, 18
    sheet = Image.new("RGB", (3 * width + 4 * gap, 2 * (height + header) + 3 * gap), (5, 12, 27))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    for index, (family, label, card) in enumerate(cards):
        row, column = divmod(index, 3)
        x = gap + column * (width + gap)
        y = gap + row * (height + header + gap)
        draw.text((x, y), f"{family}  |  {label}", fill=(230, 242, 255), font=font)
        sheet.paste(card, (x, y + header))
    sheet.save(output / "ablation-routing-overlays.png", quality=95)
    sheet.save(output / "ablation-routing-overlays.jpg", quality=88, optimize=True)


def run(config_path: Path) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output = ROOT / "results" / config["run_id"]
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite evidence: {output}")
    output.mkdir(parents=True)
    os.environ["YOLO_CONFIG_DIR"] = str(ROOT / ".runtime" / "ultralytics")
    os.environ["YOLO_AUTOINSTALL"] = "false"
    import torch
    from ultralytics import YOLO

    device = torch.device("cuda:0" if config["device"] != "cpu" and torch.cuda.is_available() else "cpu")
    image_paths = sorted(Path(config["image_dir"]).glob("*.jpg"))
    image_paths = [image_paths[i] for i in config["sample_indices"]]
    seed_results = []
    captures = []
    comparisons = []
    fingerprints = {}
    overlay_paths = []
    for family, base_family_cfg in config["families"].items():
        family_cfg = {**base_family_cfg, "name": family}
        for condition, condition_cfg in config["conditions"].items():
            for seed in config["seeds"]:
                model, source = make_model(YOLO, torch, family_cfg, condition, condition_cfg, int(seed))
                fingerprints[f"{family}/{condition}/{seed}"] = {"source": str(source), "sha256": sha256(source)}
                val = model.val(data=str(Path(config["dataset_yaml"])), imgsz=int(config["validation_resolution"]),
                                batch=1, device=str(config["device"]), workers=0, plots=False, save_json=False,
                                project=str(ROOT.parent / "tmp" / "e3-ablation-validation"),
                                name=f"{family}-{condition}-seed{seed}", exist_ok=True, verbose=False)
                network = model.model.to(device).eval()
                arrays = {}
                local_captures = []
                local_comparisons = []
                for sample_index, image_path in zip(config["sample_indices"], image_paths):
                    original = Image.open(image_path).convert("RGB")
                    for spec in config["transformations"]:
                        transformed = apply_transform(original, spec)
                        canvas, geometry = letterbox(transformed, int(config["route_resolution"]))
                        tensor = torch.from_numpy(canvas.astype(np.float32).transpose(2, 0, 1) / 255.0).unsqueeze(0).to(device)
                        collector = SpatialRouterCollector(family, family_cfg["router_class"])
                        collector.register(network)
                        try:
                            with torch.inference_mode():
                                network(tensor)
                        finally:
                            collector.remove()
                        for record in collector.records:
                            weights = record.weights[0]
                            key = (sample_index, spec["name"], record.module_name)
                            arrays[key] = weights
                            diag = routing_diagnostics(weights)
                            row = {"family": family, "condition": condition, "seed": seed, "sample_index": sample_index,
                                   "sample_name": image_path.name, "transform": spec["name"], "module": record.module_name, **diag}
                            captures.append(row)
                            local_captures.append(row)
                            if seed == 0 and sample_index == config["sample_indices"][0] and spec["name"] == "identity" and record.module_name.endswith("19.m.0.router"):
                                overlay = output / f"overlay-{family}-{condition}.png"
                                save_dominant_overlay(original, weights, geometry, str(overlay), alpha=0.58)
                                overlay_paths.append(str(overlay.name))
                for (sample_index, transform, module), candidate in arrays.items():
                    if transform == "identity":
                        continue
                    row = {"family": family, "condition": condition, "seed": seed, "sample_index": sample_index,
                           "transform": transform, "module": module,
                           **comparison(arrays[(sample_index, "identity", module)], candidate)}
                    comparisons.append(row)
                    local_comparisons.append(row)
                identity = [r for r in local_captures if r["transform"] == "identity"]
                seed_results.append({
                    "family": family, "condition": condition, "seed": seed,
                    "mAP50": float(val.box.map50), "mAP50_95": float(val.box.map),
                    "active_experts": float(np.mean([r["active_dominant_experts"] for r in identity])),
                    "all_experts_active_rate": float(np.mean([r["active_dominant_experts"] == len(r["mean_expert_probability"]) for r in identity])),
                    "normalized_entropy": float(np.mean([r["normalized_entropy"]["mean"] for r in identity])),
                    "top1_margin": float(np.mean([r["top1_margin"]["mean"] for r in identity])),
                    "spatial_variation": float(np.mean([r["neighbor_probability_l1_mean"] for r in identity])),
                    "appearance_agreement": float(np.mean([r["dominant_expert_agreement"] for r in local_comparisons])),
                    "appearance_probability_mae": float(np.mean([r["probability_mae"] for r in local_comparisons])),
                })
                del model, network
                torch.cuda.empty_cache()
    metrics = ["mAP50", "mAP50_95", "active_experts", "all_experts_active_rate", "normalized_entropy",
               "top1_margin", "spatial_variation", "appearance_agreement", "appearance_probability_mae"]
    aggregate = defaultdict(dict)
    for family in config["families"]:
        for condition in config["conditions"]:
            rows = [r for r in seed_results if r["family"] == family and r["condition"] == condition]
            aggregate[family][condition] = {metric: mean_std([r[metric] for r in rows]) for metric in metrics}
    summary = {
        "created_utc": datetime.now(UTC).isoformat(), "status": "PASS", "run_id": config["run_id"],
        "protocol": {"families": list(config["families"]), "conditions": config["conditions"], "seeds": config["seeds"],
                     "images": [p.name for p in image_paths], "route_resolution": config["route_resolution"],
                     "validation_resolution": config["validation_resolution"], "transformations": config["transformations"]},
        "conditions": config["conditions"], "seed_results": seed_results, "aggregate": dict(aggregate),
        "decision_rule": config["decision_rule"],
        "decision": {family: evaluate_decision(aggregate, config["decision_rule"], family) for family in config["families"]},
        "counts": {"captures": len(captures), "appearance_comparisons": len(comparisons)},
        "checkpoint_fingerprints": fingerprints, "representative_overlays": overlay_paths,
        "interpretation_boundary": "Expert IDs are permutation-symmetric across independently initialized seeds; cross-seed expert-ID agreement is not reported. COCO8 detection scores are pipeline checks, not generalization claims.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output / "captures.json").write_text(json.dumps(captures) + "\n", encoding="utf-8")
    (output / "comparisons.json").write_text(json.dumps(comparisons) + "\n", encoding="utf-8")
    for name in ("captures.json", "comparisons.json"):
        with (output / f"{name}.gz").open("wb") as raw:
            with gzip.GzipFile(filename=name, mode="wb", fileobj=raw, compresslevel=9, mtime=0) as archive:
                archive.write((output / name).read_bytes())
    (output / "config.resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    save_figures(summary, output)
    manifest = {str(p.relative_to(output)).replace("\\", "/"): sha256(p) for p in sorted(output.iterdir()) if p.is_file()}
    (output / "manifest.sha256.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "decision": summary["decision"], "counts": summary["counts"]}, indent=2))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "ablation.yaml")
    run(parser.parse_args().config.resolve())


if __name__ == "__main__":
    main()
