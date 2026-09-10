"""Analyze trained MOT/MOA spatial routers under controlled appearance changes."""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageEnhance, ImageFilter

from .capture import SpatialRouterCollector, routing_diagnostics
from .geometry import letterbox
from .io_utils import sha256_file, write_json, write_manifest
from .plotting import save_dominant_overlay
from .supplemental import _save_sheet
from .trained_demo import build_trained_demo

ROOT = Path(__file__).resolve().parents[2]
YOLO_ROOT = (ROOT.parent / "YOLO-Master").resolve()
DATASET = (ROOT.parent / "datasets" / "coco8" / "images" / "val").resolve()
COLORS = {"mot": "#20c7d9", "moa": "#a56cff"}


def apply_transform(image: Image.Image, spec: dict) -> Image.Image:
    image = image.convert("RGB")
    kind = spec["kind"]
    if kind == "identity":
        return image.copy()
    if kind == "brightness":
        return ImageEnhance.Brightness(image).enhance(float(spec["factor"]))
    if kind == "contrast":
        return ImageEnhance.Contrast(image).enhance(float(spec["factor"]))
    if kind == "gaussian_blur":
        return image.filter(ImageFilter.GaussianBlur(float(spec["radius"])))
    raise ValueError(f"unsupported transformation: {kind}")


def comparison(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    if reference.shape != candidate.shape:
        raise ValueError(f"unaligned routing grids: {reference.shape} != {candidate.shape}")
    delta = np.abs(reference.astype(np.float64) - candidate.astype(np.float64))
    ref_dominant = np.argmax(reference, axis=0)
    candidate_dominant = np.argmax(candidate, axis=0)
    return {
        "probability_mae": float(delta.mean()),
        "total_variation_distance_mean": float((0.5 * delta.sum(axis=0)).mean()),
        "dominant_expert_agreement": float((ref_dominant == candidate_dominant).mean()),
        "dominant_switch_fraction": float((ref_dominant != candidate_dominant).mean()),
    }


def setup_plot():
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": "#050c1b", "axes.facecolor": "#09172d", "savefig.facecolor": "#050c1b",
        "text.color": "#eaf4ff", "axes.labelcolor": "#b9cce0", "xtick.color": "#9bb8d6",
        "ytick.color": "#9bb8d6", "axes.edgecolor": "#21678d", "font.size": 10,
    })
    return plt


def save_appearance(summary: dict, output: Path) -> None:
    plt = setup_plot()
    transforms = summary["transformations"]
    x = np.arange(len(transforms))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), constrained_layout=True)
    width = 0.36
    for index, family in enumerate(("mot", "moa")):
        values = [summary["appearance"][family][name]["dominant_expert_agreement_mean"] * 100 for name in transforms]
        errors = [summary["appearance"][family][name]["dominant_expert_agreement_std"] * 100 for name in transforms]
        bars = axes[0].bar(x + (index - 0.5) * width, values, width, yerr=errors, capsize=3, label=family.upper(), color=COLORS[family])
        axes[0].bar_label(bars, fmt="%.1f", fontsize=8, padding=2)
        maes = [summary["appearance"][family][name]["probability_mae_mean"] for name in transforms]
        mae_errors = [summary["appearance"][family][name]["probability_mae_std"] for name in transforms]
        bars = axes[1].bar(x + (index - 0.5) * width, maes, width, yerr=mae_errors, capsize=3, label=family.upper(), color=COLORS[family])
        axes[1].bar_label(bars, labels=[f"{value:.4f}" for value in maes], fontsize=7, padding=2, rotation=90)
    labels = [name.replace("_", "\n") for name in transforms]
    axes[0].set(title="Dominant expert agreement", ylabel="Agreement (%)", xticks=x, xticklabels=labels, ylim=(0, 102))
    axes[1].set(title="Router probability sensitivity", ylabel="Mean absolute probability change", xticks=x, xticklabels=labels)
    for axis in axes:
        axis.grid(axis="y", alpha=0.15)
        axis.legend(frameon=False)
    fig.suptitle("E3 P2 / TRAINED APPEARANCE SENSITIVITY\nmean ± descriptive SD across 16 image×layer comparisons", color="#00e5ff", fontsize=18)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def save_appearance_inputs(image: Image.Image, specs: list[dict], output: Path) -> None:
    """Archive the exact visual perturbations used by the trained analysis."""
    plt = setup_plot()
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    for axis, spec in zip(axes.flat, specs):
        axis.imshow(apply_transform(image, spec))
        axis.set_title(spec["name"].replace("_", " ").upper())
        axis.axis("off")
    fig.suptitle("E3 P2 / APPEARANCE INPUT AUDIT\nSame original geometry · exact configured transformations", color="#00e5ff", fontsize=18)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def enrich_summary(summary: dict, comparisons: list[dict]) -> dict:
    """Add dispersion and absolute per-layer response to the published means."""
    absolute = {family: {} for family in summary["modules"]}
    for family, modules in summary["modules"].items():
        for transform in summary["transformations"]:
            rows = [row for row in comparisons if row["family"] == family and row["transform"] == transform]
            appearance = summary["appearance"][family][transform]
            appearance["probability_mae_std"] = float(np.std([row["probability_mae"] for row in rows], ddof=1))
            appearance["dominant_expert_agreement_std"] = float(
                np.std([row["dominant_expert_agreement"] for row in rows], ddof=1)
            )
            absolute[family][transform] = {
                module: float(np.mean([row["probability_mae"] for row in rows if row["module"] == module]))
                for module in modules
            }
    summary["absolute_layer_sensitivity"] = absolute
    summary["dispersion_unit"] = "descriptive sample SD across 4 images × 4 router layers; not an independent CI"
    return summary


def save_absolute_sensitivity(summary: dict, output: Path) -> None:
    """Plot absolute layer MAE on one shared log scale to ground relative attribution."""
    plt = setup_plot()
    transforms = summary["transformations"]
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), constrained_layout=True)
    image = None
    for axis, family in zip(axes, ("mot", "moa")):
        modules = summary["modules"][family]
        matrix = np.asarray([
            [summary["absolute_layer_sensitivity"][family][transform][module] for transform in transforms]
            for module in modules
        ])
        image = axis.imshow(np.log10(np.maximum(matrix, 1e-8)), vmin=-8, vmax=-1, cmap="viridis", aspect="auto")
        axis.set(title=f"{family.upper()} absolute probability MAE", yticks=np.arange(len(modules)), yticklabels=modules,
                 xticks=np.arange(len(transforms)), xticklabels=[name.replace("_", "\n") for name in transforms])
        for row, column in np.ndindex(matrix.shape):
            axis.text(column, row, f"{matrix[row, column]:.2e}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(image, ax=axes, label="log10(mean absolute probability change)", shrink=0.8)
    fig.suptitle("E3 P2 / ABSOLUTE ROUTER SENSITIVITY\nShared scale prevents relative attribution from exaggerating tiny changes", color="#00e5ff", fontsize=18)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def save_attribution(summary: dict, output: Path) -> None:
    plt = setup_plot()
    transforms = summary["transformations"]
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), constrained_layout=True)
    palette = ["#00d7e9", "#8f65db", "#ffc400", "#2ce07b"]
    for axis, family in zip(axes, ("mot", "moa")):
        modules = summary["modules"][family]
        bottom = np.zeros(len(transforms))
        for index, module in enumerate(modules):
            values = [summary["attribution"][family][name][module] * 100 for name in transforms]
            bars = axis.bar(np.arange(len(transforms)), values, bottom=bottom, label=module, color=palette[index % len(palette)])
            labels = [f"{value:.0f}%" if value >= 9 else "" for value in values]
            axis.bar_label(bars, labels=labels, label_type="center", fontsize=8, color="#071020")
            bottom += values
        axis.set(title=f"{family.upper()} layer share of probability MAE", ylabel="Share (%)", xticks=np.arange(len(transforms)), xticklabels=[name.replace("_", "\n") for name in transforms], ylim=(0, 100))
        axis.legend(ncols=4, frameon=False, fontsize=8)
    fig.suptitle("E3 P2 / TRAINED ROUTER ATTRIBUTION\nDescriptive layer contribution; not causal attribution", color="#00e5ff", fontsize=18)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def save_scatter(comparisons: list[dict], output: Path) -> None:
    plt = setup_plot()
    fig, axis = plt.subplots(figsize=(9, 6), constrained_layout=True)
    for family in ("mot", "moa"):
        rows = [item for item in comparisons if item["family"] == family]
        axis.scatter([row["probability_mae"] for row in rows], [row["dominant_switch_fraction"] * 100 for row in rows], s=55, alpha=0.72, label=family.upper(), c=COLORS[family], edgecolors="#eaf4ff", linewidths=0.35)
    axis.set(title="Probability shift vs dominant-expert switching", xlabel="Probability MAE", ylabel="Dominant switch fraction (%)")
    axis.grid(alpha=0.15)
    axis.legend(frameon=False)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def save_usage(summary: dict, output: Path) -> None:
    plt = setup_plot()
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), constrained_layout=True)
    palette = ["#00d7e9", "#8f65db", "#ffc400", "#2ce07b", "#ff5277", "#4f79ff"]
    for axis, family in zip(axes, ("mot", "moa")):
        modules = summary["modules"][family]
        max_experts = max(len(summary["identity_usage"][family][module]) for module in modules)
        x = np.arange(len(modules))
        width = 0.75 / max_experts
        for expert in range(max_experts):
            values = [summary["identity_usage"][family][module][expert] * 100 if expert < len(summary["identity_usage"][family][module]) else 0 for module in modules]
            bars = axis.bar(x + (expert - (max_experts - 1) / 2) * width, values, width, label=f"E{expert}", color=palette[expert])
            axis.bar_label(bars, labels=[f"{value:.1f}" if value >= 2 else "" for value in values], fontsize=7, padding=2, rotation=90)
        axis.set(title=f"{family.upper()} mean expert probability on identity images", ylabel="Probability mass (%)", xticks=x, xticklabels=modules)
        axis.legend(ncols=max_experts, frameon=False)
    fig.suptitle("E3 P2 / TRAINED EXPERT USAGE", color="#00e5ff", fontsize=18)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def run(config_path: Path) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output = ROOT / "artifacts" / "p2" / config["run_id"]
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite evidence: {output}")
    output.mkdir(parents=True)
    os.environ["YOLO_CONFIG_DIR"] = str(ROOT / ".runtime" / "ultralytics")
    os.environ["YOLO_AUTOINSTALL"] = "false"
    sys.path.insert(0, str(YOLO_ROOT))
    import torch
    from ultralytics import YOLO

    image_paths = sorted(DATASET.glob("*.jpg"))
    image_paths = [image_paths[index] for index in config["sample_indices"]]
    arrays: dict[tuple[str, int, str, str], np.ndarray] = {}
    captures = []
    overlay_cards = []
    profiles = {"mot": "_MoTRouter", "moa": "_MoARouter"}
    for family, checkpoint_value in config["checkpoints"].items():
        checkpoint = Path(checkpoint_value)
        model = YOLO(checkpoint).model.cpu().eval()
        for sample_index, image_path in zip(config["sample_indices"], image_paths):
            original = Image.open(image_path).convert("RGB")
            for spec in config["transformations"]:
                transformed = apply_transform(original, spec)
                canvas, geometry = letterbox(transformed, int(config["resolution"]))
                tensor = torch.from_numpy(canvas.astype(np.float32).transpose(2, 0, 1) / 255.0).unsqueeze(0)
                collector = SpatialRouterCollector(family, profiles[family])
                collector.register(model)
                try:
                    with torch.inference_mode():
                        model(tensor)
                finally:
                    collector.remove()
                for record in collector.records:
                    weights = record.weights[0]
                    key = (family, sample_index, spec["name"], record.module_name)
                    arrays[key] = weights
                    captures.append({"family": family, "sample_index": sample_index, "sample_name": image_path.name, "transform": spec["name"], "module": record.module_name, **routing_diagnostics(weights)})
                    if sample_index == config["sample_indices"][0] and spec["name"] == "identity":
                        path = output / f"overlay-{family}-{record.module_name.replace('.', '-')}.png"
                        counts = save_dominant_overlay(original, weights, geometry, str(path), alpha=0.58)
                        overlay_cards.append({"path": str(path), "caption": f"{family.upper()} | {record.module_name} | dominant {counts}"})
    comparisons = []
    candidates = [item["name"] for item in config["transformations"] if item["name"] != "identity"]
    for (family, sample_index, transform, module), candidate in arrays.items():
        if transform == "identity":
            continue
        metrics = comparison(arrays[(family, sample_index, "identity", module)], candidate)
        comparisons.append({"family": family, "sample_index": sample_index, "transform": transform, "module": module, **metrics})
    modules = {family: sorted({item["module"] for item in captures if item["family"] == family}) for family in ("mot", "moa")}
    appearance = {family: {} for family in modules}
    attribution = {family: {} for family in modules}
    identity_usage = {family: {} for family in modules}
    for family, family_modules in modules.items():
        for transform in candidates:
            rows = [item for item in comparisons if item["family"] == family and item["transform"] == transform]
            appearance[family][transform] = {
                "comparison_count": len(rows),
                "probability_mae_mean": float(np.mean([row["probability_mae"] for row in rows])),
                "dominant_expert_agreement_mean": float(np.mean([row["dominant_expert_agreement"] for row in rows])),
                "dominant_switch_fraction_mean": float(np.mean([row["dominant_switch_fraction"] for row in rows])),
            }
            layer_mae = {module: float(np.mean([row["probability_mae"] for row in rows if row["module"] == module])) for module in family_modules}
            denominator = sum(layer_mae.values())
            attribution[family][transform] = {module: value / denominator if denominator else 0.0 for module, value in layer_mae.items()}
        for module in family_modules:
            rows = [item for item in captures if item["family"] == family and item["transform"] == "identity" and item["module"] == module]
            identity_usage[family][module] = np.mean(np.asarray([row["mean_expert_probability"] for row in rows]), axis=0).tolist()
    summary = {
        "status": "PASS",
        "scope": "trained single-checkpoint appearance sensitivity and spatial routing audit",
        "resolution": int(config["resolution"]),
        "sample_count": len(image_paths),
        "checkpoint_seed_count": 1,
        "single_seed_limitation": True,
        "transformations": candidates,
        "capture_count": len(captures),
        "comparison_count": len(comparisons),
        "modules": modules,
        "appearance": appearance,
        "attribution": attribution,
        "identity_usage": identity_usage,
        "checkpoints": {family: {"sha256": sha256_file(Path(path)), "published": False} for family, path in config["checkpoints"].items()},
        "interpretation_boundary": "Attribution is descriptive share of probability change. Dominant colors are argmax categories, not learned semantic classes. COCO8 and one seed cannot establish robustness or specialization.",
    }
    enrich_summary(summary, comparisons)
    write_json(output / "summary.json", summary)
    write_json(output / "comparisons.json", comparisons)
    write_json(output / "captures.json", captures)
    save_appearance(summary, output / "appearance-sensitivity.png")
    save_appearance_inputs(Image.open(image_paths[0]).convert("RGB"), config["transformations"], output / "appearance-inputs.png")
    save_attribution(summary, output / "router-attribution.png")
    save_absolute_sensitivity(summary, output / "router-absolute-sensitivity.png")
    save_scatter(comparisons, output / "sensitivity-scatter.png")
    save_usage(summary, output / "expert-usage-bars.png")
    _save_sheet(overlay_cards, "TRAINED MOT / MOA ROUTING", "Same COCO8 image · 320px · true dominant experts · all spatial router layers", output / "trained-routing-overlays.png", columns=4)
    mot_cards = []
    for card in overlay_cards:
        if not card["caption"].startswith("MOT"):
            continue
        parts = card["caption"].split(" | ")
        layer = parts[1].split(".")[1]
        counts = parts[-1].removeprefix("dominant ")
        active = sum(value > 0 for value in ast.literal_eval(counts))
        mot_cards.append({"path": card["path"], "caption": f"MOT L{layer} · active {active}/3 · counts {counts}"})
    _save_sheet(
        mot_cards,
        "TRAINED MOT / LAYER-BY-LAYER",
        "Same checkpoint and image · true argmax colors · active experts legitimately vary by layer",
        output / "trained-mot-layer-focus.png",
        columns=4,
    )
    build_trained_demo(output)
    (output / "config.resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    write_manifest(output)
    print(json.dumps(summary, indent=2))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    run(parser.parse_args().config.resolve())


if __name__ == "__main__":
    main()
