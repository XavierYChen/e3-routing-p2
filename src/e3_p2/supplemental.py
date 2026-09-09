"""Build truthful supplemental figures from audited arrays and an optional trained MOT checkpoint."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .capture import SpatialRouterCollector, routing_diagnostics
from .geometry import LetterboxMeta, letterbox
from .io_utils import sha256_file, write_json
from .plotting import save_dominant_overlay

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FORMAL_RUN = PROJECT_ROOT / "artifacts" / "p2" / "p2-v2-five-family-final"
ROBUSTNESS_RUN = PROJECT_ROOT / "artifacts" / "p2" / "p2r-20260909-cpu-resolution-flip-v2"
OUTPUT = PROJECT_ROOT / "artifacts" / "p2" / "supplemental"
CHECKPOINT = Path(r"D:\AI\tmp\e3-p2-local-checkpoints\mot-coco8-seed0-best.pt")
SOURCE_ROOT = (PROJECT_ROOT / ".." / "YOLO-Master").resolve()


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = ["C:/Windows/Fonts/seguisb.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf", "DejaVuSans.ttf"]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _save_sheet(cards: list[dict[str, str]], title: str, subtitle: str, path: Path, columns: int) -> None:
    card_width, image_height, caption_height = 500, 390, 76
    rows = (len(cards) + columns - 1) // columns
    width = columns * card_width + 100
    height = 155 + rows * (image_height + caption_height) + 55
    sheet = Image.new("RGB", (width, height), "#050c1b")
    draw = ImageDraw.Draw(sheet)
    draw.rounded_rectangle((24, 24, width - 24, height - 24), radius=26, fill="#09172d", outline="#21678d", width=3)
    draw.text((58, 48), title, fill="#00e5ff", font=_font(32, True))
    draw.text((60, 97), subtitle, fill="#9bb8d6", font=_font(17))
    for index, card in enumerate(cards):
        row, column = divmod(index, columns)
        left = 50 + column * card_width
        top = 145 + row * (image_height + caption_height)
        image = Image.open(card["path"]).convert("RGB")
        image.thumbnail((card_width - 36, image_height - 20), Image.Resampling.LANCZOS)
        sheet.paste(image, (left + (card_width - image.width) // 2, top + (image_height - image.height) // 2))
        draw.text((left + 18, top + image_height + 12), card["caption"], fill="#eaf4ff", font=_font(17))
    sheet.quantize(colors=256, method=Image.Quantize.MEDIANCUT).save(path, optimize=True)


def _meta(value: dict[str, object]) -> LetterboxMeta:
    return LetterboxMeta(**{key: value[key] for key in LetterboxMeta.__dataclass_fields__})


def _moa_resolution_figure() -> None:
    raw = np.load(ROBUSTNESS_RUN / "stability-routing-raw.npz")
    captures = json.loads((ROBUSTNESS_RUN / "spatial-captures.json").read_text(encoding="utf-8"))
    image = Image.open(ROBUSTNESS_RUN / "inputs" / "sample-0--000000000036.jpg").convert("RGB")
    cards = []
    for size in (64, 128, 256):
        record = next(
            item for item in captures
            if item["family"] == "moa" and item["seed"] == 0 and item["sample_index"] == 0
            and item["resolution"] == size and item["transformation"] == "identity"
            and item["module"] == "model.16.m.0.router"
        )
        weights = raw[record["raw_keys"]["weights"]][0]
        destination = OUTPUT / f"moa-resolution-{size}.png"
        counts = save_dominant_overlay(image, weights, _meta(record["geometry"]), str(destination), alpha=0.58)
        diagnostics = routing_diagnostics(weights)
        cards.append({
            "path": str(destination),
            "caption": f"MOA {size}px | tokens {diagnostics['token_count']} | dominant {counts}",
        })
    _save_sheet(
        cards,
        "MOA / RESOLUTION DETAIL",
        "Same image · same seed · same router layer · true argmax after exact letterbox alignment",
        OUTPUT / "moa-resolution-overview.png",
        columns=3,
    )


def _trained_mot_figure() -> None:
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"trained MOT checkpoint is unavailable: {CHECKPOINT}")
    os.environ["YOLO_CONFIG_DIR"] = str(PROJECT_ROOT / ".runtime" / "ultralytics")
    sys.path.insert(0, str(SOURCE_ROOT))
    import torch
    from ultralytics import YOLO

    model = YOLO(CHECKPOINT).model.cpu().eval()
    image_paths = sorted((PROJECT_ROOT / ".." / "datasets" / "coco8" / "images" / "val").glob("*.jpg"))
    records = []
    rendered = []
    for sample_index, image_path in enumerate(image_paths):
        image = Image.open(image_path).convert("RGB")
        canvas, geometry = letterbox(image, 160)
        tensor = torch.from_numpy(canvas.astype(np.float32).transpose(2, 0, 1) / 255.0).unsqueeze(0)
        collector = SpatialRouterCollector("mot", "_MoTRouter")
        collector.register(model)
        try:
            with torch.inference_mode():
                model(tensor)
        finally:
            collector.remove()
        for capture in collector.records:
            diagnostics = routing_diagnostics(capture.weights[0])
            records.append({
                "sample_index": sample_index,
                "sample_name": image_path.name,
                "module": capture.module_name,
                **diagnostics,
            })
            if capture.module_name == "model.19.m.0.router":
                destination = OUTPUT / f"trained-mot-sample-{sample_index}-model-19.png"
                counts = save_dominant_overlay(
                    image, capture.weights[0], geometry, str(destination), alpha=0.58
                )
                rendered.append({
                    "path": str(destination),
                    "caption": f"MOT trained | sample {sample_index} | dominant {counts}",
                })
    _save_sheet(
        rendered,
        "MOT / TRAINED CHECKPOINT SPATIAL ROUTING",
        "model.19.m.0.router · four COCO8 images · colors are true dominant experts",
        OUTPUT / "trained-mot-spatial-overview.png",
        columns=2,
    )
    module_summary = {}
    for module in sorted({item["module"] for item in records}):
        items = [item for item in records if item["module"] == module]
        module_summary[module] = {
            "active_dominant_experts_max": max(item["active_dominant_experts"] for item in items),
            "dominant_token_count_total": np.sum(
                np.asarray([item["dominant_token_count"] for item in items], dtype=np.int64), axis=0
            ).tolist(),
            "top1_margin_mean": float(np.mean([item["top1_margin"]["mean"] for item in items])),
            "neighbor_probability_l1_mean": float(
                np.mean([item["neighbor_probability_l1_mean"] for item in items])
            ),
        }
    write_json(OUTPUT / "trained-mot-audit.json", {
        "status": "PASS",
        "checkpoint_path_local_only": str(CHECKPOINT),
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "checkpoint_published": False,
        "input_size": 160,
        "sample_count": len(image_paths),
        "capture_count": len(records),
        "interpretation_boundary": (
            "Colors are true dominant-expert argmax. Small margins mean a multicolor map is not, by itself, "
            "evidence of stable semantic specialization."
        ),
        "by_module": module_summary,
        "records": records,
    })


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    _moa_resolution_figure()
    _trained_mot_figure()
    print(OUTPUT)


if __name__ == "__main__":
    main()
