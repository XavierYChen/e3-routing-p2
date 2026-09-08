"""Run the E3 P2 spatial routing evidence pipeline."""

from __future__ import annotations

import argparse, hashlib, json, os, platform, random, subprocess, sys, time
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from .collector import SpatialRoutingCollector
from .render import make_demo, render_comparison, render_family, write_viewer
from .spatial import SCHEMA_VERSION, image_identity, letterbox, resolve_allowed_file, validate_record

ROOT = Path(__file__).resolve().parents[1]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--yolo-root", type=Path, default=Path(os.environ.get("E3_YOLO_MASTER_ROOT", ROOT.parent / "YOLO-Master")))
    p.add_argument("--dataset-root", type=Path, default=ROOT.parent / "datasets" / "coco8")
    p.add_argument("--image", type=Path)
    p.add_argument("--output", type=Path, default=ROOT / "results" / datetime.now(UTC).strftime("run-%Y%m%d-%H%M%S"))
    p.add_argument("--imgsz", type=int, default=160)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--mot-weights", type=Path, default=ROOT / "checkpoints" / "mot-coco8-seed0" / "weights" / "best.pt")
    return p


def git_value(root: Path, *args: str) -> str:
    cp = subprocess.run(["git", "-c", f"safe.directory={root.as_posix()}", "-C", str(root), *args], capture_output=True, text=True)
    return cp.stdout.strip()


def load_model(config: Path, device, weights: Path | None = None):
    from ultralytics.nn.tasks import DetectionModel
    if weights is not None and weights.is_file():
        from ultralytics import YOLO
        return YOLO(str(weights)).model.to(device).eval().float()
    return DetectionModel(str(config), ch=3, verbose=False).to(device).eval().float()


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    args.yolo_root, args.dataset_root = args.yolo_root.resolve(), args.dataset_root.resolve()
    if not (args.yolo_root / "ultralytics").is_dir(): raise SystemExit("YOLO-Master checkout not found")
    sys.path.insert(0, str(args.yolo_root))
    os.environ.update({"YOLO_AUTOINSTALL": "false", "MOE_SNAPSHOT_INTERVAL": "1"})
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "env" / "runtime"))
    import torch
    candidates = sorted((args.dataset_root / "images" / "val").glob("*.jpg"))
    raw_image = args.image or (candidates[0] if candidates else None)
    if raw_image is None: raise SystemExit("no COCO8 validation image found")
    image_path = resolve_allowed_file(raw_image, [args.dataset_root])
    image = cv2.imread(str(image_path))
    if image is None: raise SystemExit("image could not be decoded")
    boxed, transform = letterbox(image, args.imgsz)
    rgb = np.ascontiguousarray(boxed[..., ::-1].transpose(2, 0, 1))
    batch = torch.from_numpy(rgb).unsqueeze(0).to(args.device, dtype=torch.float32).div_(255)
    profiles = {"moe": "yolo26-master-n.yaml", "mot": "yolo26-master-mot-n.yaml", "latent": "yolo26-master-latent-n.yaml"}
    args.output.mkdir(parents=True, exist_ok=False)
    art_dir = args.output / "artifacts"; art_dir.mkdir()
    cv2.imwrite(str(art_dir / "source.jpg"), image)
    records, rendered, timings = [], {}, {}
    for family, filename in profiles.items():
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        config = args.yolo_root / "ultralytics" / "cfg" / "models" / "26" / filename
        model = load_model(config, args.device, args.mot_weights if family == "mot" else None)
        start = time.perf_counter()
        with SpatialRoutingCollector(model, family) as collector, torch.no_grad(): model(batch)
        timings[family] = round((time.perf_counter() - start) * 1000, 3)
        capture = collector.representative()
        family_render = render_family(image, transform, family, capture, art_dir)
        rendered[family] = family_render
        record = {
            "schema_version": SCHEMA_VERSION, "family": family, "family_display": family.upper(),
            "layer": capture["layer"], "source": capture["source"],
            "spatial_granularity": capture["spatial_granularity"],
            "feature_grid_hw": list(capture["maps"].shape[-2:]), "num_experts": capture["num_experts"],
            "top_k": capture["top_k"], "probability_sum_max_error": capture["probability_sum_max_error"],
            "mean_router_probs": [round(float(x), 8) for x in capture["maps"].mean((1,2))],
            "transform": transform, "artifacts": family_render["files"],
            "representative_expert_probability": family_render["selected_stats"],
            "semantics": "pixel-aligned token probabilities" if family == "mot" else "image-level mixture broadcast for display",
            "downstream_consumers": ["B1", "D1", "A3", "WebUI"],
        }
        validate_record(record); records.append(record)
    comparison = args.output / "routing_spatial_comparison.png"
    render_comparison(image, rendered, comparison)
    write_viewer(records, args.output)
    make_demo(comparison, records, args.output / "e3_p2_demo_120s.mp4")
    identity = image_identity(image_path, args.dataset_root)
    configs = {}
    for f, name in profiles.items():
        p = args.yolo_root / "ultralytics" / "cfg" / "models" / "26" / name
        configs[f] = {"path": f"ultralytics/cfg/models/26/{name}", "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    if args.mot_weights.is_file():
        configs["mot"]["checkpoint"] = {"path": args.mot_weights.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(args.mot_weights.read_bytes()).hexdigest(), "training": "COCO8, 10 epochs, seed 0; visualization checkpoint only"}
    payload = {
        "status": "passed", "schema_version": SCHEMA_VERSION, "run_id": args.output.name,
        "input": identity, "runtime": {"device": args.device, "seed": args.seed, "imgsz": args.imgsz, "batch": 1, "dtype": "float32", "forward_ms": timings},
        "version": {"yolo_master_commit": git_value(args.yolo_root, "rev-parse", "HEAD"), "yolo_master_describe": git_value(args.yolo_root, "describe", "--tags", "--always", "--dirty"), "worktree_dirty": bool(git_value(args.yolo_root, "status", "--short")), "python": platform.python_version(), "torch": torch.__version__, "opencv": cv2.__version__},
        "model_configs": configs, "records": records,
        "coverage": {"implemented": ["moe", "mot", "latent"], "unsupported": {"moa": "adapter pending; spatial router exists but not included in the P0 three-family contract", "molora": "requires an attached PEFT checkpoint and adapter-specific coordinate contract"}},
        "statistics": {"seeds": [args.seed], "claim": "functional visualization only; no accuracy improvement or decline is claimed"},
        "security": {"path_whitelist": [args.dataset_root.name], "absolute_paths_logged": False, "arbitrary_shell": False},
    }
    (args.output / "spatial_records.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    cap = cv2.VideoCapture(str(args.output / "e3_p2_demo_120s.mp4")); frames, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS); cap.release()
    summary = {"status": "passed", "families": [r["family"] for r in records], "token_level_families": [r["family"] for r in records if r["spatial_granularity"] == "token"], "image_level_families": [r["family"] for r in records if r["spatial_granularity"] == "image"], "video_seconds": frames / fps, "records": len(records)}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    files = {p.relative_to(args.output).as_posix(): {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size} for p in sorted(args.output.rglob("*")) if p.is_file()}
    (args.output / "manifest.sha256.json").write_text(json.dumps({"algorithm":"sha256","files":files}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary)); return 0


if __name__ == "__main__": raise SystemExit(main())
