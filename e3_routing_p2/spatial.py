"""Geometry, schema validation, and path-safety helpers."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

SCHEMA_VERSION = "e3.spatial_routing.v1"
SUPPORTED_FAMILIES = frozenset({"moe", "mot", "latent"})


def resolve_allowed_file(path: Path, allowed_roots: list[Path]) -> Path:
    """Resolve an input file and reject anything outside explicit roots."""
    target = path.expanduser().resolve(strict=True)
    roots = [root.expanduser().resolve(strict=True) for root in allowed_roots]
    if not target.is_file() or not any(target == root or root in target.parents for root in roots):
        raise ValueError("input path is outside the configured whitelist")
    return target


def image_identity(path: Path, dataset_root: Path) -> dict[str, str]:
    """Return a portable, non-sensitive identity without an absolute path."""
    return {
        "dataset_relative_path": path.relative_to(dataset_root).as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def letterbox(image_bgr: np.ndarray, size: int) -> tuple[np.ndarray, dict[str, Any]]:
    h, w = image_bgr.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(image_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    left, top = (size - nw) // 2, (size - nh) // 2
    right, bottom = size - nw - left, size - nh - top
    canvas = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return canvas, {
        "method": "letterbox",
        "scale": scale,
        "padding_ltrb": [left, top, right, bottom],
        "resized_wh": [nw, nh],
        "input_wh": [size, size],
        "original_wh": [w, h],
    }


def project_to_original(values: np.ndarray, transform: dict[str, Any]) -> np.ndarray:
    """Upsample a feature grid, remove letterbox padding, and restore original pixels."""
    size = transform["input_wh"][0]
    up = cv2.resize(values.astype(np.float32), (size, size), interpolation=cv2.INTER_LINEAR)
    left, top, right, bottom = transform["padding_ltrb"]
    crop = up[top : size - bottom, left : size - right]
    return cv2.resize(crop, tuple(transform["original_wh"]), interpolation=cv2.INTER_LINEAR)


def normalized_entropy(maps: np.ndarray) -> np.ndarray:
    probs = np.clip(maps.astype(np.float32), 1e-8, 1.0)
    return -(probs * np.log(probs)).sum(axis=0) / math.log(max(probs.shape[0], 2))


def validate_record(record: dict[str, Any]) -> None:
    required = {
        "schema_version", "family", "family_display", "layer", "source", "spatial_granularity",
        "feature_grid_hw", "num_experts", "top_k", "probability_sum_max_error", "artifacts",
    }
    missing = sorted(required - record.keys())
    if missing:
        raise ValueError(f"missing schema fields: {missing}")
    if record["schema_version"] != SCHEMA_VERSION or record["family"] not in SUPPORTED_FAMILIES:
        raise ValueError("unsupported schema version or family")
    if record["family_display"] != record["family"].upper():
        raise ValueError("family_display must be uppercase")
    if record["spatial_granularity"] not in {"token", "image"}:
        raise ValueError("invalid spatial granularity")
    if record["probability_sum_max_error"] > 1e-4:
        raise ValueError("router probabilities do not sum to one")

