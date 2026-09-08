"""Train an MOT visualization checkpoint on COCO8; this makes no accuracy claim."""

from __future__ import annotations

import os, random, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
YOLO_ROOT = Path(os.environ.get("E3_YOLO_MASTER_ROOT", ROOT.parent / "YOLO-Master")).resolve()
sys.path.insert(0, str(YOLO_ROOT))
os.environ.update({"YOLO_AUTOINSTALL": "false", "YOLO_CONFIG_DIR": str(ROOT / "env" / "runtime")})

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils import SETTINGS

random.seed(0); np.random.seed(0); torch.manual_seed(0)
SETTINGS.update({"datasets_dir": str(ROOT.parent / "datasets")})
model = YOLO(str(YOLO_ROOT / "ultralytics/cfg/models/26/yolo26-master-mot-n.yaml"))
model.train(
    data=str(YOLO_ROOT / "ultralytics/cfg/datasets/coco8.yaml"), epochs=10, imgsz=160, batch=2,
    device="cpu", workers=0, seed=0, deterministic=True, amp=False, cache=False,
    project=str(ROOT / "checkpoints"), name="mot-coco8-seed0", exist_ok=True,
    save=True, plots=False, verbose=False, val=False, warmup_epochs=0.0,
)
run = ROOT / "checkpoints" / "mot-coco8-seed0"
best = run / "weights" / "best.pt"
trained = YOLO(str(best)).model
router_state = {key: value.detach().cpu() for key, value in trained.state_dict().items() if ".router." in key}
torch.save({"format": "e3.mot_router_state.v1", "state_dict": router_state}, run / "mot_router_state.pt")
for generated in (run / "args.yaml", run / "weights" / "last.pt", run / "weights" / "last_healthy.pt"):
    generated.unlink(missing_ok=True)
best.unlink(missing_ok=True)
