# MOT demo checkpoint provenance

- Purpose: make the token-level routing grid visible after a small, real training run; no accuracy claim.
- Model config: `ultralytics/cfg/models/26/yolo26-master-mot-n.yaml`
- Data: COCO8 train split, 4 images.
- Budget: 10 epochs, batch 2, image size 160, CPU, float32.
- Seed: 0, deterministic mode enabled.
- Augmentation: Tencent/Ultralytics defaults recorded by the fixed training entry point.
- Entry point: `scripts/train_mot_demo.py`.
- Retained artifact: `weights/best.pt`; `results.csv` contains the training trace.
- Limitation: one seed and a tiny dataset. This checkpoint is only functional visualization evidence and must not support a claim of accuracy improvement.

The committed files intentionally omit the generated `args.yaml` because it contains machine-specific absolute paths. The complete portable configuration is captured here and in `configs/p2.yaml`.
