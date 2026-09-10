# 检查点复现命令

消融使用 P1 的 snapshot-safe 训练入口，避免运行时路由快照被序列化进检查点。对 `mot`/`moa` 和 seeds `0`/`1`/`2` 分别运行以下命令，并替换花括号：

```bat
D:\AI\envs\yolo_master\python.exe D:\AI\E3-Routing-P1\scripts\train_specialization.py ^
  --yolo-root D:\AI\YOLO-Master ^
  --family {mot_or_moa} ^
  --weights D:\AI\YOLO-Master\yolo26n.pt ^
  --data D:\AI\E3-Routing-P2\configs\coco8-local.yaml ^
  --project D:\AI\tmp\e3-ablation-checkpoints ^
  --name {family}-pretrained-10e-seed{seed} ^
  --summary D:\AI\E3-Routing-Final\results\checkpoint-training\seed-{seed}\{family}-summary.json ^
  --epochs 10 --imgsz 640 --batch 1 --device 0 --seed {seed}
```

训练脚本拒绝覆盖已有运行目录。运行完成后，把 `configs/ablation.yaml` 中相应 checkpoint 路径指向生成的 `best.pt`，再执行仓库根目录的 `run_ablation.cmd`。
