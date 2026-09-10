# PR 描述

## 改动摘要

建立 E3 路由透视镜最终研究包，串联 Smoke、P0、P1、P2 的验收证据，并新增 MOT/MOA 三条件、三种子消融。分析使用临时 forward hook 捕获真实空间路由概率，不修改腾讯模型核心 `forward`；统一保存逐层 capture、外观对照、汇总统计、完整性 manifest 和可视化。

消融显示：MOT 经迁移后训练 10 epochs，从固定单专家变为平均 2.46 个活跃专家，54.2% 捕获中三个专家全部参与，空间变化从 0 增至 0.0882，外观一致率为 96.8%，通过预注册判读线。MOA 的全专家率从 77.1% 增至 91.7%，但路由间隔下降且空间变化增幅不足，因此报告为覆盖改善、分工清晰度未获充分支持。

## 测试证据

```text
Final ablation unit tests: 4 passed
Smoke tests: passed
P0 tests: 11 passed
P1 tests: 5 passed
P2 tests: 36 passed
```

正式 P1 GPU 测量覆盖 3 个 seeds、18 个 paired runs。MOE/MOT/LATENT 中位训练减速分别为 -0.95%、+2.15%、+1.66%，均低于 10%。最终消融保存 1,728 份 capture 与 1,440 份外观 comparison，结果文件由 SHA-256 manifest 锁定。

## 消融数据

所有条件固定 COCO8、imgsz、增广、验证口径和 seeds 0/1/2：随机初始化、腾讯 `yolo26n.pt` 仅迁移、迁移后训练 10 epochs。MOT 满足空间变化与全专家活跃率两项判据，且外观一致率下降 3.17 点，未触发 5 点红线。MOA 只满足全专家活跃率判据，外观稳定性守线，但不宣称分工已变清晰。逐 seed 结果、均值/样本标准差和图表见 `docs/FINAL_RESEARCH_REPORT.md`。

## 已知局限

COCO8 太小且与预训练域重合，检测 mAP 只作流水线检查。独立 seed 间专家 ID 可置换，因此不直接比较专家编号。dominant map 对近似平票敏感，需与概率 MAE、margin 一起阅读。五族均有结构化 capability 声明，但当前只有 MOT/MOA 暴露可与像素对齐的真实空间路由格。检查点因体积和授权边界不上传，仅发布哈希、训练曲线与复现配置。
