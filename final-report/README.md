# E3 五族混合系统路由透视镜｜最终研究包

本仓库汇总 Smoke、P0、P1、P2 的验收证据，并给出独立的三种子路由分工消融。它是项目导航与最终 PR 研究报告；各阶段仍保留在独立仓库，便于逐项验收。

| 阶段 | 目标 | 主要结果 | 仓库 |
|---|---|---|---|
| Smoke | COCO8 单次 routing snapshot、字段与开销方案 | MOE/MOT/LATENT 三族通过准入，明确随机初始化边界 | [e3-routing-smoke](https://github.com/XavierYChen/e3-routing-smoke) |
| P0 | 统一 schema、三族结构化日志与静态图 | 13 条有效记录，MOE/MOT/LATENT 通过统一契约 | [e3-routing-p0](https://github.com/XavierYChen/e3-routing-p0) |
| P1 | 实时面板、三族、训练减速低于 10% | 3 seeds / 18 paired runs；三族中位减速均低于 10% | [e3-routing-p1](https://github.com/XavierYChen/e3-routing-p1) |
| P2 | token 路由叠加原图、五族覆盖、演示与分析 | MOT/MOA 真空间图；其余三族明确使用全局/非空间降级；补齐敏感性与归因图 | [e3-routing-p2](https://github.com/XavierYChen/e3-routing-p2) |
| Final | 三条件消融与总报告 | MOT 支持“训练形成空间分工”；MOA 仅支持“覆盖增加” | 本仓库 |

![三条件消融总览](results/routing-ablation-20260909/ablation-dashboard.png)

消融比较随机初始化、仅迁移腾讯 `yolo26n.pt`、迁移后训练 10 epochs 三种条件。所有组使用同一 COCO8、同一输入尺寸、同一外观扰动和 3 个 seeds。共保存 1,728 份路由捕获及 1,440 组对照。

| 族 | 条件 | mAP50–95 | 平均活跃专家 | 三专家全活跃率 | 空间变化 | 外观一致率 |
|---|---:|---:|---:|---:|---:|---:|
| MOT | 随机初始化 | 0.0000 | 1.00 | 0.0% | 0.0000 | 100.0% |
| MOT | 仅迁移 | 0.0059 | 1.00 | 0.0% | 0.0000 | 100.0% |
| MOT | 迁移 + 10 epochs | 0.0225 | 2.46 | 54.2% | 0.0882 | 96.8% |
| MOA | 随机初始化 | 0.0000 | 2.25 | 50.0% | 0.0000 | 98.4% |
| MOA | 仅迁移 | 0.0368 | 2.75 | 77.1% | 0.0047 | 98.1% |
| MOA | 迁移 + 10 epochs | 0.0359 | 2.92 | 91.7% | 0.0050 | 96.6% |

MOT 满足预先定义的 2/3 路由判据，并守住外观一致率下降不超过 5 个百分点的红线。MOA 只满足专家覆盖率判据，因此不宣称其专家分工已变得更清晰。检测指标仅用于确认流水线可工作：COCO8 验证集只有 4 张图并与预训练域重合，不能代表模型泛化性能。

完整方法、逐阶段证据、逐 seed 表格、局限和 PR 模板见 [最终研究报告](docs/FINAL_RESEARCH_REPORT.md) 与 [PR 描述](docs/PR_DESCRIPTION.md)。[检查点训练命令](scripts/TRAINING_COMMANDS.md) 记录六组固定预算训练；运行 `run_ablation.cmd` 可从本地检查点复现消融。检查点体积较大且不上传，报告记录了 SHA-256。
