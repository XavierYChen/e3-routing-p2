# 改动摘要

增加训练后 MOT/MOA 路由分析入口，加载 P1 的 10-epoch 迁移训练 checkpoint，在 4 张 COCO8 图像和 320px 输入上采集全部空间路由层。新增外观敏感性、路由层归因、概率变化散点图、专家使用柱图和八层原图叠加，并保留原有五族接口审计与交互 UI。

# 测试证据

- `python -m pytest -q`：36/36 通过。
- 192 次真实 capture、160 组 identity 对齐比较。
- MOT/MOA checkpoint SHA-256 分别为 `223a9006…7c375`、`4624536b…d081`。
- `manifest.sha256.json` 覆盖配置、逐 capture/比较数据和全部图像。
- forward hook 每次推理后移除；腾讯源码与核心 forward 无修改。

# 消融数据

| 族 | 最低主导专家一致率 | 最大 probability MAE | 主要敏感层 |
|---|---:|---:|---|
| MOT | 95.33% | 0.02098 | model.22（41.4%～48.4%） |
| MOA | 96.34% | 0.000667 | model.16（31.3%～33.2%） |

扰动包括 brightness 0.9/1.1、contrast 0.9/1.1 和 Gaussian blur 0.75。该表是单 checkpoint seed 的诊断，不作为正式多 seed 消融结论。

# 已知局限

- 当前训练后分析只有 seed 0；完整消融需至少 3 个独立 checkpoint seed。
- COCO8 太小且位于预训练域，只适合工具验收。
- 归因是各层 probability MAE 的描述性份额，不是因果归因。
- 颜色表示 token 的 argmax expert ID，不是聚类标签或专家语义；MOA 概率仍接近三等分。
- 五族均已审计，但只有 MOT/MOA 保留可逆空间轴；MOE、LATENT、MoLoRA 不生成虚构热图。
