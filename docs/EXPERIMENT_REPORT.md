# E3 P2 实验报告

## 1. 问题与假设

目标是判断 YOLO-Master 五类混合结构中哪些路由真实保留 token 空间坐标，并在不侵入模型 forward 的条件下映射回 COCO8 原图。假设是：MOT、MOA 输出二维路由网格；MOE、LATENT、MoLoRA 在空间聚合后只提供样本级权重。

## 2. 实验合同

| 项目 | 固定值 |
|---|---|
| Tencent checkout | `dab47c5e6e3d680c8d62576cf291e66431758873`，运行时另存 dirty 列表 |
| 数据 | COCO8 val 全部 4 张图及 YOLO 标注 |
| 输入 | letterbox 160×160，RGB/255 |
| 预算 | 每族同 4 图、同 seed、同 device；batch 2/4 只做等价性复核 |
| 设备 | CPU（环境文件同时记录 CUDA 可见性） |
| seed | 0，单 seed |
| 配置 | `configs/p2_v2.yaml` |
| 路由采集 | 可移除 forward hook；腾讯源码零修改 |

## 3. 验证结果

| 检查 | MOT | MOA | 门槛 | 结论 |
|---|---:|---:|---:|---|
| 完整模型 capture 数 | 16 | 16 | >0 | PASS |
| 概率和最大误差 | ≤1e-5 | ≤1e-5 | ≤1e-5 | PASS |
| hook 输出最大差值 | 0 | 0 | 0 | PASS |
| 确定性重复最大差值 | 0 | 0 | 0 | PASS |
| batch=2 最大权重差 | 0 | 2.98e-8 | ≤1e-5 | PASS |
| batch=4 最大权重差 | 0 | 2.98e-8 | ≤1e-5 | PASS |
| hook 清理 | 完成 | 完成 | 无残留 | PASS |

## 4. 路由统计

| 指标（32 captures 聚合） | MOT | MOA |
|---|---:|---:|
| token 数 | 2,500 | 2,500 |
| 主导专家份额 | `[1.0000, 0, 0]` | `[0.1840, 0.4272, 0.3888]` |
| 归一化熵均值 | 0.605693 | 1.000000 |
| Top-1 margin 均值 | 0.040000 | 3.116e-6 |
| 邻域概率 L1 均值 | 0 | 1.671e-6 |
| 前景/背景 pooled TV | 0.005226 | 3.133e-7 |
| 前景/背景 paired TV 均值 | 0 | 5.633e-7 |

MOT 的前三层在随机初始化下通常为 `[0.5,0.5,0]`，末层可出现 one-hot；未选专家的 0 来自 Top-K 稀疏化。MOA 的三个概率都极接近 `1/3`，熵为 1 且 margin 约百万分之三。因此 MOA 主导专家彩色块只表示极小差值下的 argmax 边界，不代表强置信度或训练成功。

前景/背景结果是描述性诊断。随机初始化、单 seed、仅 4 张图不足以形成统计推断；这里的负结果符合预定义判读线，不包装成性能提升。

## 5. 产物

| 文件 | 用途 |
|---|---|
| `routing-overview.png` | MOT/MOA 代表层总览 |
| `demo.html` + `demo-index.json` | 交互 UI 与索引 |
| `spatial-routing-raw.npz` | 本地生成的原始概率、logits、索引、mask；不公开上传 |
| `spatial-captures.json` | 逐样本逐层元数据 |
| `family-feasibility.json` | 五族支持矩阵与实测形状 |
| `region-routing-analysis.json` | GT 前景/背景诊断 |
| `environment.json` | Python、Torch、commit、dirty 状态 |
| `manifest.sha256.json` | 每个产物的大小与 SHA-256 |

## 6. 已知限制与下一步

- 单 seed 随机初始化只证明采集链路，不证明精度或专家分工。
- COCO8 适合 smoke/P2 工具验收，不适合泛化结论。
- 真正的训练结果需要同数据、同预算、同增广、至少 3 seed 的 checkpoint 对照；P2 工具无需训练 600 epochs 才能验收。
- 五族中只有两族有可逆空间轴；其余三族的缺口已进入 schema 和 UI。
