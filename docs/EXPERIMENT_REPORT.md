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

## 7. 三 seed 分辨率与翻转稳定性

配置为 `configs/robustness.yaml`，固定 CPU、4 张 COCO8、seed 0/1/2、64/128/256 三档输入以及 identity/horizontal flip。随机初始化下完成 576 次空间捕获和 480 组比较，所有 12 组代表性 hook/重复性约束为 PASS。

| MOA 比较 | 主导专家一致率 | 概率 MAE | 解读 |
|---|---:|---:|---|
| flip @ 64 | 82.01% | 3.99e-7 | 低分辨率下 argmax 更敏感 |
| flip @ 128 | 84.78% | 3.42e-7 | 一致率提高 |
| flip @ 256 | 85.23% | 3.05e-7 | 三档中最高 |
| 64 → 128 | 83.64% | 2.72e-7 | 跨尺度离散标签仍有变化 |
| 64 → 256 | 78.60% | 3.55e-7 | 尺度跨度更大，一致率下降 |

| 输入尺寸 | 支持前景/背景比较 | 前景 token | 背景 token | padding token |
|---:|---:|---:|---:|---:|
| 64 | 11/16 | 103 | 201 | 96 |
| 128 | 16/16 | 409 | 799 | 392 |
| 256 | 16/16 | 1,621 | 3,179 | 1,600 |

概率 MAE 远低于百万分之一，同时 top-1 margin 也在百万分之一量级。因此离散一致率必须和连续概率误差一起汇报。MOT 冷启动图完全相等，但其空间常量输入令 Pearson 相关系数无定义；这不是学得的鲁棒性。128px 已消除 64px 下全部 5 个 `INSUFFICIENT_TOKENS`，是区域分析更合适的最低档。

## 8. 训练 MOT checkpoint 补充审计

本机 checkpoint SHA-256 为 `8658764afae0a471e524f7b85f8066c36210c29118c45dac56fe00b8b6591d59`。对 4 张 COCO8、160px 输入完成 16 次 capture，权重文件不上传。

| MOT 层 | 四图主导 token 合计 | 最大活跃专家数 | Top-1 margin 均值 | 邻域概率 L1 均值 |
|---|---:|---:|---:|---:|
| model.13 | `[0, 0, 400]` | 1 | 2.09e-5 | 5.54e-8 |
| model.16 | `[1600, 0, 0]` | 1 | 4.84e-5 | 3.27e-7 |
| model.19 | `[75, 8, 317]` | **3** | 4.15e-5 | **2.96e-2** |
| model.22 | `[100, 0, 0]` | 1 | 1.0 | 0 |

训练后 `model.19` 的多色区域来自真实 Top-K 路由，证明代码可以展示 MOT 多专家空间分配。三层仍单色、model.19 margin 仍小，所以当前证据不支持“各专家已经学到稳定物体语义”的结论。进一步结论需要保存完整训练配置，并按同预算至少运行 3 个 checkpoint seed。

## 9. 公开产物边界

| 类别 | GitHub | 本地 |
|---|---|---|
| 正式图、聚合表、环境、fingerprint、manifest | ✅ | ✅ |
| 逐捕获元数据、逐项稳定性比较 | 摘要与哈希 | ✅ |
| 原始 routing weights/logits/indices | ❌ | ✅ |
| 训练 checkpoint | ❌，仅公布 SHA-256 | ✅ |

## 10. 训练后外观敏感性与路由归因

P1 新训练的 MOT/MOA checkpoint 使用相同 `yolo26n.pt` 起点、COCO8、640px、10 epochs、batch=1、seed 0。P2 分析固定 320px，对 identity 加上亮度 0.9/1.1、对比度 0.9/1.1 和 Gaussian blur 0.75，覆盖 4 图、2 族、每族 4 层。

| 证据 | 数量 |
|---|---:|
| 空间 capture | 192 |
| identity 对齐比较 | 160 |
| checkpoint seed | 1 |
| MOT checkpoint SHA-256 | `223a9006…7c375` |
| MOA checkpoint SHA-256 | `4624536b…d081` |

| 扰动 | MOT 一致率 | MOT 概率 MAE | MOA 一致率 | MOA 概率 MAE |
|---|---:|---:|---:|---:|
| brightness 0.9 | 96.34% | 0.01551 | 97.41% | 0.000490 |
| brightness 1.1 | 96.12% | 0.02008 | 96.34% | 0.000667 |
| contrast 0.9 | 96.84% | 0.01400 | 97.41% | 0.000525 |
| contrast 1.1 | 97.18% | 0.01310 | 97.24% | 0.000539 |
| blur 0.75 | 95.33% | 0.02098 | 96.43% | 0.000657 |

专家使用柱图表明 MOT 前三层三个专家都获得非零概率质量，解决了旧 checkpoint 多层 `active experts=1` 的展示问题；末层仍因 Top-K 路由令 E2 为 0。MOA 四层平均概率接近三等分，这说明其彩色空间边界的置信间隔可能仍较小，不能把颜色直接解释成语义聚类。

层归因按同一扰动下四个模块的平均 probability MAE 归一化。MOT 的 `model.22` 始终贡献最大（41.4%～48.4%）；MOA 的 `model.16` 最大（31.3%～33.2%）。该统计描述路由概率变化落在哪一层，不证明该层造成检测结果变化。

散点图同时展示连续概率 MAE 与离散 expert switch。MOT 的概率移动更大；MOA 大量点靠近 MAE=0，但仍存在换色，符合“近并列 argmax 对微小变化敏感”的机制。所有数字来自 [`trained-routing-analysis-20260909`](../artifacts/p2/trained-routing-analysis-20260909/summary.json)，当前只有一个 checkpoint seed，正式稳健性/专门化结论需在后续消融补齐至少 3 seed。
