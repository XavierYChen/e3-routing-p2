# `e3.spatial_routing.v1` 字段字典

P2 在 P0 结构化日志之上增加空间坐标、原图身份和可视化产物。所有族名在机器字段中保持小写，在图表和界面中统一显示为 **MOE / MOT / LATENT / MOA / MOLORA**。

| 字段 | 类型 | 含义 |
|---|---|---|
| `schema_version` | string | 固定为 `e3.spatial_routing.v1` |
| `family` / `family_display` | string | 规范族名 / 大写显示名 |
| `layer` | string | 被 hook 的路由器层名 |
| `source` | string | 原始路由张量的形状语义 |
| `spatial_granularity` | enum | `token` 或 `image`；消费方必须据此决定能否做空间解释 |
| `feature_grid_hw` | `[H,W]` | 原始路由网格；图像级路由固定 `[1,1]` |
| `num_experts` / `top_k` | int | 专家数与激活数 |
| `probability_sum_max_error` | float | 专家维概率和相对 1 的最大误差，门槛 `1e-4` |
| `mean_router_probs` | float[] | 该层平均专家权重 |
| `transform` | object | letterbox 比例、四边 padding、输入与原图大小，支持逆映射 |
| `artifacts` | object | 专家图、熵图、主导专家图文件名 |
| `representative_expert_probability` | object | 代表专家原始概率的 min/mean/max/range；相对对比图必须同时提供这些真值 |
| `semantics` | string | 当前图可被怎样解释 |
| `downstream_consumers` | string[] | `B1`、`D1`、`A3`、`WebUI` |

输入身份只记录相对路径和 SHA-256。绝对数据路径、用户名、令牌和环境变量不会写入证据。

## 五族覆盖和缺口

| 族 | 状态 | 缺口或接入条件 |
|---|---|---|
| MOE | 已支持 | 当前 `EfficientSpatialRouter` 最终对空间求均值得到 `[B,E]`，所以标记 image-level |
| MOT | 已支持 | `_MoTRouter` 直接提供 `[B,E,H,W]`，是真实 token-level 空间路由 |
| LATENT | 已支持 | `LatentRouter` 对尺度 token 输出 `[B,E]`，所以标记 image-level |
| MOA | 暂不支持 | 源码存在 `[B,M,H,W]` 空间路由器，但尚未进入 P0 三族 contract；需补 P0 adapter、模型层选择和回归测试后接入 |
| MoLoRA | 暂不支持 | 需要先提供已挂载 PEFT adapter 的 checkpoint，并统一 Linear/Conv 路由坐标和 merge 状态；裸检测配置中没有可采集实例 |

降级原则是先交真实三族并显式返回 unsupported。不得用随机热图、特征激活或插值后的常数图冒充缺失族的 token 路由。

`relative_contrast` 仅对单张专家图做 `(p-min)/(max-min)`，用途是观察窄区间内的空间结构。它不改变或替代绝对概率；同一记录同时提供 absolute probability 图和原始 min/mean/max/range，禁止把相对色差解释为专家概率的大幅变化。
