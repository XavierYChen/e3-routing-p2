# E3 P2 空间路由 Schema v2.0

Schema ID：`e3.spatial-routing/v2.0.0`。GitHub 机器可读入口为 `demo-index.json`、`spatial-captures.json`、`spatial-diagnostics.json` 与 `family-feasibility.json`。完整复现时还会在本地生成 `spatial-routing-raw.npz`；该原始 logits/weights 文件不公开上传。

## 五族能力矩阵

| family | runtime evidence | observed shape | spatial granularity | overlay |
|---|---|---|---|---|
| `mot` | full detector forward hook | `[B,3,H,W]` | `token` | supported |
| `moa` | full detector forward hook | `[B,3,H,W]` | `token` | supported |
| `moe` | nested router hook | `[B,2]` | `image` | unsupported |
| `latent` | runtime snapshot | `[B,4]` | `image` | unsupported |
| `molora` | official router contract | `[B,4]` | `image` | unsupported |

只有 `H>1 且 W>1`、专家轴和为 1 的 `[B,E,H,W]` 才能进入空间叠加。`[B,E]` 或带单例空间轴的张量不能广播成 token 热图。

## capture 字段

| 字段 | 类型 | 含义 |
|---|---|---|
| `family` | string | 小写机器标识；UI 使用大写 |
| `sample_index` | int | COCO8 val 排序后的样本号 |
| `module` | string | 完整模型内 router 模块路径 |
| `weights_key` | string | NPZ 中 `[B,E,H,W]` 概率数组键 |
| `logits_key` | string | NPZ 中原始 logits 键 |
| `indices_key` | string/null | MOT Top-K 索引；MOA 为 null |
| `validation.shape` | int[4] | `[B,E,H,W]` |
| `validation.max_expert_sum_error` | float | 每 token 专家概率和对 1 的最大误差 |
| `diagnostics.normalized_entropy` | object | 以 `log(E)` 归一化到 `[0,1]` |
| `diagnostics.top1_margin` | object | 最大与次大专家概率差 |
| `region_diagnostics` | object | 前景、背景、padding 与差异统计 |

MOT 的 `indices_key` 与稀疏 `weights` 会交叉验证：已选专家概率和必须为 1，未选专家最大概率必须接近 0。因此精确零是合法值。

## 空间映射

原图先按比例 letterbox 到 160×160；每张图保存整数 padding。热图先升采样到输入平面，再严格去除 padding，最后恢复原图尺寸。区域分析以 token 中心是否落入 YOLO ground-truth box 判定前景；padding 不计入前景或背景。

## 下游消费规则

B1、D1、A3 和 WebUI 必须先读取 `family-feasibility.json`。只有 `token_overlay=supported` 才读取空间数组；其余族应展示 unsupported 原因。可视化不得对每张图单独 min-max 拉伸后冒充概率强度；概率色阶固定为 `[0,1]`。
