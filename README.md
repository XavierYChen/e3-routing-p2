# E3 Routing Lens · P2

腾讯犀牛鸟 E3「五族混合系统的路由透视镜」P2 独立仓库。本版在腾讯 YOLO-Master 的**完整检测模型 forward**上，以可移除 hook 采集真实路由张量；不修改腾讯源码，不再使用旧版 router-only replay。

阶段导航：[Smoke](https://github.com/XavierYChen/e3-routing-smoke) · [P0](https://github.com/XavierYChen/e3-routing-p0) · [P1](https://github.com/XavierYChen/e3-routing-p1) · **P2（本仓库）**

![MOT 与 MOA token 路由总览](artifacts/p2/p2-v2-five-family-final/routing-overview.png)

## 验收结论

| 路由族 | 实测输出 | token 热图 | P2 处理 |
|---|---:|:---:|---|
| **MOT** | `[B,3,H,W]`，Top-K=2 | ✅ | 概率、主导专家、熵、margin、原图叠加 |
| **MOA** | `[B,3,H,W]` | ✅ | 概率、主导专家、熵、margin、原图叠加 |
| **MOE** | `[B,2]` | ❌ | 完整模型审计并记录 unsupported |
| **LATENT** | `[B,4]` | ❌ | 完整模型运行态审计并记录 unsupported |
| **MoLoRA** | `[B,4]` | ❌ | 官方三种 router 合约审计并记录 unsupported |

“五族完成”指五族都经过真实接口审计，并不等于五族都有空间 token 轴。给 MOE、LATENT 或 MoLoRA 强行画区域热图会虚构空间信息，因此本仓库明确拒绝。

## 为什么会看到 0 或鲜艳色块

MOT 的 Top-K=2 会把三个专家中未选中的一个概率精确置零，所以 `min probability = 0 (exact)` 是稀疏路由的正确结果。MOA 在随机初始化时接近均匀分配，整体均值约为 `1/3`，Top-1 margin 只有 `3.12e-6`；主导专家图把极小的大小关系画成离散颜色，视觉上很鲜艳，却不能当作已学到专家分工。WebUI 同时给出绝对概率、熵、margin 与前景/背景统计，避免只凭颜色解读。

## 本次证据

| 项目 | 结果 |
|---|---:|
| COCO8 val 图像 | 4/4 |
| 完整模型空间捕获 | 32（MOT 16 + MOA 16） |
| 可切换视图 | 192 |
| 原始数组 | 240（本地生成；GitHub 仅提交脱敏统计） |
| hook 前后 detector 最大差值 | `0` |
| 确定性重复最大差值 | `0` |
| 单图 vs batch=2/4 | PASS，最大差值 `2.98e-8` |
| 源文件 SHA-256 | 9/9 PASS |
| seed | 0（单 seed，功能证据） |

详细表格见 [实验报告](docs/EXPERIMENT_REPORT.md)，五族判定依据见 [可行性审计](docs/FEASIBILITY_AUDIT.md)，字段定义见 [Schema](docs/SCHEMA.md)。

## 打开交互 UI

双击 `run_demo.cmd`，浏览器会打开 `http://127.0.0.1:8766/demo.html`。可以切换 **MOT/MOA、4 张图片、4 个路由层、6 类视图**，也可显示 COCO 标注框并导出当前图。

两分钟现场演示按 [演示与录屏脚本](docs/DEMO.md) 操作。这里采用真实交互页面，而不是把静态图拼成假视频。

## 从零复现

1. 保持 `D:\AI\YOLO-Master`、`D:\AI\datasets\coco8` 与 `D:\AI\envs\yolo_master` 可用。
2. 双击 `run_p2_v2.cmd`。
3. 运行结束后双击 `run_demo.cmd`。

等价命令：

```bat
cd /d D:\AI\E3-Routing-P2
set "PYTHONPATH=D:\AI\E3-Routing-P2\src"
D:\AI\envs\yolo_master\python.exe -m e3_p2 run --config configs\p2_v2.yaml --run-id my-p2-run
```

配置固定 CPU、seed 0、COCO8 val 全部 4 张图、`imgsz=160` 和 batch 等价性检查。结果验证工具链，不声明检测精度提升或已形成专家专门化。

`spatial-routing-raw.npz` 会在本地正式结果目录生成，但不上传 GitHub；仓库提交逐层统计、形状、校验误差和 SHA-256 清单。这样能验收 schema 与数值口径，同时避免公开原始 logits/weights。

## 安全与来源边界

- 只读取白名单内的 YOLO-Master、COCO8 与本仓库路径。
- forward hook 在每次采集后移除；输出等价性为强制检查。
- 日志不记录凭据，不执行任意 shell，不自动上传数据。
- COCO8 图像遵循其原许可；腾讯源码不在本仓库重新分发。
- 参考项目只用于核对验收口径和交互设计，来源见 [NOTICE](NOTICE.md)。
