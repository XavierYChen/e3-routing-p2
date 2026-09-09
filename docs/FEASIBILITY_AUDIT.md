# 五族路由可行性审计

## 判读规则

预先规定：只有非单例二维空间网格 `[B,E,H,W]` 才支持 token 原图叠加；概率必须有限、非负并在专家轴上求和为 1。若路由在空间池化后只剩 `[B,E]`，结论为 unsupported，并保留实测形状作为负结果证据。

## 结果

| 族 | 检查方式 | 模块/变体 | 实测形状 | 判定 |
|---|---|---:|---:|---|
| **MOT** | 完整 YOLO detector forward + hook | 4 routers | `[1,3,5/10/20,5/10/20]` | SUPPORTED |
| **MOA** | 完整 YOLO detector forward + hook | 4 routers | `[1,3,5/10/20,5/10/20]` | SUPPORTED |
| **MOE** | 完整 detector 内嵌 routing hook | 12 captures | `[1,2]` | UNSUPPORTED |
| **LATENT** | 完整 detector 运行态 snapshot | 3 modules | `[1,4]` | UNSUPPORTED |
| **MoLoRA** | 官方 Linear/Spatial/HybridRouter | 3 variants | `[1,4]` | UNSUPPORTED |

MOT 和 MOA 的 hook 前后 detector 输出最大绝对差均为 0，证明采集没有改变 forward。两族在完全相同的 4 张 COCO8 val 图、160 输入、CPU、seed 0 上运行。

MoLoRA 中名为 `SpatialRouter` 的实现仍会在返回前聚合空间轴；“Spatial”是内部特征处理方式，不代表输出保留 token 坐标。LATENT 同样在空间池化后只保留专家轴。此处不生成伪热图。

## 已知限制

本次模型为固定 seed 的随机初始化，验证接口、几何和可复现性，不验证训练后的专家专门化。若后续获得五族已训练 checkpoint，可沿用同一审计；接口仍无空间轴的族不会因训练而变成 token 路由。
