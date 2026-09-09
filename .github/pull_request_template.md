## 改动摘要

- 五族路由接口审计；仅对真实 `[B,E,H,W]` 的 MOT/MOA 生成 token 叠加。
- 完整 detector forward hook、固定尺度图、交互 UI、GT 区域分析与复现证据。

## 测试证据

- `run_tests.cmd`
- `run_p2_v2.cmd --run-id p2-v2-five-family-final`
- 概率语义、hook 等价、确定性、batch 2/4、几何、资源清单全部 PASS。

## 消融数据

- 同 COCO8 val 4 图、同 160 输入、同 CPU、同 seed 0 对比 MOT/MOA。
- 当前是单 seed 功能证据，不声明检测精度或训练收益。

## 已知局限

- MOE、LATENT、MoLoRA 输出没有可逆 H/W token 轴，明确标记 unsupported。
- 随机初始化的 MOA 近似均匀；鲜艳 argmax 色块不代表强路由置信度。
- 动态演示由本地 WebUI 录屏，仓库不提交静态图片拼接视频。
