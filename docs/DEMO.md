# 两分钟交互演示与录屏脚本

## 启动

双击仓库根目录 `run_trained_demo.cmd`。页面在 `http://127.0.0.1:8767/trained-demo.html` 打开。该页面读取 10-epoch checkpoint 的已归档统计与叠加图。`record_demo_video.cmd` 可直接重建恰好 120 秒的 MP4，无需人工录屏。

## 120 秒讲解顺序

| 时间 | 页面操作 | 讲解重点 |
|---:|---|---|
| 0–12s | 展示训练后总览 | 192 次 capture、160 组比较、4+4 个空间层、0 次 forward 修改 |
| 12–54s | 依次切 MOT 13/16/19/22 层 | 活跃专家为 2/2/3/2；只有 19 层三色齐全，未人为补色 |
| 54–76s | 切 MOA 16/19 层 | 三专家均活跃；同时读取概率、熵和 margin，避免只看颜色 |
| 70–78s | Appearance inputs | 展示实际亮度、对比度和模糊输入，空间几何保持不变 |
| 78–88s | Appearance metrics | 单 checkpoint 下 16 个 image×layer 单元的均值与描述性 SD |
| 88–98s | Relative attribution | 分层变化贡献是描述统计，不声称因果 |
| 98–106s | Absolute MAE | 共享 log10 色阶揭示 MOT/MOA 的绝对变化量级 |
| 106–113s | Scatter | 概率 MAE 与主导专家切换率的关系 |
| 113–120s | Expert usage | 逐层专家使用率与专门化证据边界 |

视频由浏览器真实操作页面逐场景录制，并带进度条和讲解标题。提交前可对照 `e3-p2-trained-two-minute-demo.json` 校验时长、分辨率和 SHA-256。
