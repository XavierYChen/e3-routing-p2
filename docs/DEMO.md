# 两分钟交互演示与录屏脚本

## 启动

双击仓库根目录 `run_demo.cmd`。页面在 `http://127.0.0.1:8766/demo.html` 打开。推荐用 Windows 自带截图工具录屏或 OBS，画面 1920×1080，浏览器缩放 90%–100%。录屏只需操作网页，不需要展示终端。

## 120 秒讲解顺序

| 时间 | 页面操作 | 讲解重点 |
|---:|---|---|
| 0–15s | 展示顶部五族卡片 | 五族均审计；MOT/MOA 支持 token overlay，另外三族保留 unsupported 证据 |
| 15–35s | MOT，sample 0，切 Expert 0/1/2 | Top-K=2；某专家精确为 0 是预期稀疏语义 |
| 35–50s | 切换 entropy 与 margin | 固定 `[0,1]` 尺度，不用单图 min-max 夸大差异 |
| 50–70s | 切 sample 2、layer 16 | 4 张 COCO8、4 个 router 层均可交互查看 |
| 70–90s | 切 MOA dominant，再切 probability | 彩色 argmax 很鲜明，但概率约 1/3、margin 极小，不能解释成已学到专门化 |
| 90–105s | 勾选 ground-truth boxes | token 中心映射到前景/背景，letterbox padding 被排除 |
| 105–115s | 展示统计卡和 metadata | 原始形状、概率、熵、margin、FG/BG TV 均可追溯 |
| 115–120s | 回到五族卡片 | 总结：真实接口、无侵入 hook、五族缺口明确 |

录制后不要再拼接静态图。若需要提交 MP4，用浏览器录屏产生的动态操作视频；页面本身已是正式演示交付物。
