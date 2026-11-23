# Pair04 本地准备 · 2026-11-21

EXO 试片通过完整 CPU 解码、SHA-256、49 帧逐帧 PTS、裁剪溯源和生成输入目录隔离检查。Nano 仅留在本机作独立盲评。[CPU 结果](cpu_dry_run.json) · [EXO 暂存包](pair04_exo_stage.zip) · [独立解压复验](bundle_check.json)

本机缺少 CUDA、Renderer/ViPE 运行环境及相关权重。当前没有生成深度、相机轨迹、点云、先验或第一视角视频。云盘缓存状态仍待开卡后验证。[本机预检](local_gate.json) · [开卡命令和阶段门槛](runbook.md)

独立的 Responses API 规划轮完成 5 次请求，合计输入 762、输出 3,812 tokens。结果仅作为技术建议，实际接口以固定源码和本地检查为准。调用证据见 [api_usage.json](api_usage.json)。

Pair04 后续经授权的主力执行使用官方 Responses API，固定 `model=gpt-6.2-sol`、`reasoning.effort=max`。本轮 5 次规划请求已结束。ViPE 与 EgoX 的视频计算仍由 GPU 环境执行。
