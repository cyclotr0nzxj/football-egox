# 004 Pair01 / Pair05

- 004 克隆保留了原 EgoX 与 Renderer 环境、固定推理源码、共享权重和冻结输入。控制台及实例终端确认单张空闲 A800 80GB。克隆后的 CUDA、共享权重、Pair05 49 帧冻结输入及 Pair01 Renderer CUDA 检查通过。
- 两份适配包只更换活动启动器的实例身份与输出目录。上传后逐包比对 SHA-256，通过后在同一张卡上串行执行。
- Pair01 新先验于 16:25 完成。49 帧解码和实际输入处理一致性通过，先验已取回。开场可见白鞋队友，停球段红鞋点可见，但条纹和身体碎片下无法确认单球触球。按约定未进入 EgoX 推理。[先验收据](pair01/prior-render.json) · [先验验收图](pair01/prior-review.png)
- Pair05 首次于 16:28:04 启动，第一步去噪前因实际 EXO 遮罩在 CPU、注意力在 CUDA 而退出。原 CUDA 小测将遮罩预先放在 GPU，漏测了实际路径。首次无成片。[失败日志](pair05-initial-failure/inference.log)
- 修补版只将遮罩移动到注意力张量所在设备，并用 CPU 遮罩做真实 CUDA BF16 小测。小测通过后于 16:39:31 在独立目录恢复一次。机位、输入、模型和 295 条路由均沿用冻结条件。[修补版完整性记录](pair05-device-fix/local-checks.json)
- Pair05 修补版完成 50 步去噪，耗时 1,506.9 秒，退出码 0，记录 4,000 次前景 GGA 调用。已取回，原始成片与云端 SHA-256 一致，本机通过 448×448、49 帧及逐帧 PTS 检查。[云端收据](pair05/verified-result.json) · [本机验证](pair05/verified/verification.json)
- Pair05 在 F12、F24 均未通过画面验收。锁定先验含正前方队友、球和标志盘，新成片仍以远处围网与场边为主。结果保留为实验记录。Pair01 沿用首版，Pair05 本次成片不替换已有结果。Nano 留作独立盲评，本轮没有用于判断。[Pair05 成片](pair05/verified/pair05_s0024_ego.mp4) · [Pair05 画面验收](pair05/verified/manual-review.md) · [Pair01 画面验收](pair01/manual-review.md)
- 2026-11-22 17:09（北京时间），AutoDL 容器实例列表显示 004 为「已关机」。[关机凭证](004-powered-off.jpg)

复查 Pair01 可用归档的 [渲染启动命令](launch-004/pair01-remote-command.sh) 和 [实例适配包](launch-004/pair01-first-orientation-bundle-004.zip)。Pair05 成功推理使用 [遮罩修补包](pair05-device-fix/pair05-gga-foreground-bundle-004-retry1.zip)，工作根目录、输入和权重由包内启动器校验。初次启动的包作为[失败记录](launch-004/pair05-gga-foreground-bundle-004.zip)保留。
