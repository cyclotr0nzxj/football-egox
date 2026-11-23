# Pair04 A800 028 · 2026-11-21

在用户已开启的 A800 028 上从登录后的 JupyterLab 终端执行。已读回 NVIDIA A800 80 GB PCIe（81920 MiB，预检时占用 0 MiB）、驱动 580.65.06、CUDA 编译工具 12.8.93，数据盘 50 GB、系统盘 30 GB。界面显示计划停机 15:00（北京时间）。

数据盘没有历史 EgoX 或 ViPE 权重缓存。固定版 EgoX 的 transformer 14 个必需分片合计 61.077 GiB，大于本实例数据盘。当前准备范围为 ViPE Lyra 几何。远端工作目录为 `/root/autodl-tmp/football_egox/pair04-20261121`，源码使用官方 Renderer commit `17cdef688885c3ac6a4c371a2d620391cc28dab4` 的稀疏检出，只保留编译与运行所需目录。

Responses API 以 `gpt-6.2-sol`、`reasoning.effort=max`、standard、`store:false` 调用。首次达到输出上限，用户要求继续后第二次完成。执行意见是并行完成固定版安装及公开权重核验，确认 CUDA、私有输入授权和逐帧检查后才允许一次 ViPE；约在本轮开始后 55 分钟设截止。用量见 [api_usage.json](api_usage.json)。模型得到的是最少的去标识化技术状态。

北京时间 14:32 云端读回：官方 Renderer 与 MoGe 的固定源码已检出，独立 Python 3.10 环境已建成；安装官方完整锁定依赖的首次尝试限时退出。并行安装回退仍在下载工具，ViPE 扩展与 CUDA 内核见证尚未运行。15 个公开缓存/分词文件有 13 个通过完整哈希，缺 MoGe2 `model.pt` 与 VDA `video_depth_anything_vitl.pth`。试片暂存包不在云端，A800 显存占用 0 MiB，没有启动 ViPE 或 EgoX，也没有生成几何、先验或视频。[续跑与检查命令](RESUME.md)

用户随后将计划停机延至 16:00，已在同一实例的控制台读回。JupyterLab 文件页可访问，原终端出现 WebSocket 断连。本机经物理网卡已到达 SSH 端口，首次主机密钥尚未受信。提供了在 Mac 终端交互完成认证的 [SSH 脚本](connect-a800.sh)，通过独立控制 socket 复用会话。试片上传确认单独处理。

随后新开的 JupyterLab 终端恢复。用户授权 CLI 登录，门户终端读到的 ED25519 指纹与直连 SSH 相符，已严格认证并验证可复用连接。按延长窗口重启受限准备任务，已从通过官方 SHA-256 的 0.8.22 包中提取并运行 uv，同一版本锁的安装在进行。已测官方公开权重来源支持准确的 206 Range，API 建议只补传缺失字节并在整文件 SHA-256 通过前保留为分段。依用户最新要求，聊天端 Sol Ultra 子代理与 Responses API Sol Max 子代理并行协作。

公开权重使用 [断点补传脚本](relay_public_weights.py) 仅传所缺 153,587,572 字节，最终 MoGe2 与 VDA 的大小和官方 SHA-256 都通过，全部 **15/15** 离线资产已复验。按固定 Git 对象验收 Renderer 263 个运行文件和 MoGe 60 个运行文件，零缺失、零摘要偏差。

收到用户的明确确认后，仅上传并解压 727,441 字节的 EXO 暂存包。云端验证了相同的 ZIP SHA-256、EXO 隔离、完整 49 帧、25 fps、784×448 和逐帧 PTS。[云端 CPU 证据](cloud_cpu_dry_run.json)。[一次性启动与验收脚本](launch_vipe_remote.sh) 已排队等待真实 CUDA 内核见证。15:47 后停止等待新启动，运行最迟 15:50 截止。Nano 与现场手记留在本机。

依赖恢复时，原 Renderer requirements 锁保持原样。经原生子代理和 API 审查，仅将 MoGe 声明的额外依赖 Gradio 固定为 `5.29.0`，与该锁的 HTTPX、Pillow 和 Hugging Face Hub 版本相容。MoGe 两项 Git 依赖仍使用上游固定提交。须实际完成依赖检查与 CUDA 见证才放行。

北京时间 15:20：使用同一 uv 缓存、[已核验的 PyPI 补传](relay_locked_wheels.py) 及离线 wheelhouse 完成核心安装。原锁定的 117 个版本逐项比对通过；离线解析的传递依赖固定为 hf-xet 1.8.0、pytz 2026.6、tzdata 2026.6。ANTLR 和 calmsize 的 wheel 由 SHA-256 匹配 PyPI 元数据的精确源码包离线构建。MoGe 安装随后进入两项 Git 依赖处理。

用户再次将实例最晚关机时间设为 17:00。尚未开始时替换了旧等候队列，新队列仍自动等待同一 CUDA 见证，通过即启动。16:47 后停止等待新启动，16:50 为任务硬截止。现行远端队列日志为 `launch-queue-1650.log`。

用户随后主动关机。AutoDL 控制台已读回 028 实例为已关机，50 GB 数据盘使用率为 45.77%。关机前最后一次读回仍在处理 MoGe 附加依赖。当前没有 CUDA 见证及 Pair04 几何、先验渲染或生成视频的成功证据。

用户看到的候选主机同样只有 50 GB。完整 Wan 与 EgoX LoRA 共 35 项、93,453,078,396 字节（87.035 GiB），当前首选是复用 028 数据盘上的 ViPE 环境，确认同地区支持 AutoDL 文件存储后，将官方固定版 EgoX 权重放在 `/root/autodl-fs`，环境、临时文件和结果仍在 `/root/autodl-tmp`。共享盘的实际挂载状态及加载性能待验证，本轮未初始化存储、未重开实例。见 [续跑说明](RESUME.md)。

## 008 克隆实例

用户克隆并开机，随后初始化 A800 专区文件存储。控制台显示 200 GB 配额和 0 B 已用。通过 Jupyter 核对主机公钥指纹，再从 Mac 终端严格认证。已验证克隆保留数据盘，50 GB 中已用 23G、空闲 28G。A800 80GB PCIe 空闲，驱动 595.71.05。克隆初次启动时 `/root/autodl-fs` 不存在。EXO ZIP 的 SHA-256 与旧记录一致，15 项公开离线资产重新校验全部通过。MoGe 和 ViPE 的固定源码已在新机核对，补齐额外依赖并重建 CUDA 扩展进行中。Sol Max API 就续跑顺序给出判断，选择先完成 ViPE 并取回证据；共享权重下载须等待真实挂载。

ViPE Lyra 于 16:00:52–16:01:57 完成一次真实推理。固定源码、原生 `SE3.exp` CUDA 算子、15 项缓存、EXO 及实际产物均通过独立检查。启动脚本的 Python here-document 排版使一项预检未阻止启动，已改正脚本并单独重新执行该预检及产物验证。49 帧原始 RGB、深度、K、C2W 和实例掩码取回本机并逐文件复验 SHA-256。ViPE 在固定机位模式下给出单位位姿和固定焦距约 776.22 px，全部深度像素有限且为正。草地拟合给出 EXO 高度约 1.57 ViPE 单位，对照手记 1.23 m 提示尺度有偏差，本次保留原始估计。

基于 EXO 的 49 个手工头部锚点及 ViPE 真实深度构建虚拟相机，朝队友躯干，采用名义 95° 鱼眼视场和五帧平滑。PyTorch3D 0.7.9 在固定的 PyTorch 2.7 CUDA 12.8 环境通过实 GPU 鱼眼点云测试。原始渲染经首、中、末帧检查发现官方图像坐标相对地面有 90° 滚转，给轨迹施加反向滚转后重新渲染，保持 EgoX 光线栅格与官方 Renderer 的相同坐标约定。已核验校正先验 49 帧、448×448、30 fps，草地在下且队友竖直，有明显遮挡空洞与条纹。输出见 [008 先验](geometry-008/prior-008-upright/pair04_s0024/ego_Prior.mp4)，47 个中间帧也均已解码检查。云端已用固定版转换器生成 49 张 NPY，逐帧与 ViPE 读取逐元素一致。Nano 未参与。

随后重开同一 008 实例，确证 `/root/autodl-fs` 指向实际 FUSE 挂载 `/autodl-fs/data`，约 200 GiB 可用。固定版本 EgoX 仓库源码、W2C、EXO、先验和 49 个深度在 `egox-shared-008` 暂存就绪。经验证的下载器已开始把公开权重写到共享盘，独立 EgoX 环境正在按旧成功版本重建。用户确认 008 没有定时关机，已撤销由旧机 17:00 时间误沿用的临时关机进程。用户要求完整成片验收及取回后提前关机。

用户随后把范围扩至四段，沿用同一 008。追加 Pair01 的 EXO 可见 F288–F336、Pair02 F0–F48、Pair05 F24–F72。其余素材留作备选。三段均以独立视频和 49 个视觉头部锚点从 EXO 取证，经 ViPE 得到深度、K、位姿及动态实例，真实渲染 448×448、49 帧、30 fps 的先验。Pair01、Pair05 初版朝固定地标，抽查未覆盖队友。保留原稿后用 ViPE 实例和 Z 朝同一可见队友躯干，保留既有头部位置、95° 名义视场和滚转补偿，最终抽查首、中、末帧可见队友。四段的深度 NPY 与 ViPE 原始解码一致，最终 `inference-four/validated-inputs.json` 固定了 EXO、先验与深度哈希。同一官方 EgoX 进程按 Pair04、Pair01、Pair02、Pair05 顺序生成，权重只加载一次。

旧成功 EgoX 环境的 81 项包已在 008 安装，实际运行 `torch 2.5.1+cu124` CUDA 内核，并由 Diffusers 与 Decord 双重读取 EXO 与先验各 49 帧。uv 对旧版 Decord 给出了平台标签提示，当前真实导入和解码通过。共享盘的 safetensors 已用 PyTorch 成功读出确切 bfloat16 数据。四份先验和原始 ViPE 资料已带哈希回收。权重下载经过第一批校验后，直接向 FUSE 写大量并发分段时变慢并出现等待写入的 curl 子进程。已停止旧下载进程，继续保留下载成果并准备改用本地 SSD 分段、共享盘单文件写入。

数据盘暂存、共享盘单写已经在 008 实测通过正式权重校验。17:04:51 从 GPU 模式停机，17:07:45 同一实例无卡续传，实际读回无 GPU、2 GiB 内存、共享盘正常挂载。界面显示无卡 ¥0.10/小时。Wan 下载继续，LoRA 临时移至 Mac 到 008 的公开模型分块补传；首轮保留 112 MiB，外部模型 API `gpt-6.2-sol / max` 给出三路与有限重试策略，已经实际复用受校验的分块。获准的 GPU 仅在完整权重验收后重开。

18:03:34，官方固定版的 Wan 与 LoRA 共 35 项、93,453,078,396 字节全部通过校验并写入共享盘。已从 008 取回 [校验清单](multiclip-evidence/model-verification-008.json) 和 [无卡下载日志](multiclip-evidence/download-cpu-only-008.log)。本机复验所有路径、大小、已校验标记和 LFS SHA-256 与固定清单一致，回传清单和启动脚本的摘要与云端相符。`gpt-6.2-sol / max` 最新实际完成的 API 请求记录 212 个输入 token、3,956 个输出 token。

18:04:11 确认同一 008 无 GPU、下载已结束、尚无四段输出目录，随后执行无卡关机并验证 SSH 断开。当前会话的电脑控制已断开，需在控制台对原 008 使用普通开机来恢复 A800。四段独立解码通过的 [先验抽查](multiclip-evidence/four-priors-reviewed.jpg) 已在本机保存。EgoX 成片与 Nano 盲评尚未完成。现场手记的原始依据见 [拍摄信息核验](../../records/course_validation_20261019/capture_metadata.md)。

用户完成普通开机后，18:07:59 在原 008 上启动四段批次。复核 A800 PCIe 81,920 MiB、启动前显存 0 MiB、120 GiB 容器内存限制及 200 GiB 文件存储挂载。脚本于 18:08:02 确认 4 个固定输入与 35 个权重、真实 CUDA 及共享盘读取，并进入同一进程的官方模型加载。远端日志在 `egox-shared-008/run-four-final.log` 和 `egox-shared-008/four-egox-output/inference.log`。已获准在取回并验收成片后关闭同一实例。

恢复 GPU 后两次 Responses 请求使用 `gpt-6.2-sol / max`。第一次达到 5,000 个输出 token 上限。第二次完成，输入 105、输出 1,847，建议允许有明确进展的分片加载继续，三分钟无分片进展时检查 I/O 与内存，收集失败证据后只重试缺片。实时采样与逐段回收由本机执行，见 [API 用量](api_usage.json) 与 [GPU 采样](four-egox-outputs-008/gpu-telemetry.log)。

18:37 首段 Pair04 已真实生成、回传并通过校验。原始封装以毫秒记录总时长，造成约 0.326 ms 的舍入；49 帧的 PTS 与单帧时长精确。经一次完成的 `gpt-6.2-sol / max` 调用，验收仅放宽官方原片总时长元数据到 1 ms，成片逐帧及导出总时长仍保持精确。本机用真实素材与两种不合格模拟输入重测通过。用户已设置 20:00 自动关机；仍将在四段本地验收后立即关闭。

四段批次最终返回 0，官方 rank-256 LoRA 的载入与融合均有实际运行证据。由同一次权重加载处理四段，每段 50 个去噪步。运行审计用时 87 分 29 秒，PyTorch 峰值显存保留 72.17 GiB，进程峰值 RSS 106.74 GiB。原始输出均为 1232×448；本机独立解码和校验后导出的右侧生成画面均为 448×448、49 帧、30 fps。四段的云端原片摘要与本机一致，所有帧的 PTS、时长和导出裁剪一致性通过。核对 [云端结果](four-egox-outputs-008/logs/batch-result.json)、[运行审计](four-egox-outputs-008/logs/runtime_audit.json) 与 [本机验收](four-egox-outputs-008/verified/verification.json)。

19:36:27 本机验收通过后，008 无其他 GPU 计算进程，显存回到 0 MiB，执行原实例关机且命令返回 0。随后 SSH 会话断开并复查 41248 停止接入，早于计划 20:00。见 [验收关机日志](four-egox-outputs-008/logs/acceptance-and-shutdown.log)。用户确认暂缓 Pair03，本批未使用 Nano，保存作独立对照。开卡后这轮共发送 3 次去标识化 Responses API 请求，输入 462、输出 8,678 token，其中两次完成，一次达到输出上限。
