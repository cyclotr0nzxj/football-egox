# Pair04：从第三人称到虚拟第一视角

本轮准备一个 49 帧自采试片，用官方 ITW 路线验证能否从 EXO 得到虚拟第一视角。最终相机轨迹需要人工依据 EXO 几何标注，测试不要求自动头部跟踪。只有深度、虚拟轨迹、先验和模型结果都真实通过时，才能报告这段视频的端到端结果。

## 输入核验

- 已验证 EXO H.264、784×448、25 fps、无音轨，49 帧完整软件解码。时间基 `1/12800`，逐帧 PTS 为 `512i`，最后一帧为 1.92 秒，容器时长 1.96 秒。
- 从 3840×2160 的现存 4K EXO 在左边去掉 30 像素，以 `[30, 0, 3780, 2160]` 的 `xywh` 裁剪后等比缩放 `28/135`。独立从 4K 源重算，49 对同索引灰度帧 PSNR 35.77–41.65 dB，符合有损转码。原始帧坐标用像素中心变换 `u' = s(u + 0.5 − 30) − 0.5`、`v' = s(v + 0.5) − 0.5`。若未来有经标定的原始 K，才用 `K' = A K` 转换。这一仿射本身不提供 K。
- EXO F0–48 对应 4K 裁剪源可见 F24–72、同步基准 F123–171、历史 EXO 原片 F154–202。历史原片备份当前缺失。两路 4K 裁剪源去掉预滚后均为 311 个连续的 `i/25` PTS。历史双端拍掌证据支持 EXO 比 Nano 晚 31 帧（1.24 秒），人工对齐约 ±1 帧。它只能支持盲评按帧看同一时刻，不提供相机标定。
- 背景跟踪选建筑、远背景、右侧边线和下方静态草地角点，PyrLK 往返误差 ≤0.25 px，RANSAC 相似变换阈值 0.65 px。逐帧相对首帧有 882–1014 个内点，拟合画心最大位移 2.82 px、旋转 0.231°、缩放 0.999412–1.000312。按近似固定机位试跑；实际三维外参与尺度仍未知。若拟合与几何质量门槛冲突，停在 ViPE。

已有 [试片溯源](../../data/self_capture/pair04_s0024/staging_manifest.json) 和 [同步证据](../../records/capture_sync_20261019/visual_audit/final_visual_audit.json)。Nano 的参考视频留在原目录 `reference/`。准备包只带 EXO 与预检脚本。锁定轨迹、提示词、相机、种子和比较口径以后才开放 Nano 盲评。Nano 不参与输入、轨迹标注、候选选择或调参。

**现场记录。** Pair04 根据画面和现场照片对应到场景 2。手记中相机高度 **1.23 m**、到最近的右下标识盘 **6.3 m**、盘间距 **0.38 m**。R8 设置照片显示 24 mm、1/60 s、4K 25 fps，24 mm 和 1/60 s 仅代表拍下设置页时的值。估计几何后可用场记检查尺度是否合理；做定量比对前须明确测距方式、光心高度及盘的对应关系。场景 2 的盘近似共线，这些记录不能单独给出相机内参或外参，也不填入 ViPE 的标定参数。[原始手记](../../验证视频/拍摄信息/场地信息.md) · [拍摄资料核验](../../records/course_validation_20261019/capture_metadata.md)

本机重新验证（在项目根目录执行，使用已有的 `ffmpeg` 与 `ffprobe`）：

```sh
python3 -B scripts/pair04_cpu_dry_run.py
```

输出包含源 SHA、裁剪仿射、逐帧 PTS 对照及暂未运行的阶段。外部暂存树可用 `--input-root exo_only`。验证只接受固定路径和哈希的 EXO，拒绝额外文件与符号链接。

## 版本与缓存

| 组件 | 本次固定值或状态 |
| --- | --- |
| EgoX | `43c095b552a8f09d3a81c0882491f52c5cf4e953`，本机 34 份已追踪源码文件与历史哈希一致。 |
| EgoPriorRenderer / ViPE fork | EgoX 的 gitlink `17cdef688885c3ac6a4c371a2d620391cc28dab4`。ViPE 来自同一个 checkout（包版本 0.1.1），不用另一个 NVIDIA checkout 替换。 |
| Renderer 环境 | 采用该 commit 的 `envs/base.yml` 与 `envs/requirements.txt`。Python 3.10、CUDA 12.8、torch 2.7.0+cu128、torchvision 0.22.0+cu128、NumPy 2.1.2、OpenEXR 3.2.4、PyTorch3D v0.7.9（`33824be3cbc87a7dd1db0f6a9a9de9ac81b2d0ba`）。 |
| MoGe | 上游安装说明未锁版本。本项目选择 Renderer 提交日的公开源码 `07444410f1e33f402353b99d6ccd26bd31e469e8`（2.0.0），v2 接口已对照，组合尚未 GPU 实测。其依赖 utils3d 与 pipeline 各自在 MoGe 的 pyproject 中锁到 Git commit。MoGe2 模型 `Ruicheng/moge-2-vitl-normal` 的 `model.pt` 固定参考 revision `b135031bae30b5ac2ae141a0e68717795ce38340`。上游调用默认取 main，须核实缓存指向同一 blob。 |
| EgoX 生成环境 | 分开使用已验收的历史环境。Python 3.10.21、torch 2.5.1、torchvision 0.20.1、CUDA 12.4、diffusers 0.34.0、transformers 4.49.0、accelerate 1.5.2、PEFT 0.17.1、NumPy 2.2.6、decord 0.6.0、scikit-image 0.25.2。 |
| Wan 14B / EgoX LoRA | Wan `b184e23a8a16b20f108f727c902e769e873ffc73`；LoRA `28b19ef9f079bf73c84b2347fe6701e8ab7f6634`。`records/model_verification.json` 保存 35 核验项（总计约 87.035 GiB）。开卡后要在实际盘上复验尺寸与 SHA-256，再复用。 |

源码依据：[EgoX](https://github.com/DAVIAN-Robotics/EgoX/tree/43c095b552a8f09d3a81c0882491f52c5cf4e953) · [Renderer 与 ViPE](https://github.com/kdh8156/EgoX-EgoPriorRenderer/tree/17cdef688885c3ac6a4c371a2d620391cc28dab4) · [MoGe 源码](https://github.com/microsoft/MoGe/tree/07444410f1e33f402353b99d6ccd26bd31e469e8)。

历史阶段 A 的模型、依赖记录和校验单可以复用，官方样例的 K、深度、轨迹、先验与描述不能混用。ViPE 所需的 GeoCalib、DROID、GroundingDINO、SAM、DeAOT、VDA、MoGe2、PriorDA 缓存不包含在 Wan/LoRA 历史核验单里。当前 Intel Mac 无 NVIDIA CUDA，也未找到可用的 ViPE、Renderer 或权重。云盘是否保有历史权重没有当前证据。

开卡后核对以下 Lyra 缓存路径和来源，避免临时自动下载。Torch Hub 根通常为 `$TORCH_HOME/hub`，实际以 `torch.hub.get_dir()` 为准。HF 缓存以当前环境的 `HF_HUB_CACHE` 为准，需核对代码字面 repo ID 可解析的 `refs/main` 与 snapshot。

| 缓存 | 所需资产 |
| --- | --- |
| Torch Hub | `geocalib/pinhole.tar`（GeoCalib v1.0）、`droid_slam/droid.pth`、`sam/sam_vit_b_01ec64.pth`、`aot/R50_DeAOTL_PRE_YTB_DAV.pth`、`checkpoints/groundingdino_swint_ogc.pth`、`checkpoints/video_depth_anything_vitl.pth`。 |
| MoGe2 | `Ruicheng/moge-2-vitl-normal/model.pt` 为 1,323,815,904 B，固定版及查询时 main 的权重 SHA-256 同为 `280741fd09bc3f403ccff9967784c2a391b52d2c0742ae3efdb21d9f90cc1a01`。 |
| PriorDA | `Rain729/Prior-Depth-Anything` revision `25b3bfcf85209fb015e3b15259f4df8b6f13bb16` 的 `depth_anything_v2_vitl.pth`、`depth_anything_v2_vitb.pth`、`prior_depth_anything_vitb_1_1.pth`。该 fork 的 Lyra 初始化也会装载 PriorDA。 |
| GroundingDINO 文本编码器 | loader 直接使用 `bert-base-uncased`，对应 `google-bert/bert-base-uncased` revision `86b5e0934494bd15c9632b12f734a8a67f723594`。核验 `config.json`、`tokenizer.json`、`tokenizer_config.json`、`vocab.txt`、`model.safetensors` 在代码实际查找的缓存中可读。 |

仅公开元数据可确认的 7 份大型文件合计约 6.1 GB，另有 GeoCalib、SAM、DROID、DeAOT。这是待核验清单，未下载、未在云盘确认。缓存里的 GroundingDINO 权重预期 SHA-256 为 `3b3ca2563c77c69f651d7bd133e97139c186df06231157a64c507099c52bc799`，VDA-L 为 `43df27c6b396042ba34ff7b798ab279f64d204d2e86d7a373968f8fa36d0e6fa`。其余权重以后续实际来源与哈希记录为准。

## 开卡后的第一步

开卡、传入包和准备 GPU 环境需要另行授权。本次没有启动云卡、连接云盘、安装依赖或上传文件。后续将 `pair04_exo_stage.zip` 解压到新目录；以下命令从该解压目录执行。Renderer 固定 checkout 假设位于同级 `../EgoX-EgoPriorRenderer`，终端已激活独立的 Python 3.10 Renderer 环境。

```sh
python3 -B scripts/pair04_vipe_preflight.py --input-root exo_only --renderer ../EgoX-EgoPriorRenderer --run-dir first_vipe
```

此命令只读检查视频、源码、核心版本、环境里的 ViPE 入口和 NVIDIA 可见性。之后检查 CUDA 实际可用、MoGe 固定版本以及全部 Lyra 权重已命中对应缓存，确认盘空间和运行授权。ViPE 首次构建某些模型时会自动下载，`HF_HUB_OFFLINE=1` 单独不能拦住 torch.hub 或 gdown。若缓存未齐，停止并补齐单独授权，不把隐式下载算成缓存复用。

以上通过后，首条真实几何命令如下。用已选的可见 GPU，保留原始 25 fps，不传外部 K。

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
  vipe infer "$PWD/exo_only/example/self_capture/videos/pair04_s0024/exo.mp4" \
  --output "$PWD/first_vipe/vipe_results" --start_frame 0 --end_frame 48 \
  --assume_fixed_camera_pose --pipeline lyra
```

预期输出根为 `first_vipe/vipe_results/pair04_s0024`，ViPE 按输入视频父目录名组织产物。禁止把仓库里硬编码 Joker 或已有官方足球样例的 launcher 原样运行。

## 几何到生成

1. **核对 ViPE 导出。** 检查 `rgb/exo.mp4`、`pose/exo.npz`、`intrinsics/exo.npz`、`depth/exo.zip` 和实例掩码。核对 RGB、深度、K 和 pose 的 `inds` 全为 0–48，轴向 Z 和 RGB 在同一网格，有效 Z 有限且大于零，统计无效点和缺失率。记录 ViPE 估计 K 对应哪一尺寸。如果输入或输出网格不同，需要证明裁剪缩放仿射再变换 K 和重采样 Z。单目得到估计尺度，没标尺或独立标定不能称已得到精确米制几何。发现大面积无效深度或关键头部无法定位便停止。
2. **统一坐标再标头部。** ViPE pose 是 OpenCV C2W 4×4；EgoX/Renderer 的 JSON 放 W2C 3×4。齐次反投影用 `X_c = Z K⁻¹[u,v,1]ᵀ`，再用 `X_w = T_c2w X_c`。检查固定机位模式实际导出的上游姿态，并确认 Renderer 没把已变到世界的点再次当作 EXO 相机坐标乘一次 C2W。若 ViPE 的固定参考姿态为单位矩阵，可把 EXO 相机定义为世界坐标，用单位矩阵表示这个坐标定义，不能称这是实测位姿。其他情况先验证坐标归一化及两个消费者的重投影，禁止用单位阵修补。用 EXO/点云对同一名球员做 49 帧虚拟头部标注，使用官方 `vipe visualize … --ego_manual` 的交互方式且限制在本地访问（当前固定版默认监听 0.0.0.0，服务未启动）。用 `T_w2c = (T_c2w)⁻¹`；检查 R 的正交性与 det=+1、正 Z、视轴和轨迹连续性。若头部遮挡导致 49 帧不能形成可信轨迹，记录缺帧并停在该关。
3. **生成 point cloud 和先验。** 仅用 EXO RGB、Z、K 和一致的坐标生成空间点云。目标虚拟相机预先固定 448×448 的视场、主点和投影约定，保留选值依据；不复刻 Nano 或官方样例 K。Renderer 的 `FishEyeCameras` 与 GGA 的 OpenCV 去畸变/90° 光线旋转不同，至少用中心、边缘与 F0/F24/F48 的重投影核验方向和一致性。按固定 Renderer 的 `scripts/render_vipe_pointcloud.py` 使用 `--start_frame 0 --end_frame 48 --point_size 5.0 --fish_eye_rendering --use_mean_bg`，保持动态点。此 pin 的非 mean 背景路径返回值数量与调用方不一致；官方脚本用 mean 路径。渲染后验收同名目录的 `ego_Prior.mp4` 为 49 帧、448×448、30 fps，核对参考空间方向、遮挡和有无空帧。需要独立点云文件时从同一 RGB-D 和位姿导出并注明取样和遮挡范围。
4. **将深度与 meta 交给 EgoX。** 在新的生成根保留 `./example/self_capture/videos/pair04_s0024/exo.mp4` 这样的相对路径，`infer.py` 借它推导 `./example/self_capture/depth_maps/pair04_s0024/`。使用固定 Renderer 的 `scripts/convert_depth_zip_to_npy.py` 转换 Z，然后将且仅将对应的 49 份 `depth_NNNNNN.npy` 从 ZIP 名导出的子目录映射到 take 目录。EgoX 按名字排序取所有 NPY，要检查帧号、维度、有限性和 RGB 共网格。EgoX 会按深度网格强制居中 EXO 主点、相应缩放 fx/fy，再把三维网格中心裁到 448×784；必须比较这和 ViPE 的估计射线，偏离明显则先修正或停跑。meta 包含 `exo_path`、`ego_prior_path`、EXO-only 的 `prompt`、`camera_intrinsics`、`camera_extrinsics`、`ego_intrinsics`、49 个 `ego_extrinsics`，不含 GT 路径。EgoX GGA 只取 `0,4,…,48` 共 13 个时点。
5. **一次实推及锁定后盲评。** 通过前述检查再用官方 ITW 参数（seed 846514、LoRA rank 256、GGA、cos 缩放 3.0、49 帧、50 步、guidance 5.0、30 fps，且显存满足官方单卡至少 80 GB）运行固定 EgoX `infer.py`。选择单一样例 `--idx 0 --in_the_wild`；保证相对路径根正确，可从新生成根通过绝对路径调用固定仓库的 `infer.py`。验收 1232×448 的原生拼接输出，左侧 EXO 784×448，右侧生成 EGO 448×448。记录权重和源码标识、配置、日志、耗时、资源占用、视频哈希。最终依据同步证据人工复核球、人、遮挡和闪烁。虚拟相机与 Nano 平面未校准，不对错位图像报 PSNR/SSIM。

## 两种时基

几何和扩散条件按帧号对齐：EXO、Z、虚拟位姿、先验和生成的第 `i` 帧都代表同一个原始采集帧。EXO 时间 $i/25$；官方先验及输出播放时间 $i/30$；共同单位 `1/150 s` 时分别是 `6i` 和 `5i`。F24 是 0.96→0.80 秒，F48 是 1.92→1.60 秒。49 帧的 30 fps 成品时长约 1.633 秒，与 1.96 秒原片等帧数但不同播放速度。盲评用同索引，或给同 49 帧另做一个 25 fps 评估封装并复核 PTS。不要在进入 ViPE 前用 fps 滤镜做插帧、抽帧或重复帧。
