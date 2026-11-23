# Pair01 与 Pair05 下一次验证

目标是生成符合已观测事实的虚拟第一视角。已有 R8 可用于判断球员身份、画面里的球数和部分相对位置。真实头戴轨迹、被遮挡区域和接触瞬间仍需要独立证据。Nano 留到冻结输出后的评估。

| 片段 | 下一次最小检查 | 进入完整推理的条件 |
| --- | --- | --- |
| Pair05 优先 | 冻结上次确实进入推理的跟球先验和官方 VAE，只加载 VAE 对全部 49 帧做一次编解码。用同帧图检查 F12、F24 的白鞋队友、单球与标志盘，F36、F48 复查方向和球的位置。 | 若目标在往返图变得不可辨，先改进有 EXO 依据的几何输入。若仍清楚，固定先验、EXO、相机、权重和初始潜变量，用配对试验只改一条对应或条件路径，记录实际作用。成片须保住前景和构图，全 49 帧复查额外人球与跳变。 |
| Pair01 暂缓生成 | 在本机核对红鞋球员头位和近地场线，在同一坐标系检查 F00 白鞋队友及 F20–F28 红鞋、单球的同帧可见性与相对位置。只用 EXO 和既有拍摄记录，沿用场线与球径敏感性复核。 | 先验中能连续辨认脚球关系才跑下一版，F28 的真实触球仍标不确定。后段 F36–F43 特别检查单球，任一明确复制拒收。现阶段展示使用首版。 |

VAE 往返是几何条件的隔离检查，独立固定潜变量采样种子。其解码图可辨只表明该分支具备可供利用的信息，去噪可能仍被其他条件主导。几何中看不到的部分也可能在生成中被合理补全，补全质量需独立验收。

本机用现有探针通过两段先验的文件哈希、尺寸、49 帧及逐帧时间戳检查。新 008 克隆上已完成 Pair05 隔离往返，F12、F24 的队友、球和标志盘仍可辨，关机已核实。[Pair01 输入检查](pair01-input-check/verification.json) · [Pair05 输入检查](pair05-input-check/verification.json) · [Pair05 实测与关机凭证](pair05-vae-008/README.md)

下一次在已授权、已验证的运行环境复用现有探针及其相对导入。使用运行环境已有的 CUDA BF16 Python 和 ffmpeg/ffprobe。官方共享缓存中的检查目标只有 VAE 子目录的 config 和约 508 MB 权重。命令中的脚本、先验路径用运行环境中的现有对应文件，输出目录必须为新路径。

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -B /path/to/probe_wan_prior.py \
  --prior /path/to/pair05_s0024-final.mp4 \
  --sha256 c6ee04f0be3d48bcfc7564576fedfa137d2ad63f7f46ccdae711980b8713eeaa \
  --fps 30 --seed 846514 \
  --vae-dir /root/autodl-fs/football_egox/models/Wan2.1-I2V-14B-480P-Diffusers/vae \
  --vae-weights-sha256 d6e524b3fffede1787a74e81b30976dce5400c4439ba64222168e607ed19e793 \
  --output /path/to/new-pair05-vae-review
```

[探针](../../rig-refactor-20261122/probe_wan_prior.py) · [历史权重哈希](../../multiclip-evidence/model-verification-008.json) · [历史验收与反证](../README.md)

官方 Responses API 的 `gpt-6.2-astra / max` 完成独立复核。本次上限 24,000，实际输出 16,882 token。Astra 支持先做 Pair05 的隔离检查、暂缓 Pair01 完整生成。它强调 VAE 往返无法单独解释围网成因，并建议逐条隔离 EXO 条件。后续条件对照需明确被改动的支路，例如 EXO 首帧图像编码，单独记录其影响。[API 实际用量](astra-api-usage.json)
