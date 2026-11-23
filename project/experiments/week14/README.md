# 足球第一视角实验

这里保存课程报告中「向队友传球」和「沿标志盘带球」两段 EgoX 首版的实际推理条件与调整入口。两段各 49 帧。[报告与四段对照视频](../../../README.md)位于仓库首页。

## 先核对历史结果

需要 Python 3.10 或更新版及 FFmpeg（含 ffprobe）。下列命令解码历史模型输出、提取右栏第一人称画面并记录逐帧校验。可在 CPU 上运行。

```sh
python3 code/verify_egox_outputs.py reference-output checked \
  --take pair02_s0000 --take pair04_s0024 --contact-sheets
```

`reference-output/` 保存两段当时的 EgoX 双栏输出。`meta.json` 保存运行使用的相机参数与提示词。`inference-inputs.json` 记录场边裁剪片、正式先验和每帧深度的历史校验值。头戴实拍只出现在公开三栏对照的中栏。

## 用同一条件重新生成

使用 Linux CUDA 环境。原实验使用 A800 80 GB、PyTorch 2.5.1+cu124、Diffusers 0.34.0，模型文件共约 87 GiB，另留环境和缓存空间。按官方仓库安装依赖并从官方仓库下载 Wan 底座和 EgoX LoRA。`model-verification.json` 列出本项目核验的权重文件及 SHA-256。权重和云账号配置由各自环境提供。复跑按生成画面和分辨率、帧序、时长核验，CUDA 和算子环境可能带来像素差异。

```sh
git clone https://github.com/DAVIAN-Robotics/EgoX.git /path/to/EgoX
git -C /path/to/EgoX checkout 43c095b552a8f09d3a81c0882491f52c5cf4e953
```

激活准备好的 EgoX Python 环境，确保 LoRA 目录里有 `pytorch_lora_weights.safetensors`。在当前目录运行，模型路径和代码路径换成自己的绝对路径。下列命令沿用项目记录的种子、LoRA 等级和几何引导设置，可以先用 `--idx 0` 或 `--idx 1` 各取一段检查，顺序为沿标志盘带球、向队友传球。

```sh
python /path/to/EgoX/infer.py --meta_data_file meta.json \
  --model_path /path/to/Wan2.1-I2V-14B-480P-Diffusers \
  --lora_path /path/to/EgoX-LoRA \
  --lora_rank 256 --seed 846514 --use_GGA \
  --cos_sim_scaling_factor 3.0 --in_the_wild --out regenerated
```

原版推理输出为 30 fps，报告对照沿同一帧序按源 25 fps 播放。采用新的输出目录，运行后可将上节的 `reference-output` 改为 `regenerated`，按相同步骤验收。此文档的 CUDA 重跑命令本轮只核对过源码入口与输入格式，未再次生成。

## 调整虚拟机位

`geometry/` 保存两段的 ViPE 深度与实例 ZIP、相机内参及位姿，以及按源画面标注的头部轨迹。改动 `geometry/<片段>/head-anchors.json` 的头部像素或取景目标后，先用 `code/` 下对应的虚拟机位构建脚本生成新轨迹，再用固定版几何渲染器重渲染目标机位先验。修改后的相机参数和先验要配套使用。

以传球片段为例。使用安装了固定版 ViPE 的几何环境，并在当前目录运行。为改动后的轨迹和先验使用新的输出目录。

```sh
PYTHONPATH=/path/to/EgoX-EgoPriorRenderer python code/build_multiclip_virtual_view.py \
  geometry/pair02_s0000 example/self_capture/videos/pair02_s0000/exo.mp4 \
  geometry/pair02_s0000/head-anchors.json pair02-view-revised

PYTHONPATH=/path/to/EgoX-EgoPriorRenderer \
  python /path/to/EgoX-EgoPriorRenderer/scripts/render_vipe_pointcloud.py \
  --input_dir geometry/pair02_s0000 --out_dir pair02-prior-revised \
  --meta_json_path pair02-view-revised/meta.json --point_size 5.0 \
  --start_frame 0 --end_frame 48 --fish_eye_rendering --use_mean_bg
```

沿标志盘带球使用 `build_virtual_view_008.py` 和参数 `--compensate-official-roll`。该脚本的队友目标写在源码常量中。用额外拷贝试改，保留作为起点的原 `meta.json` 与正式先验。

## 代码与资源

- [本项目公开报告、四段视频及图源](../../../README.md)
- [固定版 EgoX](https://github.com/DAVIAN-Robotics/EgoX/tree/43c095b552a8f09d3a81c0882491f52c5cf4e953) · [官方 EgoX LoRA](https://huggingface.co/DAVIAN-Robotics/EgoX) · [Wan2.1 底座](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P-Diffusers)
- [固定版 EgoX-EgoPriorRenderer 与 ViPE 集成](https://github.com/kdh8156/EgoX-EgoPriorRenderer/tree/17cdef688885c3ac6a4c371a2d620391cc28dab4)
