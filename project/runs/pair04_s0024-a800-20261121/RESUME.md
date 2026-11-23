# Pair04 云端续跑

2026-11-21 四段批次已完成，19:36 已取回、验收并关闭 008。以下命令保留作复现记录。现有输出见 [本机验收清单](four-egox-outputs-008/verified/verification.json)。

## 当前 008 的四段批次

四段 EXO、深度和先验已验证，云端批次 `.../pair04-20261121/egox-shared-008/inference-four` 保留原版模型入口。顺序是 Pair04、Pair01、Pair02、Pair05。其余处理如重连、远端状态查看和回收成片都针对已经启动的同一 008 实例。

共享盘挂载确认后，先检查 `download-pipeline-four-008.log`、`download-shared.log` 和原任务进程，避免重复写入。同一模型目录有写锁。准备完毕的批次用下列命令从 `egox-shared-008` 启动。若已经在运行，沿用现有进程。

```sh
findmnt --mountpoint "$(readlink -f /root/autodl-fs)"
cd /root/autodl-tmp/football_egox/pair04-20261121/egox-shared-008
grep -Fx ALL_MODELS_VERIFIED download-shared.log
bash run_egox_four_verified_008.sh inference-four
```

模型未齐时，保留已校验文件及 `weight-staging` 暂存。共享盘单写，SSD 可提前暂存下一文件。恢复用的下载器参数如下，仅在确认旧进程已经退出后运行。

```sh
EGOX_RUN_ROOT="$PWD" EGOX_MODEL_ROOT=/root/autodl-fs/football_egox/models \
EGOX_LOCAL_SCRATCH="$PWD/weight-staging" EGOX_PIPELINE_LOCAL_SCRATCH=1 \
EGOX_USE_MODELSCOPE=1 EGOX_RANGE_WORKERS=4 EGOX_RANGE_MAX_TIME=100 \
EGOX_RANGE_SPEED_TIME=25 EGOX_RANGE_SPEED_LIMIT=131072 \
/root/miniconda3/bin/python -I -B scripts/download_models_local_pipeline.py | tee -a download-shared.log
```

批次输出在 `four-egox-output`。成片独立解码、确认右侧 448×448 生成区域、记录 SHA-256 并拉回本机后，再关闭 008。当前没有用户设定的定时关机。

按用户的节省费用要求，008 于 17:04 从 GPU 模式停机，17:07 保留同一实例与共享盘以无卡模式恢复下载，控制台显示 ¥0.10/小时。18:03 已确认 `CPU_ONLY_WEIGHTS_READY` 并取回 [35 项权重校验清单](multiclip-evidence/model-verification-008.json)。18:04 无卡关机后 SSH 已断开。对原 008 普通开机并确认空闲 A800 80 GB、共享盘挂载及现有任务状态后，执行上方启动命令。

## 50 GB 数据盘与共享权重

028 停机保留。已有 ViPE 缓存、环境和仅含 EXO 的暂存留在原 50 GB 数据盘。选择同地区支持 [AutoDL 文件存储](https://www.autodl.com/docs/fs/) 的实例后，先检查实际挂载和剩余空间。

```sh
findmnt --mountpoint /root/autodl-fs && df -h /root/autodl-fs /root/autodl-tmp
```

共享文件存储用于 Wan 与 LoRA 的 35 个官方文件，下载产物、分段和缓存空间也要计入。按 [固定清单](../../records/model_verification.json) 核验所有文件的大小和 SHA-256，再通过官方 `infer.py` 的 `--model_path` 与 `--lora_path` 直接读取共享盘。工作目录、缓存、输出与 ViPE 保持在本地盘。先实测共享盘读速和原版 safetensors 加载。旧 `scripts/download_models.py` 的空间检查针对旧工作根，使用共享盘前需将权重及分段目录与 `df` 指向同一个实际挂载卷。

若 028 所在地区不能挂载共享存储，先在 028 上完成 ViPE，再将已验证的自采几何或渲染先验交给具备存储条件的 GPU 实例跑 EgoX。新实例须重验真实 CUDA；旧机的队列、CUDA 见证、最晚停止时间不可用作新机的启动依据。Nano 继续留在本机独立评估。

## ViPE 原机预检

保留 A800 数据盘上的项目根 `/root/autodl-tmp/football_egox/pair04-20261121`。重新登录后先看以下正式门槛：

```sh
cd /root/autodl-tmp/football_egox/pair04-20261121
python -B pair04_cache_gate.py
grep -Fx VIPE_CLI_AND_CUDA_OK vipe-finish-extended.log
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
df -h .
```

权重校验须显示 `15 OF 15` 和 `PAIR04_OFFLINE_CACHE_OK`。ViPE 须完成真实 CUDA 内核见证。锁定安装日志为 `requirements-pinned.log`、`requirements-uv-fallback.log` 和 `vipe-finish.log`。公开权重的完整文件，以及仅用于断点续传的分段和 `.part`，分开保存。只有整文件校验通过的文件可用于推理。源码检出为 `Renderer-sparse` 和 `MoGe-sparse`。

收到上传确认后将本机的 `pair04_exo_stage.zip` 放到该目录，先校验其 SHA-256 为 `89b4ff66f68bb63ca8adc4aff136a58dc5916617382d9363b574cd4f0389bbf8`。解压到新 `stage` 目录，激活 `renderer-env` 后从项目根运行：

```sh
python -B stage/scripts/pair04_cpu_dry_run.py --input-root stage/exo_only
python -B stage/scripts/pair04_vipe_preflight.py \
  --input-root stage/exo_only --renderer Renderer-sparse --run-dir first_vipe
```

如果 preflight 仅因缺 PyTorch3D 不通过，应单独核对其余检查和 ViPE 的 CUDA 内核见证。PyTorch3D 属后续先验渲染阶段，完成 ViPE 并不能代表渲染环境已就绪。其余任一实际门槛失败则不运行。ViPE 的输入是原始 25 fps 的 49 帧 EXO。通过后，在有足够剩余运行时间的一次有界任务中执行：

```sh
HF_HOME="$PWD/caches/hf" HF_HUB_CACHE="$PWD/caches/hf/hub" \
TORCH_HOME="$PWD/caches/torch" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=0 vipe infer "$PWD/stage/exo_only/example/self_capture/videos/pair04_s0024/exo.mp4" \
  --output "$PWD/first_vipe/vipe_results" --start_frame 0 --end_frame 48 \
  --assume_fixed_camera_pose --pipeline lyra
```

运行后查验实际 RGB、深度、位姿、内参和帧号；用本地手记的场地测量检查尺度量级。Nano 保持独立盲评。完整流程和坐标约定见 [本地运行手册](../pair04_s0024-prep-20261121/runbook.md)。
