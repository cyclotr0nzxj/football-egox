# Pair05 B 单次启动

新 008 已完成一次 B 并关机。F12、F24 仍偏向围网，B 画面拒收。[实测、对比和关机凭证](result-008-66fc463f/README.md)

下一次仅生成 Pair05 B。以取回的 004 前景 GGA 成片作同帧历史参照。B 复用上一轮的 EXO、49 张深度、跟球外参、正式先验、GGA 路由、官方 Wan/EgoX 权重、提示词、种子 846514、50 步和引导强度 5。运行中仍计算首帧 CLIP 图像特征，唯独在交叉注意力合并处去掉该图像分支的加项，文本和原 EXO 视频条件照常输入。

历史参照来自 004，初始化潜变量未保存。B 记录各输入张量的哈希和正、负 CFG 路径的调用证据，以便之后复核。该对比用于判断新片是否改善，单次成功仍需跨片验证机制。

已有 Pair05 的 VAE 隔离检查确认 F12、F24 编码后仍能辨认队友、单球和标志盘。官方 Responses API 的 Sol max 复核也提示，围网消失可能伴随整体画面退化，验收应同时检查这些目标和其它区域。[VAE 实测](../egox-reuse-audit-20261122/next-step/pair05-vae-008/README.md) · [API 用量](sol-api-usage.json)

## 已通过的本机检查

- 核对原 EgoX 提交 43c095b、004 实际执行的启动归档、B 的 50 项包内清单和 Python 3.10 语法。
- 复验真实 EXO、正式先验及 49 张深度哈希。Aria 射线复算和 GGA 计划正常。每个 EXO 源块总预算仍为 16。
- 直接在 CPU 下调用打包的注意力算子，构造正负文本的两个小输入，验证图像项撤掉后结果与纯文本分支逐元素相同，自注意力不触发该开关。BF16 开关小测和拒绝非零图像项的检查通过。
- 用历史原片和明确构造的诊断日志走通取回验收，核对 49 帧、尺寸、PTS 和复跑收据。真实 CUDA 预检和新成片等待下次开机。

## 普通开机后

目标为 VAE 实测的新 008，实例后缀为 `66fc463f`。同名的旧 008 会被启动器拒绝。用已登录的 JupyterLab 在上传 ZIP 的目录执行下列命令。启动器内置共享权重、锁、缓存空间、源文件、BF16 及 GGA 预检，只创建一个推理 attempt。预检失败即结束。B 的推理超时为 3,000 秒。历史 004 同配置单段推理约 25 分钟，B 的实际时间待测。

```bash
bundle="$PWD/pair05-clip-b-008.zip"
run_root=/root/autodl-tmp/football_egox/pair04-20261121/egox-shared-008
printf '%s  %s\n' db98e1b8f62348ebb2d7fe0daa72f73ad418d851a23c82ee0b0c0ac5c30be044 "$bundle" |
  sha256sum -c --strict &&
  stage=$(mktemp -d "$run_root/.pair05-clip-b-launch.XXXXXXXX") &&
  unzip -q "$bundle" -d "$stage" &&
  log=$(mktemp "$run_root/pair05-clip-b-launch.XXXXXXXX") &&
  {
    nohup bash "$stage/pair05-gga-foreground-bundle/run_pair05_clip_b.sh" run > "$log" 2>&1 </dev/null &
    printf 'pid=%s log=%s\n' "$!" "$log"
  }
```

取回 `PAIR05_CLIP_B_COMPLETE` 所指目录中的 `pair05_s0024.mp4`、`verified-result.json`、`runtime_audit.json`、`inference.log`、`preflight.log`、`launch.log`、`inference-exit-code.txt`、`executed-code-and-meta.sha256` 及 `fingerprint.txt`。用包内 `verify_clip_b.py` 在本机复验后取原片右侧 `448:448:784:0`。按 0 起始索引对照原先验、历史 004 及 B。同帧核对 F12、F24 是否同时保有正前方白鞋队友、单球和对应标志盘，核对 F08、F16、F20、F28 的连续性和 F36、F48 的方向。围网仍居中、队友或球缺失、目标重复或动作严重变形均拒收。全片保留作证据，Nano 留到冻结结果的独立评估。取回并验收后关闭同一实例。
