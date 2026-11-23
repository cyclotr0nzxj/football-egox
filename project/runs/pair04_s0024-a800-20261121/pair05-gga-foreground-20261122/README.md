# Pair05 前景匹配接入 EgoX GGA

F12 和 F24 的队友、球、蓝色标志盘都进入了本机验证的 GGA 直接匹配。使用已有的正前方机位、Aria 鱼眼修正、EXO、正式几何先验和 ViPE 深度。Nano 未参与。本轮不开卡，Pair01 [继续用首版成片](../four-egox-outputs-008/verified/pair01_s0288_ego.mp4)。

| 帧 | 队友采样点 | 球采样点 | 标志盘采样点 | 直接匹配 | 单个源块最大点数 |
| --- | ---: | ---: | --- | ---: | ---: |
| F12 | 993 | 29 | 30，1 个 | 164 | 16 |
| F24 | 790 | 13 | 48，2 个，其中 1 个只露出一部分 | 131 | 16 |

[采样点与几何先验同帧图](plan/routed-points.png) · [预算及按物体计数](plan/plan-audit.json) · [CPU 实测](cpu-verification.json)

EgoX 的 49 帧按原有几何索引映射到 13 个槽位，F12 和 F24 分别使用槽位 3 和 6，从 0 起算。每个源块跨所有可见物体及目标块共用 16 个几何采样点。171 个源块以已验证样本替换固定网格样本，同一组点用于普通 GGA 的这些源块与前景匹配。其余源块沿用原有固定 16 点，CPU 对其中的向量做了逐元素复算。带前景限制的目标只有 51 个块，网格上其他目标继续使用普通注意力。替换过的源块仍可能被这些未受限目标间接读取。

F24 剔除了同时含队友和球的一个 EXO 源块，少了 1 个球采样点和 1 条球匹配。两个目标混合块分别计算队友和球的注意力，再等权平均。已用本机 PyTorch 在真实 10,192 个 EGO 查询和 28,028 个注意力键的布局上，验证 295 条可用 EXO 匹配，且 53 个身份分支只读到各自允许的直接 EXO 键。沿用更近表面遮挡检查，F12、F24 分别有 13,474、12,003 个远处前景候选被挡住。异帧与错物匹配未进入计划。另用小张量执行暂存版 `WanAttnProcessor2_0`，确认附加路由从 `attn1` 的 GGA 分支调用，缺 GGA 或启用 KV cache 时会拒绝执行。

## API 复核与验收边界

通过官方 Responses API 调用 `gpt-6.2-astra / max`，只提交脱敏技术描述。首轮耗尽 10,000 输出 token，第二轮完成，实用 14,348 输出 token。Astra 指出两条仍可能带入背景信息的路径。允许读取的 EXO 特征块可能已编码周围像素，原有 EGO 注意力也可能传递前层的跨帧或背景信息。CPU 用人为注入的特征确认两条路径都存在。实测覆盖几何和直接注意力，完整模型的内容隔离、三类物体能否同时留在成片以及球门是否仍主导构图，需要用真实权重和新成片验收。[脱敏 API 用量](../api_usage.json)

## 本机复跑

从本目录运行，临时输出与已有证据分开。沿用本机已有的两套 Python 环境，解码深度后用 CPU PyTorch 验证。两次独立构建的匹配表逐字节相等。

```bash
work=$(mktemp -d /tmp/pair05-gga.XXXXXXXX)
/usr/local/Caskroom/miniforge/base/bin/python3.12 -B build_routes.py "$work/plan"
/usr/local/Caskroom/miniforge/base/bin/python3.12 -B stage_egox.py "$work/egox"
/usr/local/Caskroom/miniforge/base/envs/dino_wm_local/bin/python -B verify_cpu.py \
  "$work/egox" "$work/plan/pair05-gga-routes.npz"
```

接入入口为暂存版 `infer.py` 的 `--foreground_gga_plan`，仅用于冻结的 Pair05 单段并要求 `--use_GGA`、关闭 KV cache。正式推理前仍需按原 Pair05 预检核验 49 个 NPY 深度、运行时和已缓存权重。官版代码放在原位置，可继续复跑既有基线。[接入源码](pair05_gga_routes.py) · [暂存入口](staged/infer.py) · [暂存文件哈希](staged-sha256.json)

## 008 单段启动与验收

使用已准备的 [Pair05 启动包](pair05-gga-foreground-bundle.zip)。仅为 Pair05 单段启用前景 GGA，沿用已查验的原跟球机位与在 F12、F24 同时看见盘子、队友和球的既有正式先验。先验、EXO、49 个深度、路由表、源码及隔离暂存均通过本机检验。原 008 开机后，在上传目录校对 ZIP 的 SHA-256 再启动。[本机启动包收据](foreground-bundle-local-readiness.json)

```bash
cd /root/autodl-tmp/football_egox/pair04-20261121/egox-shared-008 &&
  unzip -n pair05-gga-foreground-bundle.zip &&
  bash pair05-gga-foreground-bundle/run_pair05_gga_foreground_008.sh run
```

启动器先核对空闲 A800、原运行环境、缓存与冻结输入，再在全尺寸布局上跑一次 BF16 路由小测。卡上小测通过才加载 14B 权重。实际成片收据要求记录前景注意力调用次数和推理日志的 SHA-256。预检失败即停止该段。Pair01 若同一开机窗口运行，两段各自独立，且在一张卡上串行。[本机启动模拟](foreground-launch-local-checks.json)

取回 `PAIR05_FOREGROUND_COMPLETE` 对应目录的原片、`verified-result.json`、`runtime_audit.json` 和 `inference.log`。`run_sha` 取完成标记中的指纹。在本目录执行本机验收。

```bash
/usr/local/Caskroom/miniforge/base/bin/python3.12 -B review_foreground_video.py \
  --raw "$recovered/pair05_s0024.mp4" --receipt "$recovered/verified-result.json" \
  --runtime-audit "$recovered/runtime_audit.json" --inference-log "$recovered/inference.log" \
  --expected-fingerprint "$run_sha" --output-dir "$review_out"
```

先看 F12 和 F24 的同帧对照，确认正前方同时保住蓝色标志盘、白鞋队友和一颗球。再扫全片，检查是否出现重复人物或重复球、构图是否仍偏向球门、以及重点帧以外是否有跳动。GPU 产物未完成前继续保留既有版本。CPU 检查通过只说明指定匹配进入注意力，真实画面仍需验收。
