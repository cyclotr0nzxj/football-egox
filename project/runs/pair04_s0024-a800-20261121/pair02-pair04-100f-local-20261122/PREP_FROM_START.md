# Pair02 与 Pair04 全段重生成

| 视频 | 源 EXO 帧 | 三个全新推理窗口 | 成片 |
| --- | --- | --- | --- |
| Pair02 | F0–F99 | F0–F48、F28–F76、F52–F100 | 100 帧，25 fps，4.00 秒 |
| Pair04 | F24–F123 | F24–F72、F52–F100、F76–F124 | 100 帧，25 fps，4.00 秒 |

六窗均走固定版本的官方 EgoX、同一组权重和 seed 915101。复用已经核查的 EXO、深度、头轨和鱼眼几何先验。启动程序在新目录中建立 EXO-only 暂存区，固定每个窗口的顺序和输入哈希。历史成片、报告和 Nano 素材均不进入生成输入。正式出片前核对两处同帧重叠及转场。

实测每窗 50 步、约 22.9 秒每步，六窗纯去噪约 1 小时 55 分钟。旧四窗批次耗时约 87 分钟，估计六窗一次加载需约 2 小时 10 分钟。按 2–2.5 小时占卡规划，另预留 10–20 分钟用于取回和验收。单张 A800 使用一个推理进程，两个配对排在同一批中。当前官方入口固定 49 帧，单次 101 帧的显存和画面还未验证。

## 开机后的入口

用户明确开机并确认运行后，从已登录的 JupyterLab 终端启动。云端程序核对实例后缀、模型、输入、空间及独占锁。下列命令使用新批名和独立日志。

```bash
base=/root/autodl-tmp/football_egox/pair04-20261121
shared="$base/egox-shared-008"
runner="$base/pair02-pair04-100f-geometry-prep/regenerate_from_start_008.py"
run_name="pair02-pair04-from-start-$(date +%Y%m%d-%H%M%S)"
nohup "$shared/egox-env/bin/python" -B "$runner" run --run-name "$run_name" \
  > "$base/$run_name.launch.log" 2>&1 < /dev/null &
printf 'run=%s pid=%s\n' "$run_name" "$!"
```

当前版本 `regenerate_from_start_008.py` SHA-256 为 `909d706389407ec9fe5770dae29e0be5d068dbe0592a31e42f45885011a91788`。可用同一 Python 执行 `check`，只校验 EXO、几何和收据。若 Pair04 的两窗渲染收据缺失，`run` 会先调用同一验证过的校准分支渲染。流水线正常时每个窗口仍可在 `raw/` 中提前归档。发生错误时保留已有证据和日志，下一次使用独立批名。只有持有经同一版本、种子、EXO、先验与 49 帧解码校验的本轮新窗时，才可另行组合故障前后生成的新窗。

## 取回与验收

按日志中的归档 SHA-256 和 Jupyter 文件地址取回，用 `receive_008.py` 校验归档和其中逐个成员。两段的 `drafts/<Pair>/` 含 25 fps 视频、两段同帧对照和来源收据。检查首窗连续性、红鞋视角主体、队友和单球，并循环观看 Pair02 源 F49 / F76 与 Pair04 源 F73 / F100 两处切点。鱼眼边界、队友位置、球与光线的突然变化均纳入人工验收。冻结结果后才做 Nano 独立评估。报告沿用已保留的两段首版，只有新片验收通过后才决定替换。

[官方 Sol max 复核](sol-from-start-readiness-review.txt) · [实际 API 用量](api-usage-from-start.json)
