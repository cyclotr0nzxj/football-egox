# Pair05 单次前景注意力验证

这个包保留为旧架构的局部注意力诊断。当前两段都从 EXO 重新估计几何，沿用[重建结果和画面验收](../rig-refactor-20261122/README.md)作为后续实验依据。

候选复用 004 上一次完成的 Pair05 启动包。固定已估计机位、EXO、先验、49 帧深度、295 条边、Wan 2.1 与官方 EgoX LoRA，以及种子 846514。用原注意力输出与匹配 EXO 的加权输出做局部混合，使 F12 和 F24 已指定的 51 个查询块中每条目标分支的匹配 EXO 总权重至少达到 25%。原本达到下限的分支维持原样。25% 是单次验证的预设值，画面改善需另行验收。

本机用真实冻结输入和官方代码运行了封装、投影与路由检查。CPU BF16 小测在固定的 40 头布局得到 0.25 匹配 EXO、0.75 EGO 和 0 不允许 EXO，未选行逐元素不变。随机非等分数小测与独立 FP64 参照最大差约 5.42e-8。[封装检查](package-check.json) · [CPU 小测](route-tests/cpu-receipt.json)

通过官方 Responses API 让 `gpt-6.2-sol / max` 复核单次运行和验收方案。仅提交脱敏技术文本。API 支持先做实际 CUDA 预检，再检查两张关键帧和逐分支实测权重。[API 实际用量](sol-api-usage.json)

## 在 004 启动

在 Jupyter 终端进入 ZIP 上传目录。以下命令核对 ZIP 后创建独立解包目录，再异步启动。启动器先检查实例、显存、共享权重、冻结输入、BF16 局部权重与保留 EGO，推理限时 3,600 秒，最多创建一个 attempt。

```bash
bundle="$PWD/pair05-gga-attention-floor-004.zip"
run_root=/root/autodl-tmp/football_egox/pair04-20261121/egox-shared-008
printf '%s  %s\n' 4bd5ec58910dc5674e73ed805bbdb0b080db54c12a9316248159b7f2a87cbaa7 "$bundle" |
    sha256sum -c --strict &&
    stage=$(mktemp -d "$run_root/.pair05-floor25-launch-XXXXXXXX") &&
    unzip -q "$bundle" -d "$stage" &&
    log=$(mktemp "$run_root/pair05-floor25-launch-XXXXXXXX.log") &&
    {
        nohup bash "$stage/pair05-gga-foreground-bundle/run_pair05_gga_foreground_008.sh" run > "$log" 2>&1 </dev/null &
        printf 'pid=%s log=%s\n' "$!" "$log"
    }
```

记录 `PAIR05_GGA_FLOOR_CUDA_OK`、`PAIR05_FOREGROUND_STARTED` 和最终的 `PAIR05_FOREGROUND_COMPLETE`。完成收据校验 50 步共 4,000 次受限注意力调用，要求第 1、2、2000、2001、3999、4000 次的实测前后权重都能取回。根据实际调度顺序，这六次是前段、中段和末段的调用抽样。

取回完成标记指向的 attempt 下列文件及单独的启动日志。仅在已有完成标记时执行。

```bash
attempt=$(awk '/^PAIR05_FOREGROUND_COMPLETE / {print $3}' "$log" | tail -n 1)
case "$attempt" in
    "$run_root"/pair05-gga-foreground-results-004-floor25/attempt-*)
        evidence=$(mktemp "$run_root/pair05-floor25-evidence-XXXXXXXX.tar.gz") &&
            tar -czf "$evidence" -C "$attempt" -- \
                pair05_s0024.mp4 verified-result.json runtime_audit.json inference.log preflight.log \
                launch.log executed-code-and-meta.sha256 fingerprint.txt inference-exit-code.txt &&
            sha256sum "$evidence" &&
            printf '%s\n' "$evidence"
        ;;
    *) printf 'No completed attempt in launcher log.\n' >&2 ;;
esac
```

画面验收先看 F12、F24 是否同时出现白鞋队友、脚边一颗球和相应标志盘，再检查 49 帧的身份、重复物与构图连续性。实测权重达到下限仍无对应画面时，记录这次局部干预未通过。

## 008 备用包

008 包限定使用此前运行 Pair05 的实例 `cb894da57f-45e726a1`。与 004 包相比，仅更改该启动器的实例校验和独立输出路径，并相应更新清单。通过同一套本机检查。[008 封装检查](package-check-008.json)

启动流程相同，使用 `pair05-gga-attention-floor-008.zip` 和 SHA-256 `55dcaf13849cb7a7cca8f29decda87a0b14f7da7927a0ac8ad0983181d6ad2c6`。收据 attempt 位于 `pair05-gga-foreground-results-008-floor25/`。
