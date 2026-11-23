# Pair05 历史架构回放与重建输入验证

历史架构在本机可重投影的前景并没有用完。两帧旧路由只覆盖 51/10,192 个目标块。保持旧机位、旧深度与旧先验后，13 个 4 帧采样槽可编译 317/10,192 个目标块，覆盖由 0.50% 增至 3.11%。实际筛出 1,844 条同帧边和 11,767 个样本，每个同帧 EXO 源块仍至多 16 个样本。这里的「覆盖」指各槽候选目标块，无法据此推断生成质量。

- **复现**：F12 和 F24 重算所得像素映射、边、余弦和取样逐元素等于已部署的原方案。混合源块的前景身份保持相斥，明确舍去与其他物体混合的样本。毗邻帧测试中有 1,392 个像素候选可在取消时间约束时误入，同帧入口全部拒绝。
- **弱证据**：F20 的球只有 8 个样本、2 个目标块，F24 有 13 个样本、2 个目标块。F40 尽管有 61 个球样本，投影各点到其中位中心的距离第 90 百分位约 63 像素，不能视作形状可靠的球。标志盘仅在有逐帧核验的 F12 和 F24 作为路由条件。
- **历史机制限制**：GGA 的局部约束作用于指定目标查询。完整 EXO 仍然进入 `prepare_latents` 的 VAE 和首帧 `encode_image`。扩大 GGA 路由覆盖只能区分候选是否存在，不能识别模型最终依赖了哪条输入路径。[原模型条件入口](../../../../vendor/EgoX/core/finetune/models/wan_i2v/sft_trainer.py#L111)

这次回放用于解释旧系统，产出的 `timeline-support-diagnostic.npz` 不能给历史 GPU 包使用。新几何应以新的同帧像素来源验证入场，直接区分实测颜色和不可见的缺失区域，再决定是否值得运行一个可归因的生成试验。[旧架构图](old-architecture-replay/timeline-support.jpg) · [逐槽数据](old-architecture-replay/timeline-audit.json)

```bash
run=runs/pair04_s0024-a800-20261121/rig-refactor-20261122/pair05
work=$(mktemp -d /tmp/pair05-historical-timeline.XXXXXXXX)
/usr/local/Caskroom/miniforge/base/bin/python3.12 -B "$run/audit_timeline.py" "$work/replay"
```

## 新机位的来源核验

独立脚本为两套鞋底筛选后的新机位核对 49 个 EXO 和预览索引、各自的 25 和 30 fps 时间戳、冻结 EXO 和两个输出的摘要。每个投影点再回查同帧影片、冻结物体轮廓和球观测。蓝盘另用已审定的四个固定 EXO 中心附近相连的青色原像素核对。空白区域保持未观测，`verified-observed-rgb.npz` 仅包含通过检验的原始实际 RGB 与来源索引。故意在本机把一个真实的 F12 队友像素挂到球标签，核查器拒绝该像素，实际球像素不受影响。

| 核对项目 | 名义标定 | 备用标定 |
|---|---:|---:|
| 同时有队友与球源像素的帧数 | 49 | 49 |
| 物体身份与同帧 EXO 冲突的帧数 | 0 | 0 |
| F12 确认的球像素 | 36 | 30 |
| F24 确认的球像素 | 3 | 12 |

F24 两种标定下均能找到队友和两块蓝盘，球的视觉证据仍薄弱。目视名义机位的关键帧，队友结构倾斜、压扁，草地有条带和空洞。来源一致说明这些零散颜色确实来自该帧 EXO，无法保证三维结构和生成画面正确。两套输出均仅用于本机诊断。[名义标定图](rebuilt-nominal-support/rebuilt-support-review.jpg) · [备用标定图](rebuilt-alternate-support/rebuilt-support-review.jpg) · [名义数据](rebuilt-nominal-support/rebuilt-support-audit.json) · [备用数据](rebuilt-alternate-support/rebuilt-support-audit.json)

```bash
root=runs/pair04_s0024-a800-20261121/rig-refactor-20261122
work=$(mktemp -d /tmp/pair05-rebuilt-support.XXXXXXXX)
for variant in nominal alternate; do
  /usr/local/Caskroom/miniforge/base/bin/python3.12 -B "$root/pair05/verify_rebuilt_support.py" \
    "$root/pair05-reset/results-sole-filtered" "$variant" "$work/$variant"
done
```
