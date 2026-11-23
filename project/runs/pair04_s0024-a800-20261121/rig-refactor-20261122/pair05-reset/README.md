# Pair05 从 R8 重建的本机复核

重建后的几何仍未通过画面验收。新机位让人、球和至少一个标志盘在 49 帧均有直接投影像素，但 F24 的球在两套标定假设下只剩 3 和 12 个像素。在原尺寸对照中球轮廓仍不清楚。当前方案只保留作为本机诊断。[五帧对照](results-sole-filtered/rebuild-comparison.jpg) · [逐帧证据](results-sole-filtered/rebuild-receipt.json)

## 重建与反证

用 49 帧本机 R8 的帽部锚点和红鞋求头位。红鞋的图像位置与地面求交，头位在帽部射线上，地面投影靠近鞋底支撑位置。每帧由同帧白鞋队友、球和最近可见标志盘定中心方向。渲染保留实际 EXO 颜色，鱼眼采样使用帧内深度最近的源像素。先前拟合的虚拟机位只出现在比较列。Nano 未进入流程。

- 宽松的红色检测混入红手，F00 的错误候选在 y=175，实际鞋底在 y=243。F19–F31 也出现同类误取。将红色候选限定在佩戴者遮罩底边上方 12 像素内后，名义头位深度从最高约 13.50 米收敛到 6.78–7.34 米。只改这个判断，保留其他代码和输入。米数来自场地尺度假设。[宽松筛选对照](results/rebuild-receipt.json)
- 用 F00、F12、F24、F36、F48 两条场地白线的观测拟合方向，最大拟合残差 0.83 像素。假设白线正交、地平线斜率 0.025、机高 1.23 米，531 与 669 像素两种焦距都能解释该约束，重建头位相差中位 0.71 米，球的投影中心相差中位 14.14 像素。焦距及机高属于假设，颜色遮罩和脚点也有误差。旧 ViPE 深度的平面与点云一起用保留 EXO 像素射线的变换映射，其相对形状仍源于单目估计。[相机假设与现场记录](../../../../records/course_validation_20261019/capture_metadata.md#L11)
- F24 原画面有 260 个标成球的像素，全部进入两台虚拟相机的有效圆。投影收缩到 34/48 个目标位置后，最近深度只保住 3/12 个。其余位置的前表面都来自球遮罩外 1–5.2 像素的 EXO 邻域，估计深度差中位数 0.087/0.070 米。这个反例既否定了“进圆即可验收”，也说明最近深度无法单独确认球边界的真实遮挡顺序。[可复跑的球遮挡反例](visibility-counterexample.json)

只有在同帧球缘能够分辨，并且前后关系在合理的相机和深度扰动下仍稳定，才适合作为后续生成的保真约束。稀疏或冲突处应保留未知状态。

## 复跑

```bash
run=runs/pair04_s0024-a800-20261121/rig-refactor-20261122/pair05-reset
work=$(mktemp -d /tmp/pair05-rebuild.XXXXXXXX)
python=/usr/local/Caskroom/miniforge/base/bin/python3.12
OPENBLAS_NUM_THREADS=1 "$python" -B "$run/rebuild_pair05.py" "$work/geometry"
OPENBLAS_NUM_THREADS=1 "$python" -B "$run/check_ball_visibility.py" "$work/geometry" "$work/visibility.json"
```

两套候选分别输出 49 帧相机和 CPU 预览。源帧 24–72 对应本段 F00–F48，EXO 25fps 与预览 30fps 按相同帧索引核对，49 份深度的历史哈希逐个通过。`nominal-provenance.npz`、`alternate-provenance.npz` 用 `F##_target_label` 和 `F##_source_index` 记录逐帧 448×448 的已确认目标。标签 1 为白鞋队友，2 为球，3–6 为标志盘。源索引为同帧 R8 画面中的 `y * 784 + x`。未确认前景保留标签 0 和索引 −1。未使用 Nano 或 GPU。
