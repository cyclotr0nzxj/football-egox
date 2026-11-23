# Pair03 本机几何预览

本机用 Pair03 的 49 帧 R8 第三人称画面估计深度、重建红鞋球员的虚拟头部轨迹，再将同帧已观测像素投影到这台虚拟相机。朝向以同时看到白鞋队友和球为目标，由 EXO 标注构造。镜头是 114° 的近似等距鱼眼，方便检查视野。轨迹与正式预览先锁定，再打开 Nano 做对照。全程未开云实例，沿用 Pair05 原模型权重。

## 可复查的输入与结果

- [49 帧 EXO](../pair05-left-diagnosis-20261122/pair03-exo/exo.mp4) 及 [逐帧标注](annotations/pair03-exo-anchors.json)。两位球员用 25 帧人工点和 24 帧插值定位。逐帧人工检查球，18 帧清楚、13 帧部分可见、18 帧遮挡或无法确认。
- [MiDaS 原始相对逆深度](depth/midas_inverse_depth.npz) 和 [深度收据](depth/depth-receipt.json)。本机 CPU 的 49 帧推理与存盘约 11.3 秒，输入未上传。[MiDaS 官方 ONNX 流程](https://github.com/isl-org/MiDaS/blob/master/tf/run_onnx.py)
- [估计相机 Z 深度](geometry-ground-reprojection/approx_camera_z_depth.npz)、[头部轨迹](geometry-ground-reprojection/head-trajectory.json) 和 [预览冻结记录](geometry-ground-reprojection/preview-freeze.json)。深度数组为 49×448×784，0 表示无有效深度。在观察清楚的草地上以近似平面约束相对尺度。头部位置结合头部像素、落脚位置和受限深度，五帧平滑后再投影与 EXO 标注的最大差为 1.17 像素；这个读数验证像面一致性。
- [冻结的几何预览](geometry-ground-reprojection/pair03-geometry-preview.mp4)，448×448、49 帧、25 fps。直接覆盖画面面积的中位数为 62.7%。复投影用 EXO 草地纹理的逆向平面采样来降低逐点投影的条纹，人员与球仍用同帧深度和距离遮挡判断。没被采到或被身体挡住的区域保持黑色。早期纯点投影保存在 `geometry/`。

## 几何假设与可见性

按 EXO 居中裁切、24 mm 的拍摄记录估计像素焦距 530.96。首帧右侧两条场地线在图像坐标下近似为 `y = -0.12582x + 346.81` 与 `y = 0.11650x + 207.01`。把相交白线视作直角，灯杆提供粗略的滚转约束，采用地面消失线 `y = 208.81 + 0.025(x - 392)`，配合 1.23 米相机高度。用四个 EXO 标志盘做量级核对，最近盘的水平距离约 3.47 米，四段边长约 5.44–6.03 米。焦距、地面方向与米制尺度均为估计。[现场拍摄与布置记录](../../../records/course_validation_20261019/capture_metadata.md#L20)

冻结预览中，18/18 个 EXO 清楚见球帧都有队友与球的直接投影像素，球的直接像素从 4 到 83 个不等。已遮挡、部分可见与无法确认的帧未计入这个统计。像素存在只表明有对应的输入观测经过了遮挡检查；人物在视角变化后仍可能破碎，几个像素也无法辨认成球。最终画面由并排成片的肉眼复查补充判断。

## 与 Nano 的同帧对照

并排片左侧是几何预览，右侧是等比例缩放的 Nano 实拍。49 帧按裁后可见索引对齐，输入和导出逐帧时间戳都是 `i/25`，历史两机同步误差约为 ±1 帧。Nano 在预览哈希锁定后才进入对照，保留原画幅。[并排视频](comparison/pair03-geometry-vs-nano.mp4) · [关键帧](comparison/pair03-geometry-vs-nano-keyframes.jpg) · [时间戳收据](comparison/comparison-check.json)

F00 已能辨认几何里的白鞋队友轮廓和红鞋位置，投影球只剩小块颜色。F24 的 Nano 清楚看到前方队友、近脚处的球和红鞋，几何里的队友只剩断裂轮廓，球也没有稳定形状。F36、F48 的 Nano 仍拍到球，第三人称对应帧的球被挡住或无法确认，几何无法复原这些观测。逐帧复查表明两种观测可以同时投进画幅，当前几何未通过两者同时稳定可辨的画面验收。视角是真实头戴实拍和设定虚拟朝向的对照，适合审查构图与可见性。[逐帧画面复查](comparison/visual-review.md)

## 复跑

从项目根目录运行。模型文件的 SHA-256 由脚本验证，计算写入新目录。复跑使用官方 [MiDaS v2.1 Small ONNX 权重](https://github.com/isl-org/MiDaS/releases/tag/v2_1) 和本机已有的 OpenCV、ONNX Runtime 依赖。

```bash
run=runs/pair04_s0024-a800-20261121/pair03-local-geometry-20261122
work=$(mktemp -d /tmp/pair03-replay.XXXXXXXX)
py=/usr/local/Caskroom/miniforge/base/bin/python3.12
model="$HOME/.cache/football-course-cv/midas-v2_1-small/model-small.onnx"
"$py" -B "$run/estimate_depth.py" --model "$model" --output "$work/depth"
"$py" -B "$run/build_geometry.py" --depth "$work/depth/midas_inverse_depth.npz" \
  --anchors "$run/annotations/pair03-exo-anchors.json" --output "$work/geometry"
preview="$run/geometry-ground-reprojection/pair03-geometry-preview.mp4"
"$py" -B "$run/build_nano_comparison.py" "$preview" \
  --preview-sha256 b9ecd66561f6d895283786952eda77f1e22f9ea89aee8985fbcad9f3f6ad3450 \
  --output-dir "$work/comparison"
```
