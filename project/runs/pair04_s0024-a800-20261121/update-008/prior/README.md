# Pair05 上半身排除

`render_pair05.py` 接收已归档的 49 帧排除遮罩。在渲染进程内从普通背景、平均背景与逐帧动态点的深度采样中排除这些像素。ViPE 输入、实例标签、RGB、深度和相机数据保留原样。Pair04 和 Pair01 仍调用官方渲染器。

实际使用的官方脚本必须与 `17cdef688885c3ac6a4c371a2d620391cc28dab4` 的公开版本逐字节匹配，SHA-256 为 `177721241c123019e7f1932639c59efb510e314514f561ef9d54ce122f6a1ee8`。运行时还检查排除遮罩、ViPE 几何及元数据结构。[固定版本源码](https://raw.githubusercontent.com/kdh8156/EgoX-EgoPriorRenderer/17cdef688885c3ac6a4c371a2d620391cc28dab4/scripts/render_vipe_pointcloud.py)

在项目根目录用本机现有 NumPy 和 OpenCV 执行以下只读检查。

```bash
run=runs/pair04_s0024-a800-20261121
python3 -B "$run/update-008/prior/render_pair05.py" \
  --renderer records/official_renderer_geometry.py \
  --exclusion-zip "$run/ball-view-cpu-review/pair05-composition-review/pair05-wearer-upper-body.zip" \
  --input_dir "$run/multiclip-evidence/pair05_s0024/geometry/vipe_results/pair05_s0024" \
  --out_dir "$run/update-008/prior/render-check-only" \
  --meta_json_path "$run/ball-view-cpu-review/pair05_s0024-ball-view-meta.json" \
  --point_size 5.0 --start_frame 0 --end_frame 48 --fish_eye_rendering --use_mean_bg --check-only
```

本机全部 49 帧通过。排除区每帧 2,614–3,094 像素，全部属于佩戴者实例 1。渲染步长 2 下，每帧仍有 204–363 个佩戴者下半身采样和 2,537–4,212 个队友采样，均为深度有效性筛选前的计数。排除区与已标注的 49 个球区均无交叠。小型 CPU 样例覆盖三个采样分支和补丁接点缺失时的拒绝路径。运行实际先验时去掉 `--check-only`，使用已固定的渲染器环境和新输出目录，核验 `<out_dir>/pair05_s0024/ego_Prior.mp4` 是 448×448、49 帧、30 fps，且逐帧可解码。

Pair05 的 49 个球心均由 ViPE 标为背景，实际先验仍需检查球是否被静态平均冲淡。Pair01 传球和停球段有 14 帧的球心核心采样全部进入动态层、14 帧部分进入、21 帧全部进入静态层。F16、F32、F36 的 ViPE 实例 2 已进入官方动态采样，保留是否可见仍受深度和投影影响。若实际先验出现漏球，应以原 EXO 逐帧确认球的可见像素，再在独立对照中测试仅将这些观测移入动态层。现有标注用于定位和观察，遮挡像素不补绘。
