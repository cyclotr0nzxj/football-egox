# 第 8 周计算机视觉实验

本目录是早期预实验与正式实验共用输入区，不是当前提交成果。旧11页报告及其Markdown/组装JSON已移出项目；保留下面的帧、采集图、参数与预实验结果，供正式Harris/SIFT及后续研究复用。

**当前唯一正式稿**：`../week08_harris_sift/第8周_Harris与SIFT实验报告.pdf`（9页）。清理记录见 `../../records/project_cleanup_20261021/manifest.json`。

## 当前实验与材料

- 输入为 `验证视频/裁剪后验证exo/`、`验证视频/裁剪后验证ego/` 的 5 组同步片段。主实验取**配对编号 04（现场场景 2）**，共 311 个可见帧，两路实际显示时间戳对应，源视频 3840×2160、25 fps。
- `frames/scene04/` 的 `scene04` 是配对 ID，不是现场场景编号。处理工作图统一为 960×540；源视频未修改，精确时间映射见 `frames/scene04/frame_mapping.json`。
- 拍摄设置、9 张机位/菜单照片及实测几何记录：`../../records/course_validation_20261019/capture_metadata.md` 与 `capture_metadata.json`。高度、距离、焦段只能辅助几何检查，不能当作完整内外参。
- `results/keyframes/`：4 个邻近触球过渡的双视角关键帧与标注。它们由助手查看原始画面选择，带 ±2 帧动作判断不确定性，并叠加原同步约 ±1 帧的不确定性；不是自动触球检测或精确接触真值。
- `results/image_processing/`：灰度、高斯滤波、Canny、HSV 颜色候选、形态学及形状筛选、掩膜/叠加与全部 6 帧指标。固定 ROI 和规则见 `../../records/course_validation_20261019/segmentation_config_and_annotations.json`。
- `results/registration/`：ORB 特征、比率检验、RANSAC、受控透视配准与真实同机位时序匹配，含图、矩阵和独立网格指标。

## 结果及适用范围

6 个预选帧上，粗参考平均 IoU 由原始 HSV 的 **0.4512** 变为后处理的 **0.6193**。F170 选错区域、F130 足球位于 ROI 外，两帧 IoU 为 0，均计入总均值。参考多边形由助手查看未处理图像绘制，未经独立人工复核，属于粗参考；这不是精细像素真值、训练分割模型或泛化测试。

受控透视实验有 2374 对比率检验匹配、1832 个 RANSAC 内点；独立于拟合的 17×11 网格共 187 点，平均传递误差 **0.4758 px**、95 分位数 **0.8273 px**。真实同一 R8 的 F50/F200 有 2663 个内点，其平均恒等位移 5.8309 px 仅是诊断量；没有单应矩阵真值，不报告真实配准准确率。这两个实验均不能证明完整 EGO/EXO 三维视角转换。

生成部分复用 **2026-09-11 官方样例的真实 EgoX 结果**，包括 `../../artifacts/generated_ego.mp4`、官方输入与 GT；本次核对哈希，记录在 `../../records/course_validation_20261019/official_reuse_integrity.json`。原实验评估见 `../../artifacts/evaluation.json`、`../../REPORT.md`。官方样例与本次自采录像属于不同数据，未混为同场景三路对照。

本次没有运行自采 ViPE、目标轨迹、先验渲染或 EgoX，也没有新增 GPU 任务。Nano 只作为同步和实拍参考，不充当生成先验或生成首帧。未裁剪原片备份当前缺失，现有实验基于保留的基础对齐版和用户二次裁剪版；历史原片坐标不代表原片已恢复。

## 复现环境

在项目根目录使用 **Python 3.12**，依赖版本见本目录 `requirements.txt`。另需 `ffmpeg`、`ffprobe` 可从命令行调用。本次在 macOS 上运行，图注脚本使用系统 Monaco 字体，PDF 优先使用 `/System/Library/Fonts/Supplemental/Arial Unicode.ttf`，并支持可嵌入 STHeiti 回退。跨系统时需调整对应字体路径。

```sh
cd .
python3.12 -m venv .venv-week08
source .venv-week08/bin/activate
python -m pip install -r output/week08/requirements.txt
```

当前 Codex 主机也可复用已安装环境：Python 为 `/Users/mac/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3`，OpenCV 在 `/Users/mac/.cache/football-course-cv`。使用这个环境运行下面命令时，需设 `PYTHONPATH=/Users/mac/.cache/football-course-cv`。

## 从现有同步短片复现

以下命令在项目根目录执行；不会重新裁剪用户视频或启动模型推理。

1. 按已核验的可见帧与时间映射提取报告所需帧；已存在且匹配的帧会复用。

```sh
python scripts/prepare_week08_frames.py \
  --manifest records/user_ego_trim_20261019/trim_manifest.json \
  --pair 04 \
  --output output/week08/frames
```

2. 运行图像处理、固定规则分割、粗参考评估及关键帧排图。

```sh
python scripts/course_image_experiments.py \
  --config records/course_validation_20261019/segmentation_config_and_annotations.json \
  --output output/week08/results/image_processing
```

3. 运行特征、匹配及两种配准实验。随机种子固定为 42，受控误差网格不参与拟合。

```sh
python scripts/course_registration.py \
  --source output/week08/frames/scene04/exo/0050.jpg \
  --temporal output/week08/frames/scene04/exo/0200.jpg \
  --output output/week08/results/registration
```

4. 从真实指标与已有图组装报告数据。

```sh
python scripts/assemble_week08_report.py
```

5. 检查中文字体并生成 PDF 与同源 Markdown；生成后需逐页渲染检查，不能仅凭命令退出成功就认定版式验收通过。

```sh
python scripts/build_week08_pdf.py --check-fonts
python scripts/build_week08_pdf.py \
  --data output/week08/report_data.json \
  --output output/week08/第8周实验报告.pdf \
  --markdown output/week08/实验报告.md
```

准确截止日期、报告个人信息以及是否必须在本次自采素材上完成生成，仍以教师要求为准。
