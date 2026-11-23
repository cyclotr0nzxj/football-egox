# 第8周：足球双视角画面的特征稳定性分析

已完成Harris/SIFT真实实验，并保留约12.44秒的双视角动态证据；报告恢复以原实验报告的采集、处理过程和完整结果展示为主，仅加入摘要、关键词等少量论文元素。本期只在本地CPU执行，未运行自采EgoX或微调。

## 报告与数据

报告主线：**完整足球动作 → 双机采集与同步 → 动态检测 → 关键帧受控变化 → 全条件结果 → 失败案例与素材选择**。优先展示过程和完整结果，保留原图、响应图、角点、动态预览、全条件表与失败案例；补充科研图不替代这些实验过程证据。未新增实验或改变参数。

处理链：R8/Nano采集 → 拍掌同步 → 对应裁剪与选帧 → 旋转/缩放/模糊样本生成 → 灰度及Harris/SIFT → 可核查的图与指标。这里的生成是受控图像变化，不是EgoX第一视角生成。

- 输入为配对04（现场场景2）的EXO/EGO F50、F200，共4张960×540 JPEG。映射与SHA256保留；最终裁剪后的5组4K双视角素材随包附带。
- 实际运行4图×6条件×2方法共48组：恒等对照、15°/45°旋转、0.75/1.25倍缩放、9×9 sigma=2高斯模糊。
- `results/metrics.json`、CSV和`coordinates/`保留参数、H、有效域、计数与一对一位置对应。变换JPEG仅供展示，重算应从4张原始工作图重新生成。
- 核心案例：EGO F200旋转45°后，Harris的位置重复率r为68/70=97.1%，源点保留比例仅68/392=17.3%。不能只凭高重复率或点多判断方法更好。
- `第8周_Harris与SIFT实验报告.pdf`、`实验报告.md`和`report_data.json`同源；`report_validation.json`与`contact_sheet.jpg`记录排版验收。
- `paper_figures/`保留三张科研图的PNG、PDF、SVG、源数据与绘图QA，作为补充材料；`revision_notes.md`记录本轮恢复与轻量修改的口径。
- 本轮报告审阅与包内重建分别见`report_revision_review.json`、`report_portable_reproduction.json`。原`paper_revision_review.json`与`paper_portable_reproduction.json`只验证历史6页论文稿，打包时置于`evidence/history/`并明确标注历史。
- `independent_audit.json`由另一套实现依据导出坐标复核H、有效域、最大对应数与宏平均，不重跑检测器。
- `dynamic/动态特征检测_双视角对照.mp4`为完整311帧、25fps、1536×960四宫格：左R8、右Nano，上Harris、下SIFT，保留Nano音轨。`dynamic/exo.mp4`、`ego.mp4`是960×540工作输入。
- `data/source_videos_4k/trimmed_pairs/pair01`至`pair05`在提交仓库中提供最终裁剪后的4K EXO/EGO视频，共10条，按现有文件原样复制，不转码。配对04只收录一次。`manifest.csv`与`manifest.json`保存元数据、原项目相对路径及SHA256；已有裁剪验收区分可见帧数与MP4预滚帧。
- `dynamic/counts.csv`含1244条帧内检测记录；`independent_dynamic_audit.json`核验三路PTS、20次同函数抽查、5张叠加预览及F50/F200动作时刻。逐帧检测不是跟踪；工作视频与静态JPEG编码不同，点数可有差异。

## 复现

Python3.12，固定依赖见`requirements.txt`；视频另需PATH中的FFmpeg/ffprobe。从项目根目录运行，新的结果目录必须不存在或为空。

复现动态视频及核验：

```sh
python scripts/build_feature_video.py --exo output/week08_harris_sift/dynamic/exo.mp4 --ego output/week08_harris_sift/dynamic/ego.mp4 --output output/week08_harris_sift/dynamic_reproduced
python scripts/audit_feature_video.py --dynamic output/week08_harris_sift/dynamic_reproduced --static-results output/week08_harris_sift/results --output output/week08_harris_sift/dynamic_audit_reproduced.json
```

复现静态实验和报告（组装脚本自动发现正式dynamic目录；也可用`--dynamic`指定复现目录）：

```sh
python scripts/course_feature_detection.py \
  --frames-root output/week08/frames/scene04 \
  --output output/week08_harris_sift/results_reproduced
python scripts/audit_feature_results.py \
  --results output/week08_harris_sift/results_reproduced \
  --frames-root output/week08/frames/scene04 \
  --output output/week08_harris_sift/audit_reproduced.json
python scripts/assemble_feature_report.py \
  --results output/week08_harris_sift/results_reproduced \
  --output output/week08_harris_sift/report_reproduced/report_data.json \
  --paper-figures output/week08_harris_sift/paper_figures
python scripts/build_week08_pdf.py \
  --data output/week08_harris_sift/report_reproduced/report_data.json \
  --output output/week08_harris_sift/report_reproduced/report.pdf \
  --markdown output/week08_harris_sift/report_reproduced/report.md
```

补充科研图可单独复绘，报告重建不要求先完成此步骤：

```sh
python scripts/plot_week08_paper_figures.py \
  --results output/week08_harris_sift/results \
  --frames-root output/week08/frames/scene04 \
  --output output/week08_harris_sift/paper_figures_reproduced \
  --qa-tools output/week08_harris_sift/paper_figures/qa_tools
```

原检测使用Python3.12、OpenCV4.11.0和NumPy2.3.5；论文图使用Matplotlib，具体依赖见requirements。绘图脚本读取既有CSV与坐标，并可携带对齐检查工具独立复绘。PDF需可嵌入的中文TrueType字体；构建器支持`--font`或`COURSE_REPORT_FONT`。计时和字体渲染随机器变化，检测坐标与计数应可复核。

## 打包上交

本地Git仓库：`submission/week08_harris_sift/`。ZIP：`dist/第8周_Harris与SIFT_提交包.zip`。它们以项目根目录为参照。

提交仓库包含`data/`、`scripts/`、`results/`、`videos/`、`evidence/`、`report/`及完整复现说明。ZIP附4张JPEG、5组最终裁剪后的4K EXO/EGO素材共10条、两路960×540工作输入视频和一条检测成品，合计13个视频；不附.git或模型。所有MP4使用ZIP_STORED，未进行转码，源文件复制前后哈希一致。静态结果不变；上一版8页报告的独立复现保留为历史记录，本轮实验报告与动态视频的当前验收见`delivery_validation.json`。ZIP另有逐文件SHA256及CRC检查。

旧版报告与dist旧解压目录已移出项目；`output/week08/`只保留输入与预实验结果，当前正式稿为本目录9页PDF。清理记录见`records/project_cleanup_20261021/manifest.json`。第14周再开展单目深度辅助三维重建及自采EgoX验证。
