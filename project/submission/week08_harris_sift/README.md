# 第 8 周 Harris 与 SIFT 实验

这是足球项目仓库的第 8 周实验目录。克隆主仓库后即可从本目录按下文运行。

## 内容与采集流程

R8/Nano 采集 → 音画同步 → 裁剪选帧 → 受控旋转/缩放/模糊样本生成 → 灰度与 Harris/SIFT 检测 → 图与指标 → 实验报告 PDF。

`data/frames/scene04/{exo,ego}/0050.jpg` 和 `0200.jpg` 共4张960×540 JPEG用于静态受控实验。scene04指**配对编号04，现场场景2**。原视频采集为3840×2160、25fps；F50/F200对应用户最终裁剪片段的2.00/8.00秒，精确映射与SHA256见 `data/manifest.json`、`data/frame_mapping.json`。

`data/capture_setup.jpg` 是现场场景2的两张机位记录拼图。新增动态证据使用 `data/video_pair04/exo.mp4` 与 `ego.mp4` 两路960×540工作副本，覆盖约12.44秒。`evidence/` 保存采集、同步、裁剪与动态运行证据。`data/source_videos_4k/trimmed_pairs/`另外附上5组最终裁剪后的4K EXO/EGO素材，共10条；它们已同步并按用户区间裁剪，打包时不转码。本包不含模型。

- `scripts/`：Harris/SIFT实验、独立审计、论文图绘制、报告组装与PDF构建脚本。
- `results/`：正式运行的参数、48条逐图/条件/算法记录、坐标、受控变换、掩膜、图与审计。
- `report/`：以采集、处理过程和完整结果为主的正式实验报告，包含少量摘要、关键词等论文元素；另有可编辑Markdown、同源JSON与正文附图。`report/paper_figures/`保留三张科研图的PNG/PDF/SVG、图源数据和作图QA，作为补充材料。
- `data/source_videos_4k/`：裁剪后的5组4K双视角源素材，共10条；manifest.csv/JSON记录原相对路径、编码、时长、可见帧数、大小与SHA256。
- `videos/动态特征检测_双视角对照.mp4`：直接观看两视角逐帧Harris/SIFT检测；加上两条960×540工作副本，包内共13个视频。MP4使用ZIP_STORED，保持原有字节。
- `evidence/history/`：上一版静态提交记录及历史6页论文稿的审阅/复现记录；其旧脚本与报告哈希不代表本轮实验报告。当前验收见 `evidence/delivery_validation.json`、`report_revision_review.json`和`report_portable_reproduction.json`；修订说明见 `evidence/revision_notes.md`。原48组结果与动态视频保持不变。
- `manifest.json` 记录原提交的文件来源。`SHA256SUMS` 对应本目录的公开文件，含 manifest 本身。

观看 `videos/动态特征检测_双视角对照.mp4`。从包内工作副本复现动态演示到空目录：

```sh
python scripts/build_feature_video.py --exo data/video_pair04/exo.mp4 --ego data/video_pair04/ego.mp4 --output dynamic_reproduced
python scripts/audit_feature_video.py --dynamic evidence/dynamic --static-results results --output dynamic_audit_reproduced.json
```

动态审计检查时间戳、文件与计数，并用同一检测函数抽样复算；不将其称为独立算法验证。

## 科学边界

本期用同一短片的两个时刻、两个视角，共4图；旋转、缩放和模糊是**受控样本生成**，不是新拍摄视角或EgoX生成。高位置重复率也可能来自较小分母：一个旋转案例的位置重复率r为97.1%，源点保留比例却只有17.3%，应结合点数与对应数理解。实验不评价描述子、语义准确率或三维重建，4图也不能支持泛化结论。变换JPEG仅供展示，精确复现必须从4张原始工作图在内存重新生成。

动态视频是每帧重新检测和绘制特征，**不分配持续点ID，不是特征跟踪**；人物与球的运动不能据此称为已跟踪。动态展示不改变静态实验的48条记录与24份坐标。

本期实验在本地 CPU 运行。第 14 周的第一人称转换另见[足球第一视角实验](../../experiments/week14/README.md)。

## 环境与运行

在解压后的仓库根目录执行，使用Python3.12。`requirements.txt` 只安装Python包；动态视频另需系统 **FFmpeg/ffprobe**，并保证二者在PATH中可执行。无需模型依赖：

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

复现到新的空目录，避免覆盖正式结果：

```sh
python scripts/course_feature_detection.py \
  --frames-root data/frames/scene04 \
  --output results_reproduced
python scripts/audit_feature_results.py \
  --results results_reproduced \
  --frames-root data/frames/scene04 \
  --output independent_audit_reproduced.json
python scripts/assemble_feature_report.py \
  --results results_reproduced \
  --output report_reproduced/report_data.json \
  --capture-evidence data/capture_setup.jpg \
  --paper-figures report/paper_figures \
  --dynamic evidence/dynamic
python scripts/build_week08_pdf.py --check-fonts
python scripts/build_week08_pdf.py \
  --data report_reproduced/report_data.json \
  --output report_reproduced/第8周_Harris与SIFT实验报告.pdf \
  --markdown report_reproduced/实验报告.md
```

科研图作为补充材料保留；重建报告无需先重绘科研图。若需单独复绘，运行：

```sh
python scripts/plot_week08_paper_figures.py \
  --results results \
  --frames-root data/frames/scene04 \
  --output paper_figures_reproduced \
  --qa-tools report/paper_figures/qa_tools
```

PDF需要可嵌入的中文TrueType字体；macOS默认查找Arial Unicode/STHeiti，其他系统可用构建器的 `--font` 指定本机中文字体文件。算法图使用可用的系统英文字体。无需访问源项目的绝对目录或云服务器；数据与视频按包内清单提供。

坐标与重复率应可复核；CPU单次耗时是描述性记录，会随机器变化。新PDF生成后应再逐页检查排版。
