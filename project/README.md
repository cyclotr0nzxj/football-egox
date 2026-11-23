# 足球第一视角项目

本目录保留本地项目的主要结构。汇报材料与四段最终对照见[仓库首页](../README.md)。

| 内容 | 入口 |
| --- | --- |
| 足球第一视角转换实验 | [两段首版的推理、相机调整和 CPU 核验](experiments/week14/README.md) |
| 第 8 周图像实验 | [Harris/SIFT 代码、实验数据和 PDF](submission/week08_harris_sift/README.md) |
| 自采同步录像及相机设置 | [验证视频](验证视频/) |
| 项目源码与实验记录 | [scripts](scripts/) · [runs](runs/) · [records](records/) |
| 官方基线与历史记录 | [基线报告](REPORT.md) · [历史说明](PROJECT_HISTORY.md) · [状态](STATUS.md) |

## 本机检查

安装 Python 3.10+ 和 FFmpeg（含 ffprobe），从本目录运行。

```sh
python3 scripts/pair04_cpu_dry_run.py --input-root data/self_capture/pair04_s0024/egox_input
cd experiments/week14
python3 code/verify_egox_outputs.py reference-output checked \
  --take pair02_s0000 --take pair04_s0024 --contact-sheets
```

继续跑 EgoX 要在自己的 CUDA 环境安装上游依赖，并下载官方 Wan 底座和 EgoX LoRA。首版的相机、深度和先验已在实验目录，运行参数和权重校验清单随目录提供。

## 数据与上游代码

本项目采集的同步录像和最终结果随仓库提供。拍摄设置照片的公开副本移除了位置元数据，像素与本地照片一致。大文件使用 Git LFS。项目没有指定覆盖代码与视频的统一许可证。

EgoX 源码以固定提交的 Git 子模块提供。官方基线使用的第三方 Ego-Exo4D 录像和衍生画面由数据方分发。本目录保存了本项目的指标、哈希和下载脚本。查看官方数据许可后，可按历史说明运行 `python3 scripts/download_sample.py --root "$PWD"`。[Ego-Exo4D 许可](https://docs.ego-exo4d-data.org/getting-started/) · [EgoX](https://github.com/DAVIAN-Robotics/EgoX)

历史云端脚本和日志记录了当时的运行条件。维护者按自己的环境填写地址和路径。带 `example.invalid` 的地址是占位值。
