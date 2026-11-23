# 足球第一视角：从第三人称视频生成第一人称画面

计算机视觉课程实验。以单个场边机位拍摄的足球录像为输入，生成参与者位置的第一人称画面。同步的头戴实拍用于对照。

## 向队友传球

[![三栏对照动画，左为场边录像，中为头戴实拍，右为 EgoX 生成](appendix/passing-preview.gif)](https://cyclotr0nzxj.github.io/football-egox-cv-appendix/#pair02)

[播放最终版](https://cyclotr0nzxj.github.io/football-egox-cv-appendix/#pair02) · [下载原版 MP4](https://raw.githubusercontent.com/cyclotr0nzxj/football-egox-cv-appendix/main/appendix/videos/Pair02.mp4)

[阅读全文（PDF）](https://cyclotr0nzxj.github.io/football-egox-cv-appendix/report/week14_football_egox.pdf) · [报告源文件与图源](report/README.md) · [四段视频播放页](https://cyclotr0nzxj.github.io/football-egox-cv-appendix/)

## 克隆完整项目

安装 Git LFS 后克隆。大文件会随工作区检出，EgoX 通过固定版本的子模块取得。

```sh
git lfs install
git clone --recurse-submodules https://github.com/cyclotr0nzxj/football-egox-cv-appendix.git
cd football-egox-cv-appendix/project
```

[项目目录](project/README.md)提供源码、原始及同步裁剪的自采录像、实验输入与记录、首版推理命令，以及第 8 周 Harris/SIFT 实验。

## 视频附录

左栏为场边第三人称，中栏为头戴实拍第一人称，右栏为转换结果或几何预览。场边相机为 Canon EOS R8，头戴相机为 DJI Nano。视频标签中的 R8 和 Nano 分别指这两路画面。Pair 编号是文件标识，正文使用片段名称。四段均为 49 帧、25 fps、1.96 秒。拍掌对齐的人工判断误差约为一帧。

| 正文名称 | 文件编号 | 源帧 | 右栏结果 | 视频 |
| --- | --- | --- | --- | --- |
| 低头看球 | Pair01 | F288–F336 | 几何预览 | [在线播放](https://cyclotr0nzxj.github.io/football-egox-cv-appendix/#pair01) · [下载 MP4](https://raw.githubusercontent.com/cyclotr0nzxj/football-egox-cv-appendix/main/appendix/videos/Pair01.mp4) |
| 向队友传球 | Pair02 | F000–F048 | EgoX 首版 | [在线播放](https://cyclotr0nzxj.github.io/football-egox-cv-appendix/#pair02) · [下载 MP4](https://raw.githubusercontent.com/cyclotr0nzxj/football-egox-cv-appendix/main/appendix/videos/Pair02.mp4) |
| 沿标志盘带球 | Pair04 | F024–F072 | EgoX 首版 | [在线播放](https://cyclotr0nzxj.github.io/football-egox-cv-appendix/#pair04) · [下载 MP4](https://raw.githubusercontent.com/cyclotr0nzxj/football-egox-cv-appendix/main/appendix/videos/Pair04.mp4) |
| 跟球 | Pair05 | F024–F072 | 几何预览 | [在线播放](https://cyclotr0nzxj.github.io/football-egox-cv-appendix/#pair05) · [下载 MP4](https://raw.githubusercontent.com/cyclotr0nzxj/football-egox-cv-appendix/main/appendix/videos/Pair05.mp4) |

[静帧和核验清单](appendix/README.md)记录了每段视频的源帧、对照和哈希。报告附录还列出实际采用的 EgoX、ViPE 和几何渲染器。
