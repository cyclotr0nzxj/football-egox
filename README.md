# 从足球场边到第一视角

计算机视觉课程作业。用固定机位拍摄的足球视频估计单目深度，将可见的场景投到红鞋球员头部附近的虚拟相机，再用 EgoX 生成第一视角。头戴相机的实拍用于同帧对照。

[阅读全文（PDF）](report/week14_football_egox.pdf) · [报告源文件与图表依据](report/README.md)

## 视频附录

每段按「第三人称 R8｜实拍第一人称 Nano｜实验结果」排列。源帧编号来自已同步的短片。每段 49 帧、25 fps、1.96 秒。拍掌对齐的人工判断误差约为 1 帧。

| 片段 | 同步源帧 | 右栏内容 | 对照视频 |
| --- | --- | --- | --- |
| Pair01 | F288–F336 | 低头看球几何预览 | [在线播放](https://cyclotr0nzxj.github.io/football-egox-cv-appendix/#pair01) · [下载 MP4](https://raw.githubusercontent.com/cyclotr0nzxj/football-egox-cv-appendix/main/appendix/videos/Pair01.mp4) |
| Pair02 | F000–F048 | EgoX 首版 | [在线播放](https://cyclotr0nzxj.github.io/football-egox-cv-appendix/#pair02) · [下载 MP4](https://raw.githubusercontent.com/cyclotr0nzxj/football-egox-cv-appendix/main/appendix/videos/Pair02.mp4) |
| Pair04 | F024–F072 | EgoX 首版 | [在线播放](https://cyclotr0nzxj.github.io/football-egox-cv-appendix/#pair04) · [下载 MP4](https://raw.githubusercontent.com/cyclotr0nzxj/football-egox-cv-appendix/main/appendix/videos/Pair04.mp4) |
| Pair05 | F024–F072 | 跟球几何预览 | [在线播放](https://cyclotr0nzxj.github.io/football-egox-cv-appendix/#pair05) · [下载 MP4](https://raw.githubusercontent.com/cyclotr0nzxj/football-egox-cv-appendix/main/appendix/videos/Pair05.mp4) |

[浏览器播放器](https://cyclotr0nzxj.github.io/football-egox-cv-appendix/)汇集四段视频。每段中间帧及逐帧解码、时戳和 SHA-256 记录见 [画面与校验](appendix/README.md)。

## 实验方法

先用 R8 估计逐帧深度，将彩色像素反投影到固定相机的三维坐标系。场地使用跨帧汇聚的静态点，人物来自对应时刻的动态点。给虚拟相机选定红鞋球员头部路径后，观看方向仍可调整。报告同时给出点云实例和从选定外参反算的俯角曲线。

Pair01 的镜头朝向脚边，Pair05 按球的估计位置调整朝向。Pair02 和 Pair04 的首版采用前视设定，画面保留前方队友和场地。四条对照并排给出真实头戴取景与各自的实验结果。
