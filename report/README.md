# 报告源文件

[阅读全文（PDF）](week14_football_egox.pdf)

正文采用 XeLaTeX 排版。图像来自本次实验的实际帧与点云。俯角曲线由最终几何预览的相机旋转反算。

在本目录编译。

```sh
latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex
```

[图源索引](source-data.json)记录了插图帧号和文件哈希。[俯角数据](view-pitch.csv)保留了两个窗口逐帧的方向读数。

四段同步对照视频、关键帧及验证记录见 [附录 A](../appendix/README.md)。
