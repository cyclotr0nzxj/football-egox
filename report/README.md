# 报告源文件

[阅读全文（PDF）](week14_football_egox.pdf)

正文采用 XeLaTeX 排版。图像都来自本次实验的实际帧或点云，叠加说明只标示片段、帧号和画面类型。

在本目录编译。

```sh
latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex
```

[插图帧号和文件哈希](source-data.json)可用于核对画面来源。

四段同步对照视频、关键帧及验证记录见 [附录 A](../appendix/README.md)。
