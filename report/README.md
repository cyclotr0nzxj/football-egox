# 足球第一视角实验报告

- [复审版课程报告（PDF，13 页）](week14_football_egox_reviewed.pdf) 详述深度反投影、随动机位和逐帧核验，新增几何先验、生成画面与头戴实拍的同帧对照。
- [原版报告（PDF，6 页）](week14_football_egox.pdf) 概述转换方法与四段足球动作的结果。

## 原版报告源文件

[LaTeX 源文件](main.tex) · [图源索引](source-data.json)

方法图来自实验的实际图像、深度、点云和几何先验。多帧结果图从两段已发布的首版视频中提取。正式附录收录片段编号、视频入口和实现资料。

从本目录用 XeLaTeX 编译。

```sh
latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex
```

[视频附录](../appendix/README.md)
