# 第 8 周：特征匹配与图像配准

真实输入来自已同步裁剪的第 4 组 EXO，在统一的 960 × 540 工作分辨率下使用 `0050.jpg` 与 `0200.jpg`。源文件路径和版本见 `metrics.json`，图片 SHA-256 与产物自检见 `artifact_audit.json`。

## 受控透视实验

从 `0050.jpg` 人为构造一个已知透视变换；这是**受控合成实验**，不是第一人称生成或真实跨机位配准。`H_true.txt` 与 `H_est.txt` 的方向均为 source → target。

ORB 提取各 5000 个特征点，经 0.75 比率检验得到 2374 对匹配，RANSAC 保留 1832 个内点（77.17%）。评估使用预先固定的 17 × 11 网格共 187 点，网格不参与特征检测、匹配或模型拟合。独立网格传递误差：均值 0.4758 px，中位数 0.4723 px，95 分位数 0.8273 px，最大值 1.2480 px。完整逐点记录在 `controlled/grid_evaluation.csv`。

建议报告使用：`controlled/ransac_inliers.png`、`controlled/registration_checkerboard.png`、`controlled/independent_grid.png`。

## 真实同机位时序实验

将 `0050.jpg` 匹配到 `0200.jpg`，得到 2815 对比率检验匹配、2663 个 RANSAC 内点（94.60%）。估计变换在同一固定网格上的平均图像位移为 5.8309 px；它是相对恒等变换的诊断量，**不是配准误差或准确率**。这对真实图像没有单应矩阵真值。

叠加图显示背景能够对应，运动人物因时刻不同保留双影。不能用这个实验声称一张单应矩阵可以完成 EGO/EXO 的三维视角转换。

建议报告使用：`temporal/ransac_inliers.png`、`temporal/registration_overlay.png`。

## 复现

在项目根目录执行；Python 需要 OpenCV 和 NumPy：

```sh
PYTHONPATH=/Users/mac/.cache/football-course-cv \
  /Users/mac/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  scripts/course_registration.py \
  --source output/week08/frames/scene04/exo/0050.jpg \
  --temporal output/week08/frames/scene04/exo/0200.jpg \
  --output output/week08/results/registration
```

固定随机种子 42、单线程 OpenCV；本次环境为 OpenCV 4.11.0、NumPy 2.3.5。19 张 PNG 均通过可解码与尺寸检查，另用独立齐次坐标计算检查了 CSV 与指标中变换方向及误差的一致性。
