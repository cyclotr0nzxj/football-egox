# Pair02、Pair04：100 帧延拓的开机前准备

**当前结论：开机前可完成的启动、回收和验收准备已打包。** 用户明确开机后，可上传准备包及两份启动脚本，用一次入口依次执行预检、四窗 ViPE、重叠与头点检查、先验渲染和 EgoX 生成。两段独立处理，失败不自动重试。技术检查通过不代表画面通过；实际云端环境与新数据尚未运行，任何关卡失败都会停下相应片段。实例没有由本轮开机；未使用 Nano。

## 本机证据

- [四个窗口和头点清单](windows/manifest.json)：Pair02 为 F28–F76、F52–F100；Pair04 为 F52–F100、F76–F124。额外的最后一帧仅用于固定的 49 帧末窗，最终仍截到各自的 100 帧、25 fps、4 秒。EXO 输入和时戳的完整核验见 [原准备收据](../pair02-pair04-100f-prep-20261122/inputs/exo-100-verification.json)。
- 观察 EXO 接触表，两段目标球员是灰色上衣、红鞋的人；交叠区未见身份切换。自动光流曾在 Pair04 后段滑到肩部；已基于头部画面作稀疏人工修正，查看自动跟踪的逐帧裁图及修正后的重点画面。这些是**头表面候选**，不是头戴光心或真实视线。[Pair02 修正裁图 1](Pair02-corrected-review-1.jpg) · [2](Pair02-corrected-review-2.jpg) · [3](Pair02-corrected-review-3.jpg) · [4](Pair02-corrected-review-4.jpg)；[Pair04 修正裁图 1](Pair04-corrected-review-1.jpg) · [2](Pair04-corrected-review-2.jpg) · [3](Pair04-corrected-review-3.jpg) · [4](Pair04-corrected-review-4.jpg)。二维轨迹来自旧首窗锚点加光流和稀疏校正，不是逐帧独立标注。
- 本机 MiDaS v2.1 Small 处理各 101 帧，偶数首窗帧的草地拟合到旧 ViPE 逆深度，奇数首窗帧留出检验。草地相对逆深度误差中位数约 Pair02 2.54%、Pair04 2.28%，但头点 5×5 邻域推得的深度中位数分别为旧平滑头轨深度的 7.13 倍、5.61 倍。这是两种估计的差异、也可能受小区域混入背景影响；它足以拒绝把这份 MiDaS 深度直接当作人物几何。[Pair02](Pair02-depth-audit.json) · [Pair04](Pair04-depth-audit.json)
- 利用红鞋检测、旧 ViPE 地面和平面投影，建立头部深度候选。用首窗偶数帧标定一个比例，奇数帧与旧头轨 Z 的相对差中位数分别为 1.37%、1.54%。这是与**同一份旧估计**的一致性，不是对真实尺度的验证。鞋子腾空、身体倾斜、画面位移和地面误差会影响它。[Pair02 鞋点](Pair02-foot-review-1.jpg) · [后段](Pair02-foot-review-2.jpg)；[Pair04 鞋点](Pair04-foot-review-1.jpg) · [后段](Pair04-foot-review-2.jpg)。
- [Pair02 头轨候选](Pair02-trajectory-candidate.json) · [Pair04 头轨候选](Pair04-trajectory-candidate.json)：保留首窗 49 个旧外参，后段接鞋点估计并延续原先固定目标的虚拟朝向。新窗彼此共有的 25 帧外参逐项完全相同；这些是候选，不代表新 ViPE、渲染先验或生成画面通过。虚拟朝向不是实际头部视线。
- 背景特征匹配显示图像并非完全静止。以各自 100 帧片段首帧为参考，末帧 Pair02 背景中心约 `(0.33, -1.32)` 像素、Pair04 约 `(-7.22, 3.39)` 像素；Pair04 匹配内点 1292 个，中位拟合残差 0.77 像素。[测量](camera-motion.json)。这不能单独区分相机、稳定和镜头等原因，旧 `assume_fixed_camera_pose` 是近似。尺度对齐前应检查影响，不能把共用内参当作已完成三维配准。

## 启动、回收与验收

[准备包](exo-geometry-prep.zip) 含四窗 EXO、旧几何和首窗先验、候选头轨、原版权重的运行检查与全部流水线代码；[包校验收据](package-receipt.json)。[总入口](run_all_after_boot.sh) 与 [ViPE 入口](prepare_vipe_after_boot.sh) 须与准备包一起上传。脚本在已有运行目录时拒绝重复启动，先检查空闲 A800、CUDA、空间、原版模型缓存和运行环境；先运行四窗 ViPE，再按对图片段依次进行背景重叠尺度、头点/深度和渲染检查，满足技术门槛才使用官方 EgoX 权重推理。初始首窗成片复用，不再重算。

新 ViPE 深度保持原生数值，累计背景比例记作 `s = 旧深度 / 新深度`。在固定 EXO 相机坐标近似下，虚拟世界到相机矩阵的平移项除以 `s`；新 ViPE 的内参和深度同时供渲染器与 GGA 使用，避免重复缩放。[几何输入检查](stage_geometry.py) 会核对固定坐标、K、EXO 头点重投影和每帧人物掩码深度。比例 1.7 倍、连续 2×3 倍、仅头部深度错误和错误比例的 [本机检查](stage-geometry-selftest.json) 已通过；这是计算链验证，不是真实深度标定。

[背景重叠检查](check_overlap_geometry.py) 在同一源帧上拟合比例，并用交替留出帧检查；暂定工程筛选线不等于真实精度，也不能代替人物、球和画面接缝验收。[本机缩放、焦距与深度漂移测试](alignment-selftest.json)、[错源帧及错头点来源拒绝测试](rejection-selftest.json) 均通过。[完整流程模拟](render-stage-selftest.json) 使用旧首窗几何和成片假扮两窗、模拟渲染与推理命令，覆盖暂存、输出核验、归档和回收校验；**没有执行新 ViPE、渲染器或 EgoX**。[总入口失败分支模拟](driver-selftest.json) 验证一段失败仍尝试另一段，且拒绝覆盖已有运行。

通过技术检查的成片及其先验、深度、轨迹、日志和重叠证据由 [收集脚本](collect_008.py) 打包。取回后使用 [接收脚本](receive_008.py) 验证远端哈希及逐文件哈希；[拼接脚本](assemble_extension.py) 保留旧首窗，按源帧 49 和 76 两个相对位置暂定硬切，输出 100 帧、25 fps 的**待视觉验收草稿**和两个逐帧重叠对照。两个片段的 EXO 模拟检验了源帧选择、100 帧和逐帧时间戳，见 [测试收据](assembly-selftest.json)。源帧相同不代表生成内容连续，必须看完整接缝、红鞋身份和重复球后再决定是否替换首版；Nano 只在成片冻结后用于独立评估。

正常等待用户明确开机后，上传同一目录下的准备包、总入口和 ViPE 入口，执行总入口；取回全部可用证据并验收后提前关机。GPU、云端权重、四个新几何、实际生成和关机状态均须届时实测，不能用本机模拟代替。Pair01、Pair05 的几何预览与历史证据未改动。

## API

官方 Responses API，`gpt-6.2-sol`、`reasoning.effort=max`、`store:false`；只提交脱敏技术描述。五次实际合计 input 694、output 35,758 token（含 reasoning 33,894）；其中第二次达到 9,000 输出上限而 incomplete，没有形成结论。其他四次完成：[首次建议](sol-review.txt)（其中建议改变窗口起点，与用户确定的 0/28/52 不符，未采用）、[针对尺度风险的复核](sol-overlap-review.txt)、[尺度变换与索引复核](sol-geometry-implementation.txt) 和 [流水线复核](sol-pipeline-review.txt)。流水线复核误以为只有两个窗口，实际保留首窗并追加两个；其中动作条件建议不适用于本次 EgoX 视频输入；[未完成调用收据](sol-depth-review.txt)。

## 本机复跑

从项目根目录执行。首个命令读取本机缓存的官方 MiDaS ONNX 权重及 EXO；如已有相对深度缓存会复用。之后重新生成鞋点候选、头轨候选和窗口包。最终轨迹依赖已保存的稀疏修正。

```bash
work=runs/pair04_s0024-a800-20261121/pair02-pair04-100f-local-20261122
py=/usr/local/Caskroom/miniforge/base/bin/python3.12
"$py" -B "$work/depth_audit.py"
"$py" -B "$work/foot_depth_check.py"
"$py" -B "$work/build_head_candidate.py"
"$py" -B "$work/package_inputs.py"
bash -n "$work/prepare_vipe_after_boot.sh"
```

压缩包更新后应同步核对启动脚本中的固定 SHA-256，再进行任何远端操作。
