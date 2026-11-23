# EgoX 历史转换复查与几何采样对照

先固定原模型权重和公开推理实现。已有实验里，Pair01 的首版朝向丢掉停球，更新版又在生成时混淆身份并增加第二颗球。Pair05 的跟球先验位置基本正确，几何证据进入去噪后仍可能丢失。本轮用本机 CPU 回看旧成片并定位输入链路。Nano 留作独立评估。

## 历史转换

同帧表从取回的原片、正式先验和成片重新解码。每个源文件均检查 49 帧及精确逐帧 PTS，按零起始帧号并排。EXO 播放速度 25 fps，先验和成片 30 fps。[Pair01 同帧表](history/pair01-same-index.jpg) · [Pair05 同帧表](history/pair05-same-index.jpg) · [源文件与哈希](history/source-receipt.json)

| 轮次 | 同帧证据 | 能定位的问题 |
| --- | --- | --- |
| Pair01 首版 | F00 成片能看到白鞋队友和球，后续构图仍围绕队友。F20–F28 在首版外参下的估计球心 0/9 进有效圆。 | 停球段朝向在生成前就缺少球。既有 49 个 EXO 头部锚点都落在红鞋球员上，二维身份核验通过。单目头位深度、视线和真实头戴光心仍有不确定性。[首版机位复核](../pair01-pose-audit-20261122/README.md) |
| Pair01 低头更新和 A/B | 低头机位与首版相机中心相同，朝向夹角中位 87.2°。F00 的正式先验保留白鞋队友，生成混进红鞋特征。F36–F43 的正式先验仅有一个可辨球，A 八帧见双球，B 的 F36–F41 仍见双球，F42 有残片。 | 恒定输入的 A/B 只更换 GGA 反投影，证实修正鱼眼不足以消除身份混淆和双球。佩戴者排除只用于渲染，GGA 可继续收到佩戴者和异时球候选。它们是待检验的生成冲突路径。[A/B 运行及画面](../pair01-gga-ab/output-b-008-20261122/verified/manual-review.md) · [条件入口复算](../local-reassessment-20261122/REVIEW.md) |
| Pair01 004 新先验 | F00 可辨白鞋队友，F20–F28 的草地间隙和人体碎点令球缘与红鞋关系不可辨。 | 此轮在正式先验阶段拒收，未开始生成。[004 先验记录](../clone-004-20261122/pair01/manual-review.md) |
| Pair05 首版与跟球更新 | 首版正式先验只见一位清楚的前方队友，成片成了高位俯视并出现第二个身体形象。跟球更新沿用原相机中心，F12、F24 正式先验保有正前方队友、球、标志盘。更新成片里场边球门成为主体，球到 F36、F48 在偏左处可辨。 | 人体外观和构图都发生失真。更新后的朝向用于几何先验，与实际输入相符。画面偏移已出现在生成的右半幅。[原始输出核对](../update-008/output-008/diagnosis/REVIEW.md) |
| Pair05 Aria 修正与 004 前景 GGA | 镜头公式修正后成片仍偏围网。004 的 4,000 次 GGA 调用反复作用在 F12、F24 同一批 51/10,192 个目标块，两帧成片仍看不清正前方三类目标。F12、F24 先验上半有效圆近黑比例为 96.7%、98.2%。 | 证明局部路径已执行且仍未通过画面验收。真实注意力权重、值向量和各路贡献未记录。25% 增权只有准备证据。旧批次、单段和不同 GPU 环境的画面可用于复查，因果结论需要同一运行条件。[鱼眼版验收](../pair05-aria-008/output-008-20261122/verified/manual-review.md) · [004 验收](../clone-004-20261122/pair05/verified/manual-review.md) · [条件入口](../pair05-left-diagnosis-20261122/REVIEW.md) |

同批次中的 Pair04 更新条带在正式先验已有，生成后对比度升高，沿用首版。Pair02 上缘暗区同样在正式先验和输出中可见，沿用未裁切首版。[Pair04 同帧证据](../update-008/output-008/diagnosis/pair04_s0024-pipeline.jpg) · [Pair02 先验](../multiclip-evidence/pair02_s0000/prior-preview.png)

## 原论文与本机验证

EgoX 先用单路 EXO 深度和给定 EGO 轨迹渲染点云先验，再把 EXO 与先验编码为扩散条件。GGA 通过三维方向给相关源位置加权。作者的野外示例在 Viser 里手动给定期望机位，论文把自动头姿估计列为后续问题。其附录 H.5 也给出了外部视图含歧义而生成动作出错的案例。复现基线可以用仅基于 EXO 的虚拟机位，然后分别报告机位估计和生成保真。[EgoX §3、附录 F.1/F.2/H.5](https://arxiv.org/html/2512.08269) · [官方发布仓库](https://github.com/DAVIAN-Robotics/EgoX) · [官方先验渲染器](https://github.com/kdh8156/EgoX-EgoPriorRenderer)

发布代码采用每四帧取一次的时间格。对 448×784 EXO，点图双线性降到 56×98，对应 (3, 4, 11, 12)² 的 16 个原像素。四个位置换成单位方向后再池化到一个 16×16 源图块的方向。下面核对这个公开实现的**直接源像素位置**，不推断实际注意力。四组附近像素参与了几何方向计算，EXO 视频编码仍可携带其他像素的信息。[推理尺寸与射线](https://github.com/DAVIAN-Robotics/EgoX/blob/43c095b552a8f09d3a81c0882491f52c5cf4e953/infer.py#L195-L226) · [2×2 空间池化](https://github.com/DAVIAN-Robotics/EgoX/blob/43c095b552a8f09d3a81c0882491f52c5cf4e953/core/finetune/models/wan_i2v/custom_transformer.py#L600-L630)

先固定[检验协议](protocol.md)。使用 Pair01、Pair04、Pair05 共 20 个已有可见球锚点，排除本轮修正帧，保持相同 RGB 视频和几何深度赢家。比较每块同样 16 点的官方栅格和等距栅格。球心、半径是历史标注，三个布置来自同一场地，同一批参与者。结果用于检验机制，外部场景和未见对象还需独立验证。

| 场景 | 帧数 | 60% 半径的 EXO 完全漏采，官方／等距 | 深度投影后完全漏采，官方／等距 |
| --- | ---: | ---: | ---: |
| Pair01 | 9 | 0／0 | 0／0 |
| Pair04 | 6 | 0／0 | 0／0 |
| Pair05 | 5 | 0／0 | 2／3 |

全半径两种位置在投影后均无完全漏采。缩到 35% 的压力测试，等距采样在 Pair01、Pair04 减少漏采，在 Pair05 由 1/4 增至 4/4，另有 1 帧连目标深度赢家也没有。单纯铺匀采样点能缩小源图网格的空隙，实际投影后仍取决于剩下的是哪几个源像素。等距替换未通过预设判据。保留官方栅格。[逐帧测量及相位反例](sampling-receipt.json)

另对 Pair05 的真实可见球缘做诊断回放。复划后 F12、F24 在两套假设下都有队友、标志盘和单球的直接投影样本。F24 相同深度赢家中的球标签从 3→71、12→95，官方栅格分别由 0→4、0→6 候选直接样本。使用等距格时备选机位的 F24 仍为 0。RGB 逐像素相同，球缘仍稀疏。Pair01 的 R8 F20–F27 有可见脚球间隔，F28 到球心约 7 像素且轮廓接近重叠，触球标为未知。[Pair05 源像素诊断](pair05-reviewed-sampling.json) · [Pair01 R8 九帧](../occlusion-contact-20261122/pair01/results/r8-shoe-and-ball.jpg)

## 后续验证顺序

沿用官方插值、池化、权重和原始生成设置。用同一套观测记录依次核对原图物体身份与其时间、虚拟视角有效区内的近深度赢家、采样支持和去噪输出。小物体在 EXO 清楚且目标视角仍有真实像素，却在取样后失去支持时，再检验对应是否落在同物体和同一时段。把遮挡与无观测部分留作未知，另按整幅构图和运动连续性判断生成。主要判据同时记录目标位置、缺失与复制，借鉴论文 §4.1 的对象级评估。

先可复查冻结的正式先验。进入模型阶段前，复用现有 Wan VAE 离线往返探针，逐帧核对高置信度人球盘是否在编码解码后仍可辨。若可辨，再固定初始化潜变量和所有相机条件，仅测对应约束是否实际改变去噪分支的权重和值输出。若拿不到独立证据，成片可展示而保真度留待独立实拍评估。二十帧对照、两帧可见支持和历史成片均不足以宣称新方法已改善生成或跨场地泛化。现有 VAE 探针只通过了输入文件检查。[探针及运行范围](../rig-refactor-20261122/probe_wan_prior.py) · [已有文件检查](../rig-refactor-20261122/wan-prior-file-check/verification.json)

从项目根目录复跑。输出路径应为新路径。

```bash
audit=runs/pair04_s0024-a800-20261121/egox-reuse-audit-20261122
python3.12 -B "$audit/audit_source_sampling.py" /tmp/egox-sampling-replay.json
python3.12 -B "$audit/audit_source_sampling.py" /tmp/egox-pair05-replay.json --reviewed-pair05
python3.12 -B "$audit/compare_historical_runs.py" /tmp/egox-history-replay
```

本机使用 Miniforge Python 3.12、NumPy、Pillow、OpenCV 及 ffmpeg/ffprobe，复用项目已存的 EXO、ViPE 深度、虚拟外参及输出。技术设计通过官方 Responses API 的 `gpt-6.2-sol / max` 完成。Pair01 触球和 Pair05 可见来源的边界另由 `gpt-6.2-astra / max` 完成复核。用于最新历史汇总的额外 Astra 请求达到 8,000 输出 token 上限，未返回完整意见。历史结论依据本机原片和已有运行收据。API 请求仅包含脱敏文本。[Sol 用量](sol-api-usage.json) · [前景与触球用量](../occlusion-contact-20261122/api-usage.json) · [历史补审用量](astra-history-api-usage.json)
