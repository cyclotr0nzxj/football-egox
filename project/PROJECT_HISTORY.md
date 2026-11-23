# 足球第一视角项目

## 当前第8周成果（唯一最新版本）

- **查看报告**：[9页Harris/SIFT实验报告](output/week08_harris_sift/第8周_Harris与SIFT实验报告.pdf)。图、数据与生成源稿在同目录。
- **上交文件**：[完整提交包ZIP](dist/第8周_Harris与SIFT_提交包.zip)，含5组裁剪后4K配对视频、工作视频、代码、结果、证据与报告。
- **完整本地仓库**：`submission/week08_harris_sift/`，可按其中README独立复现。

`output/`是当前工作成果，`dist/`只保存最新ZIP及校验文件。已清走`dist/week08_harris_sift/`旧解压目录和旧版报告。提交仓库内的报告与output报告共用同一文件，不是另一个版本；ZIP内的报告用于完整上交。

`output/week08/`仅保留仍有用的输入帧、采集证据和预实验结果。下面的官方EgoX基线是另一项已完成工作，不属于第8周自采生成结果。第14周任务尚未启动。

## 官方EgoX阶段A基线

使用官方 Wan2.1-I2V-14B-480P-Diffusers、官方 EgoX LoRA 和足球样例 `cmu_soccer06_6_877_925`。本项目只执行推理，不训练、不微调、不重新渲染先验。

**阶段 A 已完成：首次正式推理成功（19 分 23 秒），49 帧真实结果已备份。见 [实际运行报告](REPORT.md) 与 [状态](STATUS.md)。**

## 在 Mac 查看

在本项目目录执行：

```bash
python3 scripts/serve.py --port 7860
```

打开 <http://127.0.0.1:7860>。成功结果下载回本机后，网页不再依赖云端 GPU。服务只提供 `demo/index.html` 和 `artifacts/` 内文件，不暴露项目的其他目录。

- 点击播放 / 暂停，或按空格。
- 点击上一帧 / 下一帧，或使用方向键。
- 拖动帧滑块，三路始终显示同一编号。
- 用 0.25×、0.5× 检查人、球和遮挡变化。
- 下载三路对照 MP4、生成第一人称 MP4、原始官方输出、逐帧 CSV 和评估记录。

三路使用同一个无损 PNG 帧序列绘制到同一画布。第三人称保留 784×448，GT 与生成分别保留 448×448，不为版式拉伸画面。原始官方输出的左侧是 VAE 解码的第三人称分支；对照的第一路使用原始官方 exo 视频，避免将重建图当成输入。

## 时间与图像平面

实测官方三个源 MP4 均为 **50 帧、20fps**；官方推理在输入端取前 **49 帧**，输出为 **30fps**。对照遵循源帧 0–48 对生成帧 0–48：源文件回放时间 `i/20` 秒，生成回放时间 `i/30` 秒。源末帧 49 不参与推理或评分，原始源文件完整保留。这里是 MP4 的回放时间，未独立验证原始传感器采集时间；对照视频播放时长不能直接解释为真实动作时长。

生成第一人称只取官方 1232×448 输出的右侧 `x=784..1231`；不旋转、缩放、搜索偏移、校色、掩膜或扭曲 GT / 生成画面。图像指标以额外导出编码之前的解码像素计算，包含鱼眼黑边。只有记录了当前输出对应的帧映射和图像平面核查后才允许计算 PSNR / SSIM；核查不足时保留未验证状态。

PSNR / SSIM 是这个官方样例交付图像平面上的描述性误差，不是完整论文成绩，也不能验证原始相机标定。相邻帧 MAE 包含相机和物体运动，不能直接当成闪烁分数。观察结论应结合逐帧画面阅读。

## 条件与 GT 隔离

实际输入包括官方 exo、ego_Prior、原有 prompt、49 份深度、固定第三人称相机内外参、第一人称相机内参及 49 帧外参轨迹。轨迹及几何条件已经提供，不能声称模型仅凭视频自动估计了头部运动。

官方元数据原有 `ego_gt_path`，官方 `infer.py` 不读取该字段；本项目进一步从推理专用 `football_meta.json` 中移除它。记录包装器禁止 Python 层打开 `ego_GT.mp4`，源码审查确认生成路径只读取 exo 和 prior。GT 只在评估与对照导出阶段读取，未用于首帧、提示词或模型选择。

相机内参和官方 GGA 中的硬编码畸变 / 方向处理保持原样；不为当前样例修正或优化这些参数。原始传感器标定和样例处理链中无法独立确认的部分记录在评估文件中。

## 云端目录

所有大文件位于 `/root/autodl-tmp/football_egox/`：

```text
EgoX/                 官方代码与足球样例
models/               官方 Wan 与 EgoX 权重
envs/egox/            独立 Python 3.10 环境
cache/                下载与安装缓存
scripts/              下载校验、受限启动、记录及导出工具
runs/attempt_XX/      每次正式启动的输出、配置、日志和资源采样
records/              官方版本 / 哈希 / 硬件 / 下载与启动记录
artifacts/            网页展示、视频导出及评估结果
demo/                 浏览器页面
```

`runs/attempt_XX/config.json` 的状态、退出码、日志和输出哈希共同证明真实推理；网页能打开和 CPU 测试通过不等于 GPU 推理成功。

## 固定版本与准备环境

- EgoX 源码：`43c095b552a8f09d3a81c0882491f52c5cf4e953`
- Wan HF revision：`b184e23a8a16b20f108f727c902e769e873ffc73`
- EgoX HF revision：`28b19ef9f079bf73c84b2347fe6701e8ab7f6634`
- 官方关键依赖：Diffusers 0.34.0、Transformers 4.49.0、Accelerate 1.5.2、PEFT 0.17.1。
- 完整实际环境以交付的 `records/environment_freeze.txt` 为准。

复用现有环境和完整模型，不重复下载。新环境可先创建 Python 3.10，按完整冻结列表安装；CUDA 驱动不由本项目修改。云端传输受网络影响时可使用镜像，但所有权重必须通过官方 Hugging Face 文件 SHA256，普通文件通过官方 Git blob ID。

容量计划在 `records/download_plan.json`：Wan 约 83.91GiB，EgoX 约 3.13GiB；额外预留环境 / 缓存 25GiB、样例 / 结果 5GiB、临时空间 20GiB。下载器执行前再检查剩余空间，文件使用 `.partial` 断点续传，成功后验证哈希。

```bash
cd /root/autodl-tmp/football_egox
envs/egox/bin/python scripts/download_models.py
envs/egox/bin/python scripts/download_sample.py
```

样例原始源连接失败时，可以在 Mac 使用 `download_sample.py --root /临时目录` 获取并校验这一个样例，然后通过 SSH 传到云端；不要下载其他 takes 或完整 Ego-Exo4D。

## 正式运行与重试限制

当前授权为一次正式推理、日志支持的额外修复重试最多两次。启动器通过文件锁和 `records/attempts.json` 记录每次启动；已有成功结果时拒绝再次启动，也不自动换种子。

```bash
cd /root/autodl-tmp/football_egox
envs/egox/bin/python scripts/run_baseline.py --reason '首次官方阶段 A 基线'
```

实际调用官方 `infer.py`，只通过外层包装记录 LoRA 加载与资源信息。设置为 GPU 0、seed 42、49 帧、50 步、guidance 5.0、LoRA rank 256、GGA 开启、cosine scaling 3.0、不启用 KV cache、不量化。包装器不改变模型输入、输出或采样随机数。

如启动失败，先读对应 `inference.log` 和资源记录，修复必要依赖 / 运行环境问题，再给出明确 `--reason` 启动下一次。不要手动删除计数、绕过成功保护或为好看结果继续抽样。

## 生成对照与评估

本次检查、评估和导出已完成；以下命令用于复现导出，不会启动 GPU 推理。

成功后检查原始输出帧数、图像区域、方向及对应关系，将核查记录保存到 `records/alignment_review.json`，绑定真实输出 SHA256。

```bash
cd /root/autodl-tmp/football_egox
envs/egox/bin/python scripts/build_artifacts.py \
  --alignment-review records/alignment_review.json
envs/egox/bin/python scripts/serve.py --port 7860
```

云端服务只监听 `127.0.0.1`。需要直接查看云端页面时，从 Mac 建立 SSH 隧道（本机 7860 已占用则使用下面的 7861）：

```bash
ssh -N -p 23955 \
  -L 127.0.0.1:7861:127.0.0.1:7860 \
  root@example.invalid
```

打开 <http://127.0.0.1:7861>。当前 Mac DNS 代理导致域名连接失败时，已验证的 SSH 主机地址配置见 `scripts/cloud.sh`；该脚本不保存密码或私钥。地址与会话过期后应重新核对用户提供的实例，不连接其他实例。

## 备份与关机

把 `artifacts/`、`runs/`、`records/` 和本项目代码备份回 Mac 后，就可在本地继续查看，不需要占用 GPU。模型权重保留在云端数据盘，不复制约 87GiB 到 Mac。

**实验结束后，请在 AutoDL 控制台人工关机。** 退出 SSH 或关闭网页不会关机。最终结果与备份位置、已完成的验证和剩余限制会更新到 `STATUS.md`。本项目不会自动租机、扩容、修改计费、执行阶段 B 或新增多种子任务。

## 官方来源

- [EgoX 源码与推理要求](https://github.com/DAVIAN-Robotics/EgoX)
- [固定版本官方推理脚本](https://github.com/DAVIAN-Robotics/EgoX/blob/43c095b552a8f09d3a81c0882491f52c5cf4e953/scripts/infer_ego4d.sh)
- [官方足球样例](https://github.com/DAVIAN-Robotics/EgoX/tree/43c095b552a8f09d3a81c0882491f52c5cf4e953/example/egoexo4D/videos/cmu_soccer06_6_877_925)
- [官方 Wan 模型](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P-Diffusers)
- [官方 EgoX 权重](https://huggingface.co/DAVIAN-Robotics/EgoX)
