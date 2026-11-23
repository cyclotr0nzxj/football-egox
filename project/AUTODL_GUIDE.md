# AutoDL 租用与 Codex 交接指南

适用：Mac + 能执行本地终端命令的 Codex/Astra → AutoDL 单卡实例。
资料核对：2026-09-11。当前库存、镜像菜单、实际分配和价格以租用页面为准。本指南未代租实例、未安装模型、未完成 GPU 实测。

## 1. 现在只为阶段 A 租机

| 租用项目 | 建议选择 |
|---|---|
| 产品 | 容器实例，按量计费；先不包月 |
| GPU 数量 | 1 张 |
| GPU | 优先 A100 **80GB**；可选有库存的 H100 **80GB** 或其他满足至少 80GB 的 NVIDIA 单卡 |
| 主机内存 | 优先单卡分配约 128GB；不要把整台物理机内存当作实例内存 |
| CPU | 建议至少 8 核，16 核更宽裕 |
| 数据盘 | **总容量 250GB**，或最接近且不低于它的档位；需有可扩容余量 |
| 镜像 | 官方 PyTorch/Miniconda 基础镜像，Linux x86_64，优先 Ubuntu 22.04 |
| Python / CUDA | 优先 Python 3.10、CUDA 12.1+；最终由 Agent 建立匹配官方依赖的独立环境 |

80GB 显存、Python 3.10 和 CUDA 12.1+ 是 EgoX 官方推理条件；CPU、128GB 主机内存与250GB磁盘是工程预算，不是保证所有设置都能运行的实测门槛。[G1]

A100 有不同显存版本，要看清 **80GB**，不要误租40GB。多张小显存卡不能直接替代当前单卡配置。不要为了凑主机内存先多租 GPU，应先找单卡内存配额更合适的主机。AutoDL 的 CPU/内存通常按所租卡数分配。[G2]

B 的官方训练要求更高，先不租；等数据小批处理和完整训练步骤资源检查方案准备好，再单独决定。当前单卡80GB不是已验证的B训练方案。

## 2. 在控制台创建实例

进入 AutoDL 控制台 → 我的实例/容器实例 → **租用新实例**。选择按量计费、地区、GPU型号和1张GPU，再选空闲主机。[G3]

重点核对三个地方：**实际单卡显存、分配给这一张卡的主机内存、硬盘一列的“可扩容”容量**。地区不必先限定，优先满足显存/内存/硬盘、且网络能访问官方软件和模型源的主机。

镜像选“官方基础镜像”中的 PyTorch 或 Miniconda。若有 Python3.10/CUDA12.1+ 的组合优先；没有完全同名的组合，不必去找陌生社区的“一键EgoX镜像”。可选兼容的官方基础镜像，由 Agent 新建Python3.10环境并核对驱动/PyTorch。不要自己重装NVIDIA驱动，也不必在Mac装CUDA。

数据盘配置要分清“总容量”与“额外扩容”：官方通常提供50GB数据盘，因此目标总容量250GB时，若页面填写的是额外容量，应填200GB；若页面填写总量，则填250GB。具体以该实例显示的基础容量为准。创建后也可在实例存储管理中扩容。[G4]

提交前查看GPU与数据盘各自的费用明细。实例运行即开始计费，不是等模型运行才计费。[G3] 不给出固定每小时价格，因为不同主机、地区和会员条件会变化。

## 3. 先配置SSH密钥，之后不需要把密码发给Agent

推荐在运行 Codex 的同一台 Mac 上完成以下操作。专用私钥留在Mac；只将公钥交给AutoDL。

### 3.1 在Mac终端生成专用密钥

先检查是否已有同名文件：

```bash
ls -l ~/.ssh/autodl_egox ~/.ssh/autodl_egox.pub
```

若文件已存在，复用或另选名称，**不要覆盖**。首次创建时执行：

```bash
mkdir -p ~/.ssh
chmod 700 ~/.ssh
ssh-keygen -t rsa -b 4096 -f ~/.ssh/autodl_egox -C "football-egox"
```

建议设置私钥口令。生成后，将私钥加入本机SSH agent，并复制公钥：

```bash
ssh-add ~/.ssh/autodl_egox
pbcopy < ~/.ssh/autodl_egox.pub
```

`ssh-add`如提示找不到SSH agent，先处理本机认证环境，不把私钥口令写入脚本。后续Codex还必须能够访问同一认证环境；本机测试成功不自动保证Agent可用。

### 3.2 在AutoDL粘贴公钥

在 **控制台 → 容器实例 → 设置密钥登录** 入口，粘贴刚复制的完整公钥并保存。平台官方支持这种免密码配置；对已经运行的实例按页面提示生效，必要时在尚无任务运行时重启。不要把无`.pub`后缀的私钥文件上传。[G5]

## 4. 你只做一次连接测试

从实例卡片复制它提供的SSH主机和端口。以下命令中的“实际端口”“实际主机”必须替换；不要照抄文档里的占位字样。

**在Mac终端执行：**

```bash
ssh -i ~/.ssh/autodl_egox -p 实际端口 root@实际主机
```

首次连接先核对主机指纹再接受。需要核验时，可从AutoDL控制台打开该实例JupyterLab终端，查看对应主机密钥的指纹，例如：

```bash
ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub
```

按SSH提示的密钥类型对应检查；没有该文件时不要跳过核验，可查实例使用的其他主机公钥或咨询平台。不要配置 `StrictHostKeyChecking=no`。

进入远端后，执行以下只读检查：

```bash
hostname
nvidia-smi
free -h
df -h / /root/autodl-tmp
```

确认GPU型号/总显存、可用内存和数据盘容量合理，再输入 `exit` 返回Mac。

最后在Mac验证无人值守认证：

```bash
ssh -o BatchMode=yes -o ConnectTimeout=15 \
  -i ~/.ssh/autodl_egox -p 实际端口 root@实际主机 \
  'echo SSH_OK; hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader'
```

能看到 `SSH_OK` 和GPU信息，且没有要求输入密码，才算本机非交互认证通过。失败时先处理密钥、实例状态或网络，不通过把密码发给Codex来绕过问题。

## 5. 把项目和连接信息交给Codex

把本包的 `PROJECT.md` 和 `AGENTS.md` 放到本地项目工作区。已有旧原型可放在同一工作区供审查；不需要先把模型和视频下载到Mac。已有AGENTS.md应合并有效规则，不盲目覆盖。

在目标模式粘贴 `PHASE_A_GOAL_PROMPT.md` 的正文，并附上其中的SSH交接信息。私钥只给**路径**，不给内容。默认远端目录：

```text
/root/autodl-tmp/football_egox
```

剩下的实际目录布局由Agent决定。它应在云端建立环境、下载模型及单个官方样例，然后做真实推理、三路对照和评估。你不必先手工装EgoX、ViPE、CUDA或全部Python包。

**重要：SSH命令是地址，不是已经连接的证明。** 推荐使用可以执行本机终端的Codex。你对话中的Agent若运行在隔离云环境，它不会自动拥有Mac上的私钥或本机网络；需它从实际运行位置测试访问，并按客户端提供的权限提示授权。Codex的终端网络和文件访问受沙箱/审批控制，不推荐为图省事关闭全部保护。[G6]

无法直接SSH时，Agent应明确阻塞并交付可在AutoDL终端执行的脚本；不要假称已部署。凭据或许可要求只能由你完成，Agent不能代签。

## 6. 下载、磁盘与费用怎么控制

阶段A只要该足球样例和必要模型，不需要全量数据集。Agent下载前应查询文件清单及大小、估计环境和缓存占用；不因文件夹已存在就认为下载完整。

模型、环境、下载缓存、临时文件、结果都尽量放 `/root/autodl-tmp/football_egox/`。这是数据盘，不是系统盘。不要在未检查容量时把大文件默认写入 `/root/.cache` 或系统Conda环境。[G2]

关键代码、运行记录、生成与对照视频应另备份。可使用已初始化并确认挂载的 `/root/autodl-fs`，或将少量结果拷回Mac。`/root/autodl-tmp` 关机通常保留，但实例释放、平台清理或本地磁盘故障仍可能导致丢失；保存系统镜像也不包含数据盘。[G2][G4][G7]

GPU按量实例不用时要在控制台**关机**。关闭浏览器、退出SSH、结束Python不等于关机；付费扩容数据盘停机后仍计费。先备份，长期不用再决定缩容或释放。具体费用和保留期限以控制台当前规则为准。[G3][G4]

当前目标默认允许一次正式推理、失败后最多两次有依据的修复重试。完成后Agent不继续抽样，提醒你关机；它不得自动扩容、买新机或无限重试。

## 7. 阶段B的数据：可申请，但暂不全量下载

A的公开样例位于EgoX仓库，目录可确认含 `exo.mp4`、`ego_GT.mp4`、`ego_Prior.mp4`；深度与元数据也应在下载后校验。A不依赖先下载完整受许可数据集，公开样例仍应遵守上游适用条款。[G8]

B需从Ego-Exo4D申请原始数据。流程是：访问官方申请入口，阅读并接受许可，等待获批邮件，按邮件配置下载凭据。可按个人身份申请；无权代表学校时不要代学校签署。官方说明数据使用不收费，重分发等行为有限制；云GPU、磁盘与可能的网络费用另算。[G9]

申请入口：`https://ego4d.dev/request/ego-exo4d`

官方目前说明下载凭据有14天有效期、可重新申请续期；以实际收到的邮件为准。凭据由你在云端交互式配置，不放入项目或聊天。A可以先做，B凭据没有下来不应阻塞A。[G9]

拿到许可、并由Agent安装官方CLI后，可由你执行交互式凭据配置：

```bash
aws configure --profile egoexo
```

先让Agent只下载元数据：

```bash
egoexo -o /root/autodl-tmp/football_egox/data/egoexo4d \
  --parts metadata --s3_profile egoexo
```

后续从元数据选出足球take的真实UID，再按UID和视角小批下载。例如下面的UID占位符必须先替换：

```bash
egoexo -o /root/autodl-tmp/football_egox/data/egoexo4d \
  --parts downscaled_takes/448 --views ego exo \
  --uids 实际take_UID --s3_profile egoexo
```

这是候选视频获取，不是完整训练数据处理。校准、轨迹、深度和先验要另行核验和补齐；视频缩放也要与相机参数对应。所有命令执行前以当前 `egoexo --help` 确认。[G10]

**不要执行不带 `--parts` 的默认全量下载，也不要直接选 `--parts all`。** 官方默认集合是多TB级，远超阶段A磁盘预算。B先列许可、UID、文件大小、可配对情况和数据划分，再批准扩展；未核实到作者完整预处理训练集的直接下载，不承诺原数据下载后即可微调。[G1][G10]

## 官方资料

[G1] EgoX环境与模型：`https://github.com/DAVIAN-Robotics/EgoX`

[G2] AutoDL资源分配与路径：`https://www.autodl.com/docs/env/`

[G3] AutoDL创建实例：`https://www.autodl.com/docs/quick_start/`（本次可读官方备用：`https://backup.autodl.com/docs/quick_start/`）

[G4] 数据盘扩缩容与计费：`https://www.autodl.com/docs/local_disk/`

[G5] SSH与密钥登录：`https://www.autodl.com/docs/ssh/`

[G6] Codex终端权限：`https://learn.chatgpt.com/docs/agent-approvals-security`

[G7] 系统镜像范围：`https://backup.autodl.com/docs/image/`

[G8] 官方足球样例：`https://github.com/DAVIAN-Robotics/EgoX/tree/main/example/egoexo4D/videos/cmu_soccer06_6_877_925`

[G9] Ego-Exo4D许可与凭据：`https://docs.ego-exo4d-data.org/getting-started/`

[G10] 下载筛选：`https://docs.ego-exo4d-data.org/download/`

[G11] 浏览器SSH隧道：`https://www.autodl.com/docs/ssh_proxy/`

## 软件运行后，Mac怎样看网页

若Agent按约定在云端 `127.0.0.1:7860` 启动服务，你在Mac另开终端执行：

```bash
ssh -N -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:7860:127.0.0.1:7860 \
  -i ~/.ssh/autodl_egox -p 实际端口 root@实际主机
```

该终端无输出且保持连接通常是正常的。Mac浏览器打开 `http://127.0.0.1:7860`。若本机端口被占用，让Agent给出另一端口的对应命令；不要为此直接把服务暴露到公网。[G11]
