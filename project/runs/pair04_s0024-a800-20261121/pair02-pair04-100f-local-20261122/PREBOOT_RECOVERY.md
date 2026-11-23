# Pair02 / Pair04：下一次开机前准备

## 当前状态

008 `cb894da57f-66fc463f` 已关机。上次新增两段 ViPE 因各自独立估计的焦距与首窗不一致，重叠检查未通过，未产生新增 EgoX 成片；失败输出保留在云端数据盘，尚未取回。上次重试脚本未确认上传或执行。

本轮不运行云端。新版 `run_fixed_k_008.sh` SHA-256：

```text
a55525a3d29383ae1e0fb72e9fe97c6b42b350868e25288b7db0452cdec4f8de
```

它在 ViPE 重建时固定到每段首窗的 **EXO 估计**内参；这个值不是实测标定真值，也不来自 Nano。保留旧几何和旧检查文件，新输出为 `geometry-fixed` / `overlap-fixed.json`。固定内参后，脚本检查重叠背景深度比例、留出帧误差、头部深度与投影。每个留出帧的相对误差中位数须在 20% 内，像素误差第 90 百分位须在 40% 内。这是初步一致性门槛，任一窗失败就停止该段，另一段仍可运行。两段共享一张卡，计算串行。旧 `overlap.json` 的存在不会阻止重试；上次 K 异常发生在报告写出前，**不能假定当时一定生成过该文件**。

脚本启动前核验已上传包的完整 SHA-256、包内每个落盘输入及四个复用的首窗资产，并检查 GPU、空间、锁和原环境；固定内参的四个几何任务最多各运行 900 秒。几何通过后调用原有渲染与官方 LoRA 推理入口，收集实际生成的证据。自动的技术检查不能代替人工检查人物、球、先验、接缝和最终视频。Nano 只在成片冻结后用于独立评估。

## 连接和重启时的动作

现在的网络是 Shadowrocket 场景模式：路由读回默认 `utun6`，Wi-Fi 物理接口 `en0`。同一已关机实例的 Jupyter 网关经系统代理三次为 1 次 TLS 超时、2 次 404；绑定 `en0` 的直连三次均收到 404。404 是实例关机时网关的响应，不能证明运行中的终端或 WebSocket 已恢复，也不能确定先前断连的唯一原因。项目历史记载 Jupyter 与 SSH 曾实际成功；当前 SSH agent 无身份，不代表此前 SSH 不可用。

开机后先核对控制台的实例 ID、当前 SSH 端口、物理接口及服务器主机密钥。优先恢复现有已登录 Jupyter；如需 SSH，`connect_008_physical.sh PORT` 以 `en0` 为单条连接的出口、独立 DNS 确认真实 IPv4，并保持严格主机密钥校验；端口是开机后控制台当前值。该脚本不配置或读取密码、私钥，也不更改 Shadowrocket、系统路由或账号公钥。已找到历史 41248 端口的公钥记录；仅当新端口完全相同才尝试复用，服务器实际密钥不一致会停止。没有匹配信任记录时需先从独立可信渠道核对；使用已有认证方式完成登录。现时未对已关机实例完成 SSH 或 Jupyter 的端到端复测，不能保证开机即连通。

确认是目标实例后，优先把 [启动命令](paste_launch_fixed_k_008.sh) 全部粘贴到已认证的远端 Bash 终端。它会复用云端原包，按 SHA-256 校验写入的重试脚本，再后台启动。本机已用无 GPU 的替身任务验证启动、重复提交拒绝和损坏脚本拒绝。如果使用文件上传，将新版 `run_fixed_k_008.sh` 放到同一云端目录后执行下方命令。

```bash
(
set -eu
set -o pipefail
cd /root/autodl-tmp/football_egox/pair04-20261121/egox-shared-008/pair05-vae-66fc463f.FlNTz4tU
if printf '%s  %s\n' a55525a3d29383ae1e0fb72e9fe97c6b42b350868e25288b7db0452cdec4f8de run_fixed_k_008.sh | sha256sum -c -; then
  set -C
  if test ! -e fixed-k-launch.log; then
    exec 3>fixed-k-launch.log
    nohup bash ./run_fixed_k_008.sh </dev/null >&3 2>&1 3>&- &
    echo "FIXED_K_LAUNCH_PID=$!"
  else
    echo '已有启动日志，先检查，拒绝覆盖。' >&2
  fi
fi
)
```

校验或已有日志检查不通过时不会启动。读取 `fixed-k-launch.log`、进程和 `nvidia-smi` 确认任务真实启动。若连接不通、预检失败、被占卡或某项几何检查未通过，不应宣称转换已启动或成功。按实际日志继续处理；先取回证据，再验收并及时关机。

## 本机复核

- 固定 K 参数格式与项目所用 [官方 Renderer 说明](https://github.com/kdh8156/EgoX-EgoPriorRenderer) 一致：JSON 3×3，设置后禁用内参优化；仍须在实际固定源码环境验证导出 K 和尺度。
- `fixed-k-driver-selftest.json` 是模拟调度：旧证据保留、输入损坏拒绝、重复启动拒绝、单个留出帧超限时拦下 Pair02 且 Pair04 继续，均通过。
- `stage-geometry-selftest.json` 使用既有首窗几何检查比例传递、头部深度误差拒绝及投影；`assembly-selftest.json` 使用 EXO 合成画面测试 100 帧、25 fps、4 秒和两段重叠的帧索引，均不是新 EgoX 成片。
- Responses API 使用 `gpt-6.2-sol`、`reasoning.effort=max`、`store:false`，仅发送脱敏技术文字；输入共 502 tokens，输出共 18,582 tokens，含一次达到输出上限的补充请求。其余两次完成。各次实测用量见 `sol-preboot-retry-review.txt`、`sol-preboot-final-review.txt`、`sol-preboot-final-review-2.txt` 的首行。
