# Pair01 首版朝向修正候选

首版成片开头的白鞋队友和来球仍可辨认，中后段的球已出画。建议保留首版的红鞋球员头部位置和开场视线，随后顺着来球转头、低头，把画面重点移到脚前。停球时允许白鞋队友暂时离开画面。[首版成片](../four-egox-outputs-008/verified/pair01_s0288_ego.mp4) · [五帧对照](results/orientation-comparison.jpg)

| 朝向 | 估计球心在有效圆内 | 估计球心与队友躯干中心都在有效圆内 | F20–F28 估计球心在有效圆内 |
| --- | ---: | ---: | ---: |
| 首版 | 8/49 | 8/49 | 0/9 |
| 首版只压低 30° | 14/49 | 14/49 | 0/9 |
| 之前的持续低头方案 | 49/49 | 2/49 | 9/9 |
| 新候选 | 49/49 | 21/49 | 9/9 |

统计基于 EXO 标注、ViPE 头轨、场地平面和 Aria 鱼眼前向投影。F20–F28 的两目标中心相隔约 95°–124°。新候选将停球阶段的画面优先给球，后段能让队友再回到有效圆内。球心在圈内只表示朝向覆盖了预计位置。遮挡、球的形状及球与鞋的接触还要看成片。[逐帧检查](results/verification.json)

新候选完整保留 F00–F03 的外参。之后朝球转动，并把相邻帧的视线变化限制在 6°。49 帧头部位置、原镜头参数、EXO 路径和提示词均保持首版值。对照图右栏是把实际 EXO 像素用 ViPE 深度投到候选机位的 CPU 预览，并沿用既有上半身遮罩便于看脚前。黄色圈表示估计球心，没有补画球或草地。Nano 未参与。[候选外参](results/camera-candidate.json)

复跑时从本目录执行 `/usr/local/Caskroom/miniforge/base/bin/python3.12 -B review_orientation.py /tmp/pair01-first-orientation-review`，输出目录应为新目录。

如需生成成片，先用候选外参重渲染先验，验证 F20–F28 的单球和红鞋触球证据，再复用首版运行参数做 Pair01 单段。首版是四段批次的第二段，单段即便使用相同种子也无法保证进程随机状态相同。拿新成片与首版比较构图即可，精确量化朝向本身的作用还需要同条件单段对照。本轮未开卡或生成成片。[首版运行参数](../run_pair04_egox_008.sh#L174)

## 008 开卡启动包

[启动包](pair01-first-orientation-bundle.zip)与[本机预检收据](local-preflight.json)已保存。已核对 49 个原 EXO 深度、7 项渲染缓存、逐帧输入时间戳、候选外参与开场相机，以及官方 EgoX 的源码和审计入口。本机用历史先验作夹具，验证了媒体暂存、改包拒绝、无视觉验收或先验指纹错误时拒绝继续、以及防止同一次尝试重复起推。用历史 A 作夹具走通成片取回后的验收。实际的新先验与新成片尚未生成。

使用原 008 普通开机，按项目指令核验 SSH 端点与主机密钥。把 ZIP 上传到原 `egox-shared-008`，核对远端 SHA-256 与本机收据一致后执行以下命令。该命令自动预检并渲染先验，停在画面验收处。

```bash
cd /root/autodl-tmp/football_egox/pair04-20261121/egox-shared-008 &&
  unzip -n pair01-first-orientation-bundle.zip &&
  ./egox-env/bin/python -B pair01-first-orientation-bundle/run_candidate_008.py render
```

在 `PAIR01_PRIOR_REVIEW_REQUIRED` 报出的尝试目录取回 `prior-render.json`、`prior-review.png`、`render.log` 和 `prior/pair01_s0288/ego_Prior.mp4`。使用原版 Renderer 的场景处理，只更换朝向。之前 CPU 图右侧沿用了上半身遮罩，正式纯朝向试验沿用首版的前景处理。先验经完整解码、49 帧和逐帧 `i/30` 检测。验收图由 EgoX 使用的同版本解码器读取，预检确认逐帧和空间预处理保留像素位置与顺序。[EgoX 读取路径](../../../vendor/EgoX/core/inference/wan.py) · [Diffusers v0.34.0 的 Wan 处理器](https://github.com/huggingface/diffusers/blob/v0.34.0/src/diffusers/pipelines/wan/pipeline_wan_i2v.py#L162-L164)

看图及需要时逐帧看先验视频。F00–F04 检查镜头前仍是白鞋队友。F20–F28 确认脚前可见同一颗球，能辨认红鞋和实际触球关系。F36–F43 确认单球可辨、连续且无重复。无法判断时记录结果并停在先验阶段。通过后在同一尝试目录写 `prior-visual-review.json`，填写实际观察。

```json
{
  "prior_sha256": "复制 prior-render.json 里的实际 prior_sha256",
  "opening_white_shoe_teammate": true,
  "foot_ball_contact_visible": true,
  "single_visible_ball_F36_F43": true,
  "observations": "记录实际的帧号、鞋和球的位置"
}
```

预检和视觉验收都通过后，再启动一次官方 EgoX 单段。`attempt` 用 `PAIR01_PRIOR_REVIEW_REQUIRED` 输出的绝对路径。

```bash
./egox-env/bin/python -B pair01-first-orientation-bundle/run_candidate_008.py infer \
  --attempt "$attempt" --visual-review "$attempt/prior-visual-review.json"
```

从该尝试取回候选原片 `pair01_s0288.mp4`、`verified-result.json`、先验及日志。在本目录执行下面的本机验收。`run_sha` 取 `inference-start.json` 的 `fingerprint`，`prior_sha` 取 `prior-render.json` 的 `prior_sha256`。

```bash
/usr/local/Caskroom/miniforge/base/bin/python3.12 -B review_candidate_video.py \
  --raw "$recovered/pair01_s0288.mp4" --receipt "$recovered/verified-result.json" \
  --prior "$recovered/prior/pair01_s0288/ego_Prior.mp4" --output-dir "$review_out" \
  --expected-fingerprint "$run_sha" --expected-prior-sha256 "$prior_sha"
```

本机脚本确认原片 SHA、49 帧逐帧 PTS、完整解码和右侧 448×448 导出，生成 EXO、候选先验、首版和候选四列对照图。全片检查连贯性、错人和重复球。新片画面通过才替换首版。已授权运行结束并取回结果后关闭 008。

官方 Responses API 的 `gpt-6.2-astra / max` 已完成复核，输入 172、输出 3,785 token。复核建议先检查生成器实际读入的先验帧，再把画面通过与本次固定条件相绑定。单次通过只能说明该次产物达标，无法保证可重复改善。[脱敏 API 用量](../api_usage.json)
