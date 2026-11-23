# Pair01 鱼眼反投影对照

只跑 Pair01。A、B 均用红鞋球员低头看球的 49 帧外参和上次正式渲染的最终先验。两次各自启动，固定随机种子 846514、模型、LoRA、EXO、逐帧深度、提示词、50 步和 GGA 其余参数。A 用历史推理源码，B 只替换 GGA 中虚拟相机的反投影。历史失败成片供回顾。API 复核在转换流程外，Nano 留待成片锁定后盲评。

## 008 单独运行 B 的方案与结果

B 已于 13:42:15 生成并取回，用已保存的同一指纹 A 同帧对照。两项画面验收均未通过，修正版仍混淆红鞋球员视角，并且在 F36–F41 清楚生成两颗球。[实际画面验收](output-b-008-20261122/verified/manual-review.md)

A 已完成，并已取回原片和正式收据。A 画面未通过验收，F00 出现站在镜头前的红鞋人物，F20–F28 能看出低头却看不清脚与球的关系，F36–F43 每帧可辨双球。[A 原片的验收记录](output-a-008-20261122/verified/manual-review.md)

开卡前的 `gpt-6.2-astra / max` 脱敏复核消耗输入 252、输出 7,404 token。修正前后的连续回投有效圆 p90 分别为 38.25 和 0.0000124 px。修正方向可能改善前景与几何先验的对齐，佩戴者身份仍可能被生成混淆。若重复出球来自同一个球的源特征和几何提示落在两处，反投影对齐有机会减轻。射线自检不包含栅格化与注意力池化，后续实测保留了身份混淆和双球。[实际运行的射线检查](output-a-008-20261122/logs/projection_receipt.json) · [API 用量](../api_usage.json)

本次在 008 A800 上传 [B 启动脚本](run_pair01_b_after_a_008.sh) 后，从远端 `/root/autodl-tmp/football_egox/pair04-20261121/egox-shared-008/` 启动

```bash
bash pair01-b-launch-tuUmhixo/run_pair01_b_after_a_008.sh pair01-a-launch-dlvm3OfG/pair01-gga-ab-bundle
```

入口复用 A 实际执行的原始包和预检，核对 A 收据、原片及运行指纹，只调度 B。旧包、运行环境或 A 收据校验失败时立即退出。本机用真实 A 收据和原始包模拟通过只启动 B 的分支，改动指纹、收据或打包清单时均拦截。B 产物在 `pair01-gga-ab-results/B/attempt-*`，取回后与 `output-a-008-20261122` 中已保存的 A 原片和收据一起运行下方的本机验收命令。

- **机位** 看 F00 白鞋队友是否在几何先验对应的右缘，红鞋佩戴者是否仍被画到前方。人物消失时记录为无法确认。F20–F28 核对低头下的自身脚和球是否更清楚。
- **双球** 在 F36–F43 记下 8 帧中清楚单球、仍有双球、球不可辨各有几帧。再看 49 帧全片有无双球移到其他帧，并核对单球轨迹是否连贯。

两项分开判定。两项都通过也只支持这段视频在该随机种子下的改善。

## 本机检查

实际 EXO、正式先验和由原 ViPE 压缩包解出的 49 个深度 NPY 均匹配上次推理输入。EXO 为 784×448、25 fps，先验为 448×448、30 fps，逐帧 PTS 分别为 `i/25` 与 `i/30`。两组直接按原始索引对应。替换临时目录中 F36 的深度文件后，检查拒绝使用该输入。

本机鱼眼回投检查中，A 的有效圆 p90 为约 38.25 px，B 小于 0.001 px。该指标只验证射线换算。实际 CUDA、GPU 随机性和生成质量待上机验证。

现成[启动包](pair01-gga-ab-bundle.zip)为 25,714 字节，SHA-256 为 `6dfad9e75be2c7ef44e205caec67399bef9edbde43bbc57b07f60a09624b7837`。重复打包得到相同哈希，解包与包内清单校验通过。启动模拟覆盖预检、B 失败后复用 A、恢复缺失收据和拦截被改动的输入。回收验收用历史原片副本走通，测试成片仅在临时目录。

需要重建本机启动包时，从项目根目录执行以下命令。输出路径需为空，脚本不会覆盖已有包。

```bash
/usr/local/Caskroom/miniforge/base/bin/python3.12 -B \
  runs/pair04_s0024-a800-20261121/pair01-gga-ab/prepare_bundle.py \
  --output /tmp/pair01-gga-ab-bundle.zip
```

## 原 A/B 启动方案存档

使用原 008 A800 80 GB 普通开机。根据控制台显示重新确认 SSH 端点并使用已有主机密钥比对。Clash TUN 开启时按项目规定核实物理接口和真实 IPv4，再为单条 SSH 连接设置 `ProxyCommand`。无需重装依赖、重渲先验或重新下载权重。

通过已认证的 SSH 连接上传本目录的 `pair01-gga-ab-bundle.zip` 至远端 `/root/autodl-tmp/football_egox/pair04-20261121/egox-shared-008/`。校对本机与服务器包 SHA-256 后，在远端执行

```bash
cd /root/autodl-tmp/football_egox/pair04-20261121/egox-shared-008 &&
  unzip -n pair01-gga-ab-bundle.zip &&
  bash pair01-gga-ab-bundle/run_pair01_gga_ab_008.sh preflight
```

预检核对实例、A800、CUDA、共享盘、35 项模型权重的验证记录、现存环境、可用内存和数据盘，并逐一比对 EXO、先验、深度的哈希。通过后在同一目录启动

```bash
bash pair01-gga-ab-bundle/run_pair01_gga_ab_008.sh run
```

两组串行运行，避免单卡装载两套模型。以上次加载与去噪日志估算合计约 80–90 分钟，另外留出预检、取回和验收时间。实际耗时记入运行审计。失败尝试保留日志，复跑前检查失败原因和剩余开机时间。

## 取回与验收

从 `pair01-gga-ab-results` 取回两组已核验原始 MP4、各自的 `verified-result.json` 及运行日志。用本机验收脚本比对远端哈希，检查 49 帧和逐帧 PTS，从原始视频的右侧裁出 448×448 第一视角。

每组路径为 `A/attempt-*` 或 `B/attempt-*`，以运行日志中 `EGOX_ARM_COMPLETE`、`EGOX_ARM_REUSED` 或 `EGOX_ARM_RECOVERED` 对应的目录为准。

```bash
/usr/local/Caskroom/miniforge/base/bin/python3.12 -B \
  runs/pair04_s0024-a800-20261121/pair01-gga-ab/review_pair01_ab.py \
  --raw-a /path/to/A/pair01_s0288.mp4 --receipt-a /path/to/A/verified-result.json \
  --raw-b /path/to/B/pair01_s0288.mp4 --receipt-b /path/to/B/verified-result.json \
  --output-dir /path/to/new-local-review
```

人工核对 F00 的队友位置、F20–F28 是否保持红鞋球员低头看球的机位，以及 F36–F43 每帧的球数。全片检查其他帧是否也出现双球。看不清的帧标记为无法判断，机位无法确认或出现明确双球的成片拒收。若修正后同帧仍有视角失真和双球，记录这次反投影修正不足以解决 Pair01 的成片问题。两组文件取回并核对完整性后，保留失败片供比较，按用户要求提前结束并关闭 008。
