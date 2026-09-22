# 价值推理、优势标签与接管对比

在 `value_function` conda 环境中运行。脚本名字按仓库约定保留 `calculate_advantage.py`。所有相对路径按工程根目录解析。除显式使用 `--write-back` 外，不修改原始数据、数据划分或 checkpoint。

## 输入与命令

```bash
python scripts/calculate_advantage.py --output artifacts/advantage/new-test
python scripts/calculate_advantage.py --split all --horizon 1 10 50 --output artifacts/advantage/new-all
python scripts/calculate_advantage.py --split test --max-episodes 2 --output artifacts/advantage/two-episodes
python scripts/calculate_advantage.py --reuse-values artifacts/advantage/new-test --threshold 0.01 --output artifacts/advantage/new-threshold

# 对齐 RECAP/RLinf：train 拟合 top 30% 正例阈值 + 接管段强制 positive；应用到全部
python scripts/calculate_advantage.py --split all --label-rule percentile --percentile 30 \
  --reference-split train --output artifacts/advantage/recap-all

# 把 advantage 列直接写进帧文件（副本），下游无需再 join
python scripts/calculate_advantage.py --split all --label-rule percentile \
  --write-labeled-frames --output artifacts/advantage/recap-labeled
```

- `--checkpoint`：默认 `checkpoints/optimized/best_model.pt`，需冻结 SigLIP 的分布模型 checkpoint。模型结构、相机和预处理取自其中配置。兼容旧的错误 scalar 名称，但严格检查实际分布权重；真正的旧标量权重不兼容。
- `--split train|val|test|all`：默认 test。val 参与最佳 checkpoint 选择，不等于独立测试。all 会包含训练数据，输出每帧保留 split。
- `--episode 数据集名:编号`：可重复，限定所选 split 内的完整轨迹。例如 `--episode 2026.09.16_error:26` 还需选择该轨迹所在 split，或 `--split all`。
- `--max-episodes N`：按数据加载顺序选前 N 条完整轨迹；不截断帧、不随机抽样。
- `--horizon 1 10 50`：一个或多个正整数，单位是帧；默认 50，不是 50 秒。
- `--threshold`：`--label-rule fixed` 下的固定阈值，默认 0。严格大于阈值为 advantage，其余为 disadvantage。阈值是额外标签规则，不替代奖励代价。
- `--label-rule fixed|percentile`：默认 fixed。percentile 使用 RLinf RECAP 的 top-fraction 规则，见下节标签规则。
- `--percentile`：默认 30，`--label-rule percentile` 下正例比例的百分数（top 30% 为正，阈值取优势分数第 70 百分位）。
- `--reference-split train|val|test`：默认 train，用于拟合分位阈值的 split，必须包含在本次评分范围内；不参与拟合的 split 只是应用已得到的阈值。
- `--force-intervention-positive / --no-force-intervention-positive`：默认开启。开启时接管帧强制标为 advantage，对齐 RECAP 对人工纠正的处理。
- `--write-labeled-frames`：把 advantage 列直接追加到每条已评分轨迹的帧文件里，写出到 `--labeled-dir`（默认 `<output>/labeled_frames`），目录结构镜像原数据集。下游读一个文件即可，不需要再 join。
- `--write-back`：隐含 `--write-labeled-frames`，直接覆盖原始帧文件；首次覆盖前会生成一次 `<文件>.bak` 备份。
- `--reuse-values`：复用一个已完成且价值文件校验一致的结果目录；保留它的数据选择与 checkpoint 身份，不再读取 `--checkpoint`，不能同时改变 split/episode/max-episodes。修改前瞻长度和阈值无需重复图像推理。`--write-labeled-frames` 需要该目录里存在 `frame_files.json`；框架会拒绝从未写入该文件的旧结果，按提示去掉 `--reuse-values` 重新评分即可。
- `--output`：新结果目录；默认带时间戳。不覆盖非空目录。

输入还包括：checkpoint 配置指定的 SigLIP 权重、适配 YAML、train/val/test 划分、原始轨迹 parquet/video、回报 sidecar 与合同。检查 split 不相交、训练成员与 checkpoint 一致、编码器权重及预处理文件一致、轨迹帧连续、时间戳、奖励和终止标记一致。缓存按身份校验，缺失时单进程生成以规避本机解码 worker 崩溃。

## 标签规则（对齐 RECAP / RLinf）

二值化指示量 `I_t` 决定 advantage / disadvantage 标签，与 RLinf RECAP 管线一致：

```text
固定阈值（--label-rule fixed，默认，向下兼容）
I[t] = A[t] > threshold                      # threshold 默认 0；严格 >，等于阈值归 disadvantage

分位阈值（--label-rule percentile，对齐 RLinf positive_quantile）
threshold = percentile(参考 split 的 advantage_continuous, 100 - --percentile)
                                             # --percentile 30 → 第 70 百分位，top 30% 为正
I[t] = A[t] >= threshold                     # RECAP 约定用 >=

接管覆盖（--force-intervention-positive，默认开启）
若 intervention[t] == 1：I[t] = True         # 人工纠正动作一律视为改进
```

- 阈值从**优势分数 `advantage_continuous` 自身**拟合，不是价值预测值 V（V 与 A 尺度不同，用 V 分位会得到退化的全正标签）。与 RLinf `quantile_threshold` / `positive_quantile=0.3` 同一口径：top 30% 为正。
- `--percentile` 是**正例比例的百分数**（默认 30 表示 top 30%），必须在 `(0, 100)` 开区间。
- `--reference-split`（默认 train）必须包含在本次评分范围内；只拟合阈值，不参与拟合的 split 只是应用该阈值。同一阈值对全部 horizon 生效。
- 接管覆盖只对 `intervention == 1` 的帧生效；缺失（NaN）不视为接管。
- percentile 模式用 `>=`（RECAP）；fixed 模式保持严格 `>`。被接管强制置正的帧由 `advantage_forced` 列标记。

## 计算定义

对长度 L 的轨迹中第 t 帧，N 为前瞻帧数，S 为回报尺度：

```text
k = min(N, L - t)
R = sum(gamma**i * reward[t+i] for i in range(k))
next_value = V[t+k] if t+k < L else 0
A[t] = R/S + gamma**k * next_value - V[t]
I[t] = 标签规则(A[t], threshold, intervention[t])     # 见上节「标签规则」
```

当前 gamma=1、S=4000，非终止帧奖励 -1，成功末帧 0，失败末帧 -2000。普通完整 50 帧窗口是 `V[t+50]-V[t]-0.0125`。终点附近缩短窗口，使用真实终止奖励，终止后价值为 0，不越到下一条轨迹；不会重复加失败惩罚。

这是沿已有轨迹计算的多步价值残差。正值表示实际奖励加后续估值高于当前估值，不等于“动作正确”；成功轨迹也可能负，失败轨迹也可能正。使用未来窗口，不能当作在线告警。精确回报标签作为 V 时，所有窗口的残差应接近 0，此性质已加入测试。

## 输出接口

| 文件 | 内容 |
|---|---|
| `values.parquet` | 每帧一行：dataset_id、episode_index、frame_index 唯一键；timestamp（秒）、fps、split、success、reward_raw、intervention（缺失为 NaN）、value |
| `scores.parquet` | 每帧每个 horizon 一行：唯一键与时间、split、success、intervention、value，以及 horizon、effective_horizon、reward_sum_raw（已折扣）、value_next、terminal_reached、advantage_continuous、threshold、label、is_advantage、advantage_forced |
| `advantages.parquet` | timestep-level RECAP metadata，`scores.parquet` 的瘦投影：唯一键与时间、split、horizon、threshold、advantage_continuous、advantage（bool）、advantage_forced（bool）。供训练侧 dataloader 按 (dataset_id, episode_index, frame_index[, horizon]) 查表附 label；与 scores 同行数，不修改原始数据 |
| `frame_files.json` | 每条已评分轨迹的原始帧文件路径与所属数据集根目录；`--reuse-values` 下重写 labeled frames 需要它 |
| `labeled_frames/` | 仅 `--write-labeled-frames` 时生成。每条轨迹一份帧文件，结构与原数据集一致，末尾追加 `advantage`、`advantage_positive` 列（多 horizon 时带 `_<N>` 后缀；`advantage` 为 float32 连续值，`advantage_positive` 为 bool 二值标签）。标签与 `intervention` 同处一个文件，下游无需 join |
| `labeled_frames.json` | 实际写出的文件清单：dataset_id、episode_index、帧数、目标路径、备份路径 |
| `episodes.json` | 每轨迹/前瞻长度：帧数、平均价值、平均优势、优势标签比例、接管强制置正帧数、接管/非接管帧均值、完整接管窗口数 |
| `interventions.json` | 按前瞻长度记录接管起点前后 ±1 秒，61 个插值点、事件数与事件等权均值；无事件则 mean=null |
| `plots/*.png` | 每轨迹三行折线图：价值、多步优势及阈值、接管标记 |
| `interventions.png` / `index.html` | 接管起点汇总图与可展开逐轨迹报告 |
| `manifest.json` | checkpoint SHA256、配置、划分文件哈希、缓存身份、尺度、阈值、标签规则（`label_rule`、`percentile`、`positive_fraction`、`reference_split`、`label_inclusive`、`force_intervention_positive`）、输出文件校验及完成状态 |
| `validation.json` | 帧数、轨迹数、评分行数、advantages 行数、labeled frames 文件数、唯一性与有限值检查；model_quality_validated=false |

仅 `manifest.status=complete` 表示完成；异常退出时留下的目录不应当作有效输出。复用结果校验 values.parquet 的哈希。PNG 只是展示，机器处理以 parquet 为准。

接管比较保留缺失状态，不把缺失当作“未接管”。只把 0→1 作为接管开始；轨迹第一帧已经接管，以及不足完整 ±1 秒的起点不进入事件均值。人工接管是描述性参考，不是错误真值；该对比不证明因果或预测能力。注意这一描述性对比与标签规则里的接管覆盖是两件事：前者统计接管前后的平均优势，后者无条件把接管帧标为 advantage，二者都不代表接管一定更好。

## 验收边界

流程跑通表示数据关联、模型输出、数值公式及导出接口可工作。`--reference-split` 拟合的分位阈值只允许来自非测试 split，默认 train，没有在测试集调优。该脚本不加载 π 模型，不训练优势条件策略，不生成动作，也不宣称价值信号已经可靠。写成标签列只是导出方式的简化：`advantage` 是模型多步优势按阈值规则二值化的结果，不是动作正确性的真值。`--write-back` 会原地改写原始数据文件，从而改变其特征缓存身份并触发重新编码，非必要请用默认的副本方式。换成带语言条件的模型需要扩展模型输入合同。
