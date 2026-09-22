# 价值推理、优势标签与接管对比

在 `value_function` conda 环境中运行。脚本名字按仓库约定保留 `calculate_advantage.py`。所有相对路径按工程根目录解析，不修改原始数据、数据划分或 checkpoint。

## 输入与命令

```bash
python scripts/calculate_advantage.py --output artifacts/advantage/new-test
python scripts/calculate_advantage.py --split all --horizon 1 10 50 --output artifacts/advantage/new-all
python scripts/calculate_advantage.py --split test --max-episodes 2 --output artifacts/advantage/two-episodes
python scripts/calculate_advantage.py --reuse-values artifacts/advantage/new-test --threshold 0.01 --output artifacts/advantage/new-threshold
```

- `--checkpoint`：默认 `checkpoints/optimized/best_model.pt`，需冻结 SigLIP 的分布模型 checkpoint。模型结构、相机和预处理取自其中配置。兼容旧的错误 scalar 名称，但严格检查实际分布权重；真正的旧标量权重不兼容。
- `--split train|val|test|all`：默认 test。val 参与最佳 checkpoint 选择，不等于独立测试。all 会包含训练数据，输出每帧保留 split。
- `--episode 数据集名:编号`：可重复，限定所选 split 内的完整轨迹。例如 `--episode 2026.09.16_error:26` 还需选择该轨迹所在 split，或 `--split all`。
- `--max-episodes N`：按数据加载顺序选前 N 条完整轨迹；不截断帧、不随机抽样。
- `--horizon 1 10 50`：一个或多个正整数，单位是帧；默认 50，不是 50 秒。
- `--threshold`：默认 0，严格大于阈值为 advantage，其余为 disadvantage。阈值是额外标签规则，不替代奖励代价。
- `--output`：新结果目录；默认带时间戳。不覆盖非空目录。
- `--reuse-values`：复用一个已完成且价值文件校验一致的结果目录；保留它的数据选择与 checkpoint 身份，不再读取 `--checkpoint`，不能同时改变 split/episode/max-episodes。修改前瞻长度和阈值无需重复图像推理。

输入还包括：checkpoint 配置指定的 SigLIP 权重、适配 YAML、train/val/test 划分、原始轨迹 parquet/video、回报 sidecar 与合同。检查 split 不相交、训练成员与 checkpoint 一致、编码器权重及预处理文件一致、轨迹帧连续、时间戳、奖励和终止标记一致。缓存按身份校验，缺失时单进程生成以规避本机解码 worker 崩溃。

## 计算定义

对长度 L 的轨迹中第 t 帧，N 为前瞻帧数，S 为回报尺度：

```text
k = min(N, L - t)
R = sum(gamma**i * reward[t+i] for i in range(k))
next_value = V[t+k] if t+k < L else 0
A[t] = R/S + gamma**k * next_value - V[t]
label = advantage if A[t] > threshold else disadvantage
```

当前 gamma=1、S=4000，非终止帧奖励 -1，成功末帧 0，失败末帧 -2000。普通完整 50 帧窗口是 `V[t+50]-V[t]-0.0125`。终点附近缩短窗口，使用真实终止奖励，终止后价值为 0，不越到下一条轨迹；不会重复加失败惩罚。

这是沿已有轨迹计算的多步价值残差。正值表示实际奖励加后续估值高于当前估值，不等于“动作正确”；成功轨迹也可能负，失败轨迹也可能正。使用未来窗口，不能当作在线告警。精确回报标签作为 V 时，所有窗口的残差应接近 0，此性质已加入测试。

## 输出接口

| 文件 | 内容 |
|---|---|
| `values.parquet` | 每帧一行：dataset_id、episode_index、frame_index 唯一键；timestamp（秒）、fps、split、success、reward_raw、intervention（缺失为 NaN）、value |
| `scores.parquet` | 每帧每个 horizon 一行：唯一键与时间、split、success、intervention、value，以及 horizon、effective_horizon、reward_sum_raw（已折扣）、value_next、terminal_reached、advantage_continuous、threshold、label、is_advantage |
| `advantages.parquet` | timestep-level RECAP metadata，`scores.parquet` 的瘦投影：唯一键与时间、split、horizon、threshold、advantage_continuous、advantage（bool）。供训练侧 dataloader 按 (dataset_id, episode_index, frame_index[, horizon]) 查表附 label；与 scores 同行数，不修改原始数据 |
| `episodes.json` | 每轨迹/前瞻长度：帧数、平均价值、平均优势、优势标签比例、接管/非接管帧均值、完整接管窗口数 |
| `interventions.json` | 按前瞻长度记录接管起点前后 ±1 秒，61 个插值点、事件数与事件等权均值；无事件则 mean=null |
| `plots/*.png` | 每轨迹三行折线图：价值、多步优势及阈值、接管标记 |
| `interventions.png` / `index.html` | 接管起点汇总图与可展开逐轨迹报告 |
| `manifest.json` | checkpoint SHA256、配置、划分文件哈希、缓存身份、尺度、阈值、输出文件校验及完成状态 |
| `validation.json` | 帧数、轨迹数、评分行数、advantages 行数、唯一性与有限值检查；model_quality_validated=false |

仅 `manifest.status=complete` 表示完成；异常退出时留下的目录不应当作有效输出。复用结果校验 values.parquet 的哈希。PNG 只是展示，机器处理以 parquet 为准。

接管比较保留缺失状态，不把缺失当作“未接管”。只把 0→1 作为接管开始；轨迹第一帧已经接管，以及不足完整 ±1 秒的起点不进入事件均值。人工接管是描述性参考，不是错误真值；该对比不证明因果或预测能力。

## 验收边界

流程跑通表示数据关联、模型输出、数值公式及导出接口可工作。阈值固定由用户给定，没有在测试集调优。该脚本不加载 π 模型，不训练优势条件策略，不生成动作，也不宣称价值信号已经可靠。换成带语言条件的模型需要扩展模型输入合同。
