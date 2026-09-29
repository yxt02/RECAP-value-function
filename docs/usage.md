# 运行与输入输出接口

本文按“准备数据 → 训练价值模型 → 测试 → 导出优势”的顺序说明当前本地流程。命令从仓库根目录执行，使用已有 `value_function` conda 环境；原始数据与 SigLIP2、Gemma3 权重不随 Git 提交。相对路径在脚本中按仓库根目录解析。完整可选参数可运行对应脚本的 `--help` 查看。

## 1. 检查和适配数据

`config/z02_data.yaml` 指定 `data/raw/` 的四批数据、相机 `cam2`、22 维关节约定、结果覆盖及 train/val/test 划分。只读检查先运行：

```bash
python scripts/prepare_data.py --analyze
python scripts/prepare_data.py --verify-videos  # 如需解码检查视频
```

需要重新生成回报和划分时运行：

```bash
python scripts/prepare_data.py --all
# 也可单独运行 --compute_returns 或 --split_data
```

每个原始数据集应有帧 parquet、`meta/info.json`、`meta/tasks.jsonl` 及配置相机的视频。`--compute_returns` 把逐帧 `return`、`reward`、`prompt` 写入 `meta/returns_<tag>.parquet`，并写对应的 JSON 合同、结果与适配记录；`--split_data` 写 `data/splits/{train,val,test}.json` 等清单。跨数据集识别一帧必须同时使用数据集名、`episode_index` 和 `frame_index`。适配不会更改原始帧 parquet 或视频；它会更新相关 `meta/` 文件。当前 `2026.09.16_error` 的 100 条轨迹按配置全部标为失败，包括原始终帧奖励被误标的 episode 26。

## 2. 训练价值模型

`config/train_value.yaml` 是主流程配置，指定 SigLIP2、Gemma3 和任务分词器。SigLIP2 的每个图像块分别投影，与任务文字嵌入组成输入序列；小型 Gemma 价值专家通过可学习的 `CLS` 标记读取序列，预测 201 档价值分布及其连续期望。每路相机的有效位会进入前缀注意力掩码。默认使用 `gemma_1m` 专家，在线训练 SigLIP2、Gemma3 和价值专家；三部分均可单独冻结并设置学习率。`critic_expert_variant` 还支持 `gemma_50m`、`gemma_100m`、`gemma_150m`、`gemma_300m` 和 `gemma_2b`，更大专家需要另行验证显存。数据不输入关节状态或动作。训练集和验证集分别由清单限定，选择最低验证交叉熵（CE）的 checkpoint。方法上的前后差异见 [README](../README.md#方法变化改动前与改动后)。

```bash
python scripts/train.py --smoke_test      # 小样本、独立冒烟产物
python scripts/train.py                   # 完整训练
```

常用覆盖项为 `--config`、`--save_dir`、`--num_epochs`、`--max_total_steps`、`--max_steps`、`--val_steps`、`--max_samples`、`--vision_lr`、`--gemma_lr`、`--expert_lr`、`--gradient_accumulation_steps`，以及 `--freeze_vision_encoder`／`--no-freeze_vision_encoder`、`--freeze_vlm`／`--no-freeze_vlm`、`--freeze_value_expert`／`--no-freeze_value_expert`。默认使用 BF16、梯度检查点、批大小 4、累积 4 个微批次、最多 8 轮；每轮遍历完整划分，不设总优化步数上限。`max_total_steps` 计优化更新次数，`max_steps` 计每轮微批次数。当前模型在线读取图像，不能使用旧平均特征缓存；`--prepare-cache` 仅适用于旧架构。训练写入 `best_model.pt` 和逐轮 `metrics.json`；默认在 `checkpoints/recap_patch/`。正式训练会更新该目录的 checkpoint，独立实验可指定新的 `--save_dir`。`run_train.sh` 顺序执行数据准备和训练，支持 `--smoke_test`、`--dry-run`。

`checkpoints/optimized/best_model.pt` 是历史平均特征基线；它的测试结果见 [旧版结果报告](results.md)，不能当作当前新模型的结果。

## 3. 独立测试与可视化

```bash
bash run_test.sh --dry-run
bash run_test.sh
```

一键脚本依次运行 `submodules/check_cache.py`、`submodules/evaluate.py`、`submodules/render.py`；`--checkpoint` 会传给检查和评估。当前模型的检查读取少量真实图像和任务文字，验证 checkpoint 重载与推理确定性；旧模型则比较缓存与在线编码。结果可通过 `--output` 指定 JSON 路径。`evaluate` 对**完整 test** 预测，用 train 拟合简单对照，输出 `predictions.npz`、轨迹清单、`metrics.json` 和 `protocol.json`；`render` 由这些文件生成逐轨迹图、HTML 和总览，无需再次运行模型。评估需要原 CUDA/BF16 环境。单独调用时：

```bash
python submodules/evaluate.py --checkpoint checkpoints/recap_patch/best_model.pt \
  --output artifacts/evaluation/my-run
python submodules/render.py artifacts/evaluation/my-run
```

报告保存在 `artifacts/evaluation/`，不随 Git 推送。新架构尚无正式训练结果；[旧版结果报告](results.md)只适用于旧 checkpoint。`submodules/benchmark.py --mode loader|head|gpu` 是历史基线的性能测量工具，其中 `head` 需要旧特征缓存，`gpu` 需要 `submodules/reference/` 中的旧实现快照。

## 4. 计算优势和标签

`calculate_advantage.py` 默认用完整 test、50 帧前瞻和固定阈值 0。对每条长度为 L 的轨迹，在 t 帧计算：

```text
k = min(N, L - t)
R = sum(gamma**i * reward[t+i] for i in range(k))
A[t] = R / scale + gamma**k * (V[t+k] if t+k < L else 0) - V[t]
```

当前 `gamma=1`、`scale=4000`。普通 50 帧非终止窗口为 `V[t+50]-V[t]-0.0125`；末尾按实际剩余帧数和终止奖励计算，不跨轨迹。前瞻长度 N 是帧数，不是秒数，也不能直接解释为动作块长度。

```bash
# 只对未用于训练的 test 评分
python scripts/calculate_advantage.py --split test --output artifacts/advantage/test-run

# 对全部 split 评分，并用 train 的优势分数取前 30% 作为正例阈值
python scripts/calculate_advantage.py --split all --label-rule percentile \
  --percentile 30 --reference-split train --output artifacts/advantage/all-run

# 复用已完成结果中的价值预测，重新计算窗口和阈值，无需重新解码视频
python scripts/calculate_advantage.py --reuse-values artifacts/advantage/test-run \
  --horizon 1 10 50 --threshold 0.01 --output artifacts/advantage/comparison
```

| 参数 | 作用 |
|---|---|
| `--checkpoint` | 默认 `checkpoints/recap_patch/best_model.pt`；推理采用其中保存的相机、模型参数、任务文字和预处理约定。可显式指定旧 checkpoint |
| `--split train|val|test|all` | 默认 test；`all` 包含训练内评分，不能把它当独立测试 |
| `--episode 数据集名:编号`、`--max-episodes N` | 选择完整轨迹，不截断帧；前者可重复 |
| `--horizon N [N...]` | 一个或多个正整数，默认 50 |
| `--label-rule fixed`、`--threshold X` | 默认规则：`A > X` 为正例，X 默认 0；等于阈值为负例 |
| `--label-rule percentile`、`--percentile P` | 在参考 split 的**优势分数**上取第 `100-P` 百分位；`A >= 阈值` 为正例。默认 P=30 |
| `--reference-split train|val|test` | 分位模式用于拟合阈值；默认 train，必须包含在此次评分的 split 中。推荐用 train，不用 test 调阈值 |
| `--force-intervention-positive` / `--no-force-intervention-positive` | 默认开启，把 `intervention=1` 的帧强制置正；`advantage_forced` 记录来源。缺失接管标记不视作 0 |
| `--reuse-values DIR` | 读取已完成结果的 `values.parquet`，沿用其原始选择；不能同时另选 split/episode/max-episodes。仅重新计算优势、标签和图 |
| `--write-labeled-frames`、`--labeled-dir DIR` | 在副本帧文件追加 `advantage` 连续值和 `advantage_positive` 布尔值；副本不含视频，不能单独当作完整机器人数据集 |
| `--write-back` | 原地覆盖原始帧文件并首次创建 `.bak`；改变缓存身份，后续会重新编码。仅在明确需要改原数据时使用 |
| `--output DIR` | 输出新目录；拒绝覆盖非空目录 |

分位阈值和接管置正规则是离线标签规则，正例并非“动作正确”的真值；人工接管前后曲线也只是描述性比较。纯遥操作示范数据目前**没有**作为独立类型强制标为正例。[旧版结果报告](results.md)中的标签分布来自平均特征 checkpoint，不能外推至当前模型。

输出文件包括：

| 文件 | 主要内容 |
|---|---|
| `values.parquet` | 每帧一行：`dataset_id`、`episode_index`、`frame_index`、`timestamp`、`split`、`success`、`reward_raw`、`intervention`、`value` |
| `scores.parquet` | 每帧每个 N 一行：`horizon`、`effective_horizon`、`reward_sum_raw`、`value_next`、`terminal_reached`、`advantage_continuous`、阈值、正负标签及强制标记 |
| `advantages.parquet` | 给下游按 `(dataset_id, episode_index, frame_index, horizon)` 关联的较小元数据表，含布尔 `advantage` 和 `advantage_forced` |
| `episodes.json`、`interventions.json` | 每轨迹评分统计和完整 ±1 秒接管起点窗口均值 |
| `plots/`、`interventions.png`、`index.html` | 逐轨迹曲线、接管汇总图及浏览入口 |
| `manifest.json`、`validation.json` | 输入身份、规则、文件哈希、完成状态与流程检查；`model_quality_validated=false` |
| `frame_files.json`、`labeled_frames.json` | 原始帧位置，以及使用 `--write-labeled-frames` 时写出的副本清单 |

只有 `manifest.status=complete` 的目录可作为完成的结果。修改阈值时可复用价值；`--reuse-values` 配合写副本需要源结果含 `frame_files.json`。`--write-labeled-frames` 只复制帧及部分元数据，不会复制视频；若下游训练器要读取完整数据，还须保留或正确引用原数据。默认输出被 `.gitignore` 排除，不会推送到远程分支。

指定一条已有评分轨迹生成“真实回报／模型价值／接管／优势与标签”四联图：

```bash
python scripts/visualize_episode.py --result artifacts/advantage/all-run \
  --dataset 2026.09.15_2 --episode 53 --horizon 50
```

这一步只重绘图，不更改评分或模型。
