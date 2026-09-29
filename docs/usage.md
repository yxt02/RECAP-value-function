# 运行与输入输出接口

本文按“准备数据 → 训练价值模型 → 测试 → 导出优势”的顺序说明当前本地流程。命令从仓库根目录执行，使用已有 `value_function` conda 环境；原始数据和 SigLIP 权重不随 Git 提交。相对路径在脚本中按仓库根目录解析。完整可选参数可运行对应脚本的 `--help` 查看。

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

`config/train_value.yaml` 指向数据适配配置和本地 SigLIP2 权重。当前模型使用一台机器上的冻结视觉编码器及 201 档分类价值头，取档位期望为连续价值；不读取文字、状态或动作。训练集和验证集分别由清单限定，选择最低验证交叉熵（CE）的 checkpoint。

```bash
python scripts/train.py --smoke_test      # 小样本、独立冒烟产物
python scripts/train.py --prepare-cache   # 仅生成 train/val 冻结特征
python scripts/train.py                   # 完整训练
```

常用覆盖项为 `--config`、`--save_dir`、`--num_epochs`、`--max_total_steps`、`--max_steps`、`--val_steps`、`--max_samples`、`--cache_num_workers`。`--no-cache` 从图像在线编码；当前本机首次构建缓存默认 `cache_num_workers=0`，避开此前验证集解码子进程异常退出的问题。已有缓存经过身份与 SHA256 校验后复用，目录由 `cache_dir` 指定。训练写入 `best_model.pt`、实际 `config.json` 和逐轮 `metrics.json`；默认在 `checkpoints/optimized/`。正式训练会更新该目录的 checkpoint，独立实验可指定新的 `--save_dir`。`run_train.sh` 顺序执行数据准备和训练，支持 `--smoke_test`、`--dry-run`。

当前本机已有 checkpoint 用**修正前**的目标分档规则训练；仓库现行规则只影响下次训练。本次文档整理没有重训或覆盖权重。

## 3. 独立测试与可视化

```bash
bash run_test.sh --dry-run
bash run_test.sh --checkpoint checkpoints/optimized/best_model.pt
```

一键脚本依次运行 `submodules/check_cache.py`、`submodules/evaluate.py`、`submodules/render.py`；`--checkpoint` 会传给缓存检查和评估。`check_cache` 在少量真实帧上比较缓存与在线编码及 checkpoint 重载，可通过 `--output` 指定 JSON 结果。`evaluate` 对**完整 test** 预测，用 train 拟合简单对照，输出 `predictions.npz`、轨迹清单、`metrics.json` 和 `protocol.json`；`render` 由这些文件生成逐轨迹图、HTML 和总览，无需再次运行模型。评估需要原 CUDA/BF16 环境。单独调用时：

```bash
python submodules/evaluate.py --checkpoint checkpoints/optimized/best_model.pt \
  --output artifacts/evaluation/my-run
python submodules/render.py artifacts/evaluation/my-run
```

报告保存在 `artifacts/evaluation/`，不随 Git 推送。当前 checkpoint 的数字、对照和适用边界见 [结果报告](results.md)。`submodules/benchmark.py --mode loader|head|gpu` 用于性能测量；其中 `head` 需要训练特征缓存，`gpu` 需要 `submodules/reference/` 中的旧实现快照。

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
| `--checkpoint` | 默认 `checkpoints/optimized/best_model.pt`；推理采用其中保存的相机、模型参数和预处理约定 |
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

分位阈值和接管置正规则是离线标签规则，正例并非“动作正确”的真值；人工接管前后曲线也只是描述性比较。纯遥操作示范数据目前**没有**作为独立类型强制标为正例。与 RLinf／论文的差异及已导出标签的实际分布见 [结果报告](results.md)。

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
