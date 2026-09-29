# 运行与输入输出接口

本文按“准备数据 → 训练价值模型 → 测试 → 导出优势”的顺序说明当前本地流程。命令从仓库根目录执行，使用已有 `value_function` conda 环境；原始数据与 SigLIP2、Gemma3 权重不随 Git 提交。相对路径在脚本中按仓库根目录解析。完整可选参数可运行对应脚本的 `--help` 查看。

## 1. 检查和适配数据

`config/z02_data.yaml` 指定 `data/raw/` 的四批数据、相机 `cam2`／`cam3`／`cam4`、22 维关节约定、结果覆盖及 train/val/test 划分。只读检查先运行：

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

`config/train_value.yaml` 是主流程配置，指定 SigLIP2、Gemma3 和任务分词器。三路图像各按原比例缩放、黑边补齐到 224×224，再提取图像块；仅训练循环启用随机裁剪、轻微旋转和颜色增强，推理始终使用固定预处理。每路相机 256 个图像块，三路合计 768 个，与最多 200 个任务文字 token 组成输入序列；小型 Gemma 价值专家通过可学习的 `CLS` 标记读取序列，预测 201 档价值分布及其连续期望。每路相机的有效位进入前缀注意力掩码。默认使用 18 层 `gemma_50m` 专家，在线训练 SigLIP2、Gemma3 和价值专家；三部分均可单独冻结并设置学习率。`critic_expert_variant` 还支持调试用 `gemma_1m`、`gemma_100m`、`gemma_150m`、`gemma_300m` 和 `gemma_2b`，更大专家需要另行验证显存。数据不输入关节状态或动作。训练集和验证集分别由清单限定，选择最低验证交叉熵（CE）的 checkpoint。方法上的前后差异见 [README](../README.md#模型与数据流)。

```bash
python scripts/train.py --smoke_test      # 小样本、独立冒烟产物
torchrun --standalone --nproc_per_node=2 scripts/train.py  # 两张 GPU 完整训练
```

常用覆盖项为 `--config`、`--save_dir`、`--batch_size`、`--num_epochs`、`--max_total_steps`、`--max_steps`、`--val_steps`、`--max_samples`、`--vision_lr`、`--gemma_lr`、`--expert_lr`、`--gradient_accumulation_steps`，以及三个 `--freeze_*`／`--no-freeze_*` 开关。计划使用 BF16、部分敏感层 FP32、梯度检查点、每卡批大小 2、累积 4 个微批次、最多 8 轮；双卡全局有效批大小为 16。本机冒烟强制每卡批大小 1；计划的每卡批大小 2 尚未在云端验证，应先运行双卡冒烟并检查显存，再尝试 `--batch_size 4`。DDP 每卡负责自己的一份训练数据，验证分片没有重复样本，主进程写 checkpoint。`max_total_steps` 计优化更新次数，`max_steps` 计每轮每卡的微批次数。当前模型在线读取图像，不能使用旧平均特征缓存；`--prepare-cache` 仅适用于旧架构。训练写入 `best_model.pt` 和逐轮 `metrics.json`；默认在 `checkpoints/recap_patch/`。`run_train.sh` 顺序执行数据准备和训练，支持 `--smoke_test`、`--dry-run`。

数据适配合同仍记录固定 `return_scale=4000`，供原始数据一致性检查；当前价值模型按训练划分的最小回报 `−3769` 使用 `value_scale=3769`，并保存在 checkpoint。验证／测试回报若小于训练最小值，目标截到 `−1`；模型输出也只在 `[-1,0]`。若之后重新划分数据，训练最小值和尺度会重新计算。

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

报告保存在 `artifacts/evaluation/`，不随 Git 推送。新架构尚无正式训练结果。`submodules/benchmark.py --mode loader|head|gpu` 是历史基线的性能测量工具，其中 `head` 需要旧特征缓存，`gpu` 需要 `submodules/reference/` 中的旧实现快照。

## 4. 计算优势和标签

`calculate_advantage.py` 默认用完整 test、50 帧前瞻和固定阈值 0。对每条长度为 L 的轨迹，在 t 帧计算：

```text
k = min(N, L - t)
R = sum(gamma**i * reward[t+i] for i in range(k))
A[t] = R / scale + gamma**k * (V[t+k] if t+k < L else 0) - V[t]
```

当前 `gamma=1`；新模型从 checkpoint 读取 `scale=3769`，普通 50 帧非终止窗口为 `V[t+50]-V[t]-50/3769`。旧平均特征 checkpoint 仍使用其保存的 `scale=4000`。末尾按实际剩余帧数和终止奖励计算，不跨轨迹。前瞻长度 N 是帧数，不是秒数，也不能直接解释为动作块长度。

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

分位阈值和接管置正规则是离线标签规则，正例并非“动作正确”的真值；人工接管前后曲线也只是描述性比较。纯遥操作示范数据目前**没有**作为独立类型强制标为正例。当前没有正式模型可用来判断标签质量。

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


## 5. 专家读取层与梯度检查点

默认采用 RLinf 的同层对应方式：专家第 i 层读取 Gemma3 第 i 层的 K/V。`gemma_50m` 有 18 层，覆盖 18 个前缀缓存；`gemma_1m` 有 4 层，只执行前 4 层前缀，余下层保留权重以兼容加载，但被冻结且不参与计算。专家层数不能大于前缀层数。最后被读取的前缀层仅有 K/V 等上游路径参与价值损失，最后的前缀输出及归一化层没有被价值头读取。

`gradient_checkpointing=true` 对视觉编码器启用 HF 标准非重入逐层检查点；文字前缀和专家分别用 PyTorch 非重入检查点包裹。阶段之间只传递 K/V 张量元组，每次前缀计算创建新的 `DynamicCache`，每次专家计算由这些张量重建独立缓存，不复用已被追加内容的缓存。不能额外开启 Gemma3 或专家的 HF 层级检查点：本机版本会清除 `past_key_values`，破坏两阶段接口。这一方案节省激活存储，代价是增加重算；它不是 FSDP 或参数分片。回归测试比较 FP32 小模型的全部梯度；真实三相机 BF16 检查确认输出一致、梯度有限且最后一层 KV 有梯度。BF16 多分支梯度累加即使重复普通前向也可能有数值差异，不承诺逐位相同。

## 6. 旧代码兼容边界

当前 `recap_patch_gemma_expert` 从图像在线训练，始终进入 `recap_workflow.py`。旧 `siglip_mean_patch_categorical` 使用 `model.py`、`cache.py` 和 `feature_loader.py`，训练由 `training_workflow.py` 内的兼容分支处理。`benchmark.py` 和 `reference/` 只服务历史性能对照。

旧缓存身份已升级为 `siglip-mean-patch-v3`，记录补边方式、图像大小、归一化均值与方差、回报尺度及截断方式。旧版本缓存需要重新构建；随机增强不能写入可复用的固定特征缓存。图像预处理从拉伸改为补边后，即使旧 checkpoint 可以加载，其预测也不能被当成历史预处理下的同一结果。旧 checkpoint 的来源校验可能因此拒绝重用旧产物，应重新训练或保留对应历史代码环境。

`config/z02_data_error_only.yaml` 仅用于只装有失败批次时排查数据，输出到 `data/splits/error_only/`。默认完整训练不使用它；如需在该划分训练，必须显式指定对应的 `adaptation_config`。
