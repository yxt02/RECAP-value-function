# z02 视觉价值函数

当前可运行入口是 `scripts/train_value.py`：冻结 SigLIP2，按 camera 对 patch 特征取均值，再训练标量回归头，输出范围为 `[-1, 0]`。它是视觉价值基线，尚不是完整的、带语言条件和 201 个离散值区间的 RECAP critic。旧入口虽然加载了 Gemma，但 forward 没有使用它；当前入口不再分配这部分模型。

## 本机运行

使用已有 conda 环境，不需要重装依赖：

```bash
cd /home/zoyi/my_work/RECAP-value-function-main
conda activate value_function
python scripts/train_value.py --smoke_test
python scripts/train_value.py
```

实际训练配置为 **`config/train_value.yaml`**。`config/recap_value_model_sft_z02.yaml` 和 `config/model/recap_value_model.yaml` 是另一套接口的配置，修改它们不会改变此训练入口。

默认首次运行会完整缓存 train/val 的冻结视觉特征，然后训练回归头。后续运行校验缓存后直接训练。可提前生成缓存：

```bash
python scripts/train_value.py --prepare-cache
```

缓存位于 `data/cache/value_features_v1/`，目录名包含内容指纹；单 camera 的完整 train/val 特征约 0.60 GiB。特征以 FP16 存储；冻结模型的在线图片推理也应用相同的特征精度，保持与缓存训练路径一致。指纹覆盖模型权重、processor、预处理、相机顺序、数据文件时间/大小、frame 顺序、return 标签和计算精度。缓存文件带 SHA256 校验，生成完成前不会被训练读取。修改 return 后必须重新适配以更新 sidecar 合同。当前没有随机图像增强，且 encoder 冻结，才能跨 epoch 复用这些特征。

默认把小体积的特征放进显存，GPU 内存紧张时退回 CPU。无需为已缓存特征设置大量 worker；多 worker 用于首次视频解码。若关闭缓存，仍可走原始图像训练：

```bash
python scripts/train_value.py --no-cache --smoke_test
```

## 训练预算

`num_epochs` 和 `max_total_steps` 都是上限，先达到的生效。`max_steps` 是**每轮**的可选限制，正常训练为 null；`val_steps` 为 null 表示每轮使用完整验证集。`max_samples` 只用于开发时限制样本，正常训练不截断数据。

每轮结束验证并保存最低验证 MSE 对应的回归头；验证 MSE 连续 `early_stopping_patience` 轮没有至少 `early_stopping_min_delta` 的改善时早停。学习率采用 warmup + cosine，warmup 会随短测试的实际预算缩短。日志显示 global_step，避免混淆 epoch 内 step 和总 step。默认总步数是按缓存训练的 batch 设置的；若改成全量在线训练或更换 batch，需要同时重新考虑总样本曝光次数。

本机已完成一次全量缓存训练：第 20 轮 / 4,840 步早停，最佳模型来自第 16 轮，验证 MSE 约 0.01161。当前默认上限为 24 轮 / 5,000 步，缓存 batch=1024、warmup=200。

输出目录是 `checkpoints/optimized/`，包含 `best_model.pt`、实际 `config.json` 和逐轮 `metrics.json`。冻结的视觉权重不重复写入 checkpoint；加载时需要保留原 SigLIP 模型目录。checkpoint 的 format_version 为 2，不能直接套用旧入口的模型结构。

## 数据与标签

训练严格读取 `data/splits/train.json`，验证读取 `val.json`；不会把整个数据文件夹直接当成训练集，也不会用测试集调超参数。当前相机选择保留为 `cam2`。

| split | episodes | 成功 | 失败 | frames |
|---|---:|---:|---:|---:|
| train | 260 | 168 | 92 | 246,913 |
| val | 30 | 19 | 11 | 28,359 |
| test | 37 | 24 | 13 | 34,759 |

4 批原始数据共 328 条，排除 09.08 的单帧 episode 82 后是 327 条有效数据。当前四批数据均为 22 维：7 胳膊 + 1 手 + 7 胳膊 + 1 手 + 4 腰 + 2 头；状态/动作加载时在末尾补零到 32 维，视觉基线不使用这些字段。

修复后的 `2026.09.16_error` 共 100 条全部失败，episode 26 的原始 terminal reward=0 是已确认的错误标记。适配配置显式覆盖其 outcome，但不修改原始动作、状态或 reward。失败终止惩罚统一为 -2000，所有 split 用同一个 return_scale=4000；不是分别用各自数据估计归一化范围。

新增或修改原始数据后运行：

```bash
python scripts/adapt_z02.py --config config/z02_data.yaml --all
```

## 验证与性能复测

```bash
python -m unittest discover -s tests -v
python scripts/benchmark_value.py --mode loader
python scripts/benchmark_value.py --mode head  # 需要已完成的全量 train 缓存
python scripts/verify_feature_cache.py         # 比较缓存与在线编码，并检查已有 checkpoint
```

`--mode gpu` 比较本次优化前的本机脚本快照和当前模型，需保留 `artifacts/performance/baseline/train_value.py`。性能日志和测试产物位于 `artifacts/performance/`。详细测量范围、配置选择依据和训练结果见 `docs/performance.md`。

## 当前 checkpoint 独立测试

已完成全部 37 条测试轨迹 / 34,759 帧评估。测试 MSE **0.02746**，批次+帧序号对照 **0.02250**；旧批次失败误差和逐帧波动较大，暂不建议直接用于 RECAP 优势标签。详情见 [独立测试报告](docs/independent_test.md)。
