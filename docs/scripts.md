# 脚本入口与输入输出接口

`scripts/value_function.py` 是本工程唯一的命令行入口。数据准备、训练及评估实现位于 `scripts/*_workflow.py`，按功能拆分以便测试和复用；无需安装 RLinf。

代码只放在两处：`scripts/` 存放命令行入口和三个工作流，`submodules/` 存放被它们复用的核心模块（模型、缓存、评估统计、特征加载、数据加载、共享合同）。`tests/` 为单元测试。数据集的加载与回报计算已并入这两处，不再有独立的 `recap_datasets/` 或 `process/` 目录。

```bash
conda activate value_function
cd /home/zoyi/my_work/RECAP-value-function-main
python scripts/value_function.py --help
python scripts/value_function.py data --help
python scripts/value_function.py value train --help
python scripts/value_function.py report evaluate --help
```

所有相对输入和输出路径均相对**工程根目录**，也可传绝对路径；从其他目录调用时使用入口的绝对路径。每个工作流同时提供 `main(argv=None)` 路由和 `main_<command>(argv=None)` 子命令入口，传入字符串参数列表可供 Python 调用，不修改全局 `sys.argv`。训练的 `epoch`、`build_scheduler` 和数据准备的 `run` 仍可直接导入。

执行成功退出码为 0；参数错误为 2；计算或文件错误返回非零且保留错误信息，不自动跳过。帮助页列出参数；下表定义产物和写入范围。

## 1. 数据命令

### `data prepare`

**输入：** `--config`，默认 `config/z02_data.yaml`。配置指定原始数据目录、数据集、22 维映射、结果覆盖规则、回报尺度、划分比例和输出目录。各数据集须包含 `data/**/episode_*.parquet`、`meta/info.json`、`meta/tasks.jsonl` 及所配置相机的视频。

| 参数 | 行为与输出 |
|---|---|
| `--analyze` | 只读检查关节、索引、时间戳、标签和视频存在性；终端输出统计 |
| `--verify-videos` | 追加视频帧数、帧率和首中末帧解码检查；单独使用时只读 |
| `--compute_returns` | 在各数据集 `meta/` 写入 `returns_<tag>.parquet` 和对应 JSON 合同、`episode_outcomes.json`、`z02_adaptation.json`；同步相关元数据 |
| `--split_data` | 在配置的 `output_dir` 写入 `train.json`、`val.json`、`test.json`、`summary.json` 等划分信息 |
| `--all` | 检查、生成回报、生成划分；视频深度检查需另加 `--verify-videos` |

写入操作还会生成 `output_dir/adaptation_report.json`。原始 parquet 和视频不改写；适配过程会更新 `meta` 文件，变化的 JSON 按已有机制保留备份。无操作参数时显示帮助。

**回报文件接口：** 每行由 `(episode_index, frame_index)` 标识，包含 `return`、`reward`、`prompt`。跨数据集关联时必须加上数据集标识。归一化尺度和结果覆盖以配置及 sidecar 合同为准。

```bash
python scripts/value_function.py data prepare --analyze
python scripts/value_function.py data prepare --all --verify-videos
```

### `data survey`

**输入：** 现有 `data/splits/*.json`、原始状态及可用评估产物。用于现有本地任务的同批次配对分析，不是任意数据集的通用配对 API。

**输出：** `--output` 指定 JSON，默认 `artifacts/analysis/pairing_survey.json`，以及终端统计。只读源数据，不训练模型。结果包含样本清单、配对参数扫描和相关分析。

```bash
python scripts/value_function.py data survey --output artifacts/analysis/pairing_survey.json
```

## 2. 价值模型命令

### `value train`

**输入：** `--config`，默认 `config/train_value.yaml`；配置引用适配配置、划分、SigLIP 权重和训练参数。当前架构是冻结图像编码器加标量回归网络，不是完整多模态 RECAP critic。

**输出：** 配置或 `--save_dir` 指定目录下的 `config.json`、`metrics.json`、`best_model.pt`；启用缓存时还会在 `cache_dir` 生成带身份校验的特征缓存。训练会写入目标目录，独立实验应指定新的目录。

| 参数 | 意义 |
|---|---|
| `--prepare-cache` | 只准备 train/val 特征后退出，不执行优化器更新 |
| `--no-cache` | 从图像在线编码训练 |
| `--smoke_test` | 使用现有固定的小样本冒烟配置，输出到 `artifacts/performance/smoke-cached` 或 `smoke-online`；会覆盖该冒烟目录中的同名产物 |
| `--max_total_steps` | 整次训练的更新步数上限 |
| `--max_steps` / `--val_steps` | 每轮训练/验证的批次数上限 |
| `--max_samples` | 每个划分的小样本上限 |
| `--save_dir` | 输出目录；使用 `--smoke_test` 时以固定冒烟目录为准 |

其余可覆盖参数见 `--help`。GPU 优先；CPU 会切换 FP32。缓存仅适用于冻结编码器，校验不匹配时不会误用旧缓存。

**checkpoint 接口：** format_version=2，architecture=`siglip_mean_patch_scalar`，包含模型参数、配置、优化器及训练状态等。冻结 SigLIP 权重通常不打包，重新从配置指定的模型目录加载。模型输入为相机图像字典或缓存特征，输出为 `[B, 1]` 的归一化价值，范围 `[-1, 0]`。

```bash
python scripts/value_function.py value train --prepare-cache
python scripts/value_function.py value train --smoke_test
python scripts/value_function.py value train --num_epochs 1 --max_steps 2 --val_steps 1 --max_samples 64 --save_dir artifacts/smoke/custom
```

### `value check-cache`

**输入：** `--config`（默认训练配置）、匹配的完整 train 缓存、SigLIP 权重，以及配置保存目录中存在的 `best_model.pt`。需要 CUDA；缺少 checkpoint 时只检查缓存，结果明确记为 `not tested`。

**输出：** `--output` 指定 JSON，默认 `artifacts/performance/cache-verification.json`。比较有界真实帧上的在线编码、缓存预测、标签及 checkpoint 重载。输入缓存或权重不被修改。

### `value benchmark`

**输入：** `--config` 和必填 `--mode gpu|loader|head`。

- `loader`：测量原始数据读取，比较 worker 数量。
- `head`：读取已建好的完整 train 缓存，测量回归头训练；需要 CUDA。
- `gpu`：比较历史/当前视觉网络；还依赖 `submodules/reference/train_value.py` 历史实现及相关权重，需要 CUDA。

**输出：** `artifacts/performance/<mode>_benchmark.json` 及终端统计。会做临时模型更新以测吞吐，但不保存或修改正式 checkpoint；同模式结果文件会覆盖。

## 3. 评估与报告命令

以下命令通过同一评估目录传递数据，记作 `EVAL_DIR`。

### `report evaluate`

**输入：** `--checkpoint`，默认 `checkpoints/optimized/best_model.pt`；checkpoint 配置、匹配的原始数据、当前 `data/splits/{train,val,test}.json` 和编码器。当前是固定的全 test 评估，不支持任意 split；简单对照仅用 train 拟合。需要原有 CUDA/BF16 环境。

**输出：** `--output EVAL_DIR`，默认 `artifacts/evaluation/checkpoint-<权重哈希前12位>-test`，包含：

| 文件 | 内容 / 下游用途 |
|---|---|
| `protocol.json` | checkpoint 路径、哈希、划分信息和评估约定 |
| `predictions.npz` | 按轨迹拼接的一维数组：`target`、`prediction`、`timestamp`、`intervention` 及各简单对照预测 |
| `episodes.json` | 每条轨迹的 dataset、episode_index、slug、offset、frames、fps、success 和统计；用 `offset:offset+frames` 切取数组 |
| `metrics.json` | 总体/分组误差、时序诊断、缓存位置等 |
| `baseline_fit_train_only.json` | 仅在训练集拟合的对照参数 |
| `audit.json` | 权重、源代码、缓存等溯源信息 |
| `storyboards/` | 各轨迹关键帧图片 |
| `videos/` | 原始视频的符号链接；迁移报告时须保证链接目标可用 |

可能创建或复用特征缓存。不会训练或修改 checkpoint。相同目录有协议一致性检查；重跑可能覆盖评估文件，新实验建议使用新目录。该命令**不生成 HTML**；图表和报告由 `report render` 生成。

### `report render`

**输入：** `EVAL_DIR` 下的 protocol、metrics、episodes、predictions；配套图片和视频供 HTML 引用。

**输出：** `index.html`、`trajectories/*.html`、`plots/*.png`、`overview.png`、`all_trajectories.png`、`intervention_events.png`；覆盖派生图表。重新制作图表只需 `render`，不用再次运行模型。报告页面直接引用 `videos/` 下的原始视频符号链接，因此这些链接必须保持可用。

```bash
python scripts/value_function.py report evaluate --output artifacts/evaluation/new-run
python scripts/value_function.py report render artifacts/evaluation/new-run
```

**绘图环境：** 已在 `value_function` 环境中验证 matplotlib 3.10.9。首次配置环境时安装可选绘图依赖；不再依赖系统 Python 的包：

```bash
python -m pip install -r requirements-plotting.txt
python scripts/value_function.py report render artifacts/evaluation/new-run
```

入口按命令延迟导入模块，因此 `report` 命令不会加载训练工作流。matplotlib 只在 `report render` 调用 `configure_plots()` 时导入，`evaluate` 不导入绘图库。绘图需要 NumPy 和 matplotlib，其他核心流程继续使用 `value_function` 环境。

## 环境注意事项：导入期崩溃

本机 `value_function` 环境在导入 scipy（以及经由 scipy 的 transformers、matplotlib）时会偶发段错误或 `TypeError`，根因是多线程 BLAS/OpenMP 在导入期的竞争：`scipy/_lib/_docscrape.py` 解析 docstring 时崩溃。实测（每次 60–80 次独立进程）：`import scipy.special` 约 1/60、`from transformers import SiglipVisionModel` 约 2/80、`import matplotlib.pyplot` 约 12/60 失败；`import numpy`、`import torch`、`import cv2`、`import pyarrow` 均为 0/60。设置 `OPENBLAS_NUM_THREADS=1` 或 `OMP_NUM_THREADS=1` 后全部降到 0/80。

`submodules/__init__.py` 因此在导入时把 `OMP_NUM_THREADS`、`MKL_NUM_THREADS`、`OPENBLAS_NUM_THREADS` 设为 `1`，位置在任何重依赖之前；入口和测试都最先导入该包。修复前完整测试 12 次中失败 3 次，修复后连续 39 次全部通过。`submodules/runtime.py` 的 `configure_runtime` 仍按配置设置 `torch.set_num_threads`，与上述设置互不冲突。

三个工作流模块也会在导入 numpy/cv2/torch **之前** `import submodules`，因此直接运行 `python scripts/data_workflow.py ...` 与走入口一样稳定（实测各 40 次 0 失败，修复前约 1–2/40）。新增重型导入时，应放在 `import submodules` 之后。

## 旧入口迁移

旧脚本已移入 `scripts/`，不保留十个转发文件。历史实验报告中的命令作为溯源记录保留；重新执行时按下表替换，原参数保持可用。

| 旧脚本 | 新命令（前缀均为 `python scripts/value_function.py`） |
|---|---|
| adapt_z02.py | `data prepare` |
| survey_pairing.py | `data survey` |
| train_value.py | `value train` |
| benchmark_value.py | `value benchmark` |
| verify_feature_cache.py | `value check-cache` |
| evaluate_value.py | `report evaluate` |
| render_value_evaluation.py | `report render` |

`check_value_evaluation.py`、`prepare_evaluation_videos.py` 和 `serve_value_evaluation.py` 的独立重放、视频转码和本地服务功能已删除；报告页面直接引用原始视频，用任意静态文件服务即可浏览。后续优势评分和标签生成功能可在此入口增加子命令；本次只整理已有功能，不将尚未实现的优势标注描述为已完成。

## 目录整合

2026-09-21 精简了目录布局，代码只保留 `scripts/` 和 `submodules/` 两处：

| 原位置 | 现位置 |
|---|---|
| `recap_datasets/recap/contracts.py` | `submodules/contracts.py` |
| `recap_datasets/recap/simple_dataset.py` | `submodules/datasets.py` |
| `process/compute_returns.py` | 已删除；回报计算由 `data prepare --compute_returns` 完成 |

`recap_datasets/` 其余文件（`value_dataset.py`、`common.py`、`utils.py`、`loader.py`）及 `config/recap_value_model_sft_z02.yaml`、`config/model/recap_value_model.yaml` 只被彼此引用，已一并删除。合并目标选在 `submodules/` 而不是平铺进 `scripts/`，是因为 `datasets` 与已安装的 `site-packages/datasets` 同名，平铺会造成导入歧义。核心模块的相对导入保持不变，功能未变。

## 本次整合验证（2026-09-21）

- `python -m unittest discover -s tests -v`：21 项通过，包括旧训练/数据测试与新入口路由、错误传播和跨目录调用。
- 目录整合与导入稳定性修复后，完整测试连续 39 次全部通过（修复前 12 次中失败 3 次，原因见上文“环境注意事项”）；单模块、入口各路由和三个工作流直接调用也各自连续 25–60 次无失败。
- `data prepare --analyze`：4 批真实数据只读检查通过。
- `value train --smoke_test`：256 个真实训练样本、4 次优化器更新、2 个验证批次，cache/online 两条路径均通过有限梯度与参数更新检查。
- `report render`：重新生成 37 张逐轨迹图和 37 个轨迹页面，报告首页可正常打开。
- 评估验证输出位于 `artifacts/evaluation/cli-integration-smoke`；源评估输入复制或只读链接，原报告未覆盖。本次没有重新执行完整 `report evaluate` 推理或全量训练。

> 2026-09-21 清理说明：上文记录的评估和冒烟产物及特征缓存已删除，验证记录保留。使用相关命令前须重新生成其输入产物。历史性能对照源码保留在 `submodules/reference/`。
