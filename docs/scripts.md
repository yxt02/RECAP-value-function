# 脚本入口与输入输出接口

`scripts/recap.py` 是本工程唯一的命令行入口。数据准备、训练及评估实现位于 `recap_value/workflows/`，按功能拆分以便测试和复用；无需安装 RLinf。

```bash
conda activate value_function
cd /home/zoyi/my_work/RECAP-value-function-main
python scripts/recap.py --help
python scripts/recap.py data --help
python scripts/recap.py value train --help
python scripts/recap.py report evaluate --help
```

所有相对输入和输出路径均相对**工程根目录**，也可传绝对路径；从其他目录调用时使用入口的绝对路径。各工作流提供 `main(argv=None)`，传入字符串参数列表可供 Python 调用，不修改全局 `sys.argv`。训练的 `epoch`、`build_scheduler` 和数据准备的 `run` 仍可直接导入。

执行成功退出码为 0；参数错误为 2；计算或文件错误返回非零且保留错误信息，不自动跳过。`report serve` 持续运行，Ctrl+C 正常关闭服务器。帮助页列出参数；下表定义产物和写入范围。

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
| `--import-zip ZIP` | 解压到配置的 `data_dir`，拒绝路径越界和覆盖已有文件；需要后续处理时搭配 `--all` |

写入操作还会生成 `output_dir/adaptation_report.json`。原始 parquet 和视频不改写；适配过程会更新 `meta` 文件，变化的 JSON 按已有机制保留备份。无操作参数时显示帮助。

**回报文件接口：** 每行由 `(episode_index, frame_index)` 标识，包含 `return`、`reward`、`prompt`。跨数据集关联时必须加上数据集标识。归一化尺度和结果覆盖以配置及 sidecar 合同为准。

```bash
python scripts/recap.py data prepare --analyze
python scripts/recap.py data prepare --all --verify-videos
python scripts/recap.py data prepare --import-zip /absolute/new-data.zip --all
```

### `data survey`

**输入：** 现有 `data/splits/*.json`、原始状态及可用评估产物。用于现有本地任务的同批次配对分析，不是任意数据集的通用配对 API。

**输出：** `--output` 指定 JSON，默认 `artifacts/analysis/pairing_survey.json`，以及终端统计。只读源数据，不训练模型。结果包含样本清单、配对参数扫描和相关分析。

```bash
python scripts/recap.py data survey --output artifacts/analysis/pairing_survey.json
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
python scripts/recap.py value train --prepare-cache
python scripts/recap.py value train --smoke_test
python scripts/recap.py value train --num_epochs 1 --max_steps 2 --val_steps 1 --max_samples 64 --save_dir artifacts/smoke/custom
```

### `value check-cache`

**输入：** `--config`（默认训练配置）、匹配的完整 train 缓存、SigLIP 权重，以及配置保存目录中存在的 `best_model.pt`。需要 CUDA；缺少 checkpoint 时只检查缓存，结果明确记为 `not tested`。

**输出：** `--output` 指定 JSON，默认 `artifacts/performance/cache-verification.json`。比较有界真实帧上的在线编码、缓存预测、标签及 checkpoint 重载。输入缓存或权重不被修改。

### `value benchmark`

**输入：** `--config` 和必填 `--mode gpu|loader|head`。

- `loader`：测量原始数据读取，比较 worker 数量。
- `head`：读取已建好的完整 train 缓存，测量回归头训练；需要 CUDA。
- `gpu`：比较历史/当前视觉网络；还依赖 `artifacts/performance/baseline/train_value.py` 历史实现及相关权重，需要 CUDA。

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

可能创建或复用特征缓存。不会训练或修改 checkpoint。相同目录有协议一致性检查；重跑可能覆盖评估文件，新实验建议使用新目录。该命令**不自动生成 HTML 或浏览器预览视频**。

### 其余报告命令

| 命令 | 输入 | 输出 / 副作用 |
|---|---|---|
| `report check EVAL_DIR` | protocol、metrics、episodes、predictions、原 checkpoint 和完整 test 缓存 | `independent_replay.json`；CUDA 重放全部预测并核对目标、误差及权重；目前目标公式校验针对当前 −2000/4000 合同 |
| `report videos EVAL_DIR --workers 4` | episodes.json、videos 下可访问的视频链接 | `previews/*.webm`、逐视频 JSON、`video_previews.json`；复用身份匹配的预览，不改源视频 |
| `report render EVAL_DIR` | protocol、metrics、episodes、predictions；配套图片和视频供 HTML 引用 | `index.html`、`trajectories/*.html`、`plots/*.png`、`overview.png`、`all_trajectories.png`、`intervention_events.png`；覆盖派生图表 |
| `report serve EVAL_DIR --port 0` | 已生成 index.html 的目录及其资源 | 在 127.0.0.1 启动服务，0 表示自动选择端口；写入 `preview-url.txt`，终端输出 URL |

建议顺序：`evaluate → check → videos → render → serve`。重新制作图表只需 `render`，不用再次运行模型。

```bash
python scripts/recap.py report evaluate --output artifacts/evaluation/new-run
python scripts/recap.py report check artifacts/evaluation/new-run
python scripts/recap.py report videos artifacts/evaluation/new-run --workers 4
python scripts/recap.py report render artifacts/evaluation/new-run
python scripts/recap.py report serve artifacts/evaluation/new-run --port 0
```

**绘图环境：** 已在 `value_function` 环境中验证 matplotlib 3.10.9。首次配置环境时安装可选绘图依赖；不再依赖系统 Python 的包：

```bash
python -m pip install -r requirements-plotting.txt
python scripts/recap.py report render artifacts/evaluation/new-run
```

入口按命令延迟导入模块，因此绘图和服务命令不会加载 PyTorch、模型或训练依赖。视频转换需要 PyAV，绘图需要 NumPy 和 matplotlib，其他核心流程继续使用 `value_function` 环境。

## 旧入口迁移

旧脚本已移入包内，不保留十个转发文件。历史实验报告中的命令作为溯源记录保留；重新执行时按下表替换，原参数保持可用。

| 旧脚本 | 新命令（前缀均为 `python scripts/recap.py`） |
|---|---|
| adapt_z02.py | `data prepare` |
| survey_pairing.py | `data survey` |
| train_value.py | `value train` |
| benchmark_value.py | `value benchmark` |
| verify_feature_cache.py | `value check-cache` |
| evaluate_value.py | `report evaluate` |
| check_value_evaluation.py | `report check` |
| render_value_evaluation.py | `report render` |
| prepare_evaluation_videos.py | `report videos` |
| serve_value_evaluation.py | `report serve` |

后续优势评分和标签生成功能可在此入口增加子命令；本次只整理已有功能，不将尚未实现的优势标注描述为已完成。

## 本次整合验证（2026-09-21）

- `python -m unittest discover -s tests -v`：22 项通过，包括旧训练/数据测试与新入口、错误传播、跨目录调用、视频 Range 请求测试。
- `data prepare --analyze`：4 批真实数据只读检查通过。
- `value train`：64 个训练样本、2 次真实更新、1 个验证批次，保存到 `artifacts/performance/cli-integration-smoke`。
- `report check`：独立重放 34,759 帧，预测最大差异 0，原 checkpoint 未变化。
- `report render`：重新生成 37 张逐轨迹图和 37 个轨迹页面；HTTP 首页、真实 WebM 的 206 字节范围响应及 Ctrl+C 退出检查通过。
- 评估验证输出位于 `artifacts/evaluation/cli-integration-smoke`；源评估输入复制或只读链接，原报告未覆盖。本次没有重新执行完整 `report evaluate` 推理或全量训练。
