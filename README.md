# RECAP Value Function

基于 RLinf 框架的 RECAP 算法中的 Value Function 训练模块。

## 简介

本仓库包含 RECAP (RL with Experience and Corrections via Advantage-conditioned Policies) 算法中用于训练价值函数的代码。价值函数 V(s) 用于评估机器人在特定任务中的状态价值，预测从当前状态开始的累积回报。

## 架构

```
value_function/
├── config/                          # 训练配置文件
│   ├── recap_compute_returns.yaml   # 计算 return 的配置
│   ├── recap_value_model_sft.yaml   # 价值模型训练配置
│   └── model/
│       └── recap_value_model.yaml   # 模型架构配置
├── datasets/
│   └── recap/
│       └── value_dataset.py         # 数据集加载和归一化
├── examples/
│   ├── train_value.py               # 训练入口
│   └── run_value_sft.sh             # 训练启动脚本
├── hybrid_engines/
│   └── fsdp/
│       └── fsdp_model_manager.py    # FSDP 分布式训练管理
├── models/
│   └── recap/
│       ├── __init__.py              # 模型工厂
│       ├── configuration.py         # 模型配置
│       ├── data_collator.py         # 数据整理
│       ├── modeling_critic.py       # 价值模型定义
│       ├── processing.py            # 数据处理
│       └── value_expert.py          # SigLIP2 + Gemma3 + Expert
├── process/
│   └── compute_returns.py           # 计算逐帧 return
└── workers/
    └── sft/
        └── fsdp_value_sft_worker.py # FSDP 训练 Worker
```

## 核心组件

### 1. 模型架构 (`models/recap/`)

- **Vision Encoder**: SigLIP2-so400m (1152-dim)
- **Language Model**: Gemma3-270M (640-dim)
- **Critic Expert**: GemmaForCausalLM (可配置大小)
- **Value Head**: CLS token + 线性投影到 201 个 bins

### 2. 数据流程

1. **compute_returns.py**: 计算逐帧 return
   - 每步 reward = -1
   - 成功最后一步 reward = 0
   - 失败最后一步 reward = failure_reward (如 -300)
   - 通过向后迭代计算：G_t = r_t + gamma * G_{t+1}

2. **value_dataset.py**: 数据加载
   - 归一化 return 到 [-1, 0] 范围
   - 支持多种机器人类型（libero, franka, 自定义）

3. **modeling_critic.py**: 价值预测
   - 使用 201 个 bins 的分类分布
   - 输出预测的累积回报

### 3. 训练方式

- 使用 FSDP 分布式训练
- 支持梯度累积
- 支持混合精度训练 (bf16)
- 可冻结视觉编码器和 VLM

## 快速开始

### 环境要求

```bash
# 安装依赖（需要从 RLinf 完整仓库）
pip install torch transformers
pip install safetensors omegaconf hydra-core
pip install pyarrow pandas
```

### 数据准备

数据集需要是 LeRobot 格式，包含：
- 图像观测
- 状态信息
- 动作
- 任务描述

### 训练流程

#### Step 1: 计算 Returns

```bash
# 修改 config/recap_compute_returns.yaml 中的数据路径
python process/compute_returns.py --config-name recap_compute_returns
```

#### Step 2: 训练价值模型

```bash
# 修改 config/recap_value_model_sft.yaml 中的配置
# 包括数据路径、模型路径、训练参数等
bash examples/run_value_sft.sh recap_value_model_sft
```

### 关键配置

```yaml
# config/recap_value_model_sft.yaml
data:
  train_data_paths:
    - dataset_path: "/path/to/your/dataset"
      type: "sft"
      robot_type: "your_robot_type"  # libero, franka, 或自定义
      model_type: "pi05"
  tag: "fail300"  # 必须和 Step 1 的 tag 一致

actor:
  model:
    siglip_path: "/path/to/siglip2-so400m-patch14-224"
    gemma3_path: "/path/to/gemma-3-270m"
    tokenizer_path: "/path/to/gemma-3-270m"
```

---

## 适配指南

本节介绍如何将本仓库适配到你自己的模型和机器人。

### 一、数据层适配（必须调整）

#### 1. 机器人数据字段映射

**文件**：`datasets/recap/value_dataset.py`

在 `_REPACK_KEYS` 字典中添加你的机器人类型：

```python
_REPACK_KEYS = {
    # 已有类型
    "libero": {...},
    "franka": {...},
    
    # 添加你的机器人类型
    "your_robot": {
        "observation/image": "你的图像字段名",           # 如 "camera_rgb"
        "observation/wrist_image": "你的腕部图像字段名", # 如 "wrist_camera" (可选)
        "observation/state": "你的状态字段名",           # 如 "joint_positions"
        "actions": "你的动作字段名",                     # 如 "action"
        "prompt": "你的任务描述字段名",                  # 如 "instruction"
    },
}
```

**需要确认的字段**：
- [ ] 图像字段名称
- [ ] 是否有腕部摄像头
- [ ] 状态字段名称（关节位置、末端位姿等）
- [ ] 动作字段名称
- [ ] 任务描述字段名称

#### 2. 数据集格式

数据集必须是 LeRobot 格式：

```
your_dataset/
├── data/
│   ├── episode_000000.parquet
│   ├── episode_000001.parquet
│   └── ...
└── meta/
    ├── info.json          # 数据集元信息
    ├── stats.json         # 统计信息
    ├── tasks.jsonl        # 任务描述
    └── returns.parquet    # 计算得到的 return (Step 1 生成)
```

如果你的数据不是 LeRobot 格式，需要先转换。

---

### 二、模型层适配（可选调整）

#### 1. 视觉编码器

**当前**：SigLIP2-so400m (1152-dim)

**可选替代**：

| 模型 | 维度 | HuggingFace ID |
|---|---|---|
| CLIP ViT-L/14 | 1024 | `openai/clip-vit-large-patch14` |
| DINOv2-base | 768 | `facebook/dinov2-base` |
| EVA-CLIP | 1024 | `BAAI/EVA-CLIP` |
| SigLIP2 (默认) | 1152 | `google/siglip2-so400m-patch14-224` |

**如果替换，需要修改**：

**文件**：`models/recap/value_expert.py`

```python
# 原代码
from transformers import SiglipVisionModel
self.vision_tower = SiglipVisionModel.from_pretrained(siglip_path)
siglip_hidden = self.vision_tower.config.hidden_size  # 1152

# 修改为（以 CLIP 为例）
from transformers import CLIPVisionModel
self.vision_tower = CLIPVisionModel.from_pretrained(clip_path)
clip_hidden = self.vision_tower.config.hidden_size  # 1024
```

同时修改投影层维度：

```python
# 原代码
self.multi_modal_proj = nn.Linear(1152, 640, bias=True)

# 修改为
self.multi_modal_proj = nn.Linear(1024, 640, bias=True)  # 1024 是新视觉模型的维度
```

#### 2. 语言模型

**当前**：Gemma3-270M (640-dim, head_dim=256)

**可选替代**：

| 模型 | 维度 | head_dim | HuggingFace ID |
|---|---|---|---|
| Qwen2.5-0.5B | 896 | 128 | `Qwen/Qwen2.5-0.5B` |
| Phi-3-mini | 3072 | 96 | `microsoft/Phi-3-mini-4k-instruct` |
| Llama-3.2-1B | 2048 | 64 | `meta-llama/Llama-3.2-1B` |
| SmolLM-135M | 576 | 64 | `HuggingFaceTB/SmolLM-135M` |
| Gemma3 (默认) | 640 | 256 | `google/gemma-3-270m` |

**关键约束**：Expert 的 `head_dim` 必须和语言模型的 `head_dim` 一致。

**如果替换，需要修改**：

**文件**：`models/recap/value_expert.py`

```python
# 原代码
from transformers import Gemma3ForCausalLM
self.gemma3 = Gemma3ForCausalLM.from_pretrained(gemma3_path)
gemma3_hidden = self.gemma3.config.hidden_size  # 640

# 修改为（以 Qwen2.5 为例）
from transformers import Qwen2ForCausalLM
self.gemma3 = Qwen2ForCausalLM.from_pretrained(qwen_path)
qwen_hidden = self.gemma3.config.hidden_size  # 896
```

同时修改投影层和 Expert 配置，确保 `head_dim` 一致。

#### 3. Critic Expert

**当前**：`gemma_1m` (1.2M 参数，用于调试)

**可选大小**：

| 变体 | 参数量 | 适用场景 |
|---|---|---|
| `gemma_1m` | 1.2M | 调试、快速验证 |
| `gemma_50m` | 56M | 实验、小规模数据 |
| `gemma_100m` | 110M | 默认推荐 |
| `gemma_300m` | 311M | 高精度需求 |
| `gemma_2b` | 1.98B | 最高精度 |

**修改配置**：

**文件**：`config/model/recap_value_model.yaml`

```yaml
critic_expert_variant: "gemma_100m"  # 改为你想要的大小
```

---

### 三、配置层适配（必须调整）

#### 1. 模型路径配置

**文件**：`config/model/recap_value_model.yaml`

```yaml
# 修改为你的模型路径
siglip_path: /path/to/your/siglip2-so400m-patch14-224
gemma3_path: /path/to/your/gemma-3-270m
tokenizer_path: /path/to/your/gemma-3-270m

# 如果替换为其他模型
# siglip_path: /path/to/your/clip-vit-large-patch14
# gemma3_path: /path/to/your/qwen2.5-0.5b
```

#### 2. 数据配置

**文件**：`config/recap_value_model_sft.yaml`

```yaml
data:
  train_data_paths:
    - dataset_path: "/path/to/your/dataset"
      type: "sft"
      weight: 1.0
      robot_type: "your_robot"  # 你的机器人类型
      model_type: "pi05"
  
  eval_data_paths:
    - dataset_path: "/path/to/your/eval_dataset"
      max_samples: 10000
      robot_type: "your_robot"
      model_type: "pi05"
  
  # 根据你的机器人调整
  action_dim: 7                # 你的动作维度
  action_horizon: 10           # 动作预测步数
  robot_type: "your_robot"
  model_type: "pi05"
```

#### 3. 训练配置

**文件**：`config/recap_value_model_sft.yaml`

```yaml
actor:
  micro_batch_size: 32         # 根据你的 GPU 显存调整
  global_batch_size: 256       # 根据你的 GPU 数量调整
  
  model:
    precision: bf16            # 或 fp32
    freeze_vlm: false          # 是否冻结 VLM
    value_dropout: 0.0         # Dropout 率
  
  optim:
    lr: 5.0e-5                 # 学习率
    value_lr: 1.0e-4           # Value Head 学习率
    weight_decay: 1.0e-10
    lr_warmup_steps: 500
```

---

### 四、完整调整清单

#### 必须调整（不改无法运行）

| 序号 | 文件 | 内容 | 说明 |
|---|---|---|---|
| 1 | `datasets/recap/value_dataset.py` | `_REPACK_KEYS` | 添加你的机器人类型 |
| 2 | `config/model/recap_value_model.yaml` | `siglip_path`, `gemma3_path` | 模型路径 |
| 3 | `config/recap_value_model_sft.yaml` | `data.train_data_paths` | 数据路径 |
| 4 | `config/recap_value_model_sft.yaml` | `data.robot_type` | 机器人类型 |
| 5 | `config/recap_value_model_sft.yaml` | `data.action_dim` | 动作维度 |

#### 可选调整（根据需求）

| 序号 | 文件 | 内容 | 说明 |
|---|---|---|---|
| 6 | `models/recap/value_expert.py` | 视觉编码器 | 如果要换模型 |
| 7 | `models/recap/value_expert.py` | 语言模型 | 如果要换模型 |
| 8 | `models/recap/configuration.py` | Expert 配置 | 如果要换模型 |
| 9 | `config/model/recap_value_model.yaml` | `critic_expert_variant` | Expert 大小 |
| 10 | `config/recap_value_model_sft.yaml` | 训练参数 | 根据需求调整 |

---

### 五、快速适配步骤

#### 步骤 1：确认你的数据格式

```python
import pandas as pd
df = pd.read_parquet("your_dataset/data/episode_000000.parquet")
print(df.columns.tolist())
```

#### 步骤 2：添加机器人类型

在 `datasets/recap/value_dataset.py` 中添加：

```python
_REPACK_KEYS = {
    # ... 已有类型 ...
    "your_robot": {
        "observation/image": "你的图像字段",
        "observation/state": "你的状态字段",
        "actions": "你的动作字段",
        "prompt": "你的任务描述字段",
    },
}
```

#### 步骤 3：下载预训练模型

```bash
# 如果用 SigLIP2 + Gemma3
huggingface-cli download google/siglip2-so400m-patch14-224 --local-dir ./models/siglip2
huggingface-cli download google/gemma-3-270m --local-dir ./models/gemma3

# 如果用其他模型
huggingface-cli download openai/clip-vit-large-patch14 --local-dir ./models/clip
huggingface-cli download Qwen/Qwen2.5-0.5B --local-dir ./models/qwen25
```

#### 步骤 4：修改配置文件

修改 `config/model/recap_value_model.yaml` 和 `config/recap_value_model_sft.yaml`。

#### 步骤 5：运行训练

```bash
bash examples/run_value_sft.sh recap_value_model_sft
```

---

## 模型大小与推理速度

### 参数量 vs 推理速度

| 模型 | 参数量 | 相对速度 | 适用场景 |
|---|---|---|---|
| gemma_1m | ~1.2M | 100x | 调试、快速验证 |
| gemma_50m | ~56M | 10x | 实验、小规模数据 |
| gemma_100m | ~110M | 5x | 默认选择 |
| gemma_300m | ~311M | 1x | 平衡性能和速度 |
| gemma_2b | ~1.98B | 0.2x | 追求最高精度 |

### 建议

1. **开发阶段**：用 `gemma_1m` 快速迭代
2. **训练阶段**：用 `gemma_100m` 或 `gemma_300m`
3. **部署阶段**：根据延迟要求选择，或使用量化

---

## 参考

- [RLinf 官方仓库](https://github.com/RLinf/RLinf)
- [RECAP 文档](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/recap.html)
- [OpenPI](https://github.com/Physical-Intelligence/openpi)

## 许可证

本代码基于 RLinf 框架，遵循 Apache License 2.0。
