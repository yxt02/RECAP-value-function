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

## 自定义机器人

如果使用自研机器人，需要在 `datasets/recap/value_dataset.py` 中添加机器人类型：

```python
_REPACK_KEYS = {
    # 已有类型...
    "your_robot": {
        "observation/image": "your_image_field",
        "observation/state": "your_state_field",
        "actions": "your_action_field",
        "prompt": "your_prompt_field",
    },
}
```

## 参考

- [RLinf 官方仓库](https://github.com/RLinf/RLinf)
- [RECAP 文档](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/recap.html)
- [OpenPI](https://github.com/Physical-Intelligence/openpi)

## 许可证

本代码基于 RLinf 框架，遵循 Apache License 2.0。
