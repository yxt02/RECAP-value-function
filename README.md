# RECAP Value Function - z02 机器人适配版

基于 RLinf 框架的 RECAP 算法中的 Value Function 训练模块，已适配 z02 机器人数据格式。

## 项目概述

本仓库用于训练价值函数 V(s)，预测从当前状态开始的累积回报。价值函数是 RECAP 流程的核心组件，用于后续的 Advantage 计算和策略优化。

### RECAP 流程

```
┌──────────────────────┐     ┌──────────────────────┐     ┌────────────────────────┐     ┌──────────────────────┐
│  Step 1              │     │  Step 2              │     │  Step 3                │     │  Step 4              │
│  Compute Returns     │────▶│  Value Model SFT     │────▶│  Compute Advantages    │────▶│  CFG Training        │
└──────────────────────┘     └──────────────────────┘     └────────────────────────┘     └──────────────────────┘
```

本仓库实现 **Step 1** 和 **Step 2**。

---

## 适配工作

### 1. 机器人类型适配

添加 z02 机器人类型到 `_REPACK_KEYS`：

```python
"z02": {
    "observation/image": "observation.images.cam2",           # 头部摄像头（主视角）
    "observation/wrist_image": "observation.images.cam3",     # 右腕摄像头
    "observation/left_wrist_image": "observation.images.cam4", # 左腕摄像头
    "observation/state": "observation.joint_positions",        # 22维关节位置
    "actions": "action.joint_positions",                       # 22维动作
    "prompt": "prompt",                                        # 任务描述
}
```

### 2. 数据字段映射

| 数据字段 | 维度 | 说明 |
|----------|------|------|
| `observation.joint_positions` | 22 | 左臂7 + 左手二值1 + 右臂7 + 右手二值1 + 升降腰部4 + 头部2 |
| `action.joint_positions` | 22 | 与observation相同结构 |
| `observation.left_finger_positions` | 10 | 左手指位置 |
| `observation.right_finger_positions` | 10 | 右手指位置 |
| `reward` | 1 | 终帧0=成功，-10000=失败 |
| `intervention` | 1 | 0=策略控制，1=人工介入 |

### 3. 成功/失败标签

- **2026.09.08**：纯遥操作数据，全部标记为成功
- **2026.09.15**：9成功，2失败
- **2026.09.15_2**：98成功，14失败

### 4. 绕过外部依赖

由于原版依赖 `rlinf` 和 `openpi`，本仓库实现了简化版数据加载：

- `recap_datasets/recap/simple_dataset.py`：直接读取 Parquet 和视频帧
- `recap_datasets/recap/utils.py`：episode 边界、returns 加载等工具函数
- `recap_datasets/recap/common.py`：基础数据加载器组件

---

## 目录结构

```
RECAP-value-function-main/
├── config/
│   ├── model/
│   │   └── recap_value_model.yaml           # 模型架构配置
│   ├── recap_compute_returns_z02.yaml       # 计算 returns 配置
│   └── recap_value_model_sft_z02.yaml       # 价值模型训练配置
├── data/
│   ├── raw/                                  # 原始数据
│   │   ├── 2026.09.08/                       # 纯遥操作（105 episodes）
│   │   ├── 2026.09.15/                       # 混合控制（11 episodes）
│   │   └── 2026.09.15_2/                     # 混合控制（112 episodes）
│   └── splits/                               # 数据划分
│       ├── train.json                        # 训练集（181 episodes）
│       ├── val.json                          # 验证集（20 episodes）
│       ├── test.json                         # 测试集（27 episodes）
│       └── summary.json                      # 汇总信息
├── models/
│   ├── siglip2-so400m-patch14-224/           # SigLIP2 视觉编码器
│   └── gemma-3-270m/                         # Gemma3 语言模型
├── process/
│   └── compute_returns.py                    # 计算 returns 脚本
├── recap_datasets/
│   └── recap/
│       ├── simple_dataset.py                 # 简化版数据集加载器
│       ├── value_dataset.py                  # 完整版数据集加载器
│       ├── common.py                         # 基础组件
│       └── utils.py                          # 工具函数
└── scripts/
    └── data_split.py                         # 数据划分脚本
```

---

## 快速开始

### 1. 环境配置

```bash
# 创建 conda 环境
conda create -n value_function python=3.11 -y
conda activate value_function

# 安装依赖
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install transformers safetensors omegaconf hydra-core pyarrow pandas tqdm
pip install lerobot==0.3.2
pip install opencv-python-headless av
```

### 2. 下载模型

```bash
# SigLIP2
huggingface-cli download google/siglip2-so400m-patch14-224 --local-dir models/siglip2-so400m-patch14-224

# Gemma3（需要从 ModelScope 下载）
pip install modelscope
python -c "
from modelscope import snapshot_download
snapshot_download('google/gemma-3-270m', cache_dir='models')
"
mv models/models/google--gemma-3-270m/snapshots/master models/gemma-3-270m
rm -rf models/models
```

### 3. 数据准备

```bash
# 解压数据到 data/raw/
unzip 2026.09.08.zip -d data/raw/
unzip 2026.09.15.zip -d data/raw/
unzip 2026.09.15_2.zip -d data/raw/

# 更新 robot_type 为 z02
sed -i 's/"robot_type": "z0"/"robot_type": "z02"/' data/raw/*/meta/info.json
```

### 4. 数据适配（统一脚本）

```bash
# 执行所有步骤（分析 + 计算 returns + 划分数据）
python scripts/adapt_z02.py --all

# 或分步执行
python scripts/adapt_z02.py --analyze          # 分析数据
python scripts/adapt_z02.py --compute_returns  # 计算 returns
python scripts/adapt_z02.py --split_data       # 划分数据
```

**参数配置**（在脚本开头修改）：

```python
CONFIG = {
    # Returns 计算参数
    'gamma': 1.0,                    # 折扣因子（1.0 = 无折扣）
    'failure_reward': -300.0,        # 失败终止奖励（官方推荐 -300）
    'tag': 'z0',                     # Returns 文件标签
    
    # 数据划分参数
    'train_ratio': 0.8,              # 训练集比例
    'val_ratio': 0.1,                # 验证集比例
    'test_ratio': 0.1,               # 测试集比例
    'random_seed': 42,               # 随机种子
}
```

**输出**：

```
数据分析:
  2026.09.08: 105 episodes, 68021 frames, 成功=105, 失败=0
  2026.09.15: 11 episodes, 13432 frames, 成功=9, 失败=2
  2026.09.15_2: 112 episodes, 125782 frames, 成功=98, 失败=14
  总计: 228 episodes, 207235 frames, 成功=212 (93.0%), 失败=16 (7.0%)

数据划分:
  Train: 181 episodes, 164269 frames (成功=169, 失败=12)
  Val: 20 episodes, 18123 frames (成功=19, 失败=1)
  Test: 27 episodes, 24843 frames (成功=24, 失败=3)
```

### 6. 测试数据加载

```bash
python -c "
from recap_datasets.recap.simple_dataset import SimpleValueDataset

dataset = SimpleValueDataset(
    dataset_path='data/raw/2026.09.15',
    robot_type='z02',
    tag='z0',
    max_samples=10,
    cameras=['cam2', 'cam3', 'cam4'],
)

sample = dataset[0]
print(f'images keys: {list(sample[\"images\"].keys())}')
print(f'state shape: {sample[\"observation/state\"].shape}')
print(f'actions shape: {sample[\"actions\"].shape}')
print(f'target_values: {sample[\"target_values\"]:.4f}')
"
```

---

## 数据统计

### 数据来源

| 数据集 | Episodes | 成功 | 失败 | Frames | 说明 |
|--------|----------|------|------|--------|------|
| 2026.09.08 | 105 | 105 | 0 | 68,021 | 纯遥操作 |
| 2026.09.15 | 11 | 9 | 2 | 13,432 | 混合控制 |
| 2026.09.15_2 | 112 | 98 | 14 | 125,782 | 混合控制 |
| **总计** | **228** | **212** | **16** | **207,235** | |

### Intervention 分布（2026.09.15_2）

| 介入程度 | Episodes | 说明 |
|----------|----------|------|
| 无介入 (intervention全0) | 3 | 纯策略控制 |
| 低介入 (<30%) | 55 | 策略为主 |
| 中介入 (30-60%) | 50 | 人机混合 |
| 高介入 (>60%) | 4 | 人工为主 |

### Return 计算

```python
# 每步 reward = -1
# 成功最后一步 reward = 0
# 失败最后一步 reward = -10000

# 成功 episode (100帧): 初始 return = -99
# 失败 episode (100帧): 初始 return = -10099

# 归一化后: return / |return_min| → [-1, 0]
```

---

## 模型架构

### Value Expert

```
输入: 图像 + 语言指令
      ↓
SigLIP2-so400m (视觉编码器, 1152维)
      ↓
投影层 (1152 → 640)
      ↓
Gemma3-270M (语言模型, 640维)
      ↓
Critic Expert (GemmaForCausalLM)
      ↓
Value Head (201-bin 分类分布)
      ↓
输出: V(s) ∈ [-1, 0]
```

### 关键配置

```yaml
# config/model/recap_value_model.yaml
siglip_path: models/siglip2-so400m-patch14-224
gemma3_path: models/gemma-3-270m
critic_expert_variant: "gemma_1m"  # 调试用，正式训练改为 gemma_100m
num_bins: 201
v_min: -1.0
v_max: 0.0
action_dim: 32  # 22维padding到32维
```

---

## 训练配置

### 硬件要求

- GPU: 16GB 显存（RTX 4080/4090）
- 内存: 32GB+
- 磁盘: 500GB+

### 训练参数

```yaml
# config/recap_value_model_sft_z02.yaml
actor:
  micro_batch_size: 4      # 16GB GPU
  global_batch_size: 32    # 单 GPU
  model:
    precision: bf16
    freeze_vlm: true       # 冻结 VLM，只训练投影层和 value head
  optim:
    lr: 5.0e-5
    value_lr: 1.0e-4
```

---

## 已知问题

### 1. 数据不平衡

- 成功:失败 = 14:1
- 解决方案：过采样失败样本或调整采样权重

### 2. 元数据不一致

- 元数据声明 23 维，实际 22 维
- 已通过代码处理 padding

### 3. 外部依赖

- 原版依赖 `rlinf` 和 `openpi`
- 已实现简化版数据加载绕过

---

## 后续工作

### 短期

1. 实现完整的 Value Model 训练脚本
2. 解决数据不平衡问题
3. 验证训练流程

### 中期

1. 计算 Advantages (Step 3)
2. CFG 策略训练 (Step 4)
3. 评估和调优

### 长期

1. 采集更多失败数据
2. 尝试分段 Return 计算
3. 优化 Intervention 利用方式

---

## 参考

- [RLinf 官方仓库](https://github.com/RLinf/RLinf)
- [RECAP 文档](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/recap.html)
- [OpenPI](https://github.com/Physical-Intelligence/openpi)

## 许可证

本代码基于 RLinf 框架，遵循 Apache License 2.0。
