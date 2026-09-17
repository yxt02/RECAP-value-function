# RECAP Value Function - z02 机器人

基于 RECAP 算法的价值函数训练模块，已适配 z02 机器人。

## 快速开始

### 1. 环境配置

```bash
conda create -n value_function python=3.11 -y
conda activate value_function
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install transformers safetensors omegaconf hydra-core pyarrow pandas tqdm lerobot==0.3.2 opencv-python-headless av
```

### 2. 下载模型

```bash
huggingface-cli download google/siglip2-so400m-patch14-224 --local-dir models/siglip2-so400m-patch14-224
pip install modelscope && python -c "from modelscope import snapshot_download; snapshot_download('google/gemma-3-270m', cache_dir='models')"
mv models/models/google--gemma-3-270m/snapshots/master models/gemma-3-270m && rm -rf models/models
```

### 3. 数据处理

```bash
# 解压数据到 data/raw/
unzip 2026.09.08.zip -d data/raw/
unzip 2026.09.15.zip -d data/raw/
unzip 2026.09.15_2.zip -d data/raw/

# 一键处理（分析 + 计算returns + 划分数据）
python scripts/adapt_z02.py --config config/z02_data.yaml --all
```

**分步执行**：
```bash
python scripts/adapt_z02.py --config config/z02_data.yaml --analyze          # 只分析
python scripts/adapt_z02.py --config config/z02_data.yaml --compute_returns  # 只计算returns
python scripts/adapt_z02.py --config config/z02_data.yaml --split_data       # 只划分数据
```

### 4. 训练

```bash
# 冒烟测试（验证环境）
python scripts/train_value.py --smoke_test

# 使用默认配置训练（自动读取 config/train_value.yaml）
python scripts/train_value.py

# 覆盖单个参数
python scripts/train_value.py --num_epochs 10 --max_steps 200
```

## 配置文件

### 训练配置 `config/train_value.yaml`

```yaml
# 数据
data_dir: data/raw
train_datasets: [2026.09.15, 2026.09.15_2, 2026.09.16_error]
val_dataset: 2026.09.08
tag: z02_fail2000_v1
cameras: [cam2]

# 模型
siglip_path: models/siglip2-so400m-patch14-224
gemma_path: models/gemma-3-270m
freeze_vlm: true

# 训练超参数
batch_size: 2
num_epochs: 3
lr: 1.0e-4
max_samples: 500
max_steps: 50
val_steps: 20
save_dir: checkpoints
```

### 数据配置 `config/z02_data.yaml`

```yaml
tag: z02_fail2000_v1
gamma: 1.0
failure_reward: -2000.0
return_scale: 4000.0
max_episode_steps: 2000
train_ratio: 0.8
val_ratio: 0.1
test_ratio: 0.1
cameras: [cam2, cam3, cam4]
```

## 数据统计

| 数据集 | Episodes | 成功 | 失败 | 关节维度 |
|--------|----------|------|------|----------|
| 2026.09.08 | 105 | 105 | 0 | 22 |
| 2026.09.15 | 11 | 9 | 2 | 22 |
| 2026.09.15_2 | 112 | 98 | 14 | 22 |
| 2026.09.16_error | 100 | 0 | 100 | 22 |

## 目录结构

```
├── config/
│   ├── train_value.yaml      # 训练配置
│   └── z02_data.yaml          # 数据配置
├── data/raw/                  # 原始数据
├── data/splits/               # 数据划分
├── models/                    # 预训练模型
├── process/                   # returns计算
├── recap_datasets/            # 数据加载
├── scripts/
│   ├── adapt_z02.py           # 数据适配脚本
│   └── train_value.py         # 训练脚本
└── checkpoints/               # 模型保存
```