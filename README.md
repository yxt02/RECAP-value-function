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
python scripts/smoke_test.py  # 冒烟测试
```

## 配置参数

编辑 `config/z02_data.yaml`：

```yaml
tag: z02_fail2000_v1        # returns标签
gamma: 1.0                  # 折扣因子
failure_reward: -2000.0     # 失败惩罚
return_scale: 4000.0        # return归一化范围
max_episode_steps: 2000     # 最大步数
train_ratio: 0.8            # 训练集比例
val_ratio: 0.1              # 验证集比例
test_ratio: 0.1             # 测试集比例
cameras: [cam2, cam3, cam4] # 摄像头
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
├── config/                 # 配置文件
├── data/raw/               # 原始数据
├── data/splits/            # 数据划分
├── models/                 # 预训练模型
├── process/                # returns计算
├── recap_datasets/         # 数据加载
└── scripts/
    ├── adapt_z02.py        # 数据适配脚本
    └── smoke_test.py       # 冒烟测试
```