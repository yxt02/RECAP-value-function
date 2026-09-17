#!/usr/bin/env python3
"""
冒烟测试：验证数据加载和模型前向传播
"""

import sys
import os
import logging

# 设置路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_data_loading():
    """测试数据加载"""
    logger.info("=" * 60)
    logger.info("测试 1: 数据加载")
    logger.info("=" * 60)
    
    from recap_datasets.recap.simple_dataset import SimpleValueDataset
    
    # 测试单个数据集
    dataset = SimpleValueDataset(
        dataset_path=os.path.join(project_root, 'data/raw/2026.09.15'),
        robot_type='z02',
        tag='z02_fail2000_v1',
        max_samples=10,
        cameras=['cam2'],
    )
    
    assert len(dataset) == 10, f"Expected 10 samples, got {len(dataset)}"
    
    sample = dataset[0]
    
    # 检查必要的key
    required_keys = ['images', 'observation/state', 'actions', 'prompt', 'target_values']
    for key in required_keys:
        assert key in sample, f"Missing key: {key}"
    
    # 检查图像
    assert 'observation.images.cam2' in sample['images'], "Missing cam2 image"
    img = sample['images']['observation.images.cam2']
    assert img.shape == (3, 224, 224), f"Wrong image shape: {img.shape}"
    
    # 检查状态和动作
    assert sample['observation/state'].shape == (32,), f"Wrong state shape: {sample['observation/state'].shape}"
    assert sample['actions'].shape == (10, 32), f"Wrong actions shape: {sample['actions'].shape}"
    
    # 检查target_values
    assert isinstance(sample['target_values'], float), f"Wrong target_values type: {type(sample['target_values'])}"
    
    logger.info("✓ 数据加载测试通过")
    logger.info(f"  图像 shape: {img.shape}")
    logger.info(f"  状态 shape: {sample['observation/state'].shape}")
    logger.info(f"  动作 shape: {sample['actions'].shape}")
    logger.info(f"  目标值: {sample['target_values']:.4f}")
    
    return True


def test_model_loading():
    """测试模型加载"""
    logger.info("=" * 60)
    logger.info("测试 2: 模型加载")
    logger.info("=" * 60)
    
    from transformers import SiglipVisionModel, Gemma3ForCausalLM
    
    siglip_path = os.path.join(project_root, 'models/siglip2-so400m-patch14-224')
    gemma_path = os.path.join(project_root, 'models/gemma-3-270m')
    
    # 检查模型文件是否存在
    assert os.path.exists(siglip_path), f"SigLIP2 not found: {siglip_path}"
    assert os.path.exists(gemma_path), f"Gemma3 not found: {gemma_path}"
    
    # 加载模型
    logger.info("加载 SigLIP2...")
    siglip = SiglipVisionModel.from_pretrained(siglip_path)
    siglip_hidden = siglip.config.hidden_size
    
    logger.info("加载 Gemma3...")
    gemma = Gemma3ForCausalLM.from_pretrained(gemma_path)
    gemma_hidden = gemma.config.hidden_size
    
    logger.info("✓ 模型加载测试通过")
    logger.info(f"  SigLIP2 hidden_size: {siglip_hidden}")
    logger.info(f"  Gemma3 hidden_size: {gemma_hidden}")
    
    return siglip, gemma, siglip_hidden, gemma_hidden


def test_forward_pass(siglip, gemma, siglip_hidden, gemma_hidden):
    """测试前向传播"""
    import torch
    import torch.nn as nn
    
    logger.info("=" * 60)
    logger.info("测试 3: 前向传播")
    logger.info("=" * 60)
    
    # 创建投影层
    projection = nn.Linear(siglip_hidden, gemma_hidden)
    
    # 模拟输入
    batch_size = 2
    dummy_image = torch.randn(batch_size, 3, 224, 224)
    
    # SigLIP 编码
    logger.info("SigLIP 编码...")
    with torch.no_grad():
        image_features = siglip(dummy_image).last_hidden_state.mean(dim=1)  # Mean patch pooling; SigLIP has no CLS token
    
    # 投影
    logger.info("投影层...")
    projected = projection(image_features)
    
    # Gemma3 编码
    logger.info("Gemma3 编码...")
    with torch.no_grad():
        # 简单测试：将 projected 作为 inputs_embeds
        outputs = gemma(inputs_embeds=projected.unsqueeze(1))
    
    logger.info("✓ 前向传播测试通过")
    logger.info(f"  SigLIP 输出: {image_features.shape}")
    logger.info(f"  投影后: {projected.shape}")
    logger.info(f"  Gemma3 输出: {outputs.logits.shape}")
    
    return True


def test_batch_loading():
    """测试 batch 加载"""
    import torch
    from torch.utils.data import DataLoader
    from recap_datasets.recap.simple_dataset import SimpleValueDataset
    
    logger.info("=" * 60)
    logger.info("测试 4: Batch 加载")
    logger.info("=" * 60)
    
    dataset = SimpleValueDataset(
        dataset_path=os.path.join(project_root, 'data/raw/2026.09.15'),
        robot_type='z02',
        tag='z02_fail2000_v1',
        max_samples=20,
        cameras=['cam2'],
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
    )
    
    batch = next(iter(dataloader))
    
    logger.info("✓ Batch 加载测试通过")
    logger.info(f"  图像 batch shape: {batch['images']['observation.images.cam2'].shape}")
    logger.info(f"  状态 batch shape: {batch['observation/state'].shape}")
    logger.info(f"  动作 batch shape: {batch['actions'].shape}")
    logger.info(f"  目标值 batch shape: {batch['target_values'].shape}")
    logger.info(f"  目标值范围: [{batch['target_values'].min():.4f}, {batch['target_values'].max():.4f}]")
    
    return True


def test_compute_returns():
    """测试 returns 计算"""
    from pathlib import Path
    from process.compute_returns import process_dataset
    
    logger.info("=" * 60)
    logger.info("测试 5: Returns 计算")
    logger.info("=" * 60)
    
    # 检查 returns 文件是否存在（数值/标签正确性由 tests/test_data_pipeline.py 覆盖）
    returns_path = Path(project_root) / 'data/raw/2026.09.15/meta/returns_z02_fail2000_v1.parquet'
    
    if returns_path.exists():
        logger.info(f"✓ Returns 文件已存在: {returns_path}")
        
        import pyarrow.parquet as pq
        table = pq.read_table(str(returns_path))
        df = table.to_pandas()
        
        logger.info(f"  行数: {len(df)}")
        logger.info(f"  列: {list(df.columns)}")
        logger.info(f"  Return 范围: [{df['return'].min():.2f}, {df['return'].max():.2f}]")
        
        return True
    else:
        logger.warning(f"⚠ Returns 文件不存在: {returns_path}")
        logger.info("  请先运行 compute_returns.py")
        return False


def main():
    """主测试函数"""
    logger.info("=" * 60)
    logger.info("RECAP Value Function 冒烟测试")
    logger.info("=" * 60)
    
    results = {}
    
    # 测试 1: 数据加载
    try:
        results['data_loading'] = test_data_loading()
    except Exception as e:
        logger.error(f"✗ 数据加载测试失败: {e}")
        results['data_loading'] = False
    
    # 测试 2: 模型加载
    try:
        siglip, gemma, siglip_hidden, gemma_hidden = test_model_loading()
        results['model_loading'] = True
    except Exception as e:
        logger.error(f"✗ 模型加载测试失败: {e}")
        results['model_loading'] = False
        siglip, gemma, siglip_hidden, gemma_hidden = None, None, None, None
    
    # 测试 3: 前向传播
    if results.get('model_loading'):
        try:
            results['forward_pass'] = test_forward_pass(siglip, gemma, siglip_hidden, gemma_hidden)
        except Exception as e:
            logger.error(f"✗ 前向传播测试失败: {e}")
            results['forward_pass'] = False
    
    # 测试 4: Batch 加载
    try:
        results['batch_loading'] = test_batch_loading()
    except Exception as e:
        logger.error(f"✗ Batch 加载测试失败: {e}")
        results['batch_loading'] = False
    
    # 测试 5: Returns 计算
    try:
        results['compute_returns'] = test_compute_returns()
    except Exception as e:
        logger.error(f"✗ Returns 计算测试失败: {e}")
        results['compute_returns'] = False
    
    # 总结
    logger.info("=" * 60)
    logger.info("测试总结")
    logger.info("=" * 60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✓ 通过" if passed else "✗ 失败"
        logger.info(f"  {test_name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        logger.info("\n✓ 所有测试通过！")
        return 0
    else:
        logger.error("\n✗ 部分测试失败")
        return 1


if __name__ == '__main__':
    sys.exit(main())
