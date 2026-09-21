#!/usr/bin/env python3
"""
Value Function 训练脚本
支持单GPU训练，使用SigLIP2 + Gemma3架构
"""
import argparse
import gc
import logging
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from transformers import SiglipVisionModel, Gemma3ForCausalLM

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from recap_datasets.recap.simple_dataset import SimpleValueDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ValueModel(nn.Module):
    """简化版Value模型"""
    
    def __init__(self, siglip_path, gemma_path, freeze_vlm=True):
        super().__init__()
        
        # 加载预训练模型
        logger.info(f"加载 SigLIP2: {siglip_path}")
        self.siglip = SiglipVisionModel.from_pretrained(siglip_path)
        siglip_hidden = self.siglip.config.hidden_size  # 1152
        
        logger.info(f"加载 Gemma3: {gemma_path}")
        self.gemma = Gemma3ForCausalLM.from_pretrained(gemma_path)
        gemma_hidden = self.gemma.config.hidden_size  # 640
        
        # 投影层
        self.projection = nn.Linear(siglip_hidden, gemma_hidden)
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(gemma_hidden, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()  # 输出范围 [-1, 1]
        )
        
        # 冻结VLM
        if freeze_vlm:
            logger.info("冻结 VLM 参数")
            for param in self.siglip.parameters():
                param.requires_grad = False
            for param in self.gemma.parameters():
                param.requires_grad = False
        
        # 统计参数
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"总参数: {total_params:,}, 可训练: {trainable_params:,}")
    
    def forward(self, images):
        """
        Args:
            images: dict with 'observation.images.cam2' key, shape [B, 3, 224, 224]
        Returns:
            value: shape [B, 1]
        """
        # 获取主视角图像
        img = images['observation.images.cam2']
        
        # SigLIP编码
        with torch.no_grad() if not self.siglip.training else torch.enable_grad():
            img_features = self.siglip(img).last_hidden_state[:, 0, :]  # CLS token
        
        # 投影
        projected = self.projection(img_features)
        
        # Value预测
        value = self.value_head(projected)
        
        return value


def train_epoch(model, dataloader, optimizer, scheduler, device, epoch, max_steps=None):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch_idx, batch in enumerate(dataloader):
        if max_steps and batch_idx >= max_steps:
            logger.info(f"达到最大步数 {max_steps}，停止本epoch")
            break
        
        # 移动数据到设备
        images = {k: v.to(device).float() for k, v in batch['images'].items()}
        target = batch['target_values'].to(device).float().unsqueeze(-1)
        
        # 前向传播
        predicted = model(images)
        
        # 计算损失
        loss = nn.MSELoss()(predicted, target)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx % 10 == 0:
            current_lr = scheduler.get_last_lr()[0]
            logger.info(f"Epoch {epoch}, Step {batch_idx}/{max_steps or len(dataloader)}, Loss: {loss.item():.4f}, LR: {current_lr:.6f}")
        
        # 定期清理显存
        if batch_idx % 50 == 0:
            torch.cuda.empty_cache()
            gc.collect()
    
    return total_loss / max(num_batches, 1)


def validate(model, dataloader, device, max_steps=None):
    """验证"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_steps and batch_idx >= max_steps:
                break
            
            images = {k: v.to(device).float() for k, v in batch['images'].items()}
            target = batch['target_values'].to(device).float().unsqueeze(-1)
            
            predicted = model(images)
            loss = nn.MSELoss()(predicted, target)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / max(num_batches, 1)


def main():
    parser = argparse.ArgumentParser(description="Value Function 训练")
    parser.add_argument('--config', type=str, default='config/train_value.yaml', help='配置文件路径')
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--train_datasets', nargs='+', default=None)
    parser.add_argument('--val_dataset', type=str, default=None)
    parser.add_argument('--tag', type=str, default=None)
    parser.add_argument('--cameras', nargs='+', default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--num_epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--max_steps', type=int, default=None)
    parser.add_argument('--val_steps', type=int, default=None)
    parser.add_argument('--warmup_steps', type=int, default=None)
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--siglip_path', type=str, default=None)
    parser.add_argument('--gemma_path', type=str, default=None)
    parser.add_argument('--freeze_vlm', action='store_true', default=None)
    parser.add_argument('--no_freeze_vlm', dest='freeze_vlm', action='store_false')
    parser.add_argument('--smoke_test', action='store_true', help='运行冒烟测试后退出')
    
    args = parser.parse_args()
    
    # 加载配置文件
    from omegaconf import OmegaConf
    
    default_config = {
        'data_dir': 'data/raw',
        'train_datasets': ['2026.09.15', '2026.09.15_2', '2026.09.16_error'],
        'val_dataset': '2026.09.08',
        'tag': 'z02_fail2000_v1',
        'cameras': ['cam2'],
        'batch_size': 2,
        'num_epochs': 3,
        'lr': 1e-4,
        'max_samples': 500,
        'max_steps': 50,
        'val_steps': 20,
        'save_dir': 'checkpoints',
        'siglip_path': 'models/siglip2-so400m-patch14-224',
        'gemma_path': 'models/gemma-3-270m',
        'freeze_vlm': True,
        'warmup_steps': 20,
    }
    
    # 从配置文件加载
    if args.config:
        cfg = OmegaConf.load(args.config)
        config = OmegaConf.merge(default_config, cfg)
    else:
        config = OmegaConf.create(default_config)
    
    # 命令行参数覆盖配置文件
    for key, value in vars(args).items():
        if key not in ('config', 'smoke_test') and value is not None:
            config[key] = value
    
    args = config
    args.smoke_test = parser.parse_args().smoke_test
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"设备: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 创建保存目录
    save_dir = PROJECT_ROOT / args.save_dir
    save_dir.mkdir(exist_ok=True)
    
    # 加载数据集（限制样本数）
    logger.info("=" * 60)
    logger.info("加载数据集")
    logger.info("=" * 60)
    logger.info(f"每个数据集最大样本数: {args.max_samples}")
    
    train_datasets = []
    for ds_name in args.train_datasets:
        ds_path = PROJECT_ROOT / args.data_dir / ds_name
        if ds_path.exists():
            ds = SimpleValueDataset(
                dataset_path=str(ds_path),
                robot_type='z02',
                tag=args.tag,
                max_samples=args.max_samples,
                cameras=args.cameras,
            )
            train_datasets.append(ds)
            logger.info(f"  {ds_name}: {len(ds)} 样本")
    
    # 合并训练集
    from torch.utils.data import ConcatDataset
    train_dataset = ConcatDataset(train_datasets)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=0,
        pin_memory=False,
    )
    
    # 验证集
    val_path = PROJECT_ROOT / args.data_dir / args.val_dataset
    val_dataset = SimpleValueDataset(
        dataset_path=str(val_path),
        robot_type='z02',
        tag=args.tag,
        max_samples=args.max_samples,
        cameras=args.cameras,
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=0,
        pin_memory=False,
    )
    
    logger.info(f"训练集: {len(train_dataset)} 样本")
    logger.info(f"验证集: {len(val_dataset)} 样本")
    
    # 冒烟测试模式
    if args.smoke_test:
        logger.info("=" * 60)
        logger.info("冒烟测试模式")
        logger.info("=" * 60)
        
        # 测试数据加载
        batch = next(iter(train_loader))
        logger.info(f"✓ 数据加载成功")
        logger.info(f"  图像: {batch['images']['observation.images.cam2'].shape}")
        logger.info(f"  目标值: {batch['target_values'].shape}")
        
        # 测试模型加载
        model = ValueModel(
            siglip_path=str(PROJECT_ROOT / args.siglip_path),
            gemma_path=str(PROJECT_ROOT / args.gemma_path),
            freeze_vlm=args.freeze_vlm,
        ).to(device)
        
        # 测试前向传播
        images = {k: v.to(device).float() for k, v in batch['images'].items()}
        with torch.no_grad():
            output = model(images)
        logger.info(f"✓ 前向传播成功")
        logger.info(f"  输出形状: {output.shape}")
        
        # 清理
        del model, images, output
        torch.cuda.empty_cache()
        gc.collect()
        
        logger.info("=" * 60)
        logger.info("冒烟测试通过！可以开始训练。")
        logger.info("=" * 60)
        return
    
    # 创建模型
    logger.info("=" * 60)
    logger.info("创建模型")
    logger.info("=" * 60)
    
    model = ValueModel(
        siglip_path=str(PROJECT_ROOT / args.siglip_path),
        gemma_path=str(PROJECT_ROOT / args.gemma_path),
        freeze_vlm=args.freeze_vlm,
    ).to(device)
    
    # 优化器
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
    )
    
    # 学习率调度器：warmup + cosine退火
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    
    warmup_steps = args.warmup_steps
    steps_per_epoch = args.max_steps if args.max_steps else len(train_loader)
    total_steps = args.num_epochs * steps_per_epoch
    
    warmup_scheduler = LinearLR(
        optimizer, 
        start_factor=0.01, 
        end_factor=1.0, 
        total_iters=warmup_steps
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=total_steps - warmup_steps,
        eta_min=args.lr * 0.01
    )
    scheduler = SequentialLR(
        optimizer, 
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps]
    )
    
    # 训练
    logger.info("=" * 60)
    logger.info("开始训练")
    logger.info(f"  Epochs: {args.num_epochs}")
    logger.info(f"  每Epoch步数: {args.max_steps}")
    logger.info(f"  验证步数: {args.val_steps}")
    logger.info("=" * 60)
    
    best_val_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, epoch + 1, args.max_steps)
        val_loss = validate(model, val_loader, device, args.val_steps)
        
        logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = save_dir / 'best_model.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, save_path)
            logger.info(f"保存最佳模型: {save_path}")
        
        # 清理显存
        torch.cuda.empty_cache()
        gc.collect()
    
    logger.info("=" * 60)
    logger.info("训练完成！")
    logger.info(f"最佳验证损失: {best_val_loss:.4f}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()