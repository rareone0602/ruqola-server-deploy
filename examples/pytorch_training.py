#!/usr/bin/env python3
"""
PyTorch Training Example for H200 GPUs
Demonstrates best practices for efficient training on the Ruqola server.

Usage:
gpuq submit --command "python pytorch_training.py --config resnet_config.yaml" --gpus 1 --memory 40 --time 8

Features:
- Mixed precision training with AMP
- Gradient checkpointing for memory efficiency
- Optimal data loading for H200
- Comprehensive logging and checkpointing
- Multi-GPU support with DDP
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50

import yaml
import argparse
import logging
import os
import time
from pathlib import Path
import wandb  # Optional: for experiment tracking

def setup_logging(log_dir="logs"):
    """Setup comprehensive logging"""
    Path(log_dir).mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'{log_dir}/training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def setup_distributed():
    """Initialize distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
        
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(gpu)
        
        return rank, world_size, gpu
    else:
        return 0, 1, 0  # Single GPU training

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

class OptimizedDataset:
    """Optimized dataset wrapper for H200"""
    
    @staticmethod
    def create_cifar10_loaders(config, rank=0, world_size=1):
        """Create CIFAR-10 data loaders optimized for H200"""
        
        # Data augmentation
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        # Datasets
        train_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=train_transform
        )
        
        test_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=test_transform
        )
        
        # Distributed samplers
        if world_size > 1:
            train_sampler = DistributedSampler(train_dataset, rank=rank, shuffle=True)
            test_sampler = DistributedSampler(test_dataset, rank=rank, shuffle=False)
        else:
            train_sampler = None
            test_sampler = None
        
        # Optimized data loaders for H200
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=config['data']['num_workers'],
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
            drop_last=True,
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            sampler=test_sampler,
            num_workers=config['data']['num_workers'],
            pin_memory=True,
            persistent_workers=True,
        )
        
        return train_loader, test_loader, train_sampler

class MemoryEfficientResNet(nn.Module):
    """ResNet with gradient checkpointing for memory efficiency"""
    
    def __init__(self, num_classes=10, pretrained=True):
        super().__init__()
        self.backbone = resnet50(pretrained=pretrained)
        
        # Modify for CIFAR-10
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity()
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        
        # Enable gradient checkpointing for memory efficiency
        self.gradient_checkpointing = True
    
    def forward(self, x):
        if self.gradient_checkpointing and self.training:
            # Apply checkpointing to backbone layers
            x = torch.utils.checkpoint.checkpoint(self._forward_backbone, x, use_reentrant=False)
        else:
            x = self._forward_backbone(x)
        return x
    
    def _forward_backbone(self, x):
        return self.backbone(x)

def create_optimizer(model, config):
    """Create optimizer with proper settings"""
    if config['optimizer']['type'] == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['optimizer']['lr'],
            weight_decay=config['optimizer']['weight_decay'],
            eps=1e-8,
        )
    elif config['optimizer']['type'] == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=config['optimizer']['lr'],
            momentum=config['optimizer']['momentum'],
            weight_decay=config['optimizer']['weight_decay'],
        )
    else:
        raise ValueError(f"Unknown optimizer: {config['optimizer']['type']}")
    
    return optimizer

def create_scheduler(optimizer, config, steps_per_epoch):
    """Create learning rate scheduler"""
    if config['scheduler']['type'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['epochs'] * steps_per_epoch,
            eta_min=config['scheduler']['min_lr']
        )
    elif config['scheduler']['type'] == 'step':
        scheduler = optix.lr_scheduler.StepLR(
            optimizer,
            step_size=config['scheduler']['step_size'] * steps_per_epoch,
            gamma=config['scheduler']['gamma']
        )
    else:
        scheduler = None
    
    return scheduler

def train_epoch(model, train_loader, optimizer, scheduler, criterion, scaler, epoch, config, logger, rank=0):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    start_time = time.time()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        # Move to GPU with non-blocking transfer
        data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
        
        optimizer.zero_grad()
        
        # Mixed precision forward pass
        with autocast(enabled=config['training']['mixed_precision']):
            output = model(data)
            loss = criterion(output, target)
        
        # Mixed precision backward pass
        if config['training']['mixed_precision']:
            scaler.scale(loss).backward()
            
            # Gradient clipping
            if config['training']['gradient_clip'] > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip'])
            
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            
            # Gradient clipping
            if config['training']['gradient_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip'])
            
            optimizer.step()
        
        # Update scheduler
        if scheduler is not None:
            scheduler.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        # Logging
        if batch_idx % config['logging']['log_interval'] == 0 and rank == 0:
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(
                f'Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                f'Loss: {loss.item():.6f}\t'
                f'LR: {current_lr:.6f}'
            )
            
            # Optional: log to wandb
            if config.get('wandb', {}).get('enabled', False):
                wandb.log({
                    'train_loss': loss.item(),
                    'learning_rate': current_lr,
                    'epoch': epoch,
                    'step': epoch * len(train_loader) + batch_idx
                })
    
    # Epoch statistics
    epoch_time = time.time() - start_time
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    if rank == 0:
        logger.info(
            f'Epoch {epoch} Training: Loss={avg_loss:.4f}, '
            f'Accuracy={accuracy:.2f}%, Time={epoch_time:.1f}s'
        )
    
    return avg_loss, accuracy

def validate(model, test_loader, criterion, epoch, config, logger, rank=0):
    """Validate the model"""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            
            with autocast(enabled=config['training']['mixed_precision']):
                output = model(data)
                test_loss += criterion(output, target).item()
            
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    # Average across all processes for distributed training
    if dist.is_initialized():
        test_loss_tensor = torch.tensor(test_loss).cuda()
        correct_tensor = torch.tensor(correct).cuda()
        total_tensor = torch.tensor(total).cuda()
        
        dist.all_reduce(test_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        
        test_loss = test_loss_tensor.item() / dist.get_world_size()
        correct = correct_tensor.item()
        total = total_tensor.item()
    
    avg_loss = test_loss / len(test_loader)
    accuracy = 100. * correct / total
    
    if rank == 0:
        logger.info(f'Epoch {epoch} Validation: Loss={avg_loss:.4f}, Accuracy={accuracy:.2f}%')
        
        # Optional: log to wandb
        if config.get('wandb', {}).get('enabled', False):
            wandb.log({
                'val_loss': avg_loss,
                'val_accuracy': accuracy,
                'epoch': epoch
            })
    
    return avg_loss, accuracy

def save_checkpoint(model, optimizer, scheduler, scaler, epoch, loss, accuracy, config, rank=0):
    """Save training checkpoint"""
    if rank == 0:  # Only save on main process
        checkpoint_dir = Path(config['checkpoint']['dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Get model state dict (handle DDP wrapper)
        if isinstance(model, DDP):
            model_state_dict = model.module.state_dict()
        else:
            model_state_dict = model.state_dict()
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'scaler_state_dict': scaler.state_dict(),
            'loss': loss,
            'accuracy': accuracy,
            'config': config
        }
        
        # Save latest checkpoint
        latest_path = checkpoint_dir / 'latest.pth'
        torch.save(checkpoint, latest_path)
        
        # Save epoch-specific checkpoint
        if epoch % config['checkpoint']['save_interval'] == 0:
            epoch_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
            torch.save(checkpoint, epoch_path)
            
        print(f"Checkpoint saved: {latest_path}")

def load_checkpoint(model, optimizer, scheduler, scaler, checkpoint_path):
    """Load training checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    
    # Handle DDP wrapper
    if isinstance(model, DDP):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss'], checkpoint['accuracy']

def main():
    parser = argparse.ArgumentParser(description='PyTorch H200 Training Example')
    parser.add_argument('--config', required=True, help='Configuration file')
    parser.add_argument('--resume', help='Resume from checkpoint')
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training')
    args = parser.parse_args()
    
    # Setup distributed training
    rank, world_size, gpu = setup_distributed()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging (only on main process)
    if rank == 0:
        logger = setup_logging()
        logger.info(f"Starting training with config: {args.config}")
        logger.info(f"World size: {world_size}, Rank: {rank}, GPU: {gpu}")
        
        # Optional: Initialize wandb
        if config.get('wandb', {}).get('enabled', False):
            wandb.init(
                project=config['wandb']['project'],
                name=config['wandb']['run_name'],
                config=config
            )
    else:
        logger = logging.getLogger()
    
    # Set random seeds for reproducibility
    torch.manual_seed(config['training']['seed'])
    torch.cuda.manual_seed_all(config['training']['seed'])
    
    # Enable optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Create model
    model = MemoryEfficientResNet(
        num_classes=config['model']['num_classes'],
        pretrained=config['model']['pretrained']
    )
    model.cuda(gpu)
    
    # Wrap with DDP for multi-GPU training
    if world_size > 1:
        model = DDP(model, device_ids=[gpu])
    
    # Create data loaders
    train_loader, test_loader, train_sampler = OptimizedDataset.create_cifar10_loaders(
        config, rank, world_size
    )
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config, len(train_loader))
    
    # Loss function and scaler
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=config['training']['mixed_precision'])
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        start_epoch, _, _ = load_checkpoint(model, optimizer, scheduler, scaler, args.resume)
        if rank == 0:
            logger.info(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    best_accuracy = 0.0
    
    for epoch in range(start_epoch, config['training']['epochs']):
        # Set epoch for distributed sampler
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, scaler,
            epoch, config, logger, rank
        )
        
        # Validate
        val_loss, val_acc = validate(model, test_loader, criterion, epoch, config, logger, rank)
        
        # Save checkpoint
        is_best = val_acc > best_accuracy
        if is_best:
            best_accuracy = val_acc
        
        if epoch % config['checkpoint']['save_interval'] == 0 or is_best:
            save_checkpoint(
                model, optimizer, scheduler, scaler, epoch,
                val_loss, val_acc, config, rank
            )
        
        # Print memory usage periodically
        if rank == 0 and epoch % 5 == 0:
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"GPU Memory - Allocated: {allocated:.1f}GB, Reserved: {reserved:.1f}GB")
    
    # Final results
    if rank == 0:
        logger.info(f"Training completed! Best validation accuracy: {best_accuracy:.2f}%")
        
        # Optional: finish wandb run
        if config.get('wandb', {}).get('enabled', False):
            wandb.finish()
    
    # Cleanup distributed training
    if world_size > 1:
        dist.destroy_process_group()

if __name__ == '__main__':
    main()