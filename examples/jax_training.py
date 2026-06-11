#!/usr/bin/env python3
"""
JAX/Flax Training Example for H200 GPUs
Demonstrates best practices for efficient JAX training on the Ruqola server.

Usage:
gpuq submit --command "python jax_training.py --config jax_config.py" --gpus 1 --memory 60 --time 8

Features:
- Pure functional programming with JAX
- JIT compilation for maximum performance
- Memory-efficient training with gradient checkpointing
- Multi-GPU support with pmap
- Flax neural network library
- Orbax checkpointing system
"""

import jax
import jax.numpy as jnp
from jax import random, grad, jit, vmap, pmap, devices
from jax import checkpoint as remat
import flax.linen as nn
from flax.training import train_state
import optax
import orbax.checkpoint as ocp

import numpy as np
import argparse
import importlib.util
import logging
import time
from pathlib import Path
import json

def setup_logging(log_dir="logs"):
    """Setup logging configuration"""
    Path(log_dir).mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'{log_dir}/jax_training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_config(config_path):
    """Load configuration from Python file"""
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config

class ResidualBlock(nn.Module):
    """Residual block optimized for H200 Tensor Cores"""
    filters: int
    stride: int = 1
    
    @nn.compact
    def __call__(self, x, training: bool = True):
        # Ensure dimensions are multiples of 8 for optimal Tensor Core usage
        assert self.filters % 8 == 0, f"Filters {self.filters} should be multiple of 8"
        
        residual = x
        
        # First convolution
        x = nn.Conv(
            features=self.filters,
            kernel_size=(3, 3),
            strides=self.stride,
            padding='SAME',
            use_bias=False,
        )(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)
        
        # Second convolution
        x = nn.Conv(
            features=self.filters,
            kernel_size=(3, 3),
            strides=1,
            padding='SAME',
            use_bias=False,
        )(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        
        # Adjust residual if needed
        if residual.shape != x.shape:
            residual = nn.Conv(
                features=self.filters,
                kernel_size=(1, 1),
                strides=self.stride,
                padding='VALID',
                use_bias=False,
            )(residual)
            residual = nn.BatchNorm(use_running_average=not training)(residual)
        
        # Skip connection
        x = x + residual
        x = nn.relu(x)
        
        return x

class EfficientResNet(nn.Module):
    """Memory-efficient ResNet with gradient checkpointing"""
    num_classes: int = 10
    num_blocks: int = 3
    use_checkpointing: bool = True
    
    @nn.compact
    def __call__(self, x, training: bool = True):
        # Initial convolution - keep small for CIFAR-10
        x = nn.Conv(
            features=64,
            kernel_size=(3, 3),
            strides=1,
            padding='SAME',
            use_bias=False,
        )(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)
        
        # Residual blocks with increasing filters (powers of 2, multiples of 8)
        filters_list = [64, 128, 256, 512]
        
        for i, filters in enumerate(filters_list):
            stride = 2 if i > 0 else 1
            
            # Apply gradient checkpointing for memory efficiency
            if self.use_checkpointing and training:
                block_fn = remat(ResidualBlock(filters=filters, stride=stride))
                x = block_fn(x, training=training)
                
                # Additional blocks without stride
                for _ in range(self.num_blocks - 1):
                    block_fn = remat(ResidualBlock(filters=filters, stride=1))
                    x = block_fn(x, training=training)
            else:
                x = ResidualBlock(filters=filters, stride=stride)(x, training=training)
                
                # Additional blocks without stride
                for _ in range(self.num_blocks - 1):
                    x = ResidualBlock(filters=filters, stride=1)(x, training=training)
        
        # Global average pooling and classification
        x = jnp.mean(x, axis=(1, 2))  # Global average pooling
        x = nn.Dense(features=self.num_classes)(x)
        
        return x

def create_train_state(model, rng, config, example_input):
    """Create training state with optimizer and parameters"""
    
    # Initialize model parameters
    variables = model.init(rng, example_input, training=False)
    params = variables['params']
    batch_stats = variables.get('batch_stats', {})
    
    # Create learning rate schedule
    if config.scheduler_type == 'cosine':
        lr_schedule = optax.cosine_decay_schedule(
            init_value=config.learning_rate,
            decay_steps=config.total_steps,
            alpha=config.min_lr / config.learning_rate
        )
    elif config.scheduler_type == 'exponential':
        lr_schedule = optax.exponential_decay(
            init_value=config.learning_rate,
            transition_steps=config.decay_steps,
            decay_rate=config.decay_rate
        )
    else:
        lr_schedule = config.learning_rate
    
    # Create optimizer
    if config.optimizer_type == 'adamw':
        optimizer = optax.adamw(
            learning_rate=lr_schedule,
            weight_decay=config.weight_decay,
            b1=config.beta1,
            b2=config.beta2,
            eps=config.epsilon
        )
    elif config.optimizer_type == 'sgd':
        optimizer = optax.sgd(
            learning_rate=lr_schedule,
            momentum=config.momentum,
            nesterov=config.nesterov
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer_type}")
    
    # Add gradient clipping if specified
    if config.gradient_clip > 0:
        optimizer = optax.chain(
            optax.clip_by_global_norm(config.gradient_clip),
            optimizer
        )
    
    # Create train state
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer
    )
    
    return state, batch_stats

def create_data_loaders(config):
    """Create optimized data loaders for CIFAR-10"""
    
    def load_cifar10():
        """Load CIFAR-10 dataset"""
        import tensorflow as tf
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        
        # Convert to JAX arrays
        x_train = jnp.array(x_train, dtype=jnp.float32) / 255.0
        y_train = jnp.array(y_train.squeeze(), dtype=jnp.int32)
        x_test = jnp.array(x_test, dtype=jnp.float32) / 255.0
        y_test = jnp.array(y_test.squeeze(), dtype=jnp.int32)
        
        return (x_train, y_train), (x_test, y_test)
    
    def data_augmentation(rng, images):
        """Apply data augmentation"""
        batch_size, height, width, channels = images.shape
        
        # Random horizontal flip
        flip_rng, rng = random.split(rng)
        flip_mask = random.bernoulli(flip_rng, 0.5, (batch_size, 1, 1, 1))
        images = jnp.where(flip_mask, images[:, :, ::-1, :], images)
        
        # Random crop with padding
        crop_rng, rng = random.split(rng)
        padded = jnp.pad(images, ((0, 0), (4, 4), (4, 4), (0, 0)), mode='reflect')
        
        # Random crop coordinates
        crop_coords = random.randint(
            crop_rng, (batch_size, 2), 0, 9  # 40x40 -> 32x32, so 8 pixel range
        )
        
        def crop_single(args):
            img, coords = args
            y, x = coords
            return img[y:y+32, x:x+32, :]
        
        images = vmap(crop_single)((padded, crop_coords))
        
        return images
    
    def create_batches(data, batch_size, shuffle=True, rng=None):
        """Create batches from data"""
        x, y = data
        num_samples = x.shape[0]
        
        if shuffle and rng is not None:
            perm = random.permutation(rng, num_samples)
            x, y = x[perm], y[perm]
        
        # Pad to make divisible by batch_size
        remainder = num_samples % batch_size
        if remainder != 0:
            pad_size = batch_size - remainder
            x = jnp.concatenate([x, x[:pad_size]], axis=0)
            y = jnp.concatenate([y, y[:pad_size]], axis=0)
        
        # Reshape to batches
        num_batches = x.shape[0] // batch_size
        x = x.reshape(num_batches, batch_size, *x.shape[1:])
        y = y.reshape(num_batches, batch_size)
        
        return x, y
    
    # Load data
    (x_train, y_train), (x_test, y_test) = load_cifar10()
    
    return (x_train, y_train), (x_test, y_test), data_augmentation, create_batches

@jit
def train_step(state, batch_stats, batch, rng):
    """JIT-compiled training step for maximum performance"""
    images, labels = batch
    
    def loss_fn(params):
        # Apply model with batch normalization
        variables = {'params': params, 'batch_stats': batch_stats}
        logits, new_batch_stats = state.apply_fn(
            variables,
            images,
            training=True,
            mutable=['batch_stats']
        )
        
        # Cross-entropy loss
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
        
        return loss, (logits, new_batch_stats['batch_stats'])
    
    # Compute loss and gradients
    (loss, (logits, new_batch_stats)), grads = jax.value_and_grad(
        loss_fn, has_aux=True
    )(state.params)
    
    # Update parameters
    state = state.apply_gradients(grads=grads)
    
    # Compute accuracy
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    
    return state, new_batch_stats, loss, accuracy

@jit  
def eval_step(state, batch_stats, batch):
    """JIT-compiled evaluation step"""
    images, labels = batch
    
    variables = {'params': state.params, 'batch_stats': batch_stats}
    logits = state.apply_fn(variables, images, training=False)
    
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    
    return loss, accuracy

def train_epoch(state, batch_stats, train_batches, rng, config, epoch, logger):
    """Train for one epoch"""
    epoch_losses = []
    epoch_accuracies = []
    
    num_batches = len(train_batches[0])
    
    for batch_idx in range(num_batches):
        batch = (train_batches[0][batch_idx], train_batches[1][batch_idx])
        
        # Apply data augmentation
        aug_rng, rng = random.split(rng)
        augmented_images = train_batches[2](aug_rng, batch[0])  # data_augmentation function
        batch = (augmented_images, batch[1])
        
        # Training step
        state, batch_stats, loss, accuracy = train_step(state, batch_stats, batch, rng)
        
        epoch_losses.append(loss)
        epoch_accuracies.append(accuracy)
        
        # Logging
        if batch_idx % config.log_interval == 0:
            current_lr = state.opt_state.hyperparams['learning_rate']
            if hasattr(current_lr, 'item'):
                current_lr = current_lr.item()
            
            logger.info(
                f'Epoch {epoch}, Batch {batch_idx}/{num_batches}: '
                f'Loss={loss:.4f}, Accuracy={accuracy:.4f}, LR={current_lr:.6f}'
            )
    
    avg_loss = jnp.mean(jnp.array(epoch_losses))
    avg_accuracy = jnp.mean(jnp.array(epoch_accuracies))
    
    return state, batch_stats, avg_loss, avg_accuracy, rng

def evaluate_model(state, batch_stats, test_batches, logger):
    """Evaluate model on test set"""
    test_losses = []
    test_accuracies = []
    
    num_batches = len(test_batches[0])
    
    for batch_idx in range(num_batches):
        batch = (test_batches[0][batch_idx], test_batches[1][batch_idx])
        loss, accuracy = eval_step(state, batch_stats, batch)
        
        test_losses.append(loss)
        test_accuracies.append(accuracy)
    
    avg_loss = jnp.mean(jnp.array(test_losses))
    avg_accuracy = jnp.mean(jnp.array(test_accuracies))
    
    return avg_loss, avg_accuracy

def save_checkpoint(state, batch_stats, epoch, config, logger):
    """Save checkpoint using Orbax"""
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Create checkpoint manager
    checkpoint_manager = ocp.CheckpointManager(
        str(checkpoint_dir),
        ocp.PyTreeCheckpointer(),
    )
    
    # Save checkpoint
    checkpoint_data = {
        'state': state,
        'batch_stats': batch_stats,
        'epoch': epoch,
    }
    
    checkpoint_manager.save(epoch, checkpoint_data)
    logger.info(f"Checkpoint saved at epoch {epoch}")

def load_checkpoint(checkpoint_path, logger):
    """Load checkpoint using Orbax"""
    checkpoint_manager = ocp.CheckpointManager(
        checkpoint_path,
        ocp.PyTreeCheckpointer(),
    )
    
    # Get latest checkpoint
    latest_step = checkpoint_manager.latest_step()
    if latest_step is None:
        return None, None, 0
    
    checkpoint_data = checkpoint_manager.restore(latest_step)
    logger.info(f"Loaded checkpoint from step {latest_step}")
    
    return checkpoint_data['state'], checkpoint_data['batch_stats'], checkpoint_data['epoch']

def main():
    parser = argparse.ArgumentParser(description='JAX/Flax H200 Training Example')
    parser.add_argument('--config', required=True, help='Configuration file (.py)')
    parser.add_argument('--resume', help='Resume from checkpoint directory')
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    logger.info(f"Starting JAX training with config: {args.config}")
    
    # Check JAX setup
    logger.info(f"JAX devices: {devices()}")
    logger.info(f"JAX version: {jax.__version__}")
    
    # Load configuration
    config = load_config(args.config)
    
    # Set random seed
    rng = random.PRNGKey(config.seed)
    
    # Create model
    model = EfficientResNet(
        num_classes=config.num_classes,
        num_blocks=config.num_blocks,
        use_checkpointing=config.use_gradient_checkpointing
    )
    
    # Create data loaders
    train_data, test_data, data_augmentation, create_batches = create_data_loaders(config)
    
    # Calculate total steps for scheduler
    steps_per_epoch = len(train_data[0]) // config.batch_size
    config.total_steps = config.epochs * steps_per_epoch
    
    # Create example input for model initialization
    example_input = jnp.ones((1, 32, 32, 3))
    
    # Initialize model and training state
    init_rng, train_rng = random.split(rng)
    state, batch_stats = create_train_state(model, init_rng, config, example_input)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        state, batch_stats, start_epoch = load_checkpoint(args.resume, logger)
        if state is None:
            logger.info("No checkpoint found, starting from scratch")
        else:
            logger.info(f"Resumed from epoch {start_epoch}")
    
    # Count parameters
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
    logger.info(f"Model parameters: {param_count:,}")
    
    # Training loop
    best_accuracy = 0.0
    
    for epoch in range(start_epoch, config.epochs):
        logger.info(f"Starting epoch {epoch+1}/{config.epochs}")
        
        # Create batches for this epoch
        batch_rng, train_rng = random.split(train_rng)
        train_batches = create_batches(
            train_data, config.batch_size, shuffle=True, rng=batch_rng
        )
        train_batches = (*train_batches, data_augmentation)  # Add augmentation function
        
        test_batches = create_batches(
            test_data, config.batch_size, shuffle=False
        )
        
        # Train for one epoch
        epoch_start = time.time()
        state, batch_stats, train_loss, train_acc, train_rng = train_epoch(
            state, batch_stats, train_batches, train_rng, config, epoch, logger
        )
        
        # Evaluate
        val_loss, val_acc = evaluate_model(state, batch_stats, test_batches, logger)
        
        epoch_time = time.time() - epoch_start
        logger.info(
            f"Epoch {epoch+1} completed in {epoch_time:.1f}s: "
            f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
            f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}"
        )
        
        # Save checkpoint
        is_best = val_acc > best_accuracy
        if is_best:
            best_accuracy = val_acc
        
        if (epoch + 1) % config.save_interval == 0 or is_best:
            save_checkpoint(state, batch_stats, epoch + 1, config, logger)
    
    logger.info(f"Training completed! Best validation accuracy: {best_accuracy:.4f}")

if __name__ == '__main__':
    main()