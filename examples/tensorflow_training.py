#!/usr/bin/env python3
"""
TensorFlow Training Example for H200 GPUs
Demonstrates best practices for efficient TensorFlow training on the Ruqola server.

Usage:
gpuq submit --command "python tensorflow_training.py --config tf_config.json" --gpus 1 --memory 60 --time 8

Features:
- Mixed precision training (FP16)
- XLA compilation for optimization
- Multi-GPU support with MirroredStrategy
- Optimized data pipeline
- Comprehensive monitoring and checkpointing
- TensorBoard logging
"""

import tensorflow as tf
import json
import argparse
import logging
import os
import time
from pathlib import Path
import numpy as np

def setup_logging(log_dir="logs"):
    """Setup logging configuration"""
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

def setup_gpu(config):
    """Configure GPU settings optimized for H200"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    
    if gpus:
        try:
            # A hard memory cap and memory growth are mutually exclusive on the
            # same GPU, so pick one based on whether a limit is configured.
            if config.get('gpu_memory_limit'):
                # Cap memory via a logical device configuration (in MB)
                tf.config.set_logical_device_configuration(
                    gpus[0],
                    [tf.config.LogicalDeviceConfiguration(
                        memory_limit=config['gpu_memory_limit'] * 1024
                    )]
                )
                logging.info(
                    f"Configured GPU:0 with a {config['gpu_memory_limit']}GB memory limit"
                )
            else:
                # Enable memory growth to avoid allocating all GPU memory at once
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logging.info(f"Configured {len(gpus)} GPUs with memory growth enabled")
            
            # Enable mixed precision if specified
            if config.get('mixed_precision', True):
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                logging.info("Mixed precision (FP16) enabled")
            
            # Enable XLA compilation
            if config.get('xla_compile', True):
                tf.config.optimizer.set_jit(True)
                logging.info("XLA compilation enabled")
                
        except RuntimeError as e:
            logging.error(f"GPU setup error: {e}")
    else:
        logging.warning("No GPUs detected!")

def load_config(config_path):
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        return json.load(f)

def create_optimized_dataset(config):
    """Create optimized TensorFlow data pipeline for H200"""
    
    def preprocess_image(image, label):
        """Preprocess images with data augmentation"""
        # Convert to float32
        image = tf.cast(image, tf.float32)
        
        # Normalize to [0, 1]
        image = image / 255.0
        
        # Data augmentation for training
        if config['data']['augmentation']:
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_brightness(image, max_delta=0.1)
            image = tf.image.random_contrast(image, 0.9, 1.1)
            
            # Random crop and resize
            image = tf.image.resize_with_crop_or_pad(image, 40, 40)
            image = tf.image.random_crop(image, [32, 32, 3])
        
        return image, label
    
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    
    # Convert to tf.data.Dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    
    # Optimize training dataset
    train_dataset = train_dataset.shuffle(
        buffer_size=config['data']['shuffle_buffer'],
        reshuffle_each_iteration=True
    )
    train_dataset = train_dataset.map(
        preprocess_image, 
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False
    )
    train_dataset = train_dataset.batch(
        config['training']['batch_size'],
        drop_remainder=True
    )
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    
    # Optimize test dataset
    test_dataset = test_dataset.map(
        lambda x, y: (tf.cast(x, tf.float32) / 255.0, y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    test_dataset = test_dataset.batch(config['training']['batch_size'])
    test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)
    
    return train_dataset, test_dataset

def create_efficient_model(config):
    """Create memory-efficient model optimized for H200"""
    
    # Use functional API for more control
    inputs = tf.keras.layers.Input(shape=(32, 32, 3))
    
    # Initial convolution
    x = tf.keras.layers.Conv2D(
        64, (3, 3), 
        padding='same',
        kernel_initializer='he_normal'
    )(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    
    # Residual blocks with optimal dimensions for H200 Tensor Cores
    def residual_block(x, filters, stride=1):
        shortcut = x
        
        # First conv layer
        x = tf.keras.layers.Conv2D(
            filters, (3, 3), 
            strides=stride, 
            padding='same',
            kernel_initializer='he_normal'
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        
        # Second conv layer
        x = tf.keras.layers.Conv2D(
            filters, (3, 3), 
            padding='same',
            kernel_initializer='he_normal'
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        
        # Shortcut connection
        if stride != 1 or shortcut.shape[-1] != filters:
            shortcut = tf.keras.layers.Conv2D(
                filters, (1, 1), 
                strides=stride,
                kernel_initializer='he_normal'
            )(shortcut)
            shortcut = tf.keras.layers.BatchNormalization()(shortcut)
        
        x = tf.keras.layers.Add()([x, shortcut])
        x = tf.keras.layers.Activation('relu')(x)
        
        return x
    
    # Build residual blocks
    filters_list = [64, 128, 256, 512]
    for i, filters in enumerate(filters_list):
        stride = 2 if i > 0 else 1
        x = residual_block(x, filters, stride)
        x = residual_block(x, filters, 1)
    
    # Global average pooling and classification
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(config['model']['dropout_rate'])(x)
    
    # Output layer (ensure float32 for numerical stability with mixed precision)
    outputs = tf.keras.layers.Dense(
        config['model']['num_classes'],
        activation='softmax',
        dtype='float32',  # Keep output in FP32
        kernel_initializer='he_normal'
    )(x)
    
    model = tf.keras.Model(inputs, outputs)
    
    return model

def create_custom_optimizer(config):
    """Create optimizer with optimal settings for H200"""
    optimizer_config = config['optimizer']
    
    # Learning rate schedule
    if config['scheduler']['type'] == 'cosine':
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=optimizer_config['lr'],
            decay_steps=config['training']['epochs'] * config['steps_per_epoch'],
            alpha=config['scheduler']['min_lr'] / optimizer_config['lr']
        )
    elif config['scheduler']['type'] == 'exponential':
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=optimizer_config['lr'],
            decay_steps=config['scheduler']['decay_steps'],
            decay_rate=config['scheduler']['decay_rate']
        )
    else:
        lr_schedule = optimizer_config['lr']
    
    # Create optimizer
    if optimizer_config['type'] == 'adamw':
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=optimizer_config['weight_decay'],
            beta_1=optimizer_config.get('beta_1', 0.9),
            beta_2=optimizer_config.get('beta_2', 0.999),
            epsilon=optimizer_config.get('epsilon', 1e-7)
        )
    elif optimizer_config['type'] == 'adam':
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr_schedule,
            beta_1=optimizer_config.get('beta_1', 0.9),
            beta_2=optimizer_config.get('beta_2', 0.999),
            epsilon=optimizer_config.get('epsilon', 1e-7)
        )
    elif optimizer_config['type'] == 'sgd':
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=lr_schedule,
            momentum=optimizer_config.get('momentum', 0.9),
            nesterov=optimizer_config.get('nesterov', True)
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_config['type']}")
    
    # Wrap with mixed precision loss scaling
    if config.get('mixed_precision', True):
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
    
    return optimizer

@tf.function(jit_compile=True)  # Enable XLA compilation
def train_step(model, x, y, optimizer, loss_fn, train_accuracy):
    """Optimized training step with XLA compilation"""
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = loss_fn(y, predictions)
        
        # Scale loss for mixed precision
        if hasattr(optimizer, 'get_scaled_loss'):
            scaled_loss = optimizer.get_scaled_loss(loss)
        else:
            scaled_loss = loss
    
    # Compute gradients
    if hasattr(optimizer, 'get_scaled_loss'):
        scaled_gradients = tape.gradient(scaled_loss, model.trainable_variables)
        gradients = optimizer.get_unscaled_gradients(scaled_gradients)
    else:
        gradients = tape.gradient(scaled_loss, model.trainable_variables)
    
    # Apply gradients
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    # Update metrics
    train_accuracy.update_state(y, predictions)
    
    return loss

@tf.function(jit_compile=True)
def test_step(model, x, y, loss_fn, test_accuracy):
    """Optimized test step"""
    predictions = model(x, training=False)
    loss = loss_fn(y, predictions)
    test_accuracy.update_state(y, predictions)
    return loss

class MemoryMonitor:
    """Monitor GPU memory usage during training"""
    
    def __init__(self, log_interval=100):
        self.log_interval = log_interval
        self.step_count = 0
    
    def log_memory_usage(self):
        """Log current GPU memory usage"""
        if self.step_count % self.log_interval == 0:
            try:
                gpu_info = tf.config.experimental.get_memory_info('GPU:0')
                current_mb = gpu_info['current'] / 1024 / 1024
                peak_mb = gpu_info['peak'] / 1024 / 1024
                logging.info(f"GPU Memory - Current: {current_mb:.1f}MB, Peak: {peak_mb:.1f}MB")
            except:
                pass
        self.step_count += 1

def train_model(model, train_dataset, test_dataset, config, logger):
    """Main training function"""
    
    # Create optimizer and loss function
    optimizer = create_custom_optimizer(config)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    
    # Metrics
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    
    # Memory monitor
    memory_monitor = MemoryMonitor(log_interval=config['logging']['memory_log_interval'])
    
    # TensorBoard callback
    tensorboard_dir = Path(config['logging']['tensorboard_dir'])
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    best_accuracy = 0.0
    
    for epoch in range(config['training']['epochs']):
        logger.info(f"Starting epoch {epoch + 1}/{config['training']['epochs']}")
        
        # Reset metrics
        train_accuracy.reset_states()
        test_accuracy.reset_states()
        
        # Training phase
        epoch_start = time.time()
        train_loss_avg = tf.keras.metrics.Mean()
        
        for step, (x_batch, y_batch) in enumerate(train_dataset):
            loss = train_step(model, x_batch, y_batch, optimizer, loss_fn, train_accuracy)
            train_loss_avg.update_state(loss)
            
            # Logging
            memory_monitor.log_memory_usage()
            
            if step % config['logging']['log_interval'] == 0:
                current_lr = optimizer.learning_rate
                if hasattr(current_lr, 'numpy'):
                    current_lr = current_lr.numpy()
                
                logger.info(
                    f"Epoch {epoch+1}, Step {step}: "
                    f"Loss={loss:.4f}, "
                    f"Accuracy={train_accuracy.result():.4f}, "
                    f"LR={current_lr:.6f}"
                )
        
        # Validation phase
        test_loss_avg = tf.keras.metrics.Mean()
        
        for x_batch, y_batch in test_dataset:
            loss = test_step(model, x_batch, y_batch, loss_fn, test_accuracy)
            test_loss_avg.update_state(loss)
        
        # Epoch summary
        epoch_time = time.time() - epoch_start
        train_acc = train_accuracy.result()
        test_acc = test_accuracy.result()
        train_loss = train_loss_avg.result()
        test_loss = test_loss_avg.result()
        
        logger.info(
            f"Epoch {epoch+1} completed in {epoch_time:.1f}s: "
            f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
            f"Val Loss={test_loss:.4f}, Val Acc={test_acc:.4f}"
        )
        
        # Save checkpoint if best model
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            
            checkpoint_dir = Path(config['checkpoint']['dir'])
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            model.save_weights(checkpoint_dir / 'best_model.h5')
            logger.info(f"New best model saved with accuracy: {best_accuracy:.4f}")
        
        # Save regular checkpoint
        if (epoch + 1) % config['checkpoint']['save_interval'] == 0:
            checkpoint_dir = Path(config['checkpoint']['dir'])
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            model.save_weights(checkpoint_dir / f'checkpoint_epoch_{epoch+1}.h5')
            logger.info(f"Checkpoint saved at epoch {epoch+1}")
        
        # Write TensorBoard logs
        with tf.summary.create_file_writer(str(tensorboard_dir)).as_default():
            tf.summary.scalar('train_loss', train_loss, step=epoch)
            tf.summary.scalar('train_accuracy', train_acc, step=epoch)
            tf.summary.scalar('val_loss', test_loss, step=epoch)
            tf.summary.scalar('val_accuracy', test_acc, step=epoch)
            
            current_lr = optimizer.learning_rate
            if hasattr(current_lr, 'numpy'):
                current_lr = current_lr.numpy()
            tf.summary.scalar('learning_rate', current_lr, step=epoch)
    
    logger.info(f"Training completed! Best validation accuracy: {best_accuracy:.4f}")
    return best_accuracy

def main():
    parser = argparse.ArgumentParser(description='TensorFlow H200 Training Example')
    parser.add_argument('--config', required=True, help='Configuration file (JSON)')
    parser.add_argument('--resume', help='Resume from checkpoint')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    logger = setup_logging()
    logger.info(f"Starting TensorFlow training with config: {args.config}")
    
    # Setup GPU
    setup_gpu(config)
    
    # Set random seeds
    tf.random.set_seed(config['training']['seed'])
    np.random.seed(config['training']['seed'])
    
    # Create distribution strategy for multi-GPU
    if len(tf.config.experimental.list_physical_devices('GPU')) > 1:
        strategy = tf.distribute.MirroredStrategy()
        logger.info(f"Using MirroredStrategy with {strategy.num_replicas_in_sync} GPUs")
    else:
        strategy = tf.distribute.get_strategy()  # Default strategy
    
    with strategy.scope():
        # Create model
        model = create_efficient_model(config)
        
        # Create datasets
        train_dataset, test_dataset = create_optimized_dataset(config)
        
        # Calculate steps per epoch for scheduler
        config['steps_per_epoch'] = tf.data.experimental.cardinality(train_dataset).numpy()
        
        # Distribute datasets
        train_dataset = strategy.experimental_distribute_dataset(train_dataset)
        test_dataset = strategy.experimental_distribute_dataset(test_dataset)
        
        # Resume from checkpoint if specified
        if args.resume:
            logger.info(f"Resuming from checkpoint: {args.resume}")
            model.load_weights(args.resume)
        
        # Print model summary
        logger.info("Model architecture:")
        model.summary(print_fn=logger.info)
        
        # Start training
        best_accuracy = train_model(model, train_dataset, test_dataset, config, logger)
        
        logger.info(f"Final best accuracy: {best_accuracy:.4f}")

if __name__ == '__main__':
    main()