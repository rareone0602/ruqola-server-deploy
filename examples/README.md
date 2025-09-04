# Example Scripts for H200 GPU Training

This directory contains ready-to-use example scripts demonstrating best practices for deep learning on the Ruqola server's H200 GPUs.

## 📁 Contents

### Training Scripts
- **`pytorch_training.py`** - Complete PyTorch training example with ResNet
- **`tensorflow_training.py`** - TensorFlow training with mixed precision and XLA
- **`jax_training.py`** - JAX/Flax functional programming approach
- **`transformers_finetuning.py`** - Hugging Face Transformers fine-tuning with LoRA support
- **`transformers_inference.py`** - Optimized inference for large language models
- **`lora_example.py`** - Parameter-efficient fine-tuning with LoRA

### Configuration Files
- **`resnet_config.yaml`** - PyTorch training configuration
- **`tf_config.json`** - TensorFlow training configuration  
- **`jax_config.py`** - JAX/Flax configuration
- **`transformers_config.yaml`** - Transformers fine-tuning configuration
- **`lora_config.yaml`** - LoRA fine-tuning configuration

### Utilities
- **`submit_jobs.sh`** - Job submission examples and best practices
- **`README.md`** - This documentation file

## 🚀 Quick Start

### 1. PyTorch Training (Recommended for beginners)

```bash
# Copy example files to your directory
cp examples/pytorch_training.py .
cp examples/resnet_config.yaml .

# Submit training job
gpuq submit \
    --command "python pytorch_training.py --config resnet_config.yaml" \
    --gpus 1 \
    --memory 40 \
    --time 8
```

### 2. TensorFlow Training

```bash
# Copy TensorFlow files
cp examples/tensorflow_training.py .
cp examples/tf_config.json .

# Submit job
gpuq submit \
    --command "python tensorflow_training.py --config tf_config.json" \
    --gpus 1 \
    --memory 60 \
    --time 8
```

### 3. JAX/Flax Training

```bash
# Copy JAX files
cp examples/jax_training.py .
cp examples/jax_config.py .

# Submit job
gpuq submit \
    --command "python jax_training.py --config jax_config.py" \
    --gpus 1 \
    --memory 50 \
    --time 6
```

### 4. Transformers Fine-tuning (LLMs)

```bash
# Copy Transformers files
cp examples/transformers_finetuning.py .
cp examples/transformers_config.yaml .

# Submit job for large model fine-tuning
gpuq submit \
    --command "python transformers_finetuning.py --config transformers_config.yaml" \
    --gpus 2 \
    --memory 100 \
    --time 12
```

### 5. LoRA Fine-tuning (Parameter-Efficient)

```bash
# Copy LoRA files
cp examples/lora_example.py .
cp examples/lora_config.yaml .

# Submit LoRA training job
gpuq submit \
    --command "python lora_example.py --mode train --model microsoft/DialoGPT-medium --config lora_config.yaml" \
    --gpus 1 \
    --memory 40 \
    --time 6
```

## 📊 What These Examples Demonstrate

### Common Features Across All Examples:
- ✅ **Mixed precision training** (FP16) for memory efficiency
- ✅ **Optimal batch sizes** for H200 Tensor Cores (multiples of 8)
- ✅ **Memory-efficient techniques** (gradient checkpointing, accumulation)
- ✅ **Multi-GPU support** with proper distributed training
- ✅ **Comprehensive logging** and monitoring
- ✅ **Checkpointing** for long training jobs
- ✅ **Error handling** and recovery mechanisms

### Framework-Specific Optimizations:

#### PyTorch (`pytorch_training.py`)
- Automatic Mixed Precision (AMP) with GradScaler
- DistributedDataParallel for multi-GPU training
- Gradient checkpointing for memory efficiency
- Optimized DataLoader settings for H200
- WandB integration for experiment tracking

#### TensorFlow (`tensorflow_training.py`)
- XLA compilation for performance
- Mixed precision policy with loss scaling
- MirroredStrategy for multi-GPU
- Optimized tf.data pipeline
- TensorBoard logging integration

#### JAX (`jax_training.py`)
- Pure functional programming approach
- JIT compilation with `@jit` decorator
- Gradient checkpointing with `remat`
- pmap for multi-GPU parallelism
- Orbax checkpointing system

## 💾 Storage and Data Organization

### Recommended Directory Structure:
```
~/projects/my_experiment/
├── train.py              # Your training script (based on examples)
├── config.yaml           # Configuration file
├── data/                 # Dataset (use shared storage when possible)
├── checkpoints/          # Model checkpoints
├── logs/                 # Training logs
├── results/              # Final results and plots
└── models/               # Saved models
```

### Data Loading Best Practices:
```python
# Efficient data loading for H200
dataloader = DataLoader(
    dataset,
    batch_size=128,          # Multiple of 8 for Tensor Cores
    num_workers=8,           # Match CPU cores
    pin_memory=True,         # Faster GPU transfer
    persistent_workers=True, # Reduce worker restart overhead
    prefetch_factor=2,       # Prefetch batches
)
```

## 🔧 Customization Guide

### Modifying for Your Use Case

1. **Change Dataset**:
   ```python
   # Replace CIFAR-10 with your dataset
   train_dataset = YourCustomDataset(...)
   ```

2. **Modify Model Architecture**:
   ```python
   # Change model definition
   model = YourCustomModel(...)
   ```

3. **Adjust Hyperparameters**:
   ```yaml
   # In config file
   training:
     batch_size: 64      # Adjust based on memory
     learning_rate: 0.01 # Tune for your problem
   ```

4. **Add Custom Loss Functions**:
   ```python
   def custom_loss(predictions, targets):
       # Your loss implementation
       return loss
   ```

## 📈 Performance Optimization Tips

### Memory Optimization
```python
# Enable gradient checkpointing
model.gradient_checkpointing = True

# Use mixed precision
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast():
    output = model(input)
    loss = criterion(output, target)
```

### Compute Optimization
```python
# Ensure tensor dimensions are optimal for H200
def make_divisible_by_8(x):
    return ((x + 7) // 8) * 8

batch_size = make_divisible_by_8(batch_size)
hidden_dim = make_divisible_by_8(hidden_dim)
```

### Data Loading Optimization
```python
# Use memory mapping for large datasets
data = np.memmap('large_dataset.dat', dtype='float32', mode='r')

# Implement efficient transforms
def fast_transform(x):
    # Vectorized operations
    return x / 255.0  # Faster than individual pixel operations
```

## 🔍 Monitoring and Debugging

### Real-time Monitoring
```bash
# Monitor GPU usage
watch -n 5 nvidia-smi

# Monitor job progress
watch -n 10 'gpuq status | grep $USER'

# Check job logs
tail -f /tmp/gpu_queue/logs/job_XXXXX_stdout.log
```

### Memory Profiling
```python
# Check memory usage in Python
def print_memory_stats():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory - Allocated: {allocated:.1f}GB, Reserved: {reserved:.1f}GB")

# Call periodically during training
if step % 100 == 0:
    print_memory_stats()
```

## 🚨 Troubleshooting Common Issues

### Out of Memory (OOM)
```bash
# Quick fixes:
# 1. Reduce batch size
# 2. Enable gradient checkpointing  
# 3. Use mixed precision
# 4. Clear cache periodically

# In Python:
torch.cuda.empty_cache()
```

### Slow Training
```bash
# Check GPU utilization (should be >80%)
nvidia-smi

# Common causes:
# - Data loading bottleneck (increase num_workers)
# - Small batch size (increase if memory allows)
# - Inefficient model architecture
```

### Job Won't Start
```bash
# Check queue status
gpuq status

# Common issues:
# - All GPUs busy (wait or reduce resource request)
# - Requesting too much memory
# - Syntax error in command
```

## 📚 Additional Resources

### Documentation Links
- [GPU Queue System Guide](../docs/gpu-queue-guide.md)
- [H200 Specifications](../docs/h200-specs.md)
- [PyTorch Guide](../docs/pytorch-guide.md)
- [TensorFlow Guide](../docs/tensorflow-guide.md)
- [JAX Guide](../docs/jax-guide.md)
- [Best Practices](../docs/best-practices.md)
- [Troubleshooting](../docs/troubleshooting.md)

### External Resources
- [PyTorch Documentation](https://pytorch.org/docs/)
- [TensorFlow Documentation](https://tensorflow.org/guide)
- [JAX Documentation](https://jax.readthedocs.io/)
- [Flax Documentation](https://flax.readthedocs.io/)

## 🤝 Contributing

To add new examples or improve existing ones:

1. Follow the established patterns and coding style
2. Include comprehensive comments and documentation
3. Test on the H200 GPUs before submitting
4. Add appropriate configuration files
5. Update this README with your additions

## 📞 Getting Help

If you encounter issues with these examples:

1. **Check the logs**: `/tmp/gpu_queue/logs/job_XXXXX_*.log`
2. **Review documentation**: Especially the [troubleshooting guide](../docs/troubleshooting.md)
3. **Test with minimal examples**: Start with simple cases
4. **Monitor resources**: Use `nvidia-smi` and `gpuq status`
5. **Contact administrators**: If hardware or system-level issues

---

**Happy training on the H200s! 🚀**