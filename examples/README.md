# Example Scripts for H200 GPU Training

This directory contains ready-to-use example scripts demonstrating best practices for deep learning on the Ruqola server's H200 GPUs.

The server has **4 x NVIDIA H200 NVL GPUs** (indices `0,1,2,3`), each with **~141 GB of VRAM** (~564 GB total), Hopper architecture (compute capability 9.0). "Use all GPUs" means `0,1,2,3` (e.g. `CUDA_VISIBLE_DEVICES=0,1,2,3`, `torchrun --nproc_per_node=4`, `gpuq submit -g 4`).

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

> **How `gpuq submit` works:** gpuq is daemonless — `gpuq submit` runs your command in the **foreground** in the current terminal and streams its stdout/stderr straight to you. There are no per-job log files; if you want a log, redirect output yourself (see [Monitoring and Debugging](#-monitoring-and-debugging)). By default gpuq picks free GPUs and may **stack** additional jobs onto cards you already own; GPUs held by other users are off-limits until they free them. To pin exact GPUs use `--devices 1,3` — note this is **rejected immediately** if another user holds one of them, unless you add `--queue` to wait. The `--memory N` flag is a **placement floor**: it means "only put me on a GPU with at least N GB free" (not a cap or reservation). To be emailed when the job finishes, add `--notify you@example.com`. See [../docs/gpu-queue-guide.md](../docs/gpu-queue-guide.md) for the full ownership/stacking policy.

### 1. PyTorch Training (Recommended for beginners)

```bash
# Copy example files to your directory
cp examples/pytorch_training.py .
cp examples/resnet_config.yaml .

# Submit training job.
# --memory is a placement floor (min free VRAM on the chosen GPU), not a budget.
# The CIFAR-10 ResNet example only needs a few GB, so a small floor is fine.
gpuq submit \
    --command "python pytorch_training.py --config resnet_config.yaml" \
    --gpus 1 \
    --memory 8 \
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
    --memory 12 \
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
    --memory 10 \
    --time 6
```

### 4. Transformers Fine-tuning (LLMs)

```bash
# Copy Transformers files
cp examples/transformers_finetuning.py .
cp examples/transformers_config.yaml .

# Submit job for large model fine-tuning.
# --gpus 4 uses all four H200s; set the --memory floor to the per-card peak you expect.
gpuq submit \
    --command "torchrun --nproc_per_node=4 transformers_finetuning.py --config transformers_config.yaml" \
    --gpus 4 \
    --memory 40 \
    --time 12
```

### 5. LoRA Fine-tuning (Parameter-Efficient)

```bash
# Copy LoRA files
cp examples/lora_example.py .
cp examples/lora_config.yaml .

# Submit LoRA training job (a 7B LoRA fits well under 40 GB)
gpuq submit \
    --command "python lora_example.py --mode train --model microsoft/DialoGPT-medium --config lora_config.yaml" \
    --gpus 1 \
    --memory 20 \
    --time 6
```

## 📊 What These Examples Demonstrate

### Common Features Across All Examples:
- ✅ **Mixed precision training** (BF16 preferred on H200; FP16 as a legacy fallback) for memory efficiency
- ✅ **Optimal batch sizes** for H200 Tensor Cores (multiples of 8)
- ✅ **Memory-efficient techniques** (gradient checkpointing, accumulation)
- ✅ **Multi-GPU support** with proper distributed training (up to 4 GPUs)
- ✅ **Comprehensive logging** and monitoring
- ✅ **Checkpointing** for long training jobs
- ✅ **Error handling** and recovery mechanisms

> **Precision tip:** On H200 (Hopper) GPUs, **BF16** is the preferred mixed-precision dtype — it has the same Tensor-Core throughput as FP16 but its wider exponent range avoids loss-scaling and overflow headaches. Use `torch.amp.autocast('cuda', dtype=torch.bfloat16)` in PyTorch and set `bf16: true` in `transformers_config.yaml`. FP16 (with a `GradScaler`) remains a valid fallback for older code.

### Framework-Specific Optimizations:

#### PyTorch (`pytorch_training.py`)
- Automatic Mixed Precision (AMP) with the `torch.amp` API
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
- Gradient checkpointing with `jax.checkpoint` (a.k.a. `jax.remat`)
- Sharding across devices with `jax.sharding.Mesh` / `PartitionSpec`
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
    num_workers=8,           # The host has 256 logical CPUs; tune per job
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
# Enable gradient checkpointing (call the method; do not assign an attribute)
model.gradient_checkpointing_enable()

# Use mixed precision — BF16 is preferred on H200 (no GradScaler needed)
import torch

with torch.amp.autocast('cuda', dtype=torch.bfloat16):
    output = model(input)
    loss = criterion(output, target)

# FP16 fallback (requires a GradScaler to avoid underflow):
# scaler = torch.amp.GradScaler('cuda')
# with torch.amp.autocast('cuda', dtype=torch.float16):
#     ...
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
# Monitor GPU usage (all 4 H200s)
watch -n 5 nvidia-smi

# Check the queue and your jobs. The output is already compact; grepping by
# $USER is only an approximate filter (it also matches process lines).
gpuq status
watch -n 10 gpuq status

# gpuq runs your job in the FOREGROUND and streams stdout/stderr to your
# terminal — there are no per-job log files. Capture output yourself if you
# want something to tail:
gpuq submit --command "python train.py" --gpus 1 > train.log 2>&1
# then, in another shell:
tail -f train.log

# For long runs, launch inside tmux/screen (or use nohup) so the job
# survives a disconnect:
#   tmux new -s train
#   gpuq submit -- python train.py 2>&1 | tee train.log
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
# 3. Use mixed precision (BF16 on H200)
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
# Check the queue and current GPU ownership
gpuq status

# Common issues:
# - No GPU matches your request. By default `gpuq submit` does NOT wait — if no
#   free (or owned-by-you) GPU meets your --gpus / --memory request, it is
#   rejected immediately. Add --queue to wait for a slot.
# - --devices pinned to a GPU another user holds (rejected immediately unless
#   you add --queue).
# - --memory floor set too high (e.g. 100 GB skips any card with less than
#   100 GB free). Set it just above your job's real peak usage, not the GPU size.
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

1. **Check your job output**: `gpuq` streams stdout/stderr to the terminal that ran `gpuq submit` — redirect it to a file (e.g. `> train.log 2>&1`) to keep a log. There is no `/tmp/gpu_queue/logs` directory.
2. **Review documentation**: Especially the [troubleshooting guide](../docs/troubleshooting.md)
3. **Test with minimal examples**: Start with simple cases
4. **Monitor resources**: Use `nvidia-smi` and `gpuq status`
5. **Contact administrators**: If hardware or system-level issues

---

**Happy training on the H200s! 🚀**
