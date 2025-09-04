# NVIDIA H200 GPU Specifications and Capabilities

Technical specifications and optimization guidelines for the NVIDIA H200 Tensor Core GPUs in the Ruqola server.

## 📖 Table of Contents

1. [Hardware Specifications](#hardware-specifications)
2. [Memory Architecture](#memory-architecture)
3. [Compute Capabilities](#compute-capabilities)
4. [Performance Characteristics](#performance-characteristics)
5. [Optimization Guidelines](#optimization-guidelines)
6. [Comparison with Other GPUs](#comparison-with-other-gpus)
7. [Best Use Cases](#best-use-cases)

## Hardware Specifications

### NVIDIA H200 SXM Overview

Our server is equipped with **3x NVIDIA H200 SXM5** GPUs with the following specifications:

| Specification | Value |
|---------------|-------|
| **GPU Architecture** | Hopper (GH200) |
| **Process Node** | TSMC 4N (4nm) |
| **Transistors** | 80 billion |
| **SM (Streaming Multiprocessors)** | 134 |
| **CUDA Cores** | 16,896 |
| **RT Cores** | 4th Gen (134 units) |
| **Tensor Cores** | 4th Gen (528 units) |
| **Base Clock** | 1,830 MHz |
| **Boost Clock** | 2,600 MHz |

### Memory Specifications

| Memory Feature | H200 SXM5 |
|----------------|-----------|
| **Memory Type** | HBM3e |
| **Memory Capacity** | 141 GB |
| **Memory Bandwidth** | 4,800 GB/s |
| **Memory Bus Width** | 5,120-bit |
| **L2 Cache** | 50 MB |
| **Memory Clock** | 4,800 MHz |

### Power and Thermal

| Specification | Value |
|---------------|-------|
| **Total Graphics Power (TGP)** | 700W |
| **Form Factor** | SXM5 |
| **Cooling** | Liquid Cooling Required |
| **Operating Temperature** | 0°C to 35°C |

## Memory Architecture

### HBM3e Memory System

The H200's HBM3e memory system provides exceptional bandwidth and capacity:

```
┌─────────────────────────────────────┐
│           GPU Die (GH200)           │
├─────────────────────────────────────┤
│  L2 Cache: 50 MB (Shared)          │
├─────────────────────────────────────┤
│  HBM3e Memory: 141 GB              │
│  Bandwidth: 4,800 GB/s             │
│  5,120-bit Memory Interface        │
└─────────────────────────────────────┘
```

### Memory Hierarchy

1. **Registers** (per thread): ~255 registers × 32-bit
2. **Shared Memory** (per SM): 228 KB configurable
3. **L1 Cache** (per SM): 128 KB
4. **L2 Cache** (global): 50 MB
5. **HBM3e** (global): 141 GB at 4,800 GB/s

### Memory Bandwidth Utilization

```python
# Theoretical peak memory bandwidth test
import torch
import time

def memory_bandwidth_test(size_gb=10):
    device = torch.device('cuda')
    
    # Create tensors
    size = int(size_gb * 1024**3 / 4)  # float32 = 4 bytes
    a = torch.randn(size, device=device)
    b = torch.randn(size, device=device)
    
    # Warmup
    for _ in range(10):
        c = a + b
    
    torch.cuda.synchronize()
    start = time.time()
    
    # Memory bandwidth test
    for _ in range(100):
        c = a + b  # Read 2 tensors, write 1 tensor
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    # Calculate bandwidth (3 * size_gb * 100 operations / elapsed time)
    bandwidth = (3 * size_gb * 100) / elapsed
    print(f"Memory bandwidth: {bandwidth:.1f} GB/s")
    print(f"Utilization: {bandwidth/4800*100:.1f}% of peak")

memory_bandwidth_test()
```

## Compute Capabilities

### CUDA Compute Capability

The H200 supports **CUDA Compute Capability 9.0 (Hopper)**:

```bash
# Check compute capability
nvidia-smi --query-gpu=compute_cap --format=csv,noheader
# Output: 9.0
```

### Tensor Core Capabilities

#### 4th Generation Tensor Cores

| Data Type | Matrix Size | Peak Performance (per SM) |
|-----------|-------------|---------------------------|
| **FP16** | 16×16×16 | 256 TOPS |
| **BF16** | 16×16×16 | 256 TOPS |
| **TF32** | 16×16×16 | 128 TOPS |
| **FP8** | 16×16×16 | 512 TOPS |
| **INT8** | 16×16×16 | 512 TOPS |
| **INT4** | 16×16×16 | 1024 TOPS |

#### Tensor Core Usage in Deep Learning

```python
# PyTorch automatic mixed precision with H200
import torch
from torch.cuda.amp import autocast, GradScaler

model = MyModel().cuda()
optimizer = torch.optim.AdamW(model.parameters())
scaler = GradScaler()

for data, target in dataloader:
    optimizer.zero_grad()
    
    # Use autocast for forward pass
    with autocast():
        output = model(data)
        loss = criterion(output, target)
    
    # Scale loss and backward pass
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### CUDA Cores Performance

- **Single Precision (FP32)**: 67 TFLOPS
- **Half Precision (FP16)**: 134 TFLOPS
- **Double Precision (FP64)**: 34 TFLOPS

## Performance Characteristics

### Memory Bandwidth Scaling

```python
# Test memory bandwidth with different data sizes
import torch
import numpy as np
import matplotlib.pyplot as plt

def bandwidth_vs_size():
    sizes = [1, 2, 4, 8, 16, 32, 64, 100]  # GB
    bandwidths = []
    
    for size_gb in sizes:
        # Run bandwidth test
        bandwidth = memory_bandwidth_test(size_gb)
        bandwidths.append(bandwidth)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, bandwidths, 'b-o')
    plt.axhline(y=4800, color='r', linestyle='--', label='Peak Bandwidth')
    plt.xlabel('Data Size (GB)')
    plt.ylabel('Bandwidth (GB/s)')
    plt.title('H200 Memory Bandwidth vs Data Size')
    plt.legend()
    plt.grid(True)
    plt.savefig('h200_bandwidth.png')
```

### Compute Performance

#### Matrix Multiplication Performance

```python
# GEMM performance test
import torch
import time

def gemm_performance_test():
    device = torch.device('cuda')
    sizes = [1024, 2048, 4096, 8192, 16384]
    
    for size in sizes:
        a = torch.randn(size, size, device=device, dtype=torch.float16)
        b = torch.randn(size, size, device=device, dtype=torch.float16)
        
        # Warmup
        for _ in range(10):
            c = torch.mm(a, b)
        
        torch.cuda.synchronize()
        start = time.time()
        
        for _ in range(100):
            c = torch.mm(a, b)
        
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        # Calculate TFLOPS
        ops = 2 * size**3 * 100  # multiply-accumulate operations
        tflops = ops / elapsed / 1e12
        
        print(f"Matrix size {size}x{size}: {tflops:.2f} TFLOPS")

gemm_performance_test()
```

## Optimization Guidelines

### Memory Optimization

1. **Maximize Memory Utilization**:
   ```python
   # Use all available memory effectively
   batch_size = calculate_max_batch_size(model, input_size, memory_limit=130)  # Leave 11GB buffer
   ```

2. **Memory-Efficient Training**:
   ```python
   # Gradient checkpointing for large models
   model = torch.utils.checkpoint.checkpoint_sequential(model, segments=4)
   
   # Gradient accumulation for large effective batch sizes
   accumulation_steps = 4
   for i, batch in enumerate(dataloader):
       loss = model(batch) / accumulation_steps
       loss.backward()
       
       if (i + 1) % accumulation_steps == 0:
           optimizer.step()
           optimizer.zero_grad()
   ```

### Compute Optimization

1. **Use Tensor Cores**:
   ```python
   # Enable automatic mixed precision
   model = model.half()  # FP16 model
   
   # Or use autocast
   with torch.autocast('cuda'):
       output = model(input)
   ```

2. **Optimize Tensor Shapes**:
   ```python
   # Ensure tensor dimensions are multiples of 8 for optimal Tensor Core usage
   batch_size = 64  # Multiple of 8
   hidden_dim = 4096  # Multiple of 8
   seq_length = 2048  # Multiple of 8
   ```

### Data Loading Optimization

```python
# Optimized data loading for H200
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=8,  # Match CPU cores
    pin_memory=True,  # Faster GPU transfer
    persistent_workers=True,  # Reduce worker restart overhead
    prefetch_factor=2,  # Prefetch batches
)
```

## Comparison with Other GPUs

### Performance Comparison

| GPU Model | Memory | Memory BW | FP16 TFLOPS | Architecture |
|-----------|--------|-----------|-------------|--------------|
| **H200 SXM** | **141 GB** | **4,800 GB/s** | **134 TFLOPS** | **Hopper** |
| H100 SXM | 80 GB | 3,350 GB/s | 126 TFLOPS | Hopper |
| A100 SXM | 80 GB | 2,039 GB/s | 77 TFLOPS | Ampere |
| V100 SXM | 32 GB | 900 GB/s | 31 TFLOPS | Volta |
| RTX 4090 | 24 GB | 1,008 GB/s | 42 TFLOPS | Ada Lovelace |

### Memory Capacity Advantages

```python
# Models that benefit from H200's large memory:
models_by_memory = {
    "GPT-3 175B": "350+ GB",  # Needs model parallelism on other GPUs
    "LLaMA 65B": "130 GB",    # Fits on single H200!
    "Stable Diffusion XL": "12 GB",  # Much headroom for batch size
    "BERT Large": "1.3 GB",   # Can run huge batch sizes
}
```

## Best Use Cases

### Ideal Workloads for H200

1. **Large Language Models**:
   ```python
   # Train/finetune models up to ~65B parameters on single GPU
   model = AutoModelForCausalLM.from_pretrained(
       "meta-llama/Llama-2-70b-hf",
       torch_dtype=torch.float16,
       device_map="auto"
   )
   ```

2. **Computer Vision with Large Images**:
   ```python
   # Process high-resolution images with large batch sizes
   batch_size = 128  # Much larger than possible on smaller GPUs
   image_size = 1024  # Higher resolution training
   ```

3. **Scientific Computing**:
   ```python
   # Large-scale numerical simulations
   simulation_grid = torch.zeros(8192, 8192, 8192, device='cuda')  # 2TB+ data
   ```

4. **Multi-Modal Models**:
   ```python
   # Train large vision-language models
   model = VisionLanguageModel(
       vision_dim=2048,
       text_dim=4096,
       hidden_dim=8192,  # Large hidden dimensions
       num_layers=48
   )
   ```

### Optimization Strategies by Use Case

#### Large Language Models
```python
# Memory-efficient LLM training
from transformers import AutoModelForCausalLM
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    gradient_checkpointing=True,
    use_cache=False  # Save memory during training
)

# Use DeepSpeed ZeRO for even larger models
from deepspeed import initialize
model_engine, optimizer, _, _ = initialize(
    model=model,
    config="deepspeed_config.json"
)
```

#### Computer Vision
```python
# High-throughput image processing
def create_optimized_dataloader(dataset, batch_size=256):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,  # Large batch size utilizing full memory
        num_workers=12,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=3
    )

# Mixed precision training for CNNs
with autocast():
    output = model(images)
    loss = criterion(output, labels)

scaler.scale(loss).backward()
```

#### Scientific Computing
```python
# Large tensor operations
def scientific_simulation():
    # Large 3D simulation grids
    grid = torch.zeros(2048, 2048, 2048, device='cuda', dtype=torch.float32)
    
    # Physics-informed neural networks with large domains
    coordinates = torch.rand(1000000, 3, device='cuda')  # 1M sample points
    solution = physics_model(coordinates)
```

### Performance Monitoring

```python
# Monitor H200 utilization
import torch
import nvidia_ml_py3 as nvml

def monitor_gpu_usage():
    nvml.nvmlInit()
    handle = nvml.nvmlDeviceGetHandleByIndex(0)
    
    # Memory usage
    mem_info = nvml.nvmlDeviceGetMemoryInfo(handle)
    memory_used = mem_info.used / 1024**3  # GB
    memory_total = mem_info.total / 1024**3  # GB
    
    # Utilization
    util = nvml.nvmlDeviceGetUtilizationRates(handle)
    gpu_util = util.gpu
    mem_util = util.memory
    
    # Temperature
    temp = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
    
    print(f"Memory: {memory_used:.1f}/{memory_total:.1f} GB ({memory_used/memory_total*100:.1f}%)")
    print(f"GPU Utilization: {gpu_util}%")
    print(f"Memory Utilization: {mem_util}%")
    print(f"Temperature: {temp}°C")

# Run periodically during training
import threading
import time

def monitoring_thread():
    while training:
        monitor_gpu_usage()
        time.sleep(10)

monitor = threading.Thread(target=monitoring_thread)
monitor.start()
```

---

**Next Steps**: 
- For PyTorch-specific optimizations: [PyTorch with H200 Guide](pytorch-guide.md)
- For TensorFlow optimizations: [TensorFlow with H200 Guide](tensorflow-guide.md)
- For general best practices: [Best Practices Guide](best-practices.md)