# JAX/Flax with NVIDIA H200 GPUs

Complete guide for optimizing JAX and Flax workflows on the Ruqola server's H200 GPUs.

The Ruqola server (host `wsserver1`, "Mjolnir") has **4 x NVIDIA H200 NVL** GPUs (indices `0,1,2,3`), each with ~141 GB of VRAM (~564 GB total), compute capability 9.0 (Hopper). The host runs Ubuntu 24.04 with GPU driver 575.57.08 and a CUDA 12.9 driver stack. All "use every GPU" examples below assume 4 devices, but prefer deriving the count dynamically with `len(jax.devices())` instead of hard-coding it.

## Table of Contents

1. [Setup and Installation](#setup-and-installation)
2. [Basic GPU Usage](#basic-gpu-usage)
3. [Memory Optimization](#memory-optimization)
4. [Performance Optimization](#performance-optimization)
5. [Multi-GPU Training](#multi-gpu-training)
6. [Large Model Training](#large-model-training)
7. [Advanced Techniques](#advanced-techniques)
8. [Debugging and Profiling](#debugging-and-profiling)
9. [Example Scripts](#example-scripts)

## Setup and Installation

### Recommended JAX Installation

```bash
# CUDA 12 JAX wheels (server runs CUDA 12.9 driver, H200 sm_90)
pip install -U "jax[cuda12]"

# Additional packages
pip install flax optax orbax-checkpoint
pip install chex ml_collections wandb

# Verify installation
python -c "import jax; print('JAX version:', jax.__version__); print('Devices:', jax.devices())"
```

The modern `jax[cuda12]` extra bundles the CUDA libraries as pip wheels, so the
legacy `jax[cuda12_pip]` extra and the `-f .../jax_cuda_releases.html` find-links
URL are no longer needed (and have been removed in recent JAX).

### Environment Setup

```bash
# Add to ~/.bashrc or job script
export CUDA_VISIBLE_DEVICES=0  # Use first H200, or 0,1,2,3 for all
export XLA_PYTHON_CLIENT_PREALLOCATE=false  # Dynamic memory allocation
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda
```

### Verify Installation

```python
import jax
import jax.numpy as jnp
from jax import devices

print(f"JAX version: {jax.__version__}")
print(f"Available devices: {devices()}")
print(f"Default device: {jax.devices()[0]}")

# Check H200 capabilities
for i, device in enumerate(devices()):
    print(f"Device {i}: {device}")
    print(f"  Device kind: {device.device_kind}")  # e.g. 'NVIDIA H200 NVL'
    print(f"  Platform: {device.platform}")        # e.g. 'gpu'

# Simple computation test
x = jnp.array([1.0, 2.0, 3.0])
y = jnp.array([4.0, 5.0, 6.0])
result = jnp.dot(x, y)
print(f"Dot product result: {result} (computed on {result.device})")
```

> Note: array placement is read via the `.device` **property** (not a method).
> The old callable `arr.device()` was deprecated in JAX v0.4.21 and removed in
> v0.4.27, so `result.device()` now raises an error — use `result.device`.

## Basic GPU Usage

### Device Management and Basic Operations

```python
import jax
import jax.numpy as jnp
from jax import device_put, devices

# Move data to specific GPU
def move_to_device(data, device_id=0):
    device = devices()[device_id]
    return device_put(data, device)

# Basic operations
x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
y = jnp.array([[5.0, 6.0], [7.0, 8.0]])

# Matrix multiplication on GPU
result = jnp.dot(x, y)
print(f"Result device: {result.device}")

# Element-wise operations
z = x + y
w = jnp.sin(x) * jnp.cos(y)
```

### Simple Neural Network with JAX

```python
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, random

def init_network_params(layer_sizes, key):
    """Initialize neural network parameters"""
    keys = random.split(key, len(layer_sizes))
    params = []
    
    for i in range(len(layer_sizes) - 1):
        key = keys[i]
        W_key, b_key = random.split(key)
        
        # Xavier initialization
        fan_in, fan_out = layer_sizes[i], layer_sizes[i + 1]
        W = random.normal(W_key, (fan_in, fan_out)) * jnp.sqrt(2.0 / fan_in)
        b = jnp.zeros(fan_out)
        
        params.append((W, b))
    
    return params

def forward_pass(params, x):
    """Forward pass through network"""
    for W, b in params[:-1]:
        x = jnp.maximum(0, jnp.dot(x, W) + b)  # ReLU activation
    
    # Final layer (no activation)
    W, b = params[-1]
    return jnp.dot(x, W) + b

# Initialize model
key = random.PRNGKey(42)
layer_sizes = [784, 512, 256, 10]  # MNIST example
params = init_network_params(layer_sizes, key)

# Test forward pass
x_test = random.normal(key, (32, 784))  # Batch of 32 samples
output = forward_pass(params, x_test)
print(f"Output shape: {output.shape}, Device: {output.device}")
```

### Training Loop Basics

```python
import optax

def loss_fn(params, batch_x, batch_y):
    """Compute cross-entropy loss"""
    logits = forward_pass(params, batch_x)
    return optax.softmax_cross_entropy_with_integer_labels(logits, batch_y).mean()

def accuracy(params, batch_x, batch_y):
    """Compute classification accuracy"""
    logits = forward_pass(params, batch_x)
    return jnp.mean(jnp.argmax(logits, -1) == batch_y)

# Compile functions for speed
loss_fn_jit = jit(loss_fn)
accuracy_jit = jit(accuracy)

# Gradient function
grad_fn = jit(grad(loss_fn))

# Optimizer
optimizer = optax.adam(learning_rate=1e-3)
opt_state = optimizer.init(params)

# Training step
@jit
def train_step(params, opt_state, batch_x, batch_y):
    grads = grad_fn(params, batch_x, batch_y)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state

# Training loop
for epoch in range(10):
    for batch_x, batch_y in dataloader:  # Your dataloader here
        params, opt_state = train_step(params, opt_state, batch_x, batch_y)
    
    # Evaluate
    train_loss = loss_fn_jit(params, batch_x, batch_y)
    train_acc = accuracy_jit(params, batch_x, batch_y)
    print(f"Epoch {epoch+1}, Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
```

## Memory Optimization

### Dynamic Memory Allocation

```python
import os
import jax

# Enable dynamic memory allocation (recommended on the shared H200 box).
# IMPORTANT: set these BEFORE jax/jaxlib is first imported, e.g. at the very
# top of your script or in the job's environment.
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

# Pick ONE strategy — these two are mutually exclusive:
#   * PREALLOCATE=false               -> grow memory on demand (polite default
#                                        on a shared box; co-tenant jobs can be
#                                        stacked onto the same GPU by gpuq).
#   * PREALLOCATE=true + MEM_FRACTION -> a single up-front preallocation capped
#                                        at the given fraction of the card.
# MEM_FRACTION only governs the size of that up-front preallocation, so it has
# NO effect once PREALLOCATE=false. Do not set both expecting a soft cap.
#
# Example of the capped-preallocation strategy (note: ~0.9 of a 141 GB H200 is
# ~127 GB, a large reservation on a shared server — prefer a modest fraction):
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'true'
# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.5'

# Check per-device memory usage
def print_memory_usage():
    for d in jax.devices():
        try:
            stats = d.memory_stats()  # dict with bytes_in_use / bytes_limit, etc.
            in_use = stats.get('bytes_in_use', 0) / 1024**3
            limit = stats.get('bytes_limit', 0) / 1024**3
            print(f"{d}: {in_use:.2f} GiB in use / {limit:.2f} GiB limit")
        except Exception:
            print(f"{d}: memory stats not available")
```

### Gradient Checkpointing with JAX

```python
from jax import checkpoint  # public API (aka jax.remat); jax.experimental.remat is gone

def create_checkpointed_model(layers):
    """Create model with gradient checkpointing"""
    
    @checkpoint
    def checkpointed_layer(x, params):
        W, b = params
        return jnp.maximum(0, jnp.dot(x, W) + b)
    
    def forward_pass_checkpointed(params, x):
        for layer_params in params[:-1]:
            x = checkpointed_layer(x, layer_params)
        
        # Final layer without checkpointing
        W, b = params[-1]
        return jnp.dot(x, W) + b
    
    return forward_pass_checkpointed

# Use checkpointed model for memory efficiency
checkpointed_forward = create_checkpointed_model(params)

# For Flax modules, use flax.linen.remat to checkpoint a submodule instead.
```

### Mixed Precision Training

```python
import jax.numpy as jnp
from jax import grad, jit

def mixed_precision_forward(params, x):
    """Forward pass with mixed precision"""
    # Convert to half precision for computation. On Hopper, bfloat16 is usually
    # preferred over float16 for training (wider dynamic range, no loss scaling).
    x = x.astype(jnp.bfloat16)
    
    for W, b in params[:-1]:
        W = W.astype(jnp.bfloat16)
        b = b.astype(jnp.bfloat16)
        x = jnp.maximum(0, jnp.dot(x, W) + b)
    
    # Final layer in float32 for numerical stability
    W, b = params[-1]
    x = x.astype(jnp.float32)
    W = W.astype(jnp.float32)
    b = b.astype(jnp.float32)
    
    return jnp.dot(x, W) + b

def mixed_precision_loss(params, batch_x, batch_y):
    """Loss function with mixed precision"""
    logits = mixed_precision_forward(params, batch_x)
    return optax.softmax_cross_entropy_with_integer_labels(logits, batch_y).mean()

# Use mixed precision in training
mp_grad_fn = jit(grad(mixed_precision_loss))
```

### Large Batch Training with Gradient Accumulation

```python
def gradient_accumulation_step(params, opt_state, batches, accumulation_steps):
    """Accumulate gradients over multiple batches"""
    
    def compute_grads(params, batch):
        batch_x, batch_y = batch
        return grad_fn(params, batch_x, batch_y)
    
    # Compute gradients for each mini-batch
    all_grads = [compute_grads(params, batch) for batch in batches]
    
    # Average gradients
    def average_grads(grad_list):
        if isinstance(grad_list[0], (list, tuple)):
            return [average_grads([g[i] for g in grad_list]) 
                   for i in range(len(grad_list[0]))]
        else:
            return jnp.mean(jnp.stack(grad_list), axis=0)
    
    averaged_grads = average_grads(all_grads)
    
    # Apply updates
    updates, opt_state = optimizer.update(averaged_grads, opt_state)
    params = optax.apply_updates(params, updates)
    
    return params, opt_state

# Use gradient accumulation for large effective batch sizes
accumulation_steps = 4
mini_batches = [next(dataloader) for _ in range(accumulation_steps)]
params, opt_state = gradient_accumulation_step(params, opt_state, mini_batches, accumulation_steps)
```

## Performance Optimization

### JIT Compilation and XLA Optimization

```python
import jax
from jax import jit
from functools import partial

# Aggressive JIT compilation for performance
@partial(jit, static_argnums=(2,))  # Make certain arguments static
def optimized_forward_pass(params, x, num_classes):
    """Highly optimized forward pass"""
    # Ensure optimal tensor shapes for H200
    x = jnp.reshape(x, (-1, x.shape[-1]))  # Flatten if needed
    
    for W, b in params[:-1]:
        # Fused operations for better performance
        x = jax.nn.relu(jnp.dot(x, W) + b)
    
    W, b = params[-1]
    logits = jnp.dot(x, W) + b
    
    return jnp.reshape(logits, (-1, num_classes))

# Compile training step with static arguments
@partial(jit, static_argnums=(3,))
def optimized_train_step(params, opt_state, batch, num_classes):
    batch_x, batch_y = batch
    
    def loss_fn_inner(p):
        logits = optimized_forward_pass(p, batch_x, num_classes)
        return optax.softmax_cross_entropy_with_integer_labels(logits, batch_y).mean()
    
    loss, grads = jax.value_and_grad(loss_fn_inner)(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    
    return params, opt_state, loss
```

### Optimized Data Loading

```python
import numpy as np
from functools import partial

def create_data_iterator(dataset, batch_size, shuffle=True):
    """Create optimized data iterator for H200"""
    indices = np.arange(len(dataset))
    
    if shuffle:
        np.random.shuffle(indices)
    
    for start_idx in range(0, len(indices), batch_size):
        end_idx = min(start_idx + batch_size, len(indices))
        batch_indices = indices[start_idx:end_idx]
        
        # Load batch and move to GPU
        batch_x = jnp.array([dataset[i][0] for i in batch_indices])
        batch_y = jnp.array([dataset[i][1] for i in batch_indices])
        
        yield batch_x, batch_y

# Prefetch data to overlap computation and data loading
def prefetch_iterator(iterator, prefetch_size=2):
    """Prefetch batches to overlap with computation"""
    import threading
    from queue import Queue
    
    queue = Queue(maxsize=prefetch_size)
    
    def producer():
        for batch in iterator:
            queue.put(batch)
        queue.put(None)  # Sentinel
    
    thread = threading.Thread(target=producer)
    thread.start()
    
    while True:
        batch = queue.get()
        if batch is None:
            break
        yield batch
    
    thread.join()
```

### Efficient Matrix Operations

```python
def optimized_attention(query, key, value, mask=None):
    """Optimized attention mechanism for H200"""
    # Ensure tensor dimensions are optimal for Tensor Cores
    d_k = query.shape[-1]
    
    # Compute attention scores with fused operations
    scores = jnp.matmul(query, key.transpose(-2, -1)) / jnp.sqrt(d_k)
    
    if mask is not None:
        scores = jnp.where(mask, scores, jnp.finfo(scores.dtype).min)
    
    # Softmax with numerical stability
    attention_weights = jax.nn.softmax(scores, axis=-1)
    
    # Apply attention to values
    output = jnp.matmul(attention_weights, value)
    
    return output, attention_weights

# Use optimal tensor shapes (multiples of 8 for H200 Tensor Cores)
def ensure_optimal_shapes(x, target_multiple=8):
    """Pad tensors to optimal shapes for H200"""
    shape = x.shape
    new_shape = []
    
    for dim in shape:
        if dim % target_multiple != 0:
            new_dim = ((dim // target_multiple) + 1) * target_multiple
        else:
            new_dim = dim
        new_shape.append(new_dim)
    
    if new_shape != list(shape):
        pad_width = [(0, new_dim - old_dim) for new_dim, old_dim in zip(new_shape, shape)]
        x = jnp.pad(x, pad_width, mode='constant')
    
    return x
```

The four H200 NVL cards deliver roughly ~836 TFLOP/s of empirical dense BF16
tensor-core matmul throughput each, so keeping matmul dimensions as multiples of
8 (ideally 16) lets XLA hit the Tensor Cores.

## Multi-GPU Training

The server exposes 4 H200 devices, so prefer `num_devices = len(jax.devices())`
over a hard-coded count — this keeps your code correct if a card is masked out
via `CUDA_VISIBLE_DEVICES` or if the hardware changes.

### Data Parallelism with pmap

```python
import jax
from jax import pmap, devices, device_put_replicated
import jax.numpy as jnp

# Check available devices (4 on this server, unless masked)
num_devices = len(devices())
print(f"Available devices: {num_devices}")

def replicate_params(params):
    """Replicate parameters across all devices"""
    return device_put_replicated(params, devices())

# Parallel training step
@pmap
def parallel_train_step(params, opt_state, batch):
    batch_x, batch_y = batch
    
    def loss_fn(p):
        logits = forward_pass(p, batch_x)
        return optax.softmax_cross_entropy_with_integer_labels(logits, batch_y).mean()
    
    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    
    return params, opt_state, loss

# Replicate initial state
replicated_params = replicate_params(params)
replicated_opt_state = device_put_replicated(opt_state, devices())

# Create parallel data batches
def create_parallel_batches(batch, num_devices):
    """Split batch across devices"""
    batch_x, batch_y = batch
    batch_size = batch_x.shape[0]
    
    # Ensure batch size is divisible by number of devices
    per_device_batch_size = batch_size // num_devices
    
    parallel_x = batch_x[:per_device_batch_size * num_devices]
    parallel_y = batch_y[:per_device_batch_size * num_devices]
    
    parallel_x = parallel_x.reshape(num_devices, per_device_batch_size, *batch_x.shape[1:])
    parallel_y = parallel_y.reshape(num_devices, per_device_batch_size, *batch_y.shape[1:])
    
    return parallel_x, parallel_y

# Training loop with pmap
for epoch in range(num_epochs):
    for batch in dataloader:
        parallel_batch = create_parallel_batches(batch, num_devices)
        
        replicated_params, replicated_opt_state, losses = parallel_train_step(
            replicated_params, replicated_opt_state, parallel_batch
        )
        
        # Average loss across devices
        avg_loss = jnp.mean(losses)
        print(f"Average loss: {avg_loss:.4f}")
```

### Model Parallelism with a Mesh and jax.jit

Modern JAX does model/tensor parallelism through `jax.sharding` plus a sharding-
aware `jax.jit`. The old `jax.experimental.pjit.pjit`,
`jax.experimental.PartitionSpec`, and `jax.experimental.Mesh` symbols have been
removed and will raise `ImportError`/`AttributeError` on a current
`jax[cuda12]`. Use `jax.sharding.Mesh` / `PartitionSpec` / `NamedSharding`, and
pass shardings to `jax.jit` via `in_shardings`/`out_shardings`.

```python
import jax
import jax.numpy as jnp
from jax import random
from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec, NamedSharding

# Create a device mesh for model parallelism across all available H200s.
# Derive the count dynamically rather than hard-coding it (4 on this box).
num_devices = len(jax.devices())
devices_array = mesh_utils.create_device_mesh((1, num_devices))  # 1 x N for N H200s
mesh = Mesh(devices_array, ('data', 'model'))

# Helper to turn a PartitionSpec into a concrete Sharding on this mesh.
def shard(spec):
    return NamedSharding(mesh, PartitionSpec(*spec))

def create_sharded_params(layer_sizes, key):
    """Create parameters sharded across the model dimension"""
    params = []
    keys = random.split(key, len(layer_sizes))
    
    for i in range(len(layer_sizes) - 1):
        W_key, b_key = random.split(keys[i])
        
        # Shard the weight matrix along its output dimension
        W = random.normal(W_key, (layer_sizes[i], layer_sizes[i + 1]))
        W = jax.device_put(W, shard((None, 'model')))
        
        # Shard the bias along the model dimension
        b = jnp.zeros(layer_sizes[i + 1])
        b = jax.device_put(b, shard(('model',)))
        
        params.append((W, b))
    
    return params

# Sharded forward pass: a plain jitted function; shardings on the inputs flow
# through automatically (no @pjit decorator needed).
@jax.jit
def sharded_forward_pass(params, x):
    """Forward pass with model parallelism"""
    for W, b in params[:-1]:
        x = jnp.maximum(0, jnp.dot(x, W) + b)
    
    W, b = params[-1]
    return jnp.dot(x, W) + b

def _sharded_train_step(params, opt_state, batch_x, batch_y):
    def loss_fn(p):
        logits = sharded_forward_pass(p, batch_x)
        return optax.softmax_cross_entropy_with_integer_labels(logits, batch_y).mean()
    
    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    
    return params, opt_state, loss

# Use model parallelism
with mesh:
    sharded_params = create_sharded_params(layer_sizes, key)
    
    # Build the training step with explicit input/output shardings via jax.jit.
    sharded_train_step = jax.jit(
        _sharded_train_step,
        in_shardings=(
            None,                       # params keep their existing shardings
            None,                       # opt_state likewise
            shard(('data', None)),      # inputs sharded along the data axis
            shard(('data',)),           # labels sharded along the data axis
        ),
        out_shardings=(None, None, None),
    )
```

For finer control inside a jitted function, use
`jax.lax.with_sharding_constraint(x, NamedSharding(mesh, PartitionSpec(...)))`
to pin intermediate values to a particular layout.

### Multi-GPU Job Submission

`gpuq` is daemonless: `gpuq submit` runs your job in the **foreground** of the
current terminal (its stdout/stderr stay in your terminal — redirect to a file
yourself if you want a log). It does not write per-job log files.

```bash
# Submit a multi-GPU JAX job that uses all four H200 cards
gpuq submit \
  --command "python train_jax_multi.py --num-devices=4 --model-parallel" \
  --gpus 4 \           # claim 4 whole GPUs (the box has 4 x H200 NVL, ~141 GB each)
  --memory 100 \       # selection floor: only pick GPUs with >= 100 GB FREE VRAM
  --time 16            # max runtime in hours; the job is killed past this
```

A few things worth knowing about these flags (per `gpuq` itself):

- `--gpus N` claims `N` whole GPUs chosen at random among the free cards. Use
  `--gpus 4` for all four; use a smaller number for a partial run.
- `--memory GB` is **not** a per-GPU reservation or a hard cap on your job — it
  is the *minimum free VRAM a candidate GPU must have to be selected* (an
  admission floor). `--memory 100` simply means "only schedule me on a GPU with
  at least 100 GB free right now."
- `--devices 0,1` pins exact cards (the count is taken from the list). Pinning
  is rejected immediately if any listed GPU is held by another user, **unless**
  you also pass `--queue`, in which case it waits until all of them are free or
  already yours.
- `--queue` waits and polls instead of exiting when no slot is free.
- `--notify you@example.com` overrides the completion-notice address (by default
  the address read from your account). There is no `--email` flag.

```bash
# Pin specific cards and wait for them if they are busy
gpuq submit --devices 0,1 --queue -- python train_jax_multi.py --num-devices=2
```

## Large Model Training

### Flax for Large Models

```python
import flax.linen as nn
from flax.training import train_state
import optax

class LargeTransformer(nn.Module):
    vocab_size: int = 50000
    d_model: int = 4096
    num_heads: int = 32
    num_layers: int = 24
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, x, training=False):
        # Token and position embeddings
        x = nn.Embed(self.vocab_size, self.d_model)(x)
        x = x + self.get_positional_encoding(x.shape[1], self.d_model)
        
        # Dropout
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        
        # Transformer layers with checkpointing
        for _ in range(self.num_layers):
            x = TransformerBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate
            )(x, training=training)
        
        # Output projection
        x = nn.Dense(self.vocab_size)(x)
        return x
    
    def get_positional_encoding(self, seq_len, d_model):
        """Sinusoidal positional encoding"""
        pos = jnp.arange(seq_len)[:, None]
        i = jnp.arange(d_model)[None, :]
        angle_rates = 1 / jnp.power(10000, (2 * (i // 2)) / d_model)
        angle_rads = pos * angle_rates
        
        # Apply sin to even indices
        angle_rads = angle_rads.at[:, 0::2].set(jnp.sin(angle_rads[:, 0::2]))
        # Apply cos to odd indices
        angle_rads = angle_rads.at[:, 1::2].set(jnp.cos(angle_rads[:, 1::2]))
        
        return angle_rads[None, ...]

class TransformerBlock(nn.Module):
    d_model: int
    num_heads: int
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, x, training=False):
        # Multi-head attention with residual connection
        attn_output = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
            deterministic=not training
        )(x, x)
        
        x = nn.LayerNorm()(x + attn_output)
        
        # Feed-forward network with residual connection
        ff_output = nn.Dense(self.d_model * 4)(x)
        ff_output = nn.gelu(ff_output)
        ff_output = nn.Dense(self.d_model)(ff_output)
        ff_output = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(ff_output)
        
        x = nn.LayerNorm()(x + ff_output)
        
        return x

# Initialize large model
model = LargeTransformer()
key = jax.random.PRNGKey(42)
dummy_input = jnp.ones((1, 512), dtype=jnp.int32)

# Initialize parameters
variables = model.init(key, dummy_input, training=False)
params = variables['params']

print(f"Model parameters: {sum(x.size for x in jax.tree_util.tree_leaves(params))}")
```

### Memory-Efficient Training with Orbax

```python
import orbax.checkpoint as ocp
from flax.training import train_state

def create_train_state(model, learning_rate, weight_decay):
    """Create training state with optimizer"""
    # AdamW optimizer with weight decay
    optimizer = optax.adamw(
        learning_rate=learning_rate,
        weight_decay=weight_decay
    )
    
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer
    )

# Checkpoint manager (current Orbax API: pass options, save/restore with ocp.args).
# The old positional ocp.PyTreeCheckpointer() argument is deprecated.
checkpoint_manager = ocp.CheckpointManager(
    'checkpoints/',
    options=ocp.CheckpointManagerOptions(
        save_interval_steps=1000,
        max_to_keep=3,
    ),
)

# Training step with gradient clipping
@jit
def train_step_with_clipping(state, batch, dropout_rng):
    def loss_fn(params):
        logits = model.apply(
            {'params': params},
            batch['input_ids'],
            training=True,
            rngs={'dropout': dropout_rng}
        )
        
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits[:, :-1], batch['labels'][:, 1:]
        ).mean()
        
        return loss
    
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    
    # Gradient clipping
    grads = jax.tree_util.tree_map(
        lambda g: jnp.clip(g, -1.0, 1.0), grads
    )
    
    state = state.apply_gradients(grads=grads)
    
    return state, loss

# Large model training loop
state = create_train_state(model, learning_rate=1e-4, weight_decay=0.01)

for step in range(num_training_steps):
    dropout_rng, rng = jax.random.split(rng)
    
    batch = next(dataloader)
    state, loss = train_step_with_clipping(state, batch, dropout_rng)
    
    if step % 100 == 0:
        print(f"Step {step}, Loss: {loss:.4f}")
    
    # Save checkpoint (current Orbax API uses ocp.args.StandardSave)
    if step % 1000 == 0:
        checkpoint_manager.save(step, args=ocp.args.StandardSave(state))
```

## Advanced Techniques

### Custom Gradient Transformations

```python
import optax
from typing import NamedTuple

class AdaptiveGradientState(NamedTuple):
    """State for adaptive gradient scaling"""
    step: int
    grad_norm_history: jnp.ndarray
    scale_factor: float

def adaptive_gradient_scaling(window_size=100, target_norm=1.0):
    """Custom gradient transformation with adaptive scaling"""
    
    def init_fn(params):
        return AdaptiveGradientState(
            step=0,
            grad_norm_history=jnp.zeros(window_size),
            scale_factor=1.0
        )
    
    def update_fn(updates, state, params=None):
        # Compute gradient norm
        grad_norm = optax.global_norm(updates)
        
        # Update history
        new_history = state.grad_norm_history.at[state.step % window_size].set(grad_norm)
        avg_grad_norm = jnp.mean(new_history)
        
        # Compute adaptive scale factor
        scale_factor = target_norm / (avg_grad_norm + 1e-8)
        scale_factor = jnp.clip(scale_factor, 0.1, 10.0)
        
        # Scale updates
        scaled_updates = jax.tree_util.tree_map(lambda x: x * scale_factor, updates)
        
        new_state = AdaptiveGradientState(
            step=state.step + 1,
            grad_norm_history=new_history,
            scale_factor=scale_factor
        )
        
        return scaled_updates, new_state
    
    return optax.GradientTransformation(init_fn, update_fn)

# Use custom optimizer
optimizer = optax.chain(
    adaptive_gradient_scaling(),
    optax.adam(1e-3)
)
```

### Advanced Data Pipeline

```python
import jax
import tensorflow as tf

def create_tf_data_pipeline(file_pattern, batch_size, seq_len):
    """Create optimized TensorFlow data pipeline for JAX"""
    
    def parse_example(example_proto):
        features = {
            'input_ids': tf.io.FixedLenFeature([seq_len], tf.int64),
            'labels': tf.io.FixedLenFeature([seq_len], tf.int64),
        }
        return tf.io.parse_single_example(example_proto, features)
    
    # Create dataset
    dataset = tf.data.Dataset.list_files(file_pattern, shuffle=True)
    dataset = dataset.interleave(
        tf.data.TFRecordDataset,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False
    )
    
    # Parse and batch
    dataset = dataset.map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

# Convert TF dataset to JAX arrays
def tf_to_jax_iterator(tf_dataset):
    """Convert TensorFlow dataset to JAX iterator"""
    for batch in tf_dataset:
        jax_batch = {
            'input_ids': jnp.array(batch['input_ids'].numpy()),
            'labels': jnp.array(batch['labels'].numpy()),
        }
        yield jax_batch
```

### Custom Loss Functions

```python
def focal_loss(logits, labels, alpha=0.25, gamma=2.0):
    """Focal loss for handling class imbalance"""
    ce_loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    p_t = jnp.exp(-ce_loss)  # Probability of true class
    focal_weight = alpha * (1 - p_t) ** gamma
    focal_loss = focal_weight * ce_loss
    return focal_loss.mean()

def label_smoothing_loss(logits, labels, smoothing=0.1):
    """Label smoothing cross-entropy loss"""
    num_classes = logits.shape[-1]
    smooth_labels = (1 - smoothing) * jax.nn.one_hot(labels, num_classes)
    smooth_labels += smoothing / num_classes
    
    log_probs = jax.nn.log_softmax(logits)
    loss = -jnp.sum(smooth_labels * log_probs, axis=-1)
    return loss.mean()

def contrastive_loss(embeddings1, embeddings2, labels, margin=1.0):
    """Contrastive loss for similarity learning"""
    euclidean_distance = jnp.linalg.norm(embeddings1 - embeddings2, axis=1)
    
    # Loss for positive pairs (label = 1)
    positive_loss = labels * jnp.square(euclidean_distance)
    
    # Loss for negative pairs (label = 0)
    negative_loss = (1 - labels) * jnp.square(jnp.maximum(0, margin - euclidean_distance))
    
    return jnp.mean(positive_loss + negative_loss) / 2
```

## Debugging and Profiling

### JAX Profiling

There are two independent profiling mechanisms — use whichever fits, but do not
mix them. `start_trace`/`stop_trace` captures a trace of a region to disk, while
`start_server(port)` launches a long-lived profiler server (called **once** at
program startup) for on-demand capture from TensorBoard.

```python
import jax
import jax.profiler

# (a) Trace a specific region and write it to disk
def profile_training_step():
    """Profile a window of training steps"""
    jax.profiler.start_trace("/tmp/jax_trace")
    try:
        for step in range(100):
            batch = next(dataloader)
            state, loss = train_step(state, batch)
    finally:
        jax.profiler.stop_trace()

# (b) Alternatively, start the on-demand profiler server ONCE at startup and
#     capture profiles from TensorBoard while the program runs:
#
#     jax.profiler.start_server(9999)
#
#     Do this once near the top of your program — never inside the training loop
#     (re-binding the port would error).

# Use JAX debugging utilities
def debug_shapes_and_dtypes(pytree, name="PyTree"):
    """Debug tensor shapes and dtypes"""
    
    def print_info(path, x):
        print(f"{name}.{'.'.join(map(str, path))}: shape={x.shape}, dtype={x.dtype}")
    
    jax.tree_util.tree_map_with_path(print_info, pytree)

# Check for NaN/Inf values
def check_for_nans(pytree, step_name=""):
    """Check for NaN or Inf values in pytree"""
    
    def has_nan_or_inf(x):
        return jnp.any(jnp.isnan(x)) or jnp.any(jnp.isinf(x))
    
    nan_mask = jax.tree_util.tree_map(has_nan_or_inf, pytree)
    
    if any(jax.tree_util.tree_leaves(nan_mask)):
        print(f"WARNING: NaN/Inf detected at {step_name}")
        debug_shapes_and_dtypes(pytree, step_name)
```

### Memory Profiling

```python
import psutil
import threading
import time

def memory_monitor(interval=5):
    """Monitor system and GPU memory usage"""
    
    def monitor():
        while True:
            # System memory (the host has 755 GiB RAM)
            memory = psutil.virtual_memory()
            print(f"System RAM: {memory.percent:.1f}% used, {memory.available/1024**3:.1f}GB available")
            
            # JAX per-device memory
            try:
                for i, device in enumerate(jax.devices()):
                    if device.platform == 'gpu':  # platform is 'gpu'; device_kind is the model name
                        try:
                            stats = device.memory_stats()
                            in_use = stats.get('bytes_in_use', 0) / 1024**3
                            print(f"GPU {i} ({device.device_kind}): {in_use:.2f} GiB in use")
                        except Exception:
                            print(f"GPU {i}: {device}")
            except Exception as e:
                print(f"GPU memory info not available: {e}")
            
            time.sleep(interval)
    
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    return thread

# Start memory monitoring
memory_thread = memory_monitor(interval=10)
```

> Note: a JAX device's `.platform` is `'gpu'`, while `.device_kind` is the model
> string (e.g. `'NVIDIA H200 NVL'`). Test `device.platform == 'gpu'` — comparing
> `device_kind == 'gpu'` is always False and would silently skip every card.

## Example Scripts

### Complete Training Script

```python
#!/usr/bin/env python3
"""
H200-Optimized JAX/Flax Training Script
Usage: gpuq submit --command "python train_jax_h200.py --config config.py" --gpus 1 --memory 100
"""

import jax
import jax.numpy as jnp
from jax import random, grad, jit, devices
import flax.linen as nn
from flax.training import train_state
import optax
import orbax.checkpoint as ocp
import argparse
import importlib.util
import logging
from pathlib import Path

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )

def load_config(config_path):
    """Load configuration from Python file"""
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config

def create_train_state(model, params, config):
    """Create training state"""
    if config.optimizer_name == 'adamw':
        optimizer = optax.adamw(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay
        )
    elif config.optimizer_name == 'adam':
        optimizer = optax.adam(learning_rate=config.learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer_name}")
    
    # Add gradient clipping if specified
    if hasattr(config, 'gradient_clip') and config.gradient_clip > 0:
        optimizer = optax.chain(
            optax.clip_by_global_norm(config.gradient_clip),
            optimizer
        )
    
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer
    )

@jit
def train_step(state, batch, dropout_rng):
    """Compiled training step"""
    
    def loss_fn(params):
        logits = state.apply_fn(
            {'params': params},
            batch['input'],
            training=True,
            rngs={'dropout': dropout_rng}
        )
        
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits, batch['target']
        ).mean()
        
        return loss, logits
    
    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    
    # Compute accuracy
    accuracy = jnp.mean(jnp.argmax(logits, -1) == batch['target'])
    
    return state, loss, accuracy

@jit
def eval_step(state, batch):
    """Compiled evaluation step"""
    logits = state.apply_fn({'params': state.params}, batch['input'], training=False)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch['target']).mean()
    accuracy = jnp.mean(jnp.argmax(logits, -1) == batch['target'])
    return loss, accuracy

def evaluate_model(state, eval_dataloader):
    """Evaluate model on validation set"""
    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = 0
    
    for batch in eval_dataloader:
        loss, accuracy = eval_step(state, batch)
        total_loss += loss
        total_accuracy += accuracy
        num_batches += 1
    
    return total_loss / num_batches, total_accuracy / num_batches

def main():
    parser = argparse.ArgumentParser(description='JAX/Flax Training Script')
    parser.add_argument('--config', required=True, help='Configuration file')
    parser.add_argument('--resume', help='Resume from checkpoint')
    args = parser.parse_args()
    
    setup_logging()
    config = load_config(args.config)
    
    # Setup JAX
    logging.info(f"JAX devices: {devices()}")
    
    # Create model
    if config.model_type == 'transformer':
        model = LargeTransformer(**config.model_params)
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")
    
    # Initialize parameters
    key = random.PRNGKey(config.seed)
    init_key, train_key = random.split(key)
    
    dummy_input = jnp.ones((1, config.sequence_length), dtype=jnp.int32)
    variables = model.init(init_key, dummy_input, training=False)
    params = variables['params']
    
    # Count parameters
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    logging.info(f"Model parameters: {param_count:,}")
    
    # Create training state
    state = create_train_state(model, params, config)
    
    # Setup checkpointing (current Orbax API: options-only constructor)
    checkpoint_manager = ocp.CheckpointManager(
        config.checkpoint_dir,
        options=ocp.CheckpointManagerOptions(
            save_interval_steps=config.save_interval,
            max_to_keep=3,
        ),
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        state = checkpoint_manager.restore(
            int(args.resume), args=ocp.args.StandardRestore(state)
        )
        logging.info(f"Resumed from checkpoint: {args.resume}")
    
    # Create data loaders (implement based on your data)
    train_dataloader = create_dataloader(config, split='train')
    eval_dataloader = create_dataloader(config, split='eval')
    
    # Training loop
    step = 0
    for epoch in range(config.num_epochs):
        logging.info(f"Starting epoch {epoch+1}/{config.num_epochs}")
        
        # Training phase
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        num_batches = 0
        
        for batch in train_dataloader:
            # Split RNG for dropout
            train_key, dropout_key = random.split(train_key)
            
            # Training step
            state, loss, accuracy = train_step(state, batch, dropout_key)
            
            epoch_loss += loss
            epoch_accuracy += accuracy
            num_batches += 1
            step += 1
            
            # Logging
            if step % config.log_interval == 0:
                logging.info(f"Step {step}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
            
            # Evaluation
            if step % config.eval_interval == 0:
                eval_loss, eval_accuracy = evaluate_model(state, eval_dataloader)
                logging.info(f"Evaluation - Loss: {eval_loss:.4f}, Accuracy: {eval_accuracy:.4f}")
            
            # Checkpointing
            if step % config.save_interval == 0:
                checkpoint_manager.save(step, args=ocp.args.StandardSave(state))
                logging.info(f"Checkpoint saved at step {step}")
        
        # Epoch summary
        avg_loss = epoch_loss / num_batches
        avg_accuracy = epoch_accuracy / num_batches
        logging.info(f"Epoch {epoch+1} completed - Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")
    
    # Final checkpoint
    checkpoint_manager.save(step, args=ocp.args.StandardSave(state))
    logging.info("Training completed!")

if __name__ == '__main__':
    main()
```

### Configuration File (config.py)

```python
# JAX/Flax Training Configuration
import jax.numpy as jnp

# Model configuration
model_type = 'transformer'
model_params = {
    'vocab_size': 32000,
    'd_model': 1024,
    'num_heads': 16,
    'num_layers': 12,
    'dropout_rate': 0.1,
}

# Training configuration
num_epochs = 10
batch_size = 32
sequence_length = 512
learning_rate = 1e-4
weight_decay = 0.01
gradient_clip = 1.0

# Optimizer
optimizer_name = 'adamw'

# Data configuration
train_data_path = '/path/to/train_data'
eval_data_path = '/path/to/eval_data'

# Checkpointing
checkpoint_dir = './checkpoints'
save_interval = 1000
log_interval = 100
eval_interval = 500

# Miscellaneous
seed = 42
```

---

**Next Steps**:
- For best practices across frameworks: [Best Practices Guide](best-practices.md)  
- For troubleshooting: [Troubleshooting Guide](troubleshooting.md)
- For ready-to-use examples: [Example Scripts](../examples/)
