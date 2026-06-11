# TensorFlow/Keras with NVIDIA H200 GPUs

Complete guide for optimizing TensorFlow and Keras workflows on the Ruqola server's H200 GPUs.

The Ruqola server (host `wsserver1`, the NTU "Mjolnir" box) has **4 x NVIDIA H200 NVL** GPUs at indices `0,1,2,3`, each with ~141 GB VRAM (~564 GB total), compute capability **9.0** (Hopper). The host runs Ubuntu 24.04, GPU driver 575.57.08, CUDA (driver) 12.9.

## 📖 Table of Contents

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

### Recommended TensorFlow Installation

```bash
# Current TensorFlow with bundled CUDA libraries (recommended for H200).
# The [and-cuda] extra self-selects a matching CUDA 12.x + cuDNN; you do not
# pin the CUDA version yourself. The host driver is CUDA 12.9.
pip install -U "tensorflow[and-cuda]"

# If you need to pin, use a recent release (>=2.16) for the best Hopper kernels:
# pip install "tensorflow[and-cuda]>=2.16"

# Verify GPU detection
python -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"
```

The H200 reports compute capability **9.0** (Hopper, `sm_90`). Any reasonably current TF 2.x build supports it; newer releases ship better Hopper kernels.

### Environment Setup

```bash
# Add to ~/.bashrc or job script
export CUDA_VISIBLE_DEVICES=0      # Use first H200, or 0,1,2,3 for all four H200s
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_ALLOCATOR=cuda_malloc_async
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda
```

> **On the shared server, let gpuq pick your GPUs.** When you launch through
> `gpuq submit`, it sets `CUDA_VISIBLE_DEVICES` for the GPU(s) you were allocated.
> Do **not** hard-code `CUDA_VISIBLE_DEVICES` over a gpuq allocation — `gpuq audit`
> watches for jobs running on a GPU other than the one they were assigned (rebind
> detection) and will flag the override. Only set it manually for quick interactive
> work outside the queue.

### Verify Installation

```python
import tensorflow as tf

print(f"TensorFlow version: {tf.__version__}")
print(f"CUDA available: {tf.test.is_built_with_cuda()}")
print('GPU available:', bool(tf.config.list_physical_devices('GPU')))

# List GPUs
gpus = tf.config.list_physical_devices('GPU')
print(f"Number of GPUs: {len(gpus)}")  # expect 4 on this server

for i, gpu in enumerate(gpus):
    print(f"GPU {i}: {gpu}")
    details = tf.config.experimental.get_device_details(gpu)
    print(f"  Compute Capability: {details.get('compute_capability', 'N/A')}")  # (9, 0) for H200
```

### GPU Configuration

```python
import tensorflow as tf

# Configure GPU memory growth (recommended for H200)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # Memory growth must be set before GPUs are initialized
        print(e)

# Alternative: cap a GPU's VRAM with a logical device configuration.
# There is NO tf.config.experimental.set_memory_limit() — use
# set_logical_device_configuration() instead. This must run before the GPU
# is initialized (i.e. before any tensor/op touches it).
if gpus:
    try:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=100 * 1024)],  # 100 GB cap, in MB
        )
    except RuntimeError as e:
        print(e)
```

## Basic GPU Usage

### Device Placement

```python
import tensorflow as tf

# Automatic device placement
with tf.device('/GPU:0'):
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
    c = tf.matmul(a, b)

print(f"Result computed on: {c.device}")

# Check tensor location
def check_device(tensor, name):
    print(f"{name} is on device: {tensor.device}")

check_device(a, "Tensor a")
```

### Simple Model Training

```python
import tensorflow as tf
from tensorflow import keras

# Create a simple model
model = keras.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')
])

# Compile with GPU-optimized settings
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train on GPU
history = model.fit(
    x_train, y_train,
    batch_size=128,
    epochs=10,
    validation_data=(x_val, y_val),
    verbose=1
)
```

### Memory Usage Monitoring

```python
import tensorflow as tf

def print_memory_usage():
    """Print current GPU memory usage"""
    gpu_stats = tf.config.experimental.get_memory_info('GPU:0')
    current_mb = gpu_stats['current'] / 1024 / 1024
    peak_mb = gpu_stats['peak'] / 1024 / 1024
    print(f"GPU Memory - Current: {current_mb:.1f}MB, Peak: {peak_mb:.1f}MB")

# Use during training
print_memory_usage()
```

## Memory Optimization

### Mixed Precision Training

```python
import tensorflow as tf

# Enable mixed precision globally. On Hopper (H200) you can also use 'mixed_bfloat16',
# which avoids loss scaling and is numerically robust.
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

# Create model with mixed precision
model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax', dtype='float32')  # Output in float32
])

# Compile with loss scaling (needed for float16; not needed for bfloat16)
optimizer = tf.keras.optimizers.Adam()
optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

### Gradient Checkpointing

```python
import tensorflow as tf

class CheckpointedDense(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.dense = tf.keras.layers.Dense(units)
    
    def call(self, inputs):
        # Use gradient checkpointing to save memory
        return tf.recompute_grad(self.dense)(inputs)

# Build memory-efficient model
model = tf.keras.Sequential([
    CheckpointedDense(1024),
    CheckpointedDense(1024),
    CheckpointedDense(1024),
    tf.keras.layers.Dense(10)
])
```

### Dynamic Memory Allocation

```python
import tensorflow as tf

# Enable memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Use tf.function for memory efficiency
@tf.function
def optimized_train_step(model, x, y, optimizer):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y, predictions)
        loss = tf.reduce_mean(loss)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss
```

### Large Batch Training with Gradient Accumulation

```python
class GradientAccumulator:
    def __init__(self):
        self.accumulated_gradients = []
        self.gradient_count = 0
    
    def accumulate_gradients(self, gradients):
        if self.gradient_count == 0:
            self.accumulated_gradients = [tf.Variable(tf.zeros_like(g)) for g in gradients]
        
        for i, g in enumerate(gradients):
            self.accumulated_gradients[i].assign_add(g)
        
        self.gradient_count += 1
    
    def average_and_apply(self, optimizer, model_variables):
        # Average accumulated gradients
        averaged_grads = [g / self.gradient_count for g in self.accumulated_gradients]
        
        # Apply gradients
        optimizer.apply_gradients(zip(averaged_grads, model_variables))
        
        # Reset
        self.gradient_count = 0
        for g in self.accumulated_gradients:
            g.assign(tf.zeros_like(g))

def train_with_accumulation(model, dataset, optimizer, accumulation_steps=4):
    accumulator = GradientAccumulator()
    
    for step, (x_batch, y_batch) in enumerate(dataset):
        with tf.GradientTape() as tape:
            predictions = model(x_batch, training=True)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_batch, predictions)
            loss = tf.reduce_mean(loss) / accumulation_steps
        
        gradients = tape.gradient(loss, model.trainable_variables)
        accumulator.accumulate_gradients(gradients)
        
        if (step + 1) % accumulation_steps == 0:
            accumulator.average_and_apply(optimizer, model.trainable_variables)
```

## Performance Optimization

### XLA Compilation

```python
import tensorflow as tf

# Enable XLA compilation
@tf.function(jit_compile=True)
def optimized_model(x):
    return model(x)

# Or compile the entire model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
    jit_compile=True  # Enable XLA
)
```

XLA is the most portable, best-maintained way to squeeze throughput out of the H200 and is the recommended fast path for both training and inference.

### TensorRT Optimization (legacy)

> **Legacy / not recommended for new work.** The in-graph TF-TRT integration
> (`tensorflow.python.compiler.tensorrt`) is effectively unmaintained, and the
> standalone `tensorrt` pip wheel frequently mismatches the CUDA libraries
> bundled with `tensorflow[and-cuda]`, breaking the integration. For H200
> inference, prefer **XLA** (`jit_compile=True`) or export the model and serve it
> through **standalone TensorRT**. On Hopper you can also target **BF16/FP8**
> precision rather than only FP16.

```python
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt

# Convert model to TensorRT (legacy TF-TRT path)
def convert_to_tensorrt(saved_model_dir, output_dir):
    converter = trt.TrtGraphConverterV2(
        input_saved_model_dir=saved_model_dir,
        precision_mode=trt.TrtPrecisionMode.FP16,  # H200 also supports BF16/FP8
        maximum_cached_engines=1
    )
    converter.convert()
    converter.save(output_saved_model_dir=output_dir)

# Use TensorRT model
trt_model = tf.saved_model.load('trt_model_dir')
predictions = trt_model.signatures['serving_default'](input_tensor)
```

### Optimized Data Pipeline

```python
import tensorflow as tf

def create_optimized_dataset(file_pattern, batch_size=128):
    # Create dataset from files
    dataset = tf.data.Dataset.list_files(file_pattern)
    
    # Optimizations for H200
    dataset = dataset.interleave(
        tf.data.TFRecordDataset,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False
    )
    
    # Parse and preprocess
    dataset = dataset.map(
        parse_function,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Batch and prefetch
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

# Configure for optimal performance
dataset = create_optimized_dataset("train_*.tfrecord", batch_size=256)

# Additional optimizations (the host has 256 logical CPUs, so feel free to raise these)
options = tf.data.Options()
options.threading.private_threadpool_size = 8
options.threading.max_intra_op_parallelism = 8
dataset = dataset.with_options(options)
```

### Custom Training Loop

```python
@tf.function
def train_step(model, x, y, optimizer, loss_fn):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = loss_fn(y, predictions)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# Optimized training loop
def custom_training_loop(model, dataset, epochs=10):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for x_batch, y_batch in dataset:
            loss = train_step(model, x_batch, y_batch, optimizer, loss_fn)
            epoch_loss += loss
            num_batches += 1
        
        print(f"Epoch {epoch+1}, Loss: {epoch_loss/num_batches:.4f}")
```

## Multi-GPU Training

The server has 4 H200s, so data-parallel training can scale up to 4 replicas.

### MirroredStrategy (Data Parallelism)

```python
import tensorflow as tf

# Create distribution strategy
strategy = tf.distribute.MirroredStrategy()
print(f"Number of devices: {strategy.num_replicas_in_sync}")

# Create and compile model within strategy scope
with strategy.scope():
    model = create_model()
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

# Create distributed dataset
train_dataset = strategy.experimental_distribute_dataset(dataset)

# Training with strategy
model.fit(
    train_dataset,
    epochs=10,
    steps_per_epoch=steps_per_epoch
)
```

### Custom Multi-GPU Training

```python
@tf.function
def distributed_train_step(strategy, model, x, y, optimizer, loss_fn):
    def step_fn(x, y):
        with tf.GradientTape() as tape:
            predictions = model(x, training=True)
            per_replica_loss = loss_fn(y, predictions)
            loss = tf.reduce_mean(per_replica_loss)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss
    
    per_replica_loss = strategy.run(step_fn, args=(x, y))
    return strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, axis=None)

def multi_gpu_training():
    strategy = tf.distribute.MirroredStrategy()
    
    with strategy.scope():
        model = create_model()
        optimizer = tf.keras.optimizers.Adam()
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    
    # Distributed training loop
    for epoch in range(num_epochs):
        for x_batch, y_batch in distributed_dataset:
            loss = distributed_train_step(strategy, model, x_batch, y_batch, optimizer, loss_fn)
            print(f"Batch loss: {loss:.4f}")
```

### Multi-GPU Job Submission

```bash
# Submit a 4-GPU TensorFlow job (uses all four H200s).
# -m/--memory is the MINIMUM free VRAM (GB) a candidate GPU must have to be
# picked — it is an admission requirement, not a hard cap or reservation.
# The job runs in the FOREGROUND in this terminal; redirect output yourself
# if you want a log file.
gpuq submit \
  --command "python train_tensorflow.py --strategy=mirrored" \
  --gpus 4 \
  --memory 40 \
  --time 12

# For a 2-GPU run, just ask for 2:
gpuq submit --command "python train_tensorflow.py --strategy=mirrored" --gpus 2 --memory 40 --time 12
```

> Inside the job, read `strategy.num_replicas_in_sync` rather than hard-coding a
> GPU count — gpuq has already restricted `CUDA_VISIBLE_DEVICES` to the GPUs you
> were allocated, and `MirroredStrategy()` will mirror across exactly those.

## Large Model Training

### Memory-Efficient Large Models

```python
import tensorflow as tf

class LargeTransformer(tf.keras.Model):
    def __init__(self, vocab_size, d_model=4096, num_heads=32, num_layers=24):
        super().__init__()
        self.d_model = d_model
        
        # Embeddings
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)
        self.pos_encoding = self.positional_encoding(10000, d_model)
        
        # Transformer blocks with checkpointing
        self.transformer_blocks = [
            TransformerBlock(d_model, num_heads) for _ in range(num_layers)
        ]
        
        self.final_layer = tf.keras.layers.Dense(vocab_size)
    
    def call(self, x, training=False):
        seq_len = tf.shape(x)[1]
        
        # Embeddings + positional encoding
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        
        # Apply transformer blocks with checkpointing
        for i, block in enumerate(self.transformer_blocks):
            x = tf.recompute_grad(block)(x, training=training)
        
        return self.final_layer(x)

# Enable mixed precision for large models (bfloat16 is a good default on Hopper)
tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')

model = LargeTransformer(vocab_size=50000)
```

With ~141 GB of VRAM per H200, a single card holds substantial models; use
`MirroredStrategy` across the 4 GPUs (or model parallelism) only when one card is
not enough.

### Hugging Face Transformers with TensorFlow

```python
from transformers import TFAutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2-large")

# Load a large TF model. The gpt2-large repo ships native TF weights, so do NOT
# pass from_tf=True (that flag means "convert a TF checkpoint into a PyTorch
# model" and would error here).
model = TFAutoModelForCausalLM.from_pretrained(
    "gpt2-large",
    use_cache=False,  # Save memory during training
)
```

> **No `TFTrainer` / `TFTrainingArguments`.** Those classes were deprecated and
> then **removed** from Hugging Face `transformers` — importing them from a
> current release raises `ImportError`. For a TF model, train with the native
> Keras flow (`model.compile()` + `model.fit()`); if you would rather use the
> Hugging Face `Trainer` abstraction, switch to a PyTorch model and the PyTorch
> `Trainer` (see the [Transformers guide](transformers-guide.md)).

```python
import tensorflow as tf

# Enable mixed precision (bfloat16 on Hopper avoids loss scaling)
tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')

# Keras-native training for the TF model
optimizer = tf.keras.optimizers.AdamW(learning_rate=5e-5)
model.compile(optimizer=optimizer)  # HF TF models compute their own loss

model.fit(
    train_dataset,        # a tf.data.Dataset yielding the model's expected inputs
    validation_data=val_dataset,
    epochs=3,
)
model.save_pretrained("./results")
```

### Model Parallelism

> **Mesh TensorFlow is abandoned** and is not recommended on a current TF 2.x.
> For model/tensor parallelism today, use **Keras + DTensor** within TensorFlow,
> or use **JAX** (see the [JAX with H200 Guide](jax-guide.md)), which has
> first-class sharding (`jax.sharding.Mesh` / `PartitionSpec` /
> `jax.jit(in_shardings=...)`). The fragment below is illustrative legacy
> pseudo-code only and will not run as-is.

```python
# LEGACY / illustrative only — Mesh TensorFlow is unmaintained.
import tensorflow as tf
import mesh_tensorflow as mtf  # not maintained; may not install on current TF

def create_mesh_model(mesh_shape, layout):
    mesh = mtf.Mesh(tf.Graph(), 'mesh')
    # Model parallelism across H200s (pseudo-code; `inputs` is undefined)
    with mesh:
        model = mtf.layers.dense(
            inputs,
            output_dim=mtf.Dimension('vocab', 50000),
            mesh_impl=mesh
        )
    return model
```

## Advanced Techniques

### Custom Gradient Computation

```python
@tf.function
def custom_gradient_computation(model, x, y):
    with tf.GradientTape() as tape:
        predictions = model(x)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y, predictions)
        
        # Add regularization
        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in model.trainable_variables])
        total_loss = tf.reduce_mean(loss) + 0.01 * l2_loss
    
    # Compute gradients with clipping
    gradients = tape.gradient(total_loss, model.trainable_variables)
    clipped_gradients = [tf.clip_by_norm(g, 1.0) for g in gradients]
    
    return total_loss, clipped_gradients
```

### Dynamic Learning Rate Scheduling

```python
class CosineDecayWithWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, max_lr, warmup_steps, total_steps):
        super().__init__()
        self.max_lr = max_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
    
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        total_steps = tf.cast(self.total_steps, tf.float32)
        
        # Warmup phase
        warmup_lr = self.max_lr * step / warmup_steps
        
        # Cosine decay phase
        decay_steps = total_steps - warmup_steps
        cosine_decay = 0.5 * (1 + tf.cos(
            3.14159 * (step - warmup_steps) / decay_steps
        ))
        decay_lr = self.max_lr * cosine_decay
        
        return tf.where(step < warmup_steps, warmup_lr, decay_lr)

# Use custom scheduler
lr_schedule = CosineDecayWithWarmup(
    max_lr=0.001, 
    warmup_steps=1000, 
    total_steps=100000
)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
```

### Memory-Mapped Datasets

```python
import tensorflow as tf
import numpy as np

class MemoryMappedDataset:
    def __init__(self, data_path, batch_size):
        # Memory-map large datasets
        self.data = np.memmap(data_path, dtype='float32', mode='r')
        self.batch_size = batch_size
        self.length = len(self.data) // (28 * 28)  # Example for MNIST
    
    def __call__(self):
        for i in range(0, self.length, self.batch_size):
            end_idx = min(i + self.batch_size, self.length)
            batch_data = self.data[i*784:end_idx*784].reshape(-1, 28, 28)
            yield tf.constant(batch_data, dtype=tf.float32)

# Create TensorFlow dataset from memory-mapped data
dataset = tf.data.Dataset.from_generator(
    MemoryMappedDataset('large_data.dat', 128),
    output_signature=tf.TensorSpec(shape=(None, 28, 28), dtype=tf.float32)
)
```

## Debugging and Profiling

### TensorFlow Profiler

```python
import tensorflow as tf

# Enable profiling
tf.profiler.experimental.start('logs/profile')

# Your training code here
for step, (x, y) in enumerate(dataset):
    with tf.profiler.experimental.Trace('train', step_num=step):
        train_step(x, y)
    
    if step == 100:
        break

tf.profiler.experimental.stop()

# View profile in TensorBoard
# tensorboard --logdir=logs/profile
```

### Memory Usage Profiling

```python
import tensorflow as tf

def profile_memory_usage():
    # Monitor GPU memory
    def memory_callback():
        gpu_info = tf.config.experimental.get_memory_info('GPU:0')
        current_mb = gpu_info['current'] / 1024 / 1024
        peak_mb = gpu_info['peak'] / 1024 / 1024
        print(f"GPU Memory - Current: {current_mb:.1f}MB, Peak: {peak_mb:.1f}MB")
        
        # System memory
        import psutil
        ram_usage = psutil.virtual_memory().percent
        print(f"System RAM usage: {ram_usage:.1f}%")
    
    return memory_callback

# Use during training
memory_monitor = profile_memory_usage()

for epoch in range(num_epochs):
    memory_monitor()  # Call at key points
    # Training code...
```

### Debugging CUDA Errors

```python
import tensorflow as tf

# Enable eager execution for easier debugging
tf.config.run_functions_eagerly(True)

# Enable memory growth to avoid OOM
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Debug function
def debug_tensor(tensor, name="Tensor"):
    print(f"{name} - Shape: {tensor.shape}, Device: {tensor.device}")
    print(f"  Min: {tf.reduce_min(tensor):.4f}, Max: {tf.reduce_max(tensor):.4f}")
    print(f"  Mean: {tf.reduce_mean(tensor):.4f}, Std: {tf.math.reduce_std(tensor):.4f}")
    
    # Check for NaN/Inf
    if tf.reduce_any(tf.math.is_nan(tensor)):
        print(f"  WARNING: {name} contains NaN values!")
    if tf.reduce_any(tf.math.is_inf(tensor)):
        print(f"  WARNING: {name} contains Inf values!")

# Use in training loop
for x_batch, y_batch in dataset:
    debug_tensor(x_batch, "Input")
    predictions = model(x_batch)
    debug_tensor(predictions, "Predictions")
```

## Example Scripts

### Complete Training Script

```python
#!/usr/bin/env python3
"""
H200-Optimized TensorFlow Training Script
Usage: gpuq submit --command "python train_tf_h200.py --config config.json" --gpus 1 --memory 40
"""

import tensorflow as tf
import json
import argparse
import logging
from pathlib import Path
import numpy as np

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )

def setup_gpu(config):
    """Configure GPU settings for H200"""
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        try:
            # Enable memory growth
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Optionally cap VRAM via a logical device configuration.
            # NOTE: there is no tf.config.experimental.set_memory_limit();
            # use set_logical_device_configuration() instead, before the GPU
            # is initialized.
            if config.get('gpu_memory_limit'):
                tf.config.set_logical_device_configuration(
                    gpus[0],
                    [tf.config.LogicalDeviceConfiguration(
                        memory_limit=config['gpu_memory_limit'] * 1024  # MB
                    )],
                )
            
            logging.info(f"Configured {len(gpus)} GPUs")
            
        except (RuntimeError, ValueError) as e:
            # Memory growth / logical device config must be set before init;
            # catch both RuntimeError and ValueError.
            logging.error(f"GPU configuration error: {e}")

def create_model(config):
    """Create model based on configuration"""
    if config['model']['type'] == 'transformer':
        from models.transformer import TransformerModel
        model = TransformerModel(**config['model']['params'])
    elif config['model']['type'] == 'cnn':
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(128, 3, activation='relu'),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(config['model']['num_classes'])
        ])
    else:
        raise ValueError(f"Unknown model type: {config['model']['type']}")
    
    return model

def create_dataset(config):
    """Create optimized dataset pipeline"""
    def parse_function(example):
        # Your parsing logic here
        pass
    
    # Create dataset from files
    dataset = tf.data.Dataset.list_files(config['data']['train_pattern'])
    dataset = dataset.interleave(
        tf.data.TFRecordDataset,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False
    )
    
    # Parse and preprocess
    dataset = dataset.map(parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Batch and optimize
    dataset = dataset.batch(config['training']['batch_size'])
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

@tf.function
def train_step(model, x, y, optimizer, loss_fn, metric):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = loss_fn(y, predictions)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    metric.update_state(y, predictions)
    return loss

def train_epoch(model, dataset, optimizer, loss_fn, metric):
    epoch_loss = 0.0
    num_batches = 0
    
    for x_batch, y_batch in dataset:
        batch_loss = train_step(model, x_batch, y_batch, optimizer, loss_fn, metric)
        epoch_loss += batch_loss
        num_batches += 1
        
        if num_batches % 100 == 0:
            current_accuracy = metric.result().numpy()
            logging.info(f'Batch {num_batches}, Loss: {batch_loss:.4f}, Accuracy: {current_accuracy:.4f}')
    
    return epoch_loss / num_batches

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Config file path')
    parser.add_argument('--resume', help='Resume from checkpoint')
    args = parser.parse_args()
    
    setup_logging()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Setup GPU and mixed precision
    setup_gpu(config)
    
    if config['training']['mixed_precision']:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        logging.info("Mixed precision enabled")
    
    # Enable XLA if specified
    if config['training']['xla_compile']:
        tf.config.optimizer.set_jit(True)
        logging.info("XLA compilation enabled")
    
    # Create model and optimizer
    model = create_model(config)
    
    learning_rate = config['optimizer']['learning_rate']
    if config['optimizer']['type'] == 'adamw':
        optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    if config['training']['mixed_precision']:
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
    
    # Loss and metrics
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy()
    
    # Create dataset
    train_dataset = create_dataset(config)
    
    # Training loop
    for epoch in range(config['training']['epochs']):
        logging.info(f'Starting epoch {epoch+1}/{config["training"]["epochs"]}')
        
        metric.reset_state()
        avg_loss = train_epoch(model, train_dataset, optimizer, loss_fn, metric)
        
        epoch_accuracy = metric.result().numpy()
        logging.info(f'Epoch {epoch+1} completed - Loss: {avg_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')
        
        # Save checkpoint
        if (epoch + 1) % config['checkpoint']['save_interval'] == 0:
            checkpoint_path = Path(config['checkpoint']['dir']) / f'checkpoint_epoch_{epoch+1}'
            model.save_weights(str(checkpoint_path))
            logging.info(f'Checkpoint saved to {checkpoint_path}')
        
        # Memory usage
        gpu_info = tf.config.experimental.get_memory_info('GPU:0')
        memory_mb = gpu_info['current'] / 1024 / 1024
        logging.info(f'GPU memory usage: {memory_mb:.1f}MB')

if __name__ == '__main__':
    main()
```

### Configuration File (config.json)

```json
{
  "model": {
    "type": "transformer",
    "num_classes": 1000,
    "params": {
      "d_model": 512,
      "num_heads": 8,
      "num_layers": 6,
      "vocab_size": 10000
    }
  },
  "data": {
    "train_pattern": "/path/to/train_*.tfrecord",
    "val_pattern": "/path/to/val_*.tfrecord"
  },
  "training": {
    "batch_size": 128,
    "epochs": 100,
    "mixed_precision": true,
    "xla_compile": true
  },
  "optimizer": {
    "type": "adamw",
    "learning_rate": 0.001
  },
  "checkpoint": {
    "dir": "./checkpoints",
    "save_interval": 10
  },
  "gpu_memory_limit": 100
}
```

### Job Submission Script

```bash
#!/bin/bash
# submit_tf_job.sh

SCRIPT_PATH="train_tf_h200.py"
CONFIG_PATH="config.json"
GPUS=1
MEMORY=40   # MINIMUM free VRAM (GB) a candidate GPU must have to be selected
TIME=12

# gpuq is daemonless: the job runs in the FOREGROUND in this terminal. There
# are no per-job log files — redirect the output yourself if you want one.
gpuq submit \
  --command "python $SCRIPT_PATH --config $CONFIG_PATH" \
  --gpus $GPUS \
  --memory $MEMORY \
  --time $TIME

echo "TensorFlow job finished. Check live status of other jobs with: gpuq status"
```

---

**Next Steps**:
- For JAX usage: [JAX with H200 Guide](jax-guide.md)
- For framework comparisons: [Best Practices Guide](best-practices.md)
- For ready-to-use examples: [Example Scripts](../examples/)
