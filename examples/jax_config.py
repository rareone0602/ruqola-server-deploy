# JAX/Flax Training Configuration for H200

# Model configuration
num_classes = 10
num_blocks = 2  # Number of residual blocks per group
use_gradient_checkpointing = True  # Enable for memory efficiency

# Training configuration
epochs = 100
batch_size = 128  # Optimized for H200 memory and Tensor Cores
seed = 42

# Optimizer configuration
optimizer_type = "adamw"  # "adamw" or "sgd"
learning_rate = 0.001
weight_decay = 0.01
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8

# For SGD optimizer (if used)
momentum = 0.9
nesterov = True

# Learning rate scheduler
scheduler_type = "cosine"  # "cosine", "exponential", or "constant"
min_lr = 0.00001

# For exponential scheduler (if used)  
decay_steps = 1000
decay_rate = 0.96

# Training optimization
gradient_clip = 1.0  # Set to 0 to disable

# Data configuration
data_augmentation = True

# Logging configuration
log_interval = 100  # Log every N batches

# Checkpointing
checkpoint_dir = "./jax_checkpoints"
save_interval = 10  # Save every N epochs

# Memory optimization
# These are handled automatically by JAX, but you can set environment variables:
# export XLA_PYTHON_CLIENT_PREALLOCATE=false
# export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9