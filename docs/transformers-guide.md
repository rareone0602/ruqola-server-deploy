# Hugging Face Transformers with H200 GPUs

Complete guide for optimizing Hugging Face Transformers on the Ruqola server's H200 GPUs for large language models, fine-tuning, and inference.

> **Server hardware (Mjolnir / wsserver1):** 4x NVIDIA H200 NVL, ~140 GiB VRAM each (~564 GB total), compute capability 9.0 (Hopper), CUDA 12.9 (driver 575.57.08), 256 logical CPUs, 755 GiB RAM, Ubuntu 24.04 LTS. GPU indices are `0,1,2,3`. The memory-budget examples below assume ~140 GiB per card with a small headroom reserved.

## 📖 Table of Contents

1. [Setup and Installation](#setup-and-installation)
2. [H200 Memory Optimization](#h200-memory-optimization)
3. [Model Loading Strategies](#model-loading-strategies)
4. [Fine-tuning Large Models](#fine-tuning-large-models)
5. [Multi-GPU Training](#multi-gpu-training)
6. [Inference Optimization](#inference-optimization)
7. [Memory-Efficient Techniques](#memory-efficient-techniques)
8. [Advanced Features](#advanced-features)
9. [Example Workflows](#example-workflows)

## Setup and Installation

### Recommended Installation

```bash
# Install transformers with the PyTorch stack (this guide is PyTorch-only)
pip install 'transformers[torch,sentencepiece,audio,vision]'

# Additional packages for H200 optimization
pip install accelerate  # For device mapping and optimization
pip install bitsandbytes  # For 8-bit and 4-bit quantization
pip install deepspeed  # For ZeRO optimization
pip install peft  # For parameter-efficient fine-tuning (LoRA, etc.)
pip install datasets  # For dataset handling
pip install evaluate  # For evaluation metrics

# Optional: Flash Attention for memory efficiency
pip install flash-attn --no-build-isolation

# Verify installation
python -c "from transformers import pipeline; print('Transformers ready!')"
```

> Install a CUDA 12.x PyTorch wheel that matches the server stack (CUDA 12.9 driver), e.g. a `cu124`+ build, rather than older `cu121` wheels.

### Environment Setup

```bash
# Add to ~/.bashrc or job script
export CUDA_VISIBLE_DEVICES=0,1      # outside gpuq only — `gpuq submit` sets this
                                     # for you (and caps each user at 3 cards)
export HF_HOME="/path/to/shared/hf"  # Shared HF cache (models + datasets)
export HF_DATASETS_CACHE="/path/to/shared/datasets"  # Optional: separate datasets cache
export TOKENIZERS_PARALLELISM=false  # Avoid multiprocessing issues
```

> Note: when you submit jobs through `gpuq`, the queue sets `CUDA_VISIBLE_DEVICES` per job to the GPUs you were allocated. Manually exporting it in a `gpuq` job can conflict with the allocation — only set it for ad-hoc runs outside the queue.
>
> `HF_HOME` is the current cache location for both models and datasets. The older `TRANSFORMERS_CACHE` variable is deprecated and ignored in recent transformers releases.

### Basic Model Loading Test

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Test with a medium-sized model
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

print(f"Model loaded on: {model.device}")
print(f"Model memory footprint: {model.get_memory_footprint() / 1024**3:.1f} GB")
```

## H200 Memory Optimization

### Automatic Device Mapping

```python
from transformers import AutoModelForCausalLM
import torch

def load_model_optimized(model_name, max_memory_per_gpu="130GB"):
    """Load model with optimal H200 memory distribution"""

    # Define memory constraints for 4x H200 NVL (~140 GiB each; leave ~10GB headroom per GPU)
    max_memory = {
        0: max_memory_per_gpu,
        1: max_memory_per_gpu,
        2: max_memory_per_gpu,
        3: max_memory_per_gpu,
        "cpu": "50GB"  # CPU offload if needed
    }

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # Use FP16 for memory efficiency
        device_map="auto",          # Automatic device placement
        max_memory=max_memory,
        offload_folder="./offload", # CPU offload directory
        offload_state_dict=True,    # Offload for loading
    )

    return model

# Example: Load large model across H200s
model = load_model_optimized("meta-llama/Llama-2-70b-chat-hf")
```

### Memory-Efficient Loading

```python
from transformers import AutoConfig, AutoModelForCausalLM
import torch

def load_model_memory_efficient(model_name):
    """Load model with maximum memory efficiency"""

    # Load config first to check model size
    config = AutoConfig.from_pretrained(model_name)

    # Estimate memory requirements
    def estimate_model_memory(config):
        # Rough estimation: params * 2 bytes (FP16); excludes activations/KV cache
        params = getattr(config, 'n_parameters', None)
        if params is None:
            # Estimate from hidden size and layers
            hidden_size = config.hidden_size
            num_layers = getattr(config, 'num_hidden_layers', config.n_layer)
            vocab_size = config.vocab_size
            params = vocab_size * hidden_size + num_layers * hidden_size * hidden_size * 4

        memory_gb = params * 2 / 1024**3  # FP16 weights only
        return memory_gb

    estimated_memory = estimate_model_memory(config)
    print(f"Estimated model memory: {estimated_memory:.1f} GB")

    # Choose loading strategy based on size.
    # estimate_model_memory() counts weights only (no activations/KV cache),
    # so a single ~140 GiB H200 comfortably holds models whose weight estimate
    # is up to ~110-120GB. Leave room for activations.
    if estimated_memory > 120:  # Too large for one H200 once activations are added
        print("Using multi-GPU device mapping")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            offload_folder="./offload"
        )
    elif estimated_memory > 40:  # Medium size: fits on one H200 with optimizations
        print("Using single H200 with optimizations")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).cuda()
    else:  # Small model
        print("Standard loading")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16
        ).cuda()

    return model
```

## Model Loading Strategies

### Size-Based Loading Strategy

```python
MODEL_STRATEGIES = {
    # Small models (< 3B parameters) - Single H200
    "small": {
        "models": ["gpt2", "gpt2-medium", "gpt2-large", "distilbert-base", "bert-base"],
        "config": {
            "torch_dtype": torch.float16,
            "device_map": {"": 0}  # Pin to a single GPU (cuda:0)
        }
    },

    # Medium models (3B-20B parameters) - Single H200 with optimizations
    "medium": {
        "models": ["gpt2-xl", "t5-3b", "flan-t5-xl", "opt-6.7b", "bloom-7b1"],
        "config": {
            "torch_dtype": torch.float16,
            "device_map": "auto",
            "low_cpu_mem_usage": True,
        }
    },

    # Large models (20B-65B parameters) - Multi-H200
    "large": {
        "models": ["opt-30b", "falcon-40b"],
        "config": {
            "torch_dtype": torch.float16,
            "device_map": "auto",
            "max_memory": {"0": "130GB", "1": "130GB", "2": "130GB", "3": "130GB", "cpu": "50GB"},
            "offload_folder": "./offload",
            "low_cpu_mem_usage": True,
        }
    },

    # Huge models (65B+ parameters) - Multi-H200, CPU offload only if needed
    "huge": {
        "models": ["llama-2-70b", "falcon-180b", "bloom-176b"],
        "config": {
            "torch_dtype": torch.float16,
            "device_map": "auto",
            "max_memory": {"0": "130GB", "1": "130GB", "2": "130GB", "3": "130GB", "cpu": "100GB"},
            "offload_folder": "./offload",
            "offload_state_dict": True,
            "low_cpu_mem_usage": True,
        }
    }
}

def load_model_by_size(model_name):
    """Automatically choose loading strategy based on model"""

    # Determine model size category, largest-tier-first so a 70B+ model
    # is never matched by a smaller tier.
    for category in ("huge", "large", "medium", "small"):
        info = MODEL_STRATEGIES[category]
        if any(model_id in model_name.lower() for model_id in info["models"]):
            print(f"Loading {model_name} using '{category}' strategy")
            return AutoModelForCausalLM.from_pretrained(model_name, **info["config"])

    # Default to medium strategy
    print(f"Unknown model size, using 'medium' strategy")
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        **MODEL_STRATEGIES["medium"]["config"]
    )
```

> On 4x ~140 GiB H200s, a 70B FP16 model (~140GB of weights) fits comfortably across two cards with room for activations and KV cache — no CPU/disk offload is needed. The `cpu` entries above are only a fallback for 180B-class models.

### Quantization for Memory Savings

```python
from transformers import BitsAndBytesConfig

def load_quantized_model(model_name, quantization="8bit"):
    """Load model with quantization for memory efficiency"""

    if quantization == "4bit":
        # 4-bit quantization (maximum memory savings)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,  # Double quantization
        )
    elif quantization == "8bit":
        # 8-bit quantization (good balance)
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )
    else:
        bnb_config = None

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16
    )

    return model

# Example usage
model_4bit = load_quantized_model("meta-llama/Llama-2-7b-hf", "4bit")
model_8bit = load_quantized_model("meta-llama/Llama-2-13b-hf", "8bit")
```

## Fine-tuning Large Models

### Full Fine-tuning with DeepSpeed

```python
from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class H200OptimizedTrainer:
    def __init__(self, model_name, dataset, output_dir="./results"):
        self.model_name = model_name
        self.dataset = dataset
        self.output_dir = output_dir
        self.setup_model_and_tokenizer()

    def setup_model_and_tokenizer(self):
        """Setup model and tokenizer with H200 optimizations"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Add pad token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model with memory optimization
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        # Gradient checkpointing is incompatible with the KV cache, so disable it.
        # (gradient_checkpointing_enable() does NOT set use_cache=False for you.)
        self.model.config.use_cache = False
        # Enable gradient checkpointing for memory efficiency
        self.model.gradient_checkpointing_enable()

    def create_training_arguments(self):
        """Create training arguments optimized for H200"""
        return TrainingArguments(
            output_dir=self.output_dir,

            # Batch sizes optimized for H200 memory
            per_device_train_batch_size=2,
            per_device_eval_batch_size=4,
            # NOTE: with device_map="auto" the model is sharded across GPUs
            # (model/pipeline parallel), so the Trainer does NOT data-parallelize.
            # Effective batch size is per_device * accum = 2 * 8 = 16.
            # For true DDP (one full model replica per GPU), drop device_map="auto"
            # and the effective batch would be 2 * 8 * 4 = 64 across the 4 H200s.
            gradient_accumulation_steps=8,

            # Learning parameters
            num_train_epochs=3,
            learning_rate=2e-5,
            warmup_steps=100,
            weight_decay=0.01,

            # Memory and performance optimizations
            fp16=True,                    # Mixed precision
            dataloader_pin_memory=True,
            dataloader_num_workers=4,
            remove_unused_columns=False,

            # Logging and checkpointing
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=100,
            save_steps=500,
            save_total_limit=3,

            # DeepSpeed integration
            deepspeed="ds_config.json",

            # Report to wandb (optional)
            report_to="wandb",
        )

    def train(self):
        """Start training process"""
        training_args = self.create_training_arguments()

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset.get("validation"),
            tokenizer=self.tokenizer,
        )

        # Start training
        trainer.train()

        # Save final model
        trainer.save_model()

        return trainer

# DeepSpeed configuration for H200s
deepspeed_config = {
    "fp16": {
        "enabled": True,
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "zero_optimization": {
        "stage": 2,  # ZeRO Stage 2 for good balance
        "allgather_partitions": True,
        "allgather_bucket_size": 5e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 5e8,
        "contiguous_gradients": True
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 2e-5,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01
        }
    },
    "train_micro_batch_size_per_gpu": 2,
    "gradient_accumulation_steps": 8,
    "gradient_clipping": 1.0,
    "steps_per_print": 10
}

# Save DeepSpeed config
import json
with open("ds_config.json", "w") as f:
    json.dump(deepspeed_config, f, indent=2)
```

### Parameter-Efficient Fine-tuning (LoRA)

```python
from peft import LoraConfig, get_peft_model, TaskType
from peft import PeftModel, PeftConfig

def setup_lora_model(model_name, task_type="CAUSAL_LM"):
    """Setup LoRA for parameter-efficient fine-tuning"""

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # LoRA configuration optimized for large models
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,                    # Rank - balance between efficiency and performance
        lora_alpha=32,           # Scaling parameter
        lora_dropout=0.1,        # Dropout for regularization
        target_modules=[         # Target attention and MLP layers
            "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
            "gate_proj", "up_proj", "down_proj"       # MLP (for LLaMA-style models)
        ],
        bias="none",
        fan_in_fan_out=False,
    )

    # Apply LoRA to model
    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    model.print_trainable_parameters()

    return model, lora_config

class LoRATrainer(H200OptimizedTrainer):
    """Specialized trainer for LoRA fine-tuning"""

    def setup_model_and_tokenizer(self):
        """Setup LoRA model and tokenizer"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Setup LoRA model
        self.model, self.lora_config = setup_lora_model(self.model_name)

    def create_training_arguments(self):
        """LoRA-specific training arguments"""
        return TrainingArguments(
            output_dir=self.output_dir,

            # Higher batch sizes possible with LoRA
            per_device_train_batch_size=8,
            per_device_eval_batch_size=16,
            gradient_accumulation_steps=4,

            # LoRA-specific settings
            num_train_epochs=5,          # More epochs for LoRA
            learning_rate=1e-4,          # Higher LR for LoRA
            warmup_steps=100,
            weight_decay=0.01,

            # Standard optimizations
            fp16=True,
            dataloader_pin_memory=True,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=100,
            save_steps=500,
            save_total_limit=3,
        )

    def save_lora_model(self, save_path="./lora_model"):
        """Save only LoRA weights"""
        self.model.save_pretrained(save_path)
        print(f"LoRA model saved to {save_path}")

def load_lora_model_for_inference(base_model_name, lora_path):
    """Load model with LoRA weights for inference"""

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # Load and apply LoRA weights
    model = PeftModel.from_pretrained(base_model, lora_path)

    return model
```

## Multi-GPU Training

### Accelerate Integration

```python
from accelerate import Accelerator
from accelerate.utils import set_seed

def train_with_accelerate(model_name, dataset, config):
    """Training with Accelerate for multi-GPU support"""

    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        mixed_precision="fp16",
        log_with="wandb",
        project_dir="./logs"
    )

    # Set seed for reproducibility
    set_seed(42)

    # Setup model and tokenizer
    if accelerator.is_main_process:
        print(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
    )

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )

    # Setup data loader
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=default_data_collator,
        pin_memory=True,
        num_workers=4
    )

    # Prepare everything with accelerator
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    # Training loop
    model.train()
    for epoch in range(config["epochs"]):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

                if accelerator.is_main_process and step % 10 == 0:
                    print(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}")

    # Save model
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        "./final_model",
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save
    )
```

> For data-parallel runs (one full model replica per GPU), launch with `torchrun --nproc_per_node=N train.py` where `N` matches your gpuq allocation (at most 3 cards per user on this host).

### Multi-GPU Inference

```python
import torch

class MultiGPUInference:
    """Optimized inference across multiple H200s"""

    def __init__(self, model_name, num_gpus=None):
        self.model_name = model_name
        # Default to every visible GPU (4 on this server).
        self.num_gpus = num_gpus if num_gpus is not None else torch.cuda.device_count()
        self.setup_models()

    def setup_models(self):
        """Setup model replicas across GPUs"""

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # For large models, use device mapping
        if self.is_large_model():
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            self.multi_gpu_setup = False
        else:
            # For smaller models, create replicas
            self.models = []
            for gpu_id in range(self.num_gpus):
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16
                ).cuda(gpu_id)
                self.models.append(model)
            self.multi_gpu_setup = True

    def is_large_model(self):
        """Check if model needs multi-GPU device mapping"""
        large_model_indicators = [
            "70b", "65b", "176b", "180b",
            "falcon-40b", "opt-30b", "bloom-176b"
        ]
        return any(indicator in self.model_name.lower()
                  for indicator in large_model_indicators)

    def generate_batch(self, prompts, max_new_tokens=100, temperature=0.7):
        """Generate responses for batch of prompts"""

        if not self.multi_gpu_setup:
            # Single model across multiple GPUs
            return self._generate_single_model(prompts, max_new_tokens, temperature)
        else:
            # Multiple model replicas
            return self._generate_multi_replica(prompts, max_new_tokens, temperature)

    def _generate_single_model(self, prompts, max_new_tokens, temperature):
        """Generate using single model with device mapping"""

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

        # Move inputs to same device as model
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def _generate_multi_replica(self, prompts, max_new_tokens, temperature):
        """Generate using multiple model replicas"""
        import threading

        # Split prompts across GPUs
        batch_size = len(prompts)
        prompts_per_gpu = batch_size // self.num_gpus

        results = [None] * self.num_gpus
        threads = []

        def generate_on_gpu(gpu_id, gpu_prompts, result_idx):
            model = self.models[gpu_id]

            inputs = self.tokenizer(
                gpu_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(f"cuda:{gpu_id}")

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            results[result_idx] = self.tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )

        # Launch threads for each GPU
        for i in range(self.num_gpus):
            start_idx = i * prompts_per_gpu
            if i == self.num_gpus - 1:  # Last GPU takes remainder
                gpu_prompts = prompts[start_idx:]
            else:
                gpu_prompts = prompts[start_idx:start_idx + prompts_per_gpu]

            if gpu_prompts:  # Only if there are prompts for this GPU
                thread = threading.Thread(
                    target=generate_on_gpu,
                    args=(i, gpu_prompts, i)
                )
                threads.append(thread)
                thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Combine results
        combined_results = []
        for result_batch in results:
            if result_batch:
                combined_results.extend(result_batch)

        return combined_results
```

## Inference Optimization

### Optimized Text Generation

```python
from transformers import TextGenerationPipeline
import torch

class H200TextGenerator:
    """Optimized text generation for H200 GPUs"""

    def __init__(self, model_name, use_cache=True):
        self.model_name = model_name
        self.use_cache = use_cache
        self.setup_pipeline()

    def setup_pipeline(self):
        """Setup optimized generation pipeline"""

        # Load model with optimizations
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            use_cache=self.use_cache,  # Enable KV caching
        )

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Create optimized pipeline
        self.pipeline = TextGenerationPipeline(
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    def generate(self,
                 prompt,
                 max_new_tokens=100,
                 temperature=0.7,
                 top_p=0.9,
                 top_k=50,
                 repetition_penalty=1.1,
                 do_sample=True):
        """Generate text with optimized parameters"""

        # Generation parameters optimized for quality and speed.
        # Use max_new_tokens (number of NEW tokens) rather than max_length
        # (which is prompt + new tokens) so behaviour is independent of prompt length.
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "do_sample": do_sample,
            "pad_token_id": self.pipeline.tokenizer.eos_token_id,
            "use_cache": self.use_cache,
            "num_return_sequences": 1
        }

        # Generate with timing
        import time
        start_time = time.time()

        result = self.pipeline(
            prompt,
            **generation_kwargs,
            return_full_text=False  # Only return generated text
        )[0]["generated_text"]

        generation_time = time.time() - start_time
        tokens_generated = len(self.pipeline.tokenizer.encode(result))
        tokens_per_second = tokens_generated / generation_time

        return {
            "text": result,
            "generation_time": generation_time,
            "tokens_per_second": tokens_per_second,
            "tokens_generated": tokens_generated
        }

    def batch_generate(self, prompts, **kwargs):
        """Generate for multiple prompts efficiently"""

        results = []
        for prompt in prompts:
            result = self.generate(prompt, **kwargs)
            results.append(result)

        return results

# Streaming generation for long outputs
class StreamingGenerator:
    """Streaming text generation for real-time output"""

    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def stream_generate(self, prompt, max_new_tokens=500, **kwargs):
        """Generate text with streaming output.

        model.generate() is NOT a token iterator — it returns a single result
        object. To stream, drive a TextIteratorStreamer with generate() on a
        background thread, then iterate the streamer.
        """
        from transformers import TextIteratorStreamer
        from threading import Thread

        # Encode prompt
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.model.device)

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        # Generation parameters
        generation_kwargs = {
            "input_ids": input_ids,
            "max_new_tokens": max_new_tokens,
            "temperature": kwargs.pop("temperature", 0.7),
            "top_p": kwargs.pop("top_p", 0.9),
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id,
            "streamer": streamer,
            **kwargs,
        }

        # Run generation in a background thread; the streamer yields text as it
        # is produced.
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        for new_text in streamer:
            yield new_text

        thread.join()
```

### Efficient Embeddings and Classification

```python
from transformers import AutoModel
import torch.nn.functional as F

class H200EmbeddingExtractor:
    """Efficient embedding extraction for H200s"""

    def __init__(self, model_name):
        self.model_name = model_name
        self.setup_model()

    def setup_model(self):
        """Setup model for embedding extraction"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts, batch_size=32, normalize=True):
        """Extract embeddings from texts"""

        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )

            # Move to same device as model
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Get embeddings
            outputs = self.model(**inputs)

            # Mean pooling
            embeddings = self.mean_pool(
                outputs.last_hidden_state,
                inputs["attention_mask"]
            )

            # Normalize embeddings
            if normalize:
                embeddings = F.normalize(embeddings, p=2, dim=1)

            all_embeddings.append(embeddings.cpu())

        return torch.cat(all_embeddings, dim=0)

    def mean_pool(self, hidden_states, attention_mask):
        """Mean pooling with attention mask"""
        # Expand attention mask
        expanded_mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()

        # Apply mask and compute mean
        sum_embeddings = torch.sum(hidden_states * expanded_mask, 1)
        sum_mask = torch.clamp(expanded_mask.sum(1), min=1e-9)

        return sum_embeddings / sum_mask

    def semantic_search(self, query, documents, top_k=5):
        """Semantic search using embeddings"""

        # Encode query and documents
        query_embedding = self.encode([query])
        doc_embeddings = self.encode(documents)

        # Compute similarities
        similarities = torch.cosine_similarity(
            query_embedding,
            doc_embeddings,
            dim=1
        )

        # Get top-k results
        top_indices = similarities.topk(top_k).indices

        results = []
        for idx in top_indices:
            results.append({
                "document": documents[idx],
                "similarity": similarities[idx].item(),
                "index": idx.item()
            })

        return results
```

## Memory-Efficient Techniques

### Gradient Checkpointing and Accumulation

```python
def setup_memory_efficient_training(model, config):
    """Configure model for memory-efficient training"""

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    # Gradient checkpointing is incompatible with the KV cache
    model.config.use_cache = False

    # Optional: Enable Flash Attention if available
    if hasattr(model.config, 'use_flash_attention'):
        model.config.use_flash_attention = True

    # NOTE: Do NOT call model.half() here. When training under autocast +
    # GradScaler, keep weights in FP32 (autocast casts ops to FP16/BF16 on the
    # fly and GradScaler relies on FP32 master weights). Calling model.half()
    # and then using GradScaler commonly produces unstable / NaN training.
    return model

class MemoryEfficientTrainingLoop:
    """Memory-efficient training loop with various optimizations"""

    def __init__(self, model, optimizer, tokenizer, config):
        self.model = model
        self.optimizer = optimizer
        self.tokenizer = tokenizer
        self.config = config

        # Setup gradient scaler for mixed precision (current torch.amp API)
        self.scaler = torch.amp.GradScaler('cuda')

    def train_step(self, batch):
        """Single training step with memory optimizations"""

        # Enable gradient accumulation
        accumulation_steps = self.config.get("gradient_accumulation_steps", 1)

        total_loss = 0
        for micro_batch in self.split_batch(batch, accumulation_steps):

            # Mixed precision forward pass (current torch.amp API)
            with torch.amp.autocast('cuda'):
                outputs = self.model(**micro_batch)
                loss = outputs.loss / accumulation_steps

            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()
            total_loss += loss.item()

        # Gradient clipping and optimization step
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.get("max_grad_norm", 1.0)
        )

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

        return total_loss

    def split_batch(self, batch, num_splits):
        """Split batch for gradient accumulation"""
        batch_size = len(batch['input_ids'])
        micro_batch_size = batch_size // num_splits

        for i in range(num_splits):
            start_idx = i * micro_batch_size
            end_idx = start_idx + micro_batch_size if i < num_splits - 1 else batch_size

            micro_batch = {
                k: v[start_idx:end_idx] for k, v in batch.items()
            }
            yield micro_batch
```

> Avoid calling `torch.cuda.empty_cache()` inside the inner training loop — it forces a device synchronization and slows training without freeing memory the allocator would otherwise reuse. With ~140 GiB per H200 it is rarely needed.
>
> Alternatively, use BF16 autocast (`torch.amp.autocast('cuda', dtype=torch.bfloat16)`), which on Hopper H200s is fast and numerically stable and usually needs no GradScaler at all.

### CPU Offloading

```python
import psutil

def setup_cpu_offload(model, offload_dir="./cpu_offload"):
    """Setup CPU offloading for very large models"""

    from accelerate import cpu_offload, disk_offload
    import os

    os.makedirs(offload_dir, exist_ok=True)

    # Check available system memory (the host has 755 GiB RAM)
    available_memory = psutil.virtual_memory().available
    print(f"Available system memory: {available_memory / 1024**3:.1f} GB")

    if available_memory > 100 * 1024**3:  # More than 100GB RAM
        # Use CPU offloading
        model = cpu_offload(model, execution_device=0)
        print("Using CPU offloading")
    else:
        # Use disk offloading if RAM is limited
        model = disk_offload(model, offload_dir=offload_dir)
        print(f"Using disk offloading to {offload_dir}")

    return model

class OffloadedInference:
    """Inference with automatic offloading"""

    def __init__(self, model_name, offload_strategy="auto"):
        self.model_name = model_name
        self.offload_strategy = offload_strategy
        self.setup_model()

    def setup_model(self):
        """Setup model with appropriate offloading"""

        # Determine offloading strategy.
        # NOTE: each H200 NVL reports ~140 GiB total, so on this server the
        # "auto" heuristic below essentially never triggers CPU offload — it is
        # only meaningful on smaller cards. Offload is only needed here for
        # 180B-class models.
        if self.offload_strategy == "auto":
            # Check GPU memory and model size
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3

            if gpu_memory < 100:  # Less than 100GB (never true on H200 NVL)
                offload_to_cpu = True
            else:
                offload_to_cpu = False
        else:
            offload_to_cpu = self.offload_strategy == "cpu"

        # Load model
        if offload_to_cpu:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                offload_folder="./offload",
                max_memory={"0": "130GB", "1": "130GB", "2": "130GB", "3": "130GB", "cpu": "100GB"}
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def generate_with_offload(self, prompt, **kwargs):
        """Generate text with offloading optimizations"""

        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## Advanced Features

### Custom Training Objectives

```python
class CustomLossTrainer:
    """Trainer with custom loss functions for specific tasks"""

    def __init__(self, model, tokenizer, task_type="causal_lm"):
        self.model = model
        self.tokenizer = tokenizer
        self.task_type = task_type

    def contrastive_loss(self, positive_pairs, negative_pairs, temperature=0.07):
        """Contrastive learning loss for representation learning"""

        # Encode positive and negative pairs
        pos_embeddings = self.encode_pairs(positive_pairs)
        neg_embeddings = self.encode_pairs(negative_pairs)

        # Compute contrastive loss
        pos_similarities = F.cosine_similarity(
            pos_embeddings[:, 0],
            pos_embeddings[:, 1],
            dim=1
        ) / temperature

        neg_similarities = F.cosine_similarity(
            neg_embeddings[:, 0],
            neg_embeddings[:, 1],
            dim=1
        ) / temperature

        # Contrastive loss
        loss = -torch.log(
            torch.exp(pos_similarities) /
            (torch.exp(pos_similarities) + torch.exp(neg_similarities).sum())
        ).mean()

        return loss

    def instruction_tuning_loss(self, instruction, response, ignore_instruction=True):
        """Loss function for instruction tuning"""

        # Combine instruction and response
        full_text = f"{instruction}\n{response}"

        inputs = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        # Get model outputs
        outputs = self.model(**inputs, labels=inputs["input_ids"])

        if ignore_instruction:
            # Only compute loss on response tokens
            instruction_length = len(self.tokenizer.encode(instruction))

            # Create mask to ignore instruction tokens
            labels = inputs["input_ids"].clone()
            labels[:, :instruction_length] = -100  # Ignore instruction tokens

            # Recompute loss
            outputs = self.model(**inputs, labels=labels)

        return outputs.loss

    def encode_pairs(self, pairs):
        """Encode text pairs for contrastive learning"""
        embeddings = []

        for pair in pairs:
            pair_embeddings = []
            for text in pair:
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                )

                with torch.no_grad():
                    outputs = self.model(**inputs, output_hidden_states=True)
                    # Use CLS token or mean pooling
                    embedding = outputs.hidden_states[-1].mean(dim=1)
                    pair_embeddings.append(embedding)

            embeddings.append(torch.stack(pair_embeddings))

        return torch.stack(embeddings)

# Reinforcement Learning from Human Feedback (RLHF)
class RLHFTrainer:
    """Trainer for RLHF fine-tuning"""

    def __init__(self, policy_model, reward_model, ref_model):
        self.policy_model = policy_model
        self.reward_model = reward_model
        self.ref_model = ref_model

    def ppo_loss(self, prompts, responses, advantages, old_log_probs, epsilon=0.2):
        """Proximal Policy Optimization loss"""

        # Get current log probabilities
        current_log_probs = self.get_log_probs(prompts, responses)

        # Compute ratio
        ratio = torch.exp(current_log_probs - old_log_probs)

        # Clipped surrogate loss
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * advantages

        policy_loss = -torch.min(surr1, surr2).mean()

        # KL divergence constraint
        kl_penalty = self.compute_kl_penalty(prompts, responses)

        total_loss = policy_loss + 0.1 * kl_penalty

        return total_loss

    def compute_kl_penalty(self, prompts, responses):
        """Compute KL divergence penalty against reference model"""

        # Get log probabilities from policy and reference models
        policy_log_probs = self.get_log_probs(prompts, responses, self.policy_model)
        ref_log_probs = self.get_log_probs(prompts, responses, self.ref_model)

        # KL divergence
        kl_div = (torch.exp(policy_log_probs) * (policy_log_probs - ref_log_probs)).sum()

        return kl_div

    def get_log_probs(self, prompts, responses, model=None):
        """Get log probabilities for responses given prompts"""

        if model is None:
            model = self.policy_model

        # Implementation would depend on specific tokenization and model forward pass
        # This is a simplified version
        full_text = [f"{p} {r}" for p, r in zip(prompts, responses)]

        # Tokenize and get log probabilities
        # ... (implementation details)

        return log_probs
```

## Example Workflows

### Complete Fine-tuning Workflow

```python
#!/usr/bin/env python3
"""
Complete Transformers fine-tuning workflow for H200 GPUs
Supports various model sizes and training strategies
"""

from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
import torch
import json
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--output_dir", default="./results")
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--use_deepspeed", action="store_true")
    args = parser.parse_args()

    # Load dataset
    dataset = load_dataset(args.dataset_name)

    # Setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding="max_length"
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Setup model based on size and strategy
    if args.use_lora:
        # LoRA fine-tuning
        from peft import LoraConfig, get_peft_model, TaskType

        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    else:
        # Full fine-tuning
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        # Gradient checkpointing is incompatible with the KV cache.
        model.config.use_cache = False
        model.gradient_checkpointing_enable()

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=2 if not args.use_lora else 8,
        gradient_accumulation_steps=8 if not args.use_lora else 4,
        num_train_epochs=3,
        learning_rate=2e-5 if not args.use_lora else 1e-4,
        fp16=True,
        logging_steps=10,
        save_steps=500,
        eval_strategy="steps",
        eval_steps=500,
        warmup_steps=100,
        deepspeed="ds_config.json" if args.use_deepspeed else None,
        report_to="wandb"
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal language modeling
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset.get("validation"),
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    # Train
    trainer.train()

    # Save model
    trainer.save_model()

    print("Training completed!")

if __name__ == "__main__":
    main()
```

### Batch Inference Workflow

```python
#!/usr/bin/env python3
"""
Optimized batch inference workflow for H200 GPUs
Handles large-scale inference efficiently
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from tqdm import tqdm
import argparse

def batch_inference_workflow(
    model_name,
    input_file,
    output_file,
    batch_size=16,
    max_new_tokens=100,
    temperature=0.7
):
    """Efficient batch inference workflow"""

    print(f"Loading model: {model_name}")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load input data
    with open(input_file, 'r') as f:
        inputs = [line.strip() for line in f if line.strip()]

    print(f"Processing {len(inputs)} inputs with batch size {batch_size}")

    # Process in batches
    results = []

    for i in tqdm(range(0, len(inputs), batch_size)):
        batch_inputs = inputs[i:i + batch_size]

        # Tokenize batch
        encoded = tokenizer(
            batch_inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        encoded = {k: v.to(model.device) for k, v in encoded.items()}

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True
            )

        # Decode results
        batch_results = []
        for j, output in enumerate(outputs):
            # Remove input tokens from output
            input_length = len(encoded['input_ids'][j])
            generated_tokens = output[input_length:]

            generated_text = tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True
            )

            batch_results.append({
                "input": batch_inputs[j],
                "output": generated_text
            })

        results.extend(batch_results)

    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    batch_inference_workflow(
        args.model_name,
        args.input_file,
        args.output_file,
        args.batch_size,
        args.max_new_tokens,
        args.temperature
    )

if __name__ == "__main__":
    main()
```

---

**Next Steps**:
- For ready-to-use examples: [Example Scripts](../examples/)
- For general best practices: [Best Practices Guide](best-practices.md)
- For troubleshooting: [Troubleshooting Guide](troubleshooting.md)
