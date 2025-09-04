#!/usr/bin/env python3
"""
Hugging Face Transformers Fine-tuning Example for H200 GPUs
Demonstrates efficient fine-tuning of large language models with various optimization techniques.

Usage:
gpuq submit --command "python transformers_finetuning.py --config transformers_config.yaml" --gpus 2 --memory 100 --time 12

Features:
- Support for full fine-tuning and LoRA
- Multi-GPU training with DeepSpeed
- Memory optimization for H200s
- Automatic mixed precision
- Comprehensive logging and checkpointing
- Support for various model sizes
"""

import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoConfig,
    TrainingArguments, Trainer, 
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import wandb
import yaml
import argparse
import logging
import os
import json
from pathlib import Path
import numpy as np
from typing import Dict, Any, Optional

def setup_logging(log_dir="logs"):
    """Setup comprehensive logging"""
    Path(log_dir).mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'{log_dir}/transformers_training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

class ModelSizeEstimator:
    """Estimate model memory requirements and choose optimal loading strategy"""
    
    @staticmethod
    def estimate_model_size(model_name_or_config):
        """Estimate model size in GB"""
        if isinstance(model_name_or_config, str):
            config = AutoConfig.from_pretrained(model_name_or_config)
        else:
            config = model_name_or_config
        
        # Get model parameters
        hidden_size = getattr(config, 'hidden_size', getattr(config, 'd_model', 768))
        num_layers = getattr(config, 'num_hidden_layers', 
                           getattr(config, 'n_layer', 
                           getattr(config, 'num_layers', 12)))
        vocab_size = getattr(config, 'vocab_size', 50257)
        
        # Rough parameter estimation
        # Embeddings: vocab_size * hidden_size
        # Transformer layers: num_layers * hidden_size^2 * 4 (approx)
        # Output layer: hidden_size * vocab_size
        
        embedding_params = vocab_size * hidden_size
        transformer_params = num_layers * hidden_size * hidden_size * 4
        output_params = hidden_size * vocab_size
        
        total_params = embedding_params + transformer_params + output_params
        
        # Memory estimation (FP16: 2 bytes per param + activations + gradients)
        model_memory = total_params * 2 / (1024**3)  # FP16 model weights
        gradient_memory = total_params * 4 / (1024**3)  # FP32 gradients 
        optimizer_memory = total_params * 8 / (1024**3)  # AdamW states
        
        total_memory = model_memory + gradient_memory + optimizer_memory
        
        return {
            'params': total_params,
            'model_memory_gb': model_memory,
            'total_training_memory_gb': total_memory * 1.2,  # Add 20% buffer
            'size_category': ModelSizeEstimator.categorize_model_size(total_params)
        }
    
    @staticmethod
    def categorize_model_size(num_params):
        """Categorize model by size"""
        if num_params < 1e9:  # < 1B
            return 'small'
        elif num_params < 7e9:  # < 7B
            return 'medium' 
        elif num_params < 20e9:  # < 20B
            return 'large'
        elif num_params < 70e9:  # < 70B
            return 'very_large'
        else:
            return 'huge'

class H200OptimizedModelLoader:
    """Optimized model loading strategies for H200 GPUs"""
    
    def __init__(self, logger):
        self.logger = logger
        self.device_count = torch.cuda.device_count()
        self.total_gpu_memory = self.get_total_gpu_memory()
    
    def get_total_gpu_memory(self):
        """Get total GPU memory across all devices"""
        total = 0
        for i in range(self.device_count):
            total += torch.cuda.get_device_properties(i).total_memory
        return total / (1024**3)  # Convert to GB
    
    def load_model_optimized(self, model_name, config, use_lora=False):
        """Load model with optimal strategy for H200s"""
        
        # Estimate model size
        size_info = ModelSizeEstimator.estimate_model_size(model_name)
        self.logger.info(f"Model size estimation: {size_info}")
        
        # Choose loading strategy
        strategy = self.choose_loading_strategy(size_info, use_lora)
        self.logger.info(f"Using loading strategy: {strategy['name']}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        model = self.load_with_strategy(model_name, strategy, config)
        
        # Apply LoRA if requested
        if use_lora:
            model = self.apply_lora(model, config)
        
        return model, tokenizer, size_info
    
    def choose_loading_strategy(self, size_info, use_lora):
        """Choose optimal loading strategy based on model size and resources"""
        
        memory_needed = size_info['total_training_memory_gb']
        if use_lora:
            memory_needed *= 0.1  # LoRA needs much less memory
        
        category = size_info['size_category']
        
        strategies = {
            'single_gpu': {
                'name': 'Single GPU',
                'condition': lambda: memory_needed < 120 and category in ['small', 'medium'],
                'config': {
                    'torch_dtype': torch.float16,
                    'device_map': 'cuda:0',
                }
            },
            'auto_device_map': {
                'name': 'Auto Device Mapping',
                'condition': lambda: memory_needed < 300 and category in ['medium', 'large'],
                'config': {
                    'torch_dtype': torch.float16,
                    'device_map': 'auto',
                    'low_cpu_mem_usage': True,
                }
            },
            'multi_gpu_with_offload': {
                'name': 'Multi-GPU with CPU Offload',
                'condition': lambda: memory_needed < 500 and category in ['large', 'very_large'],
                'config': {
                    'torch_dtype': torch.float16,
                    'device_map': 'auto',
                    'max_memory': {i: "75GB" for i in range(self.device_count)},
                    'offload_folder': './offload',
                    'low_cpu_mem_usage': True,
                }
            },
            'quantized_loading': {
                'name': 'Quantized Loading (8-bit)',
                'condition': lambda: category == 'huge',
                'config': {
                    'load_in_8bit': True,
                    'device_map': 'auto',
                    'torch_dtype': torch.float16,
                }
            }
        }
        
        # Choose first matching strategy
        for strategy_name, strategy in strategies.items():
            if strategy['condition']():
                return strategy
        
        # Default to most aggressive strategy
        return strategies['multi_gpu_with_offload']
    
    def load_with_strategy(self, model_name, strategy, config):
        """Load model with specified strategy"""
        
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **strategy['config']
            )
            
            # Enable gradient checkpointing for memory efficiency
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
            
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load with strategy {strategy['name']}: {e}")
            # Fallback to most conservative loading
            return self.load_fallback(model_name)
    
    def load_fallback(self, model_name):
        """Fallback loading method"""
        self.logger.info("Using fallback loading strategy")
        
        return AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map='auto'
        )
    
    def apply_lora(self, model, config):
        """Apply LoRA configuration to model"""
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=config['lora']['rank'],
            lora_alpha=config['lora']['alpha'],
            lora_dropout=config['lora']['dropout'],
            target_modules=config['lora']['target_modules'],
            bias=config['lora'].get('bias', 'none'),
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        return model

class DatasetProcessor:
    """Process and prepare datasets for training"""
    
    def __init__(self, tokenizer, config, logger):
        self.tokenizer = tokenizer
        self.config = config
        self.logger = logger
    
    def load_and_process_dataset(self):
        """Load and process dataset based on configuration"""
        
        dataset_config = self.config['dataset']
        
        if dataset_config['type'] == 'huggingface':
            # Load from Hugging Face Hub
            dataset = load_dataset(
                dataset_config['name'],
                dataset_config.get('subset'),
                split=dataset_config.get('split', 'train')
            )
        elif dataset_config['type'] == 'local':
            # Load local dataset
            dataset = load_dataset(
                dataset_config['format'],
                data_files=dataset_config['files']
            )
        else:
            raise ValueError(f"Unknown dataset type: {dataset_config['type']}")
        
        # Process dataset
        if dataset_config.get('text_column'):
            dataset = self.process_text_dataset(dataset, dataset_config['text_column'])
        elif dataset_config.get('instruction_format'):
            dataset = self.process_instruction_dataset(dataset, dataset_config['instruction_format'])
        
        # Split dataset if needed
        if 'validation_split' in dataset_config and 'validation' not in dataset:
            dataset = dataset.train_test_split(
                test_size=dataset_config['validation_split'],
                seed=self.config['training']['seed']
            )
            dataset['validation'] = dataset.pop('test')
        
        return dataset
    
    def process_text_dataset(self, dataset, text_column):
        """Process dataset with simple text column"""
        
        def tokenize_function(examples):
            # Tokenize texts
            tokenized = self.tokenizer(
                examples[text_column],
                truncation=True,
                max_length=self.config['training']['max_length'],
                padding='max_length',
                return_tensors=None
            )
            
            # For causal LM, labels are the same as input_ids
            tokenized['labels'] = tokenized['input_ids'].copy()
            
            return tokenized
        
        return dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
    
    def process_instruction_dataset(self, dataset, format_config):
        """Process instruction-following dataset"""
        
        def format_instructions(examples):
            formatted_texts = []
            
            for i in range(len(examples[format_config['instruction_column']])):
                instruction = examples[format_config['instruction_column']][i]
                response = examples[format_config['response_column']][i]
                
                # Format according to template
                if format_config['template'] == 'alpaca':
                    formatted_text = f"### Instruction:\n{instruction}\n\n### Response:\n{response}"
                elif format_config['template'] == 'chat':
                    formatted_text = f"Human: {instruction}\n\nAssistant: {response}"
                else:
                    formatted_text = f"{instruction}\n{response}"
                
                formatted_texts.append(formatted_text)
            
            return {'text': formatted_texts}
        
        dataset = dataset.map(format_instructions, batched=True)
        return self.process_text_dataset(dataset, 'text')

class H200TrainingOptimizer:
    """Optimize training configuration for H200 GPUs"""
    
    def __init__(self, model_info, config, logger):
        self.model_info = model_info
        self.config = config
        self.logger = logger
        self.device_count = torch.cuda.device_count()
    
    def optimize_batch_size(self):
        """Optimize batch size based on model size and available memory"""
        
        base_batch_size = self.config['training']['per_device_batch_size']
        memory_needed = self.model_info['total_training_memory_gb']
        
        # Adjust batch size based on available memory per GPU
        memory_per_gpu = 140  # H200 has ~140GB available
        
        if memory_needed > memory_per_gpu * 0.8:  # Using >80% memory
            # Reduce batch size
            adjusted_batch_size = max(1, base_batch_size // 2)
            self.logger.info(f"Reducing batch size to {adjusted_batch_size} due to memory constraints")
        elif memory_needed < memory_per_gpu * 0.4:  # Using <40% memory
            # Increase batch size
            adjusted_batch_size = min(16, base_batch_size * 2)
            self.logger.info(f"Increasing batch size to {adjusted_batch_size} for better GPU utilization")
        else:
            adjusted_batch_size = base_batch_size
        
        return adjusted_batch_size
    
    def create_training_arguments(self):
        """Create optimized training arguments for H200"""
        
        optimized_batch_size = self.optimize_batch_size()
        
        # Calculate effective batch size
        gradient_accumulation_steps = max(
            1, 
            self.config['training']['effective_batch_size'] // 
            (optimized_batch_size * self.device_count)
        )
        
        return TrainingArguments(
            output_dir=self.config['training']['output_dir'],
            
            # Batch size configuration
            per_device_train_batch_size=optimized_batch_size,
            per_device_eval_batch_size=optimized_batch_size * 2,
            gradient_accumulation_steps=gradient_accumulation_steps,
            
            # Training schedule
            num_train_epochs=self.config['training']['epochs'],
            learning_rate=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay'],
            warmup_steps=self.config['training']['warmup_steps'],
            lr_scheduler_type=self.config['training']['lr_scheduler_type'],
            
            # Memory and performance optimizations
            fp16=self.config['training']['fp16'],
            bf16=self.config['training'].get('bf16', False),
            gradient_checkpointing=True,
            dataloader_pin_memory=True,
            dataloader_num_workers=4,
            remove_unused_columns=False,
            
            # Logging and saving
            logging_dir=f"{self.config['training']['output_dir']}/logs",
            logging_steps=self.config['training']['logging_steps'],
            evaluation_strategy=self.config['training']['evaluation_strategy'],
            eval_steps=self.config['training']['eval_steps'],
            save_strategy=self.config['training']['save_strategy'],
            save_steps=self.config['training']['save_steps'],
            save_total_limit=self.config['training']['save_total_limit'],
            
            # Early stopping
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            
            # DeepSpeed configuration
            deepspeed=self.config['training'].get('deepspeed_config'),
            
            # Experiment tracking
            report_to=self.config['training'].get('report_to', ['tensorboard']),
            run_name=self.config['training'].get('run_name'),
            
            # Advanced optimizations
            max_grad_norm=self.config['training'].get('max_grad_norm', 1.0),
            seed=self.config['training']['seed'],
        )

def main():
    parser = argparse.ArgumentParser(description='Transformers Fine-tuning for H200')
    parser.add_argument('--config', required=True, help='Configuration file path')
    parser.add_argument('--resume_from_checkpoint', help='Resume from checkpoint')
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training')
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    logger.info(f"Starting Transformers fine-tuning with config: {args.config}")
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize wandb if configured
    if config.get('wandb', {}).get('enabled', False):
        wandb.init(
            project=config['wandb']['project'],
            name=config['wandb']['run_name'],
            config=config
        )
    
    # Setup model loader
    model_loader = H200OptimizedModelLoader(logger)
    
    # Load model and tokenizer
    logger.info(f"Loading model: {config['model']['name']}")
    model, tokenizer, model_info = model_loader.load_model_optimized(
        config['model']['name'],
        config,
        use_lora=config['model'].get('use_lora', False)
    )
    
    # Process dataset
    logger.info("Processing dataset...")
    dataset_processor = DatasetProcessor(tokenizer, config, logger)
    dataset = dataset_processor.load_and_process_dataset()
    
    logger.info(f"Dataset loaded: {dataset}")
    
    # Setup training optimizer
    training_optimizer = H200TrainingOptimizer(model_info, config, logger)
    training_args = training_optimizer.create_training_arguments()
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal language modeling
        return_tensors="pt"
    )
    
    # Callbacks
    callbacks = []
    if config['training'].get('early_stopping'):
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=config['training']['early_stopping']['patience'],
            early_stopping_threshold=config['training']['early_stopping']['threshold']
        ))
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset.get('train'),
        eval_dataset=dataset.get('validation'),
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
    )
    
    # Resume from checkpoint if specified
    if args.resume_from_checkpoint:
        logger.info(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
    
    # Log training information
    logger.info(f"Training arguments: {training_args}")
    logger.info(f"Model memory footprint: {model.get_memory_footprint() / 1024**3:.2f} GB")
    
    # Start training
    logger.info("Starting training...")
    try:
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    
    # Save final model
    logger.info("Saving final model...")
    trainer.save_model()
    
    # Save tokenizer
    tokenizer.save_pretrained(training_args.output_dir)
    
    # Final evaluation
    if dataset.get('validation'):
        logger.info("Running final evaluation...")
        eval_results = trainer.evaluate()
        logger.info(f"Final evaluation results: {eval_results}")
    
    # Save LoRA weights separately if using LoRA
    if config['model'].get('use_lora', False):
        lora_path = os.path.join(training_args.output_dir, "lora_weights")
        model.save_pretrained(lora_path)
        logger.info(f"LoRA weights saved to: {lora_path}")
    
    logger.info("Training completed successfully!")
    
    # Cleanup wandb
    if config.get('wandb', {}).get('enabled', False):
        wandb.finish()

if __name__ == '__main__':
    main()