#!/usr/bin/env python3
"""
LoRA (Low-Rank Adaptation) Fine-tuning Example for H200 GPUs
Demonstrates parameter-efficient fine-tuning of large language models.

Usage:
gpuq submit --command "python lora_example.py --model meta-llama/Llama-2-7b-hf --dataset alpaca" --gpus 1 --memory 50 --time 8

Features:
- Parameter-efficient fine-tuning with LoRA
- Support for instruction-following datasets
- Memory-efficient training on H200s
- Easy merging and deployment of LoRA weights
- Quantized training support for even larger models
"""

import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig, get_peft_model, TaskType,
    PeftModel, prepare_model_for_kbit_training
)
from datasets import load_dataset, Dataset
import json
import argparse
import logging
from pathlib import Path
import yaml
from typing import Dict, List, Any

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

class LoRATrainingConfig:
    """Configuration for LoRA training"""
    
    def __init__(self, config_dict: Dict[str, Any]):
        self.model_name = config_dict['model']['name']
        self.quantization = config_dict['model'].get('quantization', None)
        
        # LoRA parameters
        self.lora_r = config_dict['lora']['rank']
        self.lora_alpha = config_dict['lora']['alpha'] 
        self.lora_dropout = config_dict['lora']['dropout']
        self.target_modules = config_dict['lora']['target_modules']
        self.bias = config_dict['lora'].get('bias', 'none')
        
        # Training parameters
        self.output_dir = config_dict['training']['output_dir']
        self.num_epochs = config_dict['training']['epochs']
        self.batch_size = config_dict['training']['batch_size']
        self.learning_rate = config_dict['training']['learning_rate']
        self.max_length = config_dict['training']['max_length']
        
        # Dataset parameters
        self.dataset_name = config_dict['dataset']['name']
        self.dataset_config = config_dict['dataset']

class InstructionDataProcessor:
    """Process instruction-following datasets for LoRA training"""
    
    def __init__(self, tokenizer, config: LoRATrainingConfig):
        self.tokenizer = tokenizer
        self.config = config
        self.logger = setup_logging()
    
    def load_alpaca_dataset(self):
        """Load and process Alpaca-style instruction dataset"""
        
        try:
            dataset = load_dataset("tatsu-lab/alpaca", split="train")
        except:
            # Fallback to a smaller dataset for testing
            self.logger.warning("Could not load alpaca dataset, using sample data")
            dataset = self.create_sample_dataset()
        
        return dataset
    
    def load_custom_dataset(self, dataset_path: str):
        """Load custom instruction dataset"""
        
        if dataset_path.endswith('.json'):
            with open(dataset_path, 'r') as f:
                data = json.load(f)
            
            dataset = Dataset.from_list(data)
        else:
            dataset = load_dataset(dataset_path, split="train")
        
        return dataset
    
    def create_sample_dataset(self):
        """Create a small sample dataset for testing"""
        
        sample_data = [
            {
                "instruction": "Explain what machine learning is in simple terms.",
                "input": "",
                "output": "Machine learning is a way for computers to learn and make decisions from data, without being explicitly programmed for every situation."
            },
            {
                "instruction": "Write a Python function to calculate the factorial of a number.",
                "input": "",
                "output": "def factorial(n):\n    if n == 0 or n == 1:\n        return 1\n    else:\n        return n * factorial(n - 1)"
            },
            {
                "instruction": "Translate the following English text to French.",
                "input": "Hello, how are you today?",
                "output": "Bonjour, comment allez-vous aujourd'hui ?"
            }
        ] * 100  # Repeat for more training data
        
        return Dataset.from_list(sample_data)
    
    def format_instruction(self, example):
        """Format instruction data into training format"""
        
        instruction = example['instruction']
        input_text = example.get('input', '')
        output = example['output']
        
        # Create prompt in Alpaca format
        if input_text:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
        
        return prompt
    
    def tokenize_dataset(self, dataset):
        """Tokenize the dataset for training"""
        
        def tokenize_function(examples):
            # Format instructions
            texts = [self.format_instruction(example) for example in examples]
            
            # Tokenize
            tokenized = self.tokenizer(
                texts,
                truncation=True,
                max_length=self.config.max_length,
                padding='max_length',
                return_tensors=None
            )
            
            # For causal LM, labels are the same as input_ids
            tokenized['labels'] = tokenized['input_ids'].copy()
            
            return tokenized
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        return tokenized_dataset

class LoRAModelSetup:
    """Setup LoRA model with optimal configuration for H200"""
    
    def __init__(self, config: LoRATrainingConfig):
        self.config = config
        self.logger = setup_logging()
    
    def load_base_model(self):
        """Load base model with optional quantization"""
        
        self.logger.info(f"Loading base model: {self.config.model_name}")
        
        # Setup quantization if requested
        if self.config.quantization == '4bit':
            from transformers import BitsAndBytesConfig
            
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.float16
            )
            
            # Prepare model for k-bit training
            model = prepare_model_for_kbit_training(model)
            
        elif self.config.quantization == '8bit':
            from transformers import BitsAndBytesConfig
            
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.float16
            )
            
            model = prepare_model_for_kbit_training(model)
            
        else:
            # Standard FP16 loading
            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        
        return model
    
    def setup_lora(self, model):
        """Setup LoRA configuration and apply to model"""
        
        # LoRA configuration optimized for large language models
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules,
            bias=self.config.bias,
            fan_in_fan_out=False,  # Set to True for some models like GPT-2
        )
        
        # Apply LoRA to model
        model = get_peft_model(model, lora_config)
        
        # Print trainable parameters
        model.print_trainable_parameters()
        
        return model, lora_config

class LoRATrainer:
    """Custom trainer for LoRA fine-tuning"""
    
    def __init__(self, model, tokenizer, config: LoRATrainingConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.logger = setup_logging()
    
    def create_training_arguments(self):
        """Create training arguments optimized for H200 LoRA training"""
        
        return TrainingArguments(
            output_dir=self.config.output_dir,
            
            # Training schedule
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size * 2,
            gradient_accumulation_steps=4,  # Effective batch size
            
            # Optimizer settings (higher LR for LoRA)
            learning_rate=self.config.learning_rate,
            weight_decay=0.01,
            warmup_steps=100,
            lr_scheduler_type="cosine",
            
            # Memory and performance
            fp16=True,
            dataloader_pin_memory=True,
            gradient_checkpointing=False,  # Often not needed with LoRA
            
            # Logging and saving
            logging_dir=f"{self.config.output_dir}/logs",
            logging_steps=10,
            save_strategy="steps",
            save_steps=200,
            eval_strategy="steps",
            eval_steps=200,
            save_total_limit=3,
            
            # Reproducibility
            seed=42,
            
            # Remove unused columns
            remove_unused_columns=False,
        )
    
    def train(self, train_dataset, eval_dataset=None):
        """Start LoRA training"""
        
        training_args = self.create_training_arguments()
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal language modeling
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        
        # Start training
        self.logger.info("Starting LoRA training...")
        trainer.train()
        
        # Save final model
        trainer.save_model()
        self.tokenizer.save_pretrained(self.config.output_dir)
        
        self.logger.info(f"Training completed! Model saved to {self.config.output_dir}")
        
        return trainer

class LoRAInference:
    """Inference with LoRA-adapted model"""
    
    def __init__(self, base_model_name: str, lora_path: str):
        self.base_model_name = base_model_name
        self.lora_path = lora_path
        self.logger = setup_logging()
        
        self.load_model()
    
    def load_model(self):
        """Load base model and LoRA weights"""
        
        self.logger.info(f"Loading base model: {self.base_model_name}")
        self.logger.info(f"Loading LoRA weights from: {self.lora_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Load LoRA weights
        self.model = PeftModel.from_pretrained(base_model, self.lora_path)
        self.model.eval()
        
        self.logger.info("Model loaded successfully!")
    
    def generate(self, prompt: str, max_length: int = 200, **kwargs):
        """Generate text with LoRA model"""
        
        # Format prompt in instruction format if it doesn't contain ###
        if "### Instruction:" not in prompt:
            formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"
        else:
            formatted_prompt = prompt
        
        # Tokenize
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True
        )
        
        # Move to device
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=kwargs.get('temperature', 0.7),
                top_p=kwargs.get('top_p', 0.9),
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_length:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return {
            'prompt': prompt,
            'formatted_prompt': formatted_prompt,
            'generated_text': generated_text,
            'full_response': self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        }
    
    def merge_and_save(self, output_path: str):
        """Merge LoRA weights with base model and save"""
        
        self.logger.info(f"Merging LoRA weights and saving to: {output_path}")
        
        # Merge weights
        merged_model = self.model.merge_and_unload()
        
        # Save merged model
        merged_model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        
        self.logger.info("Merged model saved successfully!")

def load_config_file(config_path: str) -> Dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_default_config(model_name: str) -> Dict:
    """Create default configuration"""
    return {
        'model': {
            'name': model_name,
            'quantization': None  # 'none', '4bit', '8bit'
        },
        'lora': {
            'rank': 16,
            'alpha': 32,
            'dropout': 0.1,
            'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
            'bias': 'none'
        },
        'training': {
            'output_dir': './lora_results',
            'epochs': 3,
            'batch_size': 4,
            'learning_rate': 1e-4,
            'max_length': 512
        },
        'dataset': {
            'name': 'alpaca',
            'type': 'instruction'
        }
    }

def main():
    parser = argparse.ArgumentParser(description='LoRA Fine-tuning for H200')
    parser.add_argument('--mode', choices=['train', 'inference', 'merge'], 
                       default='train', help='Operation mode')
    parser.add_argument('--model', required=True, help='Base model name')
    parser.add_argument('--config', help='Configuration file (YAML)')
    
    # Training specific
    parser.add_argument('--dataset', default='alpaca', help='Dataset name')
    parser.add_argument('--output_dir', default='./lora_results', help='Output directory')
    
    # Inference specific  
    parser.add_argument('--lora_path', help='Path to LoRA weights')
    parser.add_argument('--prompt', help='Prompt for inference')
    parser.add_argument('--interactive', action='store_true', help='Interactive inference')
    
    # Merge specific
    parser.add_argument('--merge_output', help='Output path for merged model')
    
    args = parser.parse_args()
    
    logger = setup_logging()
    
    if args.mode == 'train':
        # Training mode
        logger.info("Starting LoRA training...")
        
        # Load configuration
        if args.config:
            config_dict = load_config_file(args.config)
        else:
            config_dict = create_default_config(args.model)
            config_dict['training']['output_dir'] = args.output_dir
            config_dict['dataset']['name'] = args.dataset
        
        config = LoRATrainingConfig(config_dict)
        
        # Setup model
        model_setup = LoRAModelSetup(config)
        base_model = model_setup.load_base_model()
        model, lora_config = model_setup.setup_lora(base_model)
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Process dataset
        data_processor = InstructionDataProcessor(tokenizer, config)
        
        if config.dataset_name == 'alpaca':
            dataset = data_processor.load_alpaca_dataset()
        else:
            dataset = data_processor.load_custom_dataset(config.dataset_name)
        
        # Tokenize dataset
        tokenized_dataset = data_processor.tokenize_dataset(dataset)
        
        # Split dataset
        if len(tokenized_dataset) > 100:
            split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
            train_dataset = split_dataset['train']
            eval_dataset = split_dataset['test']
        else:
            train_dataset = tokenized_dataset
            eval_dataset = None
        
        # Train model
        trainer = LoRATrainer(model, tokenizer, config)
        trainer.train(train_dataset, eval_dataset)
        
    elif args.mode == 'inference':
        # Inference mode
        if not args.lora_path:
            logger.error("--lora_path required for inference mode")
            return
        
        logger.info("Starting LoRA inference...")
        inference = LoRAInference(args.model, args.lora_path)
        
        if args.interactive:
            # Interactive mode
            print("🤖 LoRA Interactive Chat (type 'quit' to exit)")
            print("=" * 50)
            
            while True:
                try:
                    prompt = input("\n📝 Instruction: ").strip()
                    
                    if prompt.lower() in ['quit', 'exit', 'q']:
                        print("👋 Goodbye!")
                        break
                    
                    if not prompt:
                        continue
                    
                    print("🤖 Response: ", end="", flush=True)
                    result = inference.generate(prompt, max_length=200)
                    print(result['generated_text'])
                    
                except KeyboardInterrupt:
                    print("\n👋 Chat interrupted. Goodbye!")
                    break
        
        elif args.prompt:
            # Single prompt
            result = inference.generate(args.prompt)
            print(f"\nInstruction: {result['prompt']}")
            print(f"Response: {result['generated_text']}")
        
        else:
            logger.error("--prompt or --interactive required for inference")
    
    elif args.mode == 'merge':
        # Merge mode
        if not args.lora_path or not args.merge_output:
            logger.error("--lora_path and --merge_output required for merge mode")
            return
        
        logger.info("Merging LoRA weights with base model...")
        inference = LoRAInference(args.model, args.lora_path)
        inference.merge_and_save(args.merge_output)

if __name__ == '__main__':
    main()