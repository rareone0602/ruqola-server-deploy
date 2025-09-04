#!/usr/bin/env python3
"""
Hugging Face Transformers Inference Example for H200 GPUs
Optimized for high-throughput inference and interactive generation.

Usage:
gpuq submit --command "python transformers_inference.py --model microsoft/DialoGPT-medium --input prompts.txt" --gpus 1 --memory 40 --time 4

Features:
- Optimized batch inference for high throughput
- Interactive text generation
- Memory-efficient loading for large models
- Streaming generation support
- Multi-GPU inference for massive models
"""

import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    pipeline, TextStreamer
)
import json
import argparse
import logging
from pathlib import Path
import time
from typing import List, Dict, Optional, Iterator
import threading
from queue import Queue
import numpy as np

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

class H200InferenceOptimizer:
    """Optimized inference engine for H200 GPUs"""
    
    def __init__(self, 
                 model_name: str, 
                 device_map: str = "auto",
                 torch_dtype = torch.float16,
                 use_cache: bool = True):
        
        self.model_name = model_name
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.use_cache = use_cache
        self.logger = setup_logging()
        
        # Load model and tokenizer
        self.load_model()
        
        # Performance tracking
        self.total_tokens_generated = 0
        self.total_inference_time = 0
    
    def load_model(self):
        """Load model with optimal configuration for H200"""
        
        self.logger.info(f"Loading model: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Determine optimal loading strategy
        try:
            # Try loading with device mapping for large models
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
                device_map=self.device_map,
                use_cache=self.use_cache,
                low_cpu_mem_usage=True,
            )
            
            self.logger.info("Model loaded with device mapping")
            
        except Exception as e:
            self.logger.warning(f"Device mapping failed: {e}")
            
            # Fallback to standard loading
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
                use_cache=self.use_cache,
            ).cuda()
            
            self.logger.info("Model loaded on single GPU")
        
        # Set to evaluation mode
        self.model.eval()
        
        # Print memory usage
        if hasattr(self.model, 'get_memory_footprint'):
            memory_gb = self.model.get_memory_footprint() / 1024**3
            self.logger.info(f"Model memory footprint: {memory_gb:.2f} GB")
    
    def generate_single(self, 
                       prompt: str,
                       max_length: int = 100,
                       temperature: float = 0.7,
                       top_p: float = 0.9,
                       top_k: int = 50,
                       repetition_penalty: float = 1.1,
                       do_sample: bool = True,
                       return_full_text: bool = False) -> Dict:
        """Generate text for a single prompt"""
        
        start_time = time.time()
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True,
            max_length=512
        )
        
        # Move to same device as model
        if hasattr(self.model, 'device'):
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        else:
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        input_length = inputs['input_ids'].shape[1]
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=input_length + max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=self.use_cache,
                num_return_sequences=1
            )
        
        # Decode output
        if return_full_text:
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            # Return only the generated part
            generated_tokens = outputs[0][input_length:]
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Calculate metrics
        generation_time = time.time() - start_time
        tokens_generated = len(generated_tokens) if not return_full_text else len(outputs[0]) - input_length
        tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0
        
        # Update tracking
        self.total_tokens_generated += tokens_generated
        self.total_inference_time += generation_time
        
        return {
            'prompt': prompt,
            'generated_text': generated_text,
            'tokens_generated': tokens_generated,
            'generation_time': generation_time,
            'tokens_per_second': tokens_per_second,
            'full_text': self.tokenizer.decode(outputs[0], skip_special_tokens=True) if not return_full_text else generated_text
        }
    
    def generate_batch(self, 
                      prompts: List[str], 
                      batch_size: int = 8,
                      **generation_kwargs) -> List[Dict]:
        """Generate text for multiple prompts with batching"""
        
        self.logger.info(f"Processing {len(prompts)} prompts with batch size {batch_size}")
        
        results = []
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            self.logger.info(f"Processing batch {i//batch_size + 1}/{(len(prompts)-1)//batch_size + 1}")
            
            # Process batch
            batch_results = self.process_batch(batch_prompts, **generation_kwargs)
            results.extend(batch_results)
            
            # Clear cache periodically
            if i % (batch_size * 5) == 0:
                torch.cuda.empty_cache()
        
        return results
    
    def process_batch(self, prompts: List[str], **generation_kwargs) -> List[Dict]:
        """Process a single batch of prompts"""
        
        start_time = time.time()
        
        # Tokenize all prompts
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Move to device
        if hasattr(self.model, 'device'):
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        else:
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        input_lengths = inputs['attention_mask'].sum(dim=1)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=inputs['input_ids'].shape[1] + generation_kwargs.get('max_length', 100),
                temperature=generation_kwargs.get('temperature', 0.7),
                top_p=generation_kwargs.get('top_p', 0.9),
                top_k=generation_kwargs.get('top_k', 50),
                repetition_penalty=generation_kwargs.get('repetition_penalty', 1.1),
                do_sample=generation_kwargs.get('do_sample', True),
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=self.use_cache,
                num_return_sequences=1
            )
        
        # Process results
        batch_time = time.time() - start_time
        results = []
        
        for i, (prompt, output, input_length) in enumerate(zip(prompts, outputs, input_lengths)):
            # Extract generated tokens
            generated_tokens = output[input_length:]
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            tokens_generated = len(generated_tokens)
            tokens_per_second = tokens_generated / (batch_time / len(prompts))
            
            results.append({
                'prompt': prompt,
                'generated_text': generated_text,
                'tokens_generated': tokens_generated,
                'generation_time': batch_time / len(prompts),
                'tokens_per_second': tokens_per_second,
                'full_text': self.tokenizer.decode(output, skip_special_tokens=True)
            })
        
        return results
    
    def stream_generate(self, prompt: str, **generation_kwargs) -> Iterator[str]:
        """Generate text with streaming output"""
        
        # Setup streamer
        streamer = TextStreamer(
            self.tokenizer, 
            skip_prompt=True, 
            skip_special_tokens=True
        )
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if hasattr(self.model, 'device'):
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        else:
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate with streaming
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=generation_kwargs.get('max_length', 100),
                temperature=generation_kwargs.get('temperature', 0.7),
                top_p=generation_kwargs.get('top_p', 0.9),
                do_sample=generation_kwargs.get('do_sample', True),
                pad_token_id=self.tokenizer.eos_token_id,
                streamer=streamer,
                use_cache=self.use_cache
            )
        
        # Return the complete generated text
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_length:]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        avg_tokens_per_second = (
            self.total_tokens_generated / self.total_inference_time 
            if self.total_inference_time > 0 else 0
        )
        
        return {
            'total_tokens_generated': self.total_tokens_generated,
            'total_inference_time': self.total_inference_time,
            'average_tokens_per_second': avg_tokens_per_second
        }

class InteractiveChat:
    """Interactive chat interface"""
    
    def __init__(self, inference_engine: H200InferenceOptimizer):
        self.engine = inference_engine
        self.conversation_history = []
    
    def chat_loop(self):
        """Start interactive chat loop"""
        
        print("🤖 Interactive Chat Started! (type 'quit' to exit, 'clear' to clear history)")
        print("=" * 60)
        
        while True:
            try:
                # Get user input
                user_input = input("\n👤 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                elif user_input.lower() == 'clear':
                    self.conversation_history = []
                    print("🗑️ Conversation history cleared!")
                    continue
                elif user_input.lower() == 'stats':
                    stats = self.engine.get_performance_stats()
                    print(f"📊 Performance: {stats['average_tokens_per_second']:.1f} tokens/sec")
                    continue
                elif not user_input:
                    continue
                
                # Add to conversation history
                self.conversation_history.append(f"Human: {user_input}")
                
                # Prepare prompt with context
                context = "\n".join(self.conversation_history[-10:])  # Last 10 exchanges
                prompt = f"{context}\nAssistant:"
                
                print("🤖 Assistant: ", end="", flush=True)
                
                # Generate response with streaming
                response = self.engine.stream_generate(
                    prompt,
                    max_length=150,
                    temperature=0.7,
                    top_p=0.9
                )
                
                # Add response to history
                self.conversation_history.append(f"Assistant: {response}")
                
            except KeyboardInterrupt:
                print("\n\n👋 Chat interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")

def load_prompts_from_file(file_path: str) -> List[str]:
    """Load prompts from text file"""
    prompts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(line)
    return prompts

def save_results_to_file(results: List[Dict], output_path: str):
    """Save inference results to JSON file"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

def main():
    parser = argparse.ArgumentParser(description='Transformers Inference on H200')
    parser.add_argument('--model', required=True, help='Model name or path')
    parser.add_argument('--mode', choices=['single', 'batch', 'interactive'], 
                       default='interactive', help='Inference mode')
    
    # Input/output options
    parser.add_argument('--input', help='Input file with prompts (one per line)')
    parser.add_argument('--output', help='Output JSON file for results')
    parser.add_argument('--prompt', help='Single prompt for direct inference')
    
    # Generation parameters
    parser.add_argument('--max_length', type=int, default=100, help='Maximum generation length')
    parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.9, help='Nucleus sampling parameter')
    parser.add_argument('--top_k', type=int, default=50, help='Top-k sampling parameter')
    parser.add_argument('--repetition_penalty', type=float, default=1.1, help='Repetition penalty')
    
    # Performance options
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for batch inference')
    parser.add_argument('--device_map', default='auto', help='Device mapping strategy')
    parser.add_argument('--torch_dtype', default='float16', help='Model precision')
    
    args = parser.parse_args()
    
    # Convert torch_dtype string to actual type
    dtype_map = {
        'float16': torch.float16,
        'float32': torch.float32,
        'bfloat16': torch.bfloat16
    }
    torch_dtype = dtype_map.get(args.torch_dtype, torch.float16)
    
    # Initialize inference engine
    logger = setup_logging()
    logger.info(f"Initializing inference engine for model: {args.model}")
    
    engine = H200InferenceOptimizer(
        model_name=args.model,
        device_map=args.device_map,
        torch_dtype=torch_dtype,
        use_cache=True
    )
    
    # Generation parameters
    generation_kwargs = {
        'max_length': args.max_length,
        'temperature': args.temperature,
        'top_p': args.top_p,
        'top_k': args.top_k,
        'repetition_penalty': args.repetition_penalty,
        'do_sample': True
    }
    
    if args.mode == 'interactive':
        # Interactive chat mode
        chat = InteractiveChat(engine)
        chat.chat_loop()
        
    elif args.mode == 'single':
        # Single prompt inference
        if not args.prompt:
            logger.error("--prompt required for single mode")
            return
        
        logger.info("Generating response for single prompt...")
        result = engine.generate_single(args.prompt, **generation_kwargs)
        
        print(f"\nPrompt: {result['prompt']}")
        print(f"Generated: {result['generated_text']}")
        print(f"Tokens/sec: {result['tokens_per_second']:.1f}")
        
    elif args.mode == 'batch':
        # Batch inference mode
        if not args.input:
            logger.error("--input file required for batch mode")
            return
        
        # Load prompts
        logger.info(f"Loading prompts from: {args.input}")
        prompts = load_prompts_from_file(args.input)
        logger.info(f"Loaded {len(prompts)} prompts")
        
        # Run batch inference
        results = engine.generate_batch(
            prompts, 
            batch_size=args.batch_size,
            **generation_kwargs
        )
        
        # Save results
        if args.output:
            logger.info(f"Saving results to: {args.output}")
            save_results_to_file(results, args.output)
        
        # Print summary
        total_tokens = sum(r['tokens_generated'] for r in results)
        total_time = sum(r['generation_time'] for r in results)
        avg_tokens_per_sec = total_tokens / total_time if total_time > 0 else 0
        
        logger.info(f"Batch inference completed:")
        logger.info(f"  Prompts processed: {len(results)}")
        logger.info(f"  Total tokens generated: {total_tokens}")
        logger.info(f"  Average tokens/sec: {avg_tokens_per_sec:.1f}")
    
    # Print final performance stats
    stats = engine.get_performance_stats()
    logger.info(f"Final performance stats: {stats}")

if __name__ == '__main__':
    main()