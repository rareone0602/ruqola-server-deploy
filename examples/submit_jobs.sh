#!/bin/bash
"""
Job submission examples for H200 GPUs
Demonstrates various common use cases and best practices.
"""

echo "=== H200 Job Submission Examples ==="
echo
echo "Server: 4x NVIDIA H200 NVL (indices 0,1,2,3), ~141 GB VRAM each."
echo
echo "Note on flags:"
echo "  --memory N  = MINIMUM free VRAM (GB) a GPU must have to be CHOSEN (a placement"
echo "                floor, NOT a cap/reservation). Set it just above your job's peak."
echo "  --notify    = email to ping when the job finishes (there is no --email flag)."
echo "  --queue     = wait for a slot; without it a submit is REJECTED immediately when"
echo "                no free (or owned-by-you) GPU matches the request."
echo "  --devices A,B = pin exact GPUs (rejected if another user holds one, unless --queue)."
echo "gpuq runs your job in the FOREGROUND and streams output to your terminal; there are"
echo "no per-job log files. Redirect output yourself (e.g. '> train.log 2>&1') to keep a log."
echo

# Example 1: Single GPU PyTorch training
echo "1. Single GPU PyTorch Training (ResNet on CIFAR-10)"
echo "Command:"
echo "gpuq submit --command \"python pytorch_training.py --config resnet_config.yaml\" --gpus 1 --memory 8 --time 8 --notify user@example.com"
echo

# Example 2: Multi-GPU PyTorch training
echo "2. Multi-GPU PyTorch Training (Distributed, all 4 H200s)"
echo "Command:"
echo "gpuq submit --command \"torchrun --nproc_per_node=4 pytorch_training.py --config resnet_config.yaml\" --gpus 4 --memory 12 --time 12"
echo

# Example 3: TensorFlow training with XLA
echo "3. TensorFlow Training with Mixed Precision"
echo "Command:"
echo "gpuq submit --command \"python tensorflow_training.py --config tf_config.json\" --gpus 1 --memory 12 --time 8"
echo

# Example 4: JAX/Flax training
echo "4. JAX/Flax Training (Functional Programming)"
echo "Command:"
echo "gpuq submit --command \"python jax_training.py --config jax_config.py\" --gpus 1 --memory 10 --time 6"
echo

# Example 5: Jupyter notebook for interactive development
echo "5. Interactive Jupyter Notebook"
echo "Command:"
echo "gpuq submit --command \"jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root\" --gpus 1 --memory 10 --time 8"
echo "Then connect via SSH tunnel: ssh -L 8888:localhost:8888 user@server"
echo

# Example 6: Large language model fine-tuning
echo "6. Large Language Model Fine-tuning (Hypothetical, multi-GPU)"
echo "Command:"
echo "gpuq submit --command \"torchrun --nproc_per_node=4 finetune_llm.py --model llama-7b --dataset custom\" --gpus 4 --memory 40 --time 24"
echo

# Example 7: Hyperparameter sweep
echo "7. Hyperparameter Sweep (Multiple Jobs)"
echo "Script:"
cat << 'EOF'
#!/bin/bash
for lr in 0.001 0.01 0.1; do
    for wd in 0.0001 0.001 0.01; do
        job_name="lr${lr}_wd${wd}"
        gpuq submit \
            --command "python pytorch_training.py --config resnet_config.yaml --lr $lr --weight_decay $wd --name $job_name" \
            --gpus 1 \
            --memory 8 \
            --time 6
    done
done
EOF
echo

# Example 8: Data preprocessing job
echo "8. Data Preprocessing (CPU intensive)"
echo "Command:"
echo "gpuq submit --command \"python preprocess_data.py --input /data/raw --output /data/processed\" --gpus 0 --memory 20 --time 4"
echo "Note: Use --gpus 0 for CPU-only jobs"
echo

# Example 9: Model inference/evaluation
echo "9. Model Inference on Large Dataset"
echo "Command:"
echo "gpuq submit --command \"python inference.py --model checkpoints/best_model.pth --data test_set/\" --gpus 1 --memory 25 --time 2"
echo

# Example 10: Resume training from checkpoint
echo "10. Resume Training from Checkpoint"
echo "Command:"
echo "gpuq submit --command \"python pytorch_training.py --config resnet_config.yaml --resume checkpoints/latest.pth\" --gpus 1 --memory 8 --time 8"
echo

# Example 11: Large Language Model Fine-tuning with Transformers
echo "11. Large Language Model Fine-tuning (Transformers, all 4 H200s)"
echo "Command:"
echo "gpuq submit --command \"torchrun --nproc_per_node=4 transformers_finetuning.py --config transformers_config.yaml\" --gpus 4 --memory 40 --time 16"
echo

# Example 12: LoRA Parameter-Efficient Fine-tuning
echo "12. LoRA Fine-tuning (Parameter-Efficient)"
echo "Command:"
echo "gpuq submit --command \"python lora_example.py --mode train --model meta-llama/Llama-2-7b-hf --config lora_config.yaml\" --gpus 1 --memory 20 --time 8"
echo

# Example 13: Transformers Inference
echo "13. Large Model Inference (Interactive)"
echo "Command:"
echo "gpuq submit --command \"python transformers_inference.py --model microsoft/DialoGPT-medium --mode interactive\" --gpus 1 --memory 30 --time 4"
echo

# Example 14: Batch Text Generation
echo "14. Batch Text Generation"
echo "Command:"
echo "gpuq submit --command \"python transformers_inference.py --model gpt2-large --mode batch --input prompts.txt --output results.json\" --gpus 1 --memory 25 --time 2"
echo

echo "=== Job Management Commands ==="
echo
echo "Check queue status:     gpuq status"
echo "Monitor jobs:           watch -n 5 gpuq status"
echo "Keep a log:             gpuq runs in the foreground and streams output to your"
echo "                        terminal (no per-job log files). Redirect it yourself:"
echo "                        gpuq submit --command \"python train.py\" --gpus 1 > train.log 2>&1"
echo "Kill a job:             gpuq kill XXXXX        (or: gpuq kill --job-id XXXXX)"
echo "Kill all your jobs:     gpuq status | awk '\$1==\"Job\" && \$4==\"'\"\$USER\"'\"{print \$2}' | xargs -rn1 gpuq kill"
echo

echo "=== Resource Guidelines ==="
echo
echo "--memory is a PLACEMENT FLOOR (min free VRAM the chosen GPU must have), not a"
echo "budget. Set it just above your job's real peak usage so it isn't blocked from"
echo "otherwise-suitable cards. Each H200 has ~141 GB, so even very large jobs fit."
echo
echo "Small models (< 100M params):    --memory 8-16    --time 2-4"
echo "Medium models (100M-1B params):  --memory 16-32   --time 4-12"
echo "Large models (1B-10B params):    --memory 32-64   --time 8-24"
echo "Very large models (10B+ params): --memory 64-120  --time 12-24"
echo
echo "Memory usage tips:"
echo "- Use mixed precision (BF16 on H200; FP16 as a legacy fallback) to roughly halve memory"
echo "- Enable gradient checkpointing for large models"
echo "- Use gradient accumulation for large effective batch sizes"
echo "- Monitor with: nvidia-smi -l 1"
echo

echo "=== Best Practices Reminder ==="
echo
echo "✅ DO:"
echo "- Test with small datasets first"
echo "- Use appropriate batch sizes (multiples of 8)"
echo "- Enable mixed precision training"
echo "- Implement checkpointing for long jobs"
echo "- Monitor GPU utilization (should be >80%)"
echo "- Clean up failed jobs promptly"
echo
echo "❌ DON'T:"
echo "- Request more resources than needed"
echo "- Leave crashed jobs running"
echo "- Submit multiple identical jobs"
echo "- Use all GPUs unless necessary"
echo "- Ignore memory warnings"
echo

echo "For more information, see:"
echo "- Documentation: /path/to/docs/"
echo "- GPU Queue Guide: docs/gpu-queue-guide.md"
echo "- Best Practices: docs/best-practices.md"
echo "- Framework Guides: docs/pytorch-guide.md, docs/tensorflow-guide.md, docs/jax-guide.md"