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
echo "  --gpus N    = number of whole GPUs (default 1, up to 4); each is the job's alone."
echo "  --memory N  = run beside your own running job if N GB is free there, else on a"
echo "                free GPU. Leave it out for training; use it for small tests."
echo "  --devices N = add the job to your GPU N, which one of your jobs is running on now."
echo "  --detach    = return at once; the output goes to ~/gpuq-logs/<id>.log."
echo "  --notify    = also email you when the job ends (there is no --email flag)."
echo "Every submit waits its turn. Without --detach, gpuq runs your job in your terminal"
echo "and streams its output there: run it inside tmux, or redirect it (e.g. '> train.log 2>&1')."
echo

# Example 1: Single GPU PyTorch training
echo "1. Single GPU PyTorch Training (ResNet on CIFAR-10)"
echo "Command:"
echo "gpuq submit --command \"python pytorch_training.py --config resnet_config.yaml\" --gpus 1 --notify"
echo

# Example 2: Multi-GPU PyTorch training
echo "2. Multi-GPU PyTorch Training (Distributed, all 4 H200s)"
echo "Command:"
echo "gpuq submit --command \"torchrun --nproc_per_node=4 pytorch_training.py --config resnet_config.yaml\" --gpus 4"
echo

# Example 3: TensorFlow training with XLA
echo "3. TensorFlow Training with Mixed Precision"
echo "Command:"
echo "gpuq submit --command \"python tensorflow_training.py --config tf_config.json\" --gpus 1"
echo

# Example 4: JAX/Flax training
echo "4. JAX/Flax Training (Functional Programming)"
echo "Command:"
echo "gpuq submit --command \"python jax_training.py --config jax_config.py\" --gpus 1"
echo

# Example 5: Jupyter notebook for interactive development
echo "5. Interactive Jupyter Notebook"
echo "Command:"
echo "gpuq submit --command \"timeout 8h jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root\" --gpus 1"
echo "Then connect via SSH tunnel: ssh -L 8888:localhost:8888 user@server"
echo

# Example 6: Large language model fine-tuning
echo "6. Large Language Model Fine-tuning (Hypothetical, multi-GPU)"
echo "Command:"
echo "gpuq submit --command \"torchrun --nproc_per_node=4 finetune_llm.py --model llama-7b --dataset custom\" --gpus 4"
echo

# Example 7: Hyperparameter sweep
echo "7. Hyperparameter Sweep (Multiple Jobs)"
echo "Script:"
cat << 'EOF'
#!/bin/bash
for lr in 0.001 0.01 0.1; do
    for wd in 0.0001 0.001 0.01; do
        job_name="lr${lr}_wd${wd}"
        gpuq submit --detach \
            --command "python pytorch_training.py --config resnet_config.yaml --lr $lr --weight_decay $wd --name $job_name" \
            --gpus 1
    done
done
EOF
echo

# Example 8: Data preprocessing job
echo "8. Data Preprocessing (CPU intensive)"
echo "Command:"
echo "python preprocess_data.py --input /data/raw --output /data/processed"
echo "Note: gpuq is for GPU jobs only (--gpus is at least 1); run CPU-only jobs directly"
echo

# Example 9: Model inference/evaluation
echo "9. Model Inference on Large Dataset"
echo "Command:"
echo "gpuq submit --command \"python inference.py --model checkpoints/best_model.pth --data test_set/\" --gpus 1 --memory 25"
echo "Note: --memory 25 runs it beside your own running job if 25 GB is free there, else on a free GPU"
echo

# Example 10: Resume training from checkpoint
echo "10. Resume Training from Checkpoint"
echo "Command:"
echo "gpuq submit --command \"python pytorch_training.py --config resnet_config.yaml --resume checkpoints/latest.pth\" --gpus 1"
echo

# Example 11: Large Language Model Fine-tuning with Transformers
echo "11. Large Language Model Fine-tuning (Transformers, all 4 H200s)"
echo "Command:"
echo "gpuq submit --command \"torchrun --nproc_per_node=4 transformers_finetuning.py --config transformers_config.yaml\" --gpus 4"
echo

# Example 12: LoRA Parameter-Efficient Fine-tuning
echo "12. LoRA Fine-tuning (Parameter-Efficient)"
echo "Command:"
echo "gpuq submit --command \"python lora_example.py --mode train --model meta-llama/Llama-2-7b-hf --config lora_config.yaml\" --gpus 1"
echo

# Example 13: Transformers Inference
echo "13. Large Model Inference (Interactive)"
echo "Command:"
echo "gpuq submit --command \"python transformers_inference.py --model microsoft/DialoGPT-medium --mode interactive\" --gpus 1"
echo

# Example 14: Batch Text Generation
echo "14. Batch Text Generation"
echo "Command:"
echo "gpuq submit --command \"python transformers_inference.py --model gpt2-large --mode batch --input prompts.txt --output results.json\" --gpus 1"
echo

echo "=== Job Management Commands ==="
echo
echo "Check queue status:     gpuq status"
echo "Monitor jobs:           watch -n 5 gpuq status"
echo "Keep a log:             add --detach (output goes to ~/gpuq-logs/<id>.log), or"
echo "                        redirect an attached job's output yourself:"
echo "                        gpuq submit --command \"python train.py\" --gpus 1 > train.log 2>&1"
echo "Kill a job:             gpuq kill XXXXX        (or: gpuq kill --job-id XXXXX)"
echo "Kill all your jobs:     gpuq kill --mine"
echo

echo "=== Resource Guidelines ==="
echo
echo "Leave out --memory for training: without it the job gets a GPU to itself."
echo "Each H200 has ~141 GB, so even very large jobs fit."
echo "Every job may run 48 h; save checkpoints and resubmit for longer work."
echo "For a shorter limit, wrap the command: gpuq submit -- timeout 4h python train.py"
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