# Best Practices

How to work on Mjölnir without losing work or blocking others. The rules
themselves are in the [GPU Queue guide](gpu-queue-guide.md) and
[Scratch Storage](scratch-folder.md).

## Set up once

Make a venv for each project. The system has `python3` but no `python`, and
`pip install` outside a venv fails.

```bash
python3 -m venv ~/venvs/myenv
source ~/venvs/myenv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

- **Keep venvs in your home.** `~/venvs/` exists for this. In `/scratch`, the
  cleanup deletes files you never read, which can break a venv.
- **Put big caches on scratch** so they do not fill your 90 GiB home quota. Add
  this to `~/.bashrc`: `export HF_HOME=/scratch/users/$USER/hf`
- **Do not put `export CUDA_VISIBLE_DEVICES=...` in `~/.bashrc`.** gpuq sets it
  for each job, and a shell that reads `~/.bashrc` inside a job would undo that.
- **Learn tmux.** `tmux new -s work` starts a session, `Ctrl-b d` detaches, and
  `tmux attach -t work` returns. Jobs in tmux survive a dropped connection.

## Before a big job

- **Test small first:** `gpuq submit -m 20 -t 0.5 -- python train.py --epochs 1`.
- **Save checkpoints and resume from them.** No job runs longer than 48 hours.
  At its limit a job gets SIGTERM, then SIGKILL 10 seconds later.
- **Set `-m`** to the memory your job needs, not more.
- **Set `-t`** to the expected runtime plus a margin. The quota check counts it
  at submit.
- **Save the output:** `gpuq submit ... 2>&1 | tee run.log`, inside tmux.
- **Check `gpuq quota`.** Over 168 GPU-hours in the rolling 7-day window, new
  jobs wait 15 minutes and then run at low priority.
- **Tell the group** before a long run on several cards.

## While it runs

- Keep every card you hold busy. If a job is idle or no longer needed, stop it
  with `gpuq kill <id>`.
- Use 2 cards unless you need 3.
- Do not submit the same job twice.
- CPUs and RAM are shared too: 256 CPUs and 755 GiB for everyone. Give the
  DataLoader the workers it needs (8 is a good start), not the whole machine.

## Storage

| Where | What goes there |
|---|---|
| `~` (home) | code, venvs, small files. 90 GiB soft, 100 GiB hard (`quota -s`). |
| `/scratch/users/<you>/` | data, checkpoints, caches. Files unused for 180 days are deleted. |
| `/scratch/datasets/` | large data shared with the group. Never cleaned. |

- Link shared data (`ln -s`) instead of copying it.
- Delete what you no longer need.
- Keep a copy of anything important off the server.

## Performance tips

- Train in BF16. The H200 is a Hopper card (compute capability 9.0), and BF16
  rarely needs a gradient scaler.
- In PyTorch, try `torch.compile`, a fused optimizer
  (`torch.optim.AdamW(..., fused=True)`) and `F.scaled_dot_product_attention`.
- For more than one GPU, use DistributedDataParallel with `torchrun`, not
  `DataParallel`. All 4 cards are linked by NVLink; `nvidia-smi topo -m` shows it.
- The framework guides go further: [PyTorch](pytorch-guide.md),
  [TensorFlow](tensorflow-guide.md), [JAX](jax-guide.md),
  [Transformers](transformers-guide.md).

## When something goes wrong

- `gpuq kill --mine` stops all your jobs, running and queued.
- `pkill -u $USER -f train.py` stops a runaway script by name. Never run a bare
  `pkill -u $USER`: it also kills your shell and SSH session.
- Then read [Troubleshooting](troubleshooting.md).
