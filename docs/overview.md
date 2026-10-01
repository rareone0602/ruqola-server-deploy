# Ruqola Project — Compute Resources

The NUS / NTU / Oxford Ruqola project has two servers:

| | **Mjölnir** (NTU) | **The Hopper** (NUS) |
|---|---|---|
| Use it for | Everyday experiments and prototyping | Large, long training runs |
| GPUs | 4× NVIDIA H200 NVL, 143771 MiB (140.4 GiB) each | Multi-node H100/H200 cluster, run by NUS |
| Scheduler | `gpuq` | PBS Pro |
| Access | An account from the Mjölnir admin | NUS guest account, HPC application, VPN |
| Start here | [GPU Queue guide](gpu-queue-guide.md) | [Hopper access guide](hopper.md) |

This site is the reference for how Mjölnir works. For Hopper it covers only
access; NUS Research Computing owns the details.

---

## Mjölnir (NTU)

Host `wsserver1`: 4× H200 NVL (compute capability 9.0), 256 logical CPUs,
755 GiB RAM, Ubuntu 24.04.4 LTS, GPU driver 575.57.08, CUDA 12.9.

### The rules

| Rule | Setting |
|---|---|
| Starting GPU work | Only through gpuq (`gpuq submit`, or `gpuq shell` for interactive work). GPU use outside gpuq is stopped about a minute after gpuq first sees it (SIGTERM, then SIGKILL 10 s later), and you get an email. |
| Job time | 48 hours for every job. Longer work must checkpoint and resubmit. |
| Who goes first | When more people want GPUs than there are, whoever has used the fewest GPU-hours lately. Use fades by half every 7 days. |
| Your GPU | A GPU your job runs on is yours: nobody else's job is put on it. A job may ask for up to all 4. |
| Home directory | 90 GiB soft, 100 GiB hard disk quota. Check with `quota -s`. |
| `/scratch` | Files neither read nor modified for 180 days are deleted. `/scratch/datasets` is exempt. |

Details: [GPU Queue guide](gpu-queue-guide.md) · [Scratch Storage](scratch-folder.md) ·
[Notifications](notifications-faq.md).

### Which page do I need?

- First time on the server: [Bash Basics](bash-basics.md), then [Best Practices](best-practices.md)
- Running GPU jobs: [GPU Queue guide](gpu-queue-guide.md)
- An email from the server: [Notifications](notifications-faq.md)
- Storing data: [Scratch Storage](scratch-folder.md)
- Frameworks: [PyTorch](pytorch-guide.md) · [TensorFlow](tensorflow-guide.md) · [JAX](jax-guide.md) · [Transformers](transformers-guide.md) · [Examples](../examples/README.md)
- Hardware: [H200 Specs](h200-specs.md)
- Something broke: [Troubleshooting](troubleshooting.md)

---

## The Hopper (NUS)

NUS's HPC cluster for heavy training. It uses the PBS Pro scheduler, and jobs
run inside Singularity/Apptainer containers. Getting access takes four NUS
steps; follow the [Hopper access guide](hopper.md). The official "Hopper
Cluster User Guide" is NUS-Restricted; get it through NUS.

---

## Getting help

- Mjölnir: check [Troubleshooting](troubleshooting.md), then contact the Mjölnir admin.
- Hopper: see the contacts in the [Hopper access guide](hopper.md).
