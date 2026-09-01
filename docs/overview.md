# Ruqola Project — Compute Resources

Welcome to the central guide for the compute resources of the NUS / NTU / Oxford
joint **Ruqola project**. Two servers are available, at different scales:

| | **Mjölnir** (NTU) | **The Hopper** (NUS) |
|---|---|---|
| Purpose | Light-to-medium experiments, prototyping, exploratory analysis | Heavy, large-scale training and data processing |
| GPUs | **4× NVIDIA H200 NVL**, ~141 GB each (~564 GB total) | Multi-node H100/H200 cluster (NUS-managed) |
| Scheduler / sharing | `gpuq` cooperative queue (this guide) | PBS Pro batch scheduler |
| Access | NTU account on the box (admin-provisioned) | NUS guest account + HPC application + VPN |
| Start here | [GPU Queue (gpuq) guide](gpu-queue-guide.md) | [Hopper access guide](hopper.md) |

> This site is the **single source of truth** for using **Mjölnir**. For **Hopper**,
> it summarizes the access process; current cluster specifics are managed by NUS
> Research Computing and change over time — confirm them through NUS.

---

## Mjölnir (NTU) — at a glance

Mjölnir is the NTU server most Ruqola members use day-to-day. Verified hardware
(host `wsserver1`):

- **GPUs:** 4× NVIDIA H200 NVL, ~141 GB VRAM each (143 771 MiB), ~564 GB total
- **Compute capability:** 9.0 (Hopper, GH100)
- **CPU / RAM:** 256 logical CPUs, 755 GiB system RAM
- **OS / stack:** Ubuntu 24.04.4 LTS, GPU driver 575.57.08, CUDA 12.9
- **Storage:** large shared `/scratch` for datasets and virtualenvs (see [Scratch Storage](scratch-folder.md))
- **Sharing:** the `gpuq` cooperative queue — *you own the GPU(s) allocated to you*

GPU access on Mjölnir is coordinated **cooperatively** by `gpuq` (no hard
reservation system): you submit jobs through `gpuq`, which assigns you free GPUs,
and you may stack more of your own jobs on GPUs you already hold. **Always launch
GPU work through `gpuq`** — a GPU process started outside it gets a warning email
from `gpuq audit` and is **killed 15 minutes** after first detection. See the
[GPU Queue guide](gpu-queue-guide.md).

### Policy at a glance (current live settings)

| Policy | Current setting |
|---|---|
| Wall time per job | **48-hour hard cap** — `-t/--time` may not exceed it; it is also the default, so pass a realistic smaller `-t` |
| Cards per user | **3 GPUs hard cap**; holding 3 already triggers an admin warning, so treat **2** as the courtesy ceiling |
| GPU-hour quota | **168 GPU-hours per rolling 7 days** (one H200 running 24/7). Going over never rejects a job — it is **held 15 minutes** at submit and deprioritized |
| Jobs outside `gpuq` | Warning email at first detection, then **killed 15 minutes later** by the periodic `gpuq audit` sweep |

Full details and the exact lifecycles: [GPU Queue guide](gpu-queue-guide.md) ·
[Notifications FAQ](notifications-faq.md).

### Which Mjölnir doc do I need?

- New to the shell / the server → [Bash Basics](bash-basics.md), [Best Practices](best-practices.md)
- Running GPU jobs → **[GPU Queue (gpuq)](gpu-queue-guide.md)**, [Notifications FAQ](notifications-faq.md)
- Storing data → [Scratch Storage](scratch-folder.md)
- Framework setup → [PyTorch](pytorch-guide.md) · [TensorFlow](tensorflow-guide.md) · [JAX](jax-guide.md) · [Transformers](transformers-guide.md) · [Examples](../examples/README.md)
- Hardware detail → [H200 Specs](h200-specs.md)
- Something broke → [Troubleshooting](troubleshooting.md)

---

## The Hopper (NUS) — at a glance

**The Hopper** is NUS's high-performance computing (HPC) cluster, reserved for
**computationally intensive** experiments and large-scale training. It is a
multi-node H100/H200 cluster, uses the **PBS Pro** scheduler (not Slurm), and
requires jobs to run inside **Singularity/Apptainer** containers.

Getting on Hopper is a multi-step NUS process (guest account → activation → HPC
application → VPN). The full walkthrough is in the **[Hopper access guide](hopper.md)**.

> The official "Hopper Cluster User Guide" is an **NUS-Restricted** document and is
> not redistributed here. Authorized members can obtain it through NUS channels.

---

## Getting help

- Mjölnir usage questions → start with [Troubleshooting](troubleshooting.md) and the [Notifications FAQ](notifications-faq.md).
- Account / quota issues on Mjölnir → contact the server admin.
- Hopper / NUS account issues → see the contacts in the [Hopper access guide](hopper.md).
