# The Hopper (NUS) — Access Guide

**The Hopper** is the NUS high-performance computing (HPC) cluster available to the
Ruqola project. It is designed for **computationally intensive experiments**,
large-scale model training, and heavy data processing — the heavy-lifting
counterpart to NTU's [Mjölnir](overview.md) server.

Key facts:

- **Scheduler:** PBS Pro (not Slurm). Jobs are submitted with `qsub`.
- **Containers:** all jobs run inside **Singularity/Apptainer** containers.
- **Scale:** a multi-node H100/H200 cluster managed by NUS Research Computing.

> **Specifications change over time.** The cluster has grown since the project
> onboarding material was first written, so exact node counts, GPU mixes, and
> storage quotas are intentionally **not** pinned down here. Confirm current
> figures with NUS Research Computing. The official **"Hopper Cluster User Guide"**
> is an **NUS-Restricted** document — obtain it through NUS channels rather than
> from this public site.

---

## Access: the four-step process

Access involves (1) getting an NUS guest account, (2) activating it, (3) applying
for HPC access, and (4) connecting. Follow these in order.

### Part 1 — Request an NUS guest account

1. Contact **Prof. Lee Wee Sun (School of Computing, NUS)**, who will ask NUS
   administrative staff to create a guest account and internal NUS email for you.
2. The administrative staff will ask you for: last/first name, display name, title,
   visitor job title, visitor organization, organization address, email, mobile
   country code and number, purpose of access, and required period.
3. You will receive a confirmation email with your **account start date**.
4. On the start date you receive: an email with a **PIN-protected PDF**
   (ePasswordSlip) containing your password, and an **SMS with the PIN** to open it.

### Part 2 — Activate your NUS account

Complete these **in sequence** before using the account or applying for HPC access:

1. **Change password** at <https://exchange.nus.edu.sg/passwordportal>.
2. **Accept the AUP:** log in at <https://inetapps.nus.edu.sg/aup> with your new
   password and accept the NUS IT Acceptable Use Policy. This must be done within
   **14 days** of your start date or the account is locked.
3. **Set up MFA:** complete Microsoft Multifactor Authentication — see *Section 2*
   of the [NUS-ID Getting Started guide](https://nusit.nus.edu.sg/services/account-and-access/account/nus-id-getting-started/).

### Part 3 — Apply for Hopper (HPC) access

1. Once your NUS account is fully active, apply at
   <https://nusit.nus.edu.sg/hpc/get-an-hpc-account/>.
2. Use the project name **`CFP03-SF-101`** on the form.

### Part 4 — Connect

- **VPN required:** connect with the **NUS VPN (Cisco)** unless you are on the NUS
  campus network. ([NUS VPN guide](https://nusit-dwp.onbmc.com/dwp/app/#/knowledge/KBA00027608/rkm), requires NUS login.)
- **SSH in:** `ssh your_nus_id@hopper.nus.edu.sg`
- **Verify your project group:** after connecting, run `hpc project`.

---

## Working on Hopper — quick tips

- **Scheduler is PBS Pro.** Prepare a `.pbs` job script and submit it with
  `qsub train.pbs`; check status with `qstat`. Multi-node jobs require MPI and a
  separate worker script (unlike Slurm).
- **Containers are mandatory.** Run your code inside the provided
  Singularity/Apptainer images, or build/request your own. Pre-built images live
  under `/app1/common/singularity-img/hopper/`.
- **Python packages:** create a virtualenv inside a CUDA container (stored on
  `/scratch` or `/hpctmp`), or `pip install --prefix=...` to a custom path. Do
  **not** use `pip install --user`.
- **Storage:** use the large scratch space for datasets and virtualenvs; home
  directories are small and snapshotted. Scratch/temp areas are periodically
  **purged** — don't keep anything irreplaceable there. Confirm current quotas and
  purge windows with NUS.
- **Caches on scratch:** point Hugging Face and Triton caches at scratch, e.g.
  `export HF_HOME=/scratch/<your_id>/cache` and
  `export XDG_CACHE_HOME=/scratch/<your_id>/cache`, and add these lines to your job
  script too.
- **Utilization matters (Green Computing).** Jobs must actually use the resources
  they request — if you request N GPUs, your job should keep all N busy. Persistent
  under-utilization is discouraged and monitored.
- **Acknowledgement:** publications using Hopper must acknowledge NUS IT Research
  Computing, grant number **`NUSREC-HPC-00001`**.

---

## Contacts

- **Account / access:** Prof. Lee Wee Sun (NUS School of Computing).
- **Connecting / project setup:** project members **Wei** and **Duy** can advise.
- **Official guide:** request the NUS-Restricted *Hopper Cluster User Guide* via NUS.
