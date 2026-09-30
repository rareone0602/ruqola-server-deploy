# Ruqola Server Documentation

How to use the **Ruqola project** compute servers:

- **Mjölnir (NTU):** a shared host with **4× NVIDIA H200 NVL** GPUs, shared
  through the [`gpuq`](gpuq/README.md) queue.
- **The Hopper (NUS):** NUS's multi-node H100/H200 cluster (PBS Pro).

📖 **Read the docs:** <https://ighina.github.io/ruqola-server-deploy/>

## Quick commands (Mjölnir)

```bash
gpuq status                                  # GPUs, jobs, queue
gpuq submit -m 40 -t 8 -- python train.py    # 1 GPU with >= 40 GB free, 8 h limit
gpuq history                                 # how your recent jobs ended
gpuq quota                                   # your 7-day GPU-hours against the budget
gpuq kill --mine                             # stop and cancel all your jobs
nvidia-smi -l 1                              # live GPU usage
```

Run jobs inside `tmux`, from an activated venv. gpuq emails you when a job
ends. The [GPU Queue guide](docs/gpu-queue-guide.md) has the rules and options.

## How the site works

`index.html` renders the Markdown in this repo in the browser. There is no build
step; GitHub Pages serves the files as they are (`.nojekyll` turns off Jekyll).

- **Edit a page:** edit its Markdown under `docs/` (or `gpuq/README.md`,
  `examples/README.md`).
- **Add a page or change the tabs:** edit `MANIFEST` near the top of
  `assets/app.js`.
- **Preview:** browsers block `fetch()` from `file://`, so serve the folder:
  ```bash
  python3 -m http.server 8000   # then open http://localhost:8000
  ```

## Layout

```
index.html     the docs viewer
.nojekyll      serve raw .md files
assets/        app.js (tabs, router, rendering; MANIFEST), style.css, vendor/ (marked, highlight.js)
docs/          the documentation pages
gpuq/          the gpuq queue: userspace.py, installers, tests, README, sample config
examples/      runnable training examples and configs
scripts/       admin scripts: accounts, disk quotas, scratch cleanup (see scripts/README.md)
```

## Contributing

These pages describe what the live host does. When the host changes, update the
page in the same pull request. Edit the Markdown, preview it, and open a pull
request.

---

*Mjölnir, host `wsserver1`: 4× H200 NVL, 256 CPUs, 755 GiB RAM, Ubuntu 24.04.4,
driver 575.57.08, CUDA 12.9.*
