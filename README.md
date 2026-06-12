# Ruqola Server Documentation

Single source of truth for the **Ruqola project** compute servers:

- **Mjölnir (NTU)** — a shared box with **4× NVIDIA H200 NVL** GPUs (~141 GB each,
  ~564 GB total), coordinated by the [`gpuq`](gpuq/README.md) cooperative queue.
- **The Hopper (NUS)** — NUS's multi-node H100/H200 HPC cluster (PBS Pro).

📖 **Browse the docs in your browser:** <https://ighina.github.io/ruqola-server-deploy/>

This repository merges what used to live in three places (the *Compute-New-Users*
onboarding notes, the *NTU-Server-Guide* docs, and this deploy repo) into one
self-contained, browsable site.

## How the site works

The site is a **single static `index.html`** that renders the Markdown files in
this repo into a tabbed, sidebar-navigated page **in the browser** — no build step,
no Jekyll. It works as plain static files on GitHub Pages (a `.nojekyll` marker
disables Jekyll so the raw `.md` is served and fetched client-side).

- **Editing docs:** just edit the Markdown under `docs/` (or `gpuq/README.md`,
  `examples/README.md`). The site picks up changes automatically — the Markdown
  *is* the source of truth.
- **Adding a page / changing tabs:** edit the `MANIFEST` near the top of
  `assets/app.js`.
- **Preview locally** (browsers block `fetch()` over `file://`, so serve it):
  ```bash
  python3 -m http.server 8000   # then open http://localhost:8000
  ```

## Repository layout

```
index.html            # the tabbed docs viewer
.nojekyll             # serve raw .md (no Jekyll processing)
assets/
  app.js              # tabs + sidebar + router + Markdown rendering (edit MANIFEST here)
  style.css           # site theme (light/dark)
  vendor/             # marked.js + highlight.js (vendored, no CDN dependency)
docs/                 # all documentation (Markdown — the source of truth)
gpuq/                 # the gpuq queue tool: userspace.py, installers, tests, README, sample config
examples/             # runnable training examples + configs
scripts/              # admin scripts (user creation/deletion, quotas, scratch cleanup)
```

## ⚡ Quick commands (Mjölnir)

```bash
gpuq status                                   # see GPU + queue state
gpuq submit -g 1 -t 8 -- python train.py      # run a job on 1 GPU, 8h limit
gpuq submit -g 1 --notify you@example.com -- python train.py   # + email on finish
gpuq history                                  # your recent jobs (runtime, exit code)
gpuq quota                                    # your 7-day GPU-hours vs budget
gpuq kill --mine                              # stop/cancel all your jobs
nvidia-smi -l 1                               # live GPU usage
```

See the [GPU Queue guide](docs/gpu-queue-guide.md) for the full model (you own the
GPUs allocated to you; pin specific cards with `--devices`, wait for them with
`--queue`).

## Contributing

Edit the relevant Markdown, preview locally, and open a pull request. Keep docs
factually aligned with the live server and with `gpuq/userspace.py`.

---

*Mjölnir hardware verified on host `wsserver1`: 4× H200 NVL, 256 CPUs, 755 GiB RAM,
Ubuntu 24.04.4, driver 575.57.08, CUDA 12.9.*
