# Satellite Building Change Detection (LEVIR-CD)

CS57600 Machine Learning, Fall 2026 · Raviteja Gundu, Joel Adu-Kwarteng

Predicts pixel-level building-change masks from pairs of satellite images. The project compares an image-difference baseline, an early-fusion U-Net (U-Net-EF) and a Siamese U-Net, all evaluated with precision, recall, F1 and IoU on LEVIR-CD.

The full task list is in [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md).

## Where code runs

All code runs on **Google Colab** (T4 GPU). Data, checkpoints and results live on Google Drive. The repo only holds code, configs and notebooks.

## Quick start (Colab)

1. Push this repo to GitHub.
2. Open `notebooks/00_setup.ipynb` in Colab, select a T4 runtime and set `REPO_URL`.
3. Run all cells. The notebook mounts Drive, clones the repo, installs it, prints the GPU, creates the Drive folders and runs `pytest`.

## Drive layout

The project root defaults to `/content/drive/MyDrive/cs576-change-detection` and can be changed with the `CD_PROJECT_ROOT` environment variable.

```
<root>/data/raw          LEVIR-CD as downloaded: train/val/test/{A,B,label}
<root>/data/tiles        256x256 tiles (Task 3)
<root>/data/stats.json   training-set channel mean/std (Task 4)
<root>/runs/<name>       checkpoints, CSV logs, saved config
<root>/results           metrics JSON and figures
```

## Repository layout

```
configs/     YAML configs per method
src/cd/      package code (utils, data, metrics, baseline, models, training, evaluation)
scripts/     command-line entry points
notebooks/   Colab notebooks 00–05
tests/       pytest unit tests on small synthetic data
```

## Dependencies

torch and torchvision are the versions Colab preinstalls. Other packages are pinned in `requirements.txt`. The setup notebook installs those, then installs this package with `--no-deps` so Colab's CUDA build of torch is never replaced.
