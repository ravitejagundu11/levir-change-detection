# Satellite Building Change Detection (LEVIR-CD)

CS57600 Machine Learning, Fall 2026 · Raviteja Gundu, Joel Adu-Kwarteng

Predicts pixel-level building-change masks from pairs of satellite images. The project compares an image-difference baseline, an early-fusion U-Net (U-Net-EF) and a Siamese U-Net, all evaluated with precision, recall, F1 and IoU on LEVIR-CD.

## How to run

The whole project is one notebook: [`levir_change_detection.ipynb`](levir_change_detection.ipynb).

1. Upload it to Google Colab and select a **T4 GPU** runtime.
2. Run all cells and allow Google Drive access when prompted.

Nothing needs installing. The notebook only uses packages Colab preinstalls (torch, numpy, pillow, matplotlib).

## Drive layout

Data, checkpoints and results live on Google Drive, so they survive between Colab sessions. The root is `/content/drive/MyDrive/levir-change-detection` (set in `CONFIG["project_root"]`).

```
data/raw/        LEVIR-CD as downloaded: train/val/test/{A,B,label}
data/tiles.zip   256x256 tiles
data/stats.json  training-set channel mean/std
runs/<name>/     checkpoints, CSV logs, saved config
results/         metrics JSON, per-image CSVs, figures
```
