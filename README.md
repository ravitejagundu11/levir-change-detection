# Satellite Building Change Detection (LEVIR-CD)

CS57600 Machine Learning, Fall 2026 · Raviteja Gundu, Joel Adu-Kwarteng

Predicts pixel-level building-change masks from pairs of satellite images. The project compares an image-difference baseline, an early-fusion U-Net (U-Net-EF) and a Siamese U-Net, all evaluated with precision, recall, F1 and IoU on LEVIR-CD.

## How to run

The whole project is one notebook: [`levir_change_detection.ipynb`](levir_change_detection.ipynb).

1. Upload it to Google Colab and select a **T4 GPU** runtime.
2. Run all cells and allow Google Drive access when prompted.

Nothing needs installing. The notebook only uses packages Colab preinstalls (torch, numpy, pillow, matplotlib).

## Dataset

[LEVIR-CD](https://justchenhao.github.io/LEVIR/): 637 pairs of 1024×1024 Google Earth images at 0.5 m/pixel, taken 5 to 14 years apart, with binary building-change labels and the official 445/64/128 train/val/test split.

The notebook downloads it from the Kaggle mirror [`mdrifaturrahman33/levir-cd`](https://www.kaggle.com/datasets/mdrifaturrahman33/levir-cd). A copy placed in `data/raw` on Drive is used instead, if present.

LEVIR-CD is for academic use only, and its images are subject to the Google Earth terms of use. The images and tiles are kept on Drive and are not included in this repository.

### Citation

H. Chen and Z. Shi, "A Spatial-Temporal Attention-Based Method and a New Dataset for Remote Sensing Image Change Detection," *Remote Sensing*, vol. 12, no. 10, p. 1662, 2020. [doi:10.3390/rs12101662](https://doi.org/10.3390/rs12101662)

```bibtex
@Article{Chen2020,
  author  = {Chen, Hao and Shi, Zhenwei},
  title   = {A Spatial-Temporal Attention-Based Method and a New Dataset for Remote Sensing Image Change Detection},
  journal = {Remote Sensing},
  volume  = {12},
  year    = {2020},
  number  = {10},
  article-number = {1662},
  url     = {https://www.mdpi.com/2072-4292/12/10/1662},
  issn    = {2072-4292},
  doi     = {10.3390/rs12101662}
}
```

## Drive layout

Data, checkpoints and results live on Google Drive, so they survive between Colab sessions. The root is `/content/drive/MyDrive/levir-change-detection` (set in `CONFIG["project_root"]`).

```
data/raw/        LEVIR-CD as downloaded: train/val/test/{A,B,label}
data/tiles.zip   256x256 tiles
data/stats.json  training-set channel mean/std
runs/<name>/     checkpoints, CSV logs, saved config
results/         metrics JSON, per-image CSVs, figures
```
