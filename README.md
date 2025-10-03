# Super-Resolution (Super-Res)

This repository contains multiple from‑scratch implementations of **state-of-the-art image super-resolution models** in PyTorch. It started as an ESRGAN reimplementation and has grown into a modular playground for experimenting with **GAN-based**, **diffusion-based**, and **transformer-based** approaches to super-resolution.

The repo includes tools for data preparation, training, experiment tracking (Weights & Biases), and checkpoints for reproducible experiments.

---

## Features

- **ESRGAN Implementation**
  - Generator with Residual-in-Residual Dense Blocks (RRDBs)
  - VGG-style discriminator with spectral normalization
  - Warmup (pretraining) + adversarial training phases
  - Checkpointing and resume support

- **Diffusion-Based Super-Resolution**
  - Custom diffusion model and training loop
  - Configurable noise scheduler

- **Transformer-Based Super-Resolution (SwinIR)**
  - Integration placeholder for SwinIR-style transformer backbone (in progress)

- **Dataset & Utilities**
  - Patch extraction (`create_patches.py`) for DIV2K / Flickr2K
  - `gather_photos.py` for pulling images from Google Drive / Google Photos
  - Scripts for working with personal photo libraries (Apple Photos support)

- **Training & Experiment Tracking**
  - Integrated with Weights & Biases (`wandb`)
  - Automatic checkpoint saving, optimizer state tracking

---

## Quick Start

1. Create and activate the conda environment:

```bash
conda create --name super-res python=3.10 -y
conda activate super-res
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Prepare datasets (see **Data Preparation** below), then run training for a chosen model:

- ESRGAN:

```bash
wandb login
python3 esrgan/train_esrgan.py
```

- Diffusion SR:

```bash
python3 diffusion/train_diffusion.py
```

---

## Repository Structure

```
super-res/
├── create_patches.py              # Process HR images into LR-HR patch datasets
├── gather_photos.py               # Google Drive/Photos integration for dataset collection
├── requirements.txt               # Python dependencies
├── setup_env.sh                   # Script for setting up environment
├── super-res-doc.txt              # Notes and design documentation
├── README.md                      # This file

├── div2K-flickr2K-data/           # Local dataset storage
│   ├── train/
│   ├── val/
│   └── test/

├── esrgan/                        # ESRGAN implementation
│   ├── generator.py
│   ├── discriminator.py
│   ├── train_esrgan.py
│   ├── generator_pretrained.pth   # Example pretrained weights (optional)
│   ├── generator_epoch_XX.pth     # Generator checkpoints
│   ├── discriminator_epoch_XX.pth # Discriminator checkpoints
│   └── optimizer_*_epoch_XX.pth   # Optimizer states

├── diffusion/                     # Diffusion-based SR
│   ├── diffusion_model.py
│   ├── scheduler.py
│   └── train_diffusion.py

├── swin_ir/                       # Transformer-based SR (in progress)

├── legacy/                        # Archived / older code
│   ├── train.py
│   ├── train_orig.py
│   ├── discriminator.py
│   ├── generator.py
│   ├── cleanup_photos_dataset.py
│   └── prepare_dataset.py

└── wandb/                         # Experiment tracking logs
```

---

## Data Preparation

This repo is set up for the **DIV2K** and **Flickr2K** datasets (common SR benchmarks).

1. Download datasets:
   - **DIV2K:** `DIV2K_train_HR`, `DIV2K_valid_HR` from the DIV2K website.
   - **Flickr2K:** `Flickr2K_HR` from the Flickr2K release.

2. Edit `create_patches.py` and set the `TRAIN_VAL_SOURCES` / `TEST_SOURCES` to point to the downloaded folders.

3. Run the patch extraction script (CPU / IO intensive):

```bash
python3 create_patches.py
```

- The script will create a structured folder of LR/HR patches suitable for training.

---

## Training Details

### ESRGAN

Run:

```bash
wandb login
python3 esrgan/train_esrgan.py
```

Key behaviors:
- If `generator_pretrained.pth` is not present, the script runs a warmup pretraining phase.
- Checkpointing happens periodically (check the `esrgan/` folder for saved `.pth` files).
- Resume training using the `--resume_epoch` flag (example: `--resume_epoch 20`).

### Diffusion-Based SR

Run:

```bash
python3 diffusion/train_diffusion.py
```

This trains a denoising diffusion model using the scheduler in `diffusion/scheduler.py`.

### SwinIR

SwinIR code and integration are currently under development. The `swin_ir/` directory is a placeholder for the transformer-based backbone and training utilities.

---

## Using Personal Photos

- `gather_photos.py` can pull or organize downloads from Google Drive / Google Photos exports.
- Apple Photos export workflows are supported (via an AppleScript for batch export) — see internal notes in `super-res-doc.txt`.
- After exporting your photos, add the folder path to `create_patches.py` sources and re-run it to generate training patches.

---

## Roadmap

- ✅ ESRGAN: end-to-end training pipeline
- ✅ Diffusion SR: prototype implementation
- 🛠️ SwinIR: integration + training scripts
- 🛠️ Unified CLI/config: choose model family via a single entrypoint
- 🛠️ Inference scripts: batch upscaling for personal photos

---

## References

- ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks — https://arxiv.org/abs/1809.00219
- Real-ESRGAN — https://arxiv.org/abs/2107.10833
- SwinIR: Image Restoration Using Swin Transformer — https://arxiv.org/abs/2108.10257
- Denoising Diffusion Probabilistic Models — https://arxiv.org/abs/2006.11239

---


### Model Architecture

* **Generator (`generator.py`):** A deep network of 23 Residual-in-Residual Dense Blocks (RRDB) with learnable upsampling via `PixelShuffle`. This architecture allows for the extraction of incredibly detailed and hierarchical features.
* **Discriminator (`discriminator.py`):** A deep, VGG-style network that acts as a patch-based classifier to determine if an image is real or generated. Spectral Normalization is used on all convolutional layers to enforce the Lipschitz constraint and stabilize GAN training dynamics.
