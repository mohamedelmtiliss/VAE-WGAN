# Forest Fire Detection via Anomaly Detection — VAE-WGAN

This repository contains two independent but complementary implementations of an unsupervised anomaly detection system for forest fire identification using satellite imagery. Both projects are built around a hybrid **Variational Autoencoder + Wasserstein GAN (VAE-WGAN)** architecture. The core principle is identical: train the model exclusively on images of healthy forests so that it learns their statistical distribution, then flag any region that it cannot faithfully reconstruct as an anomaly (active fire or burn scar).

The two sub-projects differ primarily in their data format, number of spectral channels, and scope of the pipeline.

---

## Table of Contents

1. [Project 1 — VAE-WGAN with .NPY Dataset RGB + SWIR (Multispectral)](#project-1--vae-wgan-with-npy-dataset-multispectral)
2. [Project 2 — VAE-WGAN with PNG Dataset](#project-2--vae-wgan-with-png-dataset-rgb)
3. [Theoretical Background](#theoretical-background)
4. [Requirements](#requirements)

---

## Project 1 — VAE-WGAN with .NPY Dataset RGB + SWIR (Multispectral)

Located in: `vae-wgan (Using .NPY dataset)/`

### Overview

This is the primary and more advanced implementation. It operates on **4-channel Sentinel-2 patches** (Red, Green, Blue, SWIR — Short-Wave Infrared) stored as raw NumPy tensors in `.npy` format. The inclusion of the SWIR band is the critical design choice of this project: active fires and fresh burn scars exhibit a dramatically elevated response in the SWIR spectrum, making anomalies far more detectable than with visible bands alone.

The dataset was constructed using a hybrid acquisition strategy coupling **NASA FIRMS (MODIS/VIIRS)** fire alerts with **Sentinel-2 (ESA)** high-resolution imagery, retrieved via Google Earth Engine. Only fire alerts with a confidence index above 90 % were used to build the test set, and only Sentinel-2 tiles with less than 5 % cloud coverage were retained for training.

### Dataset Statistics

| Split | Contents | Count |
|---|---|---|
| `train/` | Clean forest patches (multi-region) | 1116 patches |
| `val_clean/` | Clean patches not seen during training | 279 patches |
| `test_fire/` | Confirmed fire patches (FIRMS-verified) | 565 patches |

Geographic coverage spans six distinct biomes: **Australia, Brazil, Canada, Morocco, Russia, and Spain**, ensuring inter-regional generalization.

### Dataset Structure

```
vae-wgan (Using .NPY dataset)/
    dataset/
        MODIS/
            modis_2024_Australia.csv     # FIRMS fire alert coordinates
            modis_2024_Brazil.csv
            modis_2024_Canada.csv
            modis_2024_Morocco.csv
            modis_2024_Spain.csv
        train/                           # 1116 clean forest patches (.npy)
        val_clean/                       # 279 clean validation patches (.npy)
        test_fire/                       # 565 confirmed fire patches (.npy)
```

Each `.npy` file contains a tensor of shape `(64, 64, 4)` representing a 64x64 pixel patch with four spectral bands. Raw Sentinel-2 reflectance values (range 0–10000) are normalized to `[0, 1]` by dividing by 10000 before entering the network.

### Notebook Structure

| Notebook | Purpose |
|---|---|
| `prepating_dataset.ipynb` | Automated data acquisition pipeline using Google Earth Engine: queries MODIS fire alerts, fetches corresponding Sentinel-2 tiles, and exports 4-channel `.npy` patches. |
| `vae-wgan-for-fire-detection.ipynb` | Baseline VAE-WGAN implementation with weight-clipping Lipschitz enforcement (original Arjovsky et al. approach). Includes training loop and anomaly scoring via reconstruction error. |
| `vae-wgan-gp-for-fire-detection.ipynb` | Improved VAE-WGAN-GP implementation using Gradient Penalty for stable Lipschitz enforcement. This is the recommended and final model. Includes full training loop, validation, and evaluation cells. |
| `visualization.ipynb` | Utility notebook to visually inspect patches. Displays the RGB composite and the SWIR heatmap side-by-side for any given `.npy` file. |

### Model Architecture

The model consists of three networks:

**Encoder**
- Input: `(Batch, 4, 64, 64)` — 4-channel tensor normalized to `[0, 1]`
- 3 convolutional blocks (Conv2d + BatchNorm2d + LeakyReLU), progressively halving spatial dimensions
- Final feature map flattened and projected to `mu` and `log_var` vectors in `R^128`
- Latent sample: `z = mu + eps * exp(0.5 * log_var)`, with `eps ~ N(0, I)`

**Generator (Decoder)**
- Input: `z` in `R^128`
- Fully connected projection followed by 3 transposed convolutional blocks (ConvTranspose2d + BatchNorm2d + ReLU)
- Output activation: **Sigmoid** to constrain reconstructed values to `[0, 1]`
- Output shape: `(Batch, 4, 64, 64)`

**Discriminator (Critic)**
- Input: either a real patch or a VAE reconstruction, shape `(Batch, 4, 64, 64)`
- 3 convolutional layers with LeakyReLU
- Output: unbounded scalar Wasserstein score

### Anomaly Detection at Inference

Given a test patch $x$, the anomaly score is the per-pixel reconstruction error.

Patches with a score above a calibrated threshold are classified as fire anomalies. The SWIR channel contributes disproportionately to this score when an active fire or burn scar is present, naturally amplifying the signal without any supervision.

### Usage

1. **Prepare the dataset**: Run `prepating_dataset.ipynb` with a valid Google Earth Engine account to download and export patches. Alternatively, point the dataset loader directly to a pre-existing folder of `.npy` files.

2. **Train (WGAN-GP)**: Open `vae-wgan-gp-for-fire-detection.ipynb`. Update the three dataset paths at the top of the *Dataset Loader* cell, then run all cells sequentially. The notebook trains for a configurable number of epochs and saves checkpoints.

3. **Visualize patches**: Use `visualization.ipynb` to inspect any `.npy` file as an RGB+SWIR side-by-side figure.

---

## Project 2 — VAE-WGAN with PNG Dataset

Located in: `VAE-WGAN (Using PNG dataset )/`

### Overview

This implementation operates on standard **3-channel RGB images** in PNG format. It serves as a baseline and as a demonstrator of the broader pipeline, which extends beyond model training to include a secondary **SVM classifier** trained on the VAE's latent space. This project is the one for which a trained model (`vae_wgan_final.pth`) and all training checkpoints are already provided and ready for inference.

The dataset is structured around the **MODIS Brazil 2024** fire dataset (`modis_2024_Brazil.csv`), making this a more regionally focused experiment.

### Project Structure

```
VAE-WGAN (Using PNG dataset )/
    model.py                         # PyTorch class definitions: Encoder, Decoder, Discriminator, VAE_WGAN
    dataset.py                       # Custom Dataset class and DataLoader factory
    train.ipynb                      # Main VAE-WGAN training loop
    train_latent_classifier_svm.ipynb # SVM training on extracted latent vectors
    evaluate.ipynb                   # Basic evaluation: reconstruction error histograms
    evaluate_advanced.ipynb          # Full evaluation: ROC curves, AUC, feature matching
    evaluate_latent.ipynb            # Latent space analysis and visualization
    demo_fire_detection.ipynb        # Inference demo on individual images
    check_data.ipynb                 # Dataset integrity and distribution check
    download_data.ipynb              # Dataset download helper
    modis_2024_Brazil.csv            # FIRMS fire alert source data for Brazil
    vae_wgan_final.pth               # Final trained model weights (ready for inference)
    checkpoints/                     # Epoch-level checkpoints (10, 20, ..., 100)
    results/                         # Output figures from evaluate.ipynb
    results_classifier/              # Output figures from SVM evaluation
    results_latent/                  # Output figures from latent space analysis
```

### Model Architecture

The architecture mirrors Project 1 with the key difference that there are **3 input/output channels** (RGB) instead of 4.

**Encoder**
- Input: `(Batch, 3, 64, 64)` — normalized to `[-1, 1]`
- 4 convolutional blocks (Conv2d + BatchNorm2d + LeakyReLU), outputting a 256-channel feature map of spatial size 4x4
- Projected to `mu` and `log_var` in `R^128`

**Decoder (Generator)**
- Input: `z` in `R^128`
- Fully connected layer + 4 transposed convolutional blocks
- Output activation: **Tanh**, output range `[-1, 1]`

**Discriminator (Critic)**
- Uses **InstanceNorm2d** (instead of BatchNorm2d) for WGAN training stability

### Extended Pipeline — Latent Space SVM Classifier

A distinguishing feature of this project is a supervised second stage built on top of the unsupervised VAE-WGAN:

1. The trained VAE encoder is frozen.
2. Latent vectors `z = E(x)` are extracted for all test images (both clean and fire).
3. An **RBF-kernel SVM** is trained on these 128-dimensional vectors with binary labels (Normal = 0, Fire = 1).

This produces an explicit classifier that leverages the geometric structure of the learned latent space. Results are saved to `results_classifier/` as confusion matrices and ROC comparisons.

### Provided Evaluation Notebooks

| Notebook | Output |
|---|---|
| `evaluate.ipynb` | Reconstruction error histograms for Normal vs. Fire distributions |
| `evaluate_advanced.ipynb` | ROC curves, AUC scores, multi-method comparison (MSE vs. Critic Score vs. SVM) |
| `evaluate_latent.ipynb` | t-SNE or PCA projection of latent vectors, cluster separation analysis |
| `demo_fire_detection.ipynb` | Side-by-side visualization of input, reconstruction, and error map for selected images |

### Quick Start (Inference Only)

A pre-trained model is available at `vae_wgan_final.pth`. To run inference directly without retraining:

1. Open `demo_fire_detection.ipynb`.
2. Set the path to the image you want to analyze.
3. Run all cells. The notebook loads `vae_wgan_final.pth`, encodes the image, reconstructs it, and displays the residual error map.

### Training from Scratch

1. Prepare the dataset using `download_data.ipynb` or organize images manually into the expected folder structure:
   ```
   modis_dataset_brazil/
       normal_reference/    # PNG images of healthy forests (training)
       fire_anomalies/      # PNG images of fire regions (testing only)
   ```
2. Verify data integrity with `check_data.ipynb`.
3. Run `train.ipynb`. Checkpoints are saved every 10 epochs to `checkpoints/`.
4. (Optional) Run `train_latent_classifier_svm.ipynb` to train the SVM on latent features.
5. Evaluate with `evaluate_advanced.ipynb`.

---

## Theoretical Background

Both projects implement the same underlying hybrid generative model. The key design choices are:

- **VAE component**: Provides a deterministic, differentiable encoder $E: x \mapsto (\mu, \sigma)$, which is essential for projecting test images into the latent space at inference time. The KL divergence term ensures the latent space is compact and well-structured, maximizing the likelihood that out-of-distribution inputs (fires) fall in low-density regions.

- **WGAN-GP component**: Replaces the blurry MSE-only reconstruction objective with an adversarial loss that drives the generator to produce sharp, realistic textures. The Gradient Penalty replaces weight clipping, ensuring stable training and a meaningful Wasserstein distance estimate throughout the optimization.

- **Anomaly detection mechanism**: Because the model is trained exclusively on normal data, it learns to faithfully reconstruct healthy forest textures. When a fire patch is encoded and decoded, the generator projects it onto the closest point of the learned healthy-forest manifold, producing a structurally clean output. The high residual error between the fire input and the clean reconstruction is the anomaly signal.

For a full mathematical derivation and experimental analysis, refer to `report wgan.tex` at the repository root.

---

## Requirements

Both projects share the same Python dependencies.

```bash
pip install torch torchvision numpy pandas scikit-learn matplotlib seaborn tqdm pillow scikit-image earthengine-api
```

| Dependency | Purpose |
|---|---|
| `torch`, `torchvision` | Model definition, training, and inference |
| `numpy` | `.npy` data loading and numerical operations |
| `pillow`, `scikit-image` | PNG image loading and preprocessing |
| `scikit-learn` | SVM classifier, ROC/AUC metrics |
| `matplotlib`, `seaborn` | Visualization and plotting |
| `earthengine-api` | Google Earth Engine data acquisition (Project 1 only) |
| `tqdm` | Training progress bars |

A CUDA-capable GPU is strongly recommended for training. Inference on CPU is feasible.
