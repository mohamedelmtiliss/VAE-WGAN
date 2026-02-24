# Forest Fire Detection using VAE-WGAN & Latent Space Classification

## 📌 Project Overview

This project implements an **Anomaly Detection System** for forest fire detection using satellite imagery (MODIS dataset). It leverages a **VAE-WGAN (Variational Autoencoder + Wasserstein GAN)** architecture to learn the distribution of normal forest landscapes. Anomalies (fires) are detected by analyzing reconstruction errors, discriminator scores, or by using a secondary **SVM Classifier** trained on the VAE's latent space features.

## 🧠 Model Architecture

The core model combines generative and adversarial networks to ensure high-quality feature extraction and robust anomaly detection.

### 1. VAE-WGAN (Main Model)

* **Encoder:** Compresses 64x64 RGB images into a 128-dimensional latent vector $(z)$. Uses Convolutional layers with Batch Normalization and LeakyReLU.
* **Decoder (Generator):** Reconstructs images from the latent vector. Uses Transposed Convolutions with Tanh activation (output range [-1, 1]).
* **Discriminator (Critic):** Distinguishes between real input images and VAE reconstructions. Uses Instance Normalization for WGAN stability.
* **Loss Function:** Combines Reconstruction Loss (MSE), KL Divergence, and Adversarial Loss (Wasserstein distance with Gradient Penalty).

### 2. Latent Classifier

* **SVM (Support Vector Machine):** A secondary classifier trained on the latent vectors $(z)$ extracted by the VAE Encoder.
* **Kernel:** RBF (Radial Basis Function).
* **Purpose:** Explicitly classifies images as "Normal" or "Fire" based on the compressed feature representation.

## 📂 Dataset Structure

The project expects a dataset folder (e.g., `modis_dataset_brazil`) organized as follows:

```
modis_dataset_brazil/
├── normal_reference/    # Images of normal forests (used for training VAE)
└── fire_anomalies/      # Images of forest fires (used for testing/evaluation)

```

* **Input Size:** Images are resized to **64x64**.
* **Normalization:** Pixel values are normalized to [-1, 1].

## 🛠️ Installation & Requirements

Ensure you have the following Python libraries installed:

```bash
pip install torch torchvision numpy pandas scikit-learn matplotlib seaborn tqdm pillow

```

## 🚀 Usage Guide

### 1. Train the VAE-WGAN

Train the unsupervised model on normal data to learn the baseline distribution.

* **File:** `train.ipynb`
* **Output:** Saves model checkpoints to `checkpoints/` and the final model as `vae_wgan_final.pth`.

### 2. Train the Latent Classifier

Train the SVM classifier using features extracted from the frozen VAE-WGAN.

* **File:** `train_latent_classifier_svm.ipynb`
* **Process:** Extracts latent vectors $(z)$ from the test set and trains an SVM.
* **Output:** Saves the classifier as `final_svm_model.pkl` and plots the confusion matrix.

### 3. Evaluation & Demo

Analyze the performance using various metrics (MSE, Critic Score, Feature Matching).

* **File:** `evaluate_advanced.ipynb` or `evaluate.ipynb`
* **Metrics:** ROC Curves, AUC scores, and histograms comparing Normal vs. Fire distributions.
* **Demo:** Run `demo_fire_detection.ipynb` to visualize predictions on specific images.

## 📊 Results & Performance

The system uses multiple scoring mechanisms to detect fires:

* **Reconstruction Error (MSE):** High error indicates an anomaly (fire).
* **Critic Score:** Low "realism" score indicates an anomaly.
* **Latent SVM:** Direct binary classification (Normal=0, Fire=1).

*Example outputs located in `results/` include:*

* `confusion_matrix_svm.png`: Visualization of SVM accuracy.
* `advanced_roc_comparison.png`: ROC curves comparing different anomaly scoring methods.
* `visual_examples.png`: Side-by-side comparison of Input vs. Reconstructed images.

## 📁 File Structure

* `model.py`: PyTorch definitions for Encoder, Decoder, Discriminator, and VAE_WGAN class.
* `dataset.py`: Custom Dataset class and DataLoader setup for handling image folders.
* `train.ipynb`: Main training loop for the VAE-WGAN.
* `train_latent_classifier_svm.ipynb`: Extraction of latent features and training of the SVM.
* `evaluate_advanced.ipynb`: Advanced metrics calculation and ROC plotting.
* `demo_fire_detection.ipynb`: Inference script for demonstrations.