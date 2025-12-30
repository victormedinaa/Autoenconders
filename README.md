# Hierarchical Vector Quantized Variational Autoencoder (VQ-VAE)

**Status**: Active Research  
**Author**: Antigravity (Assistant) for User  
**Topic**: Generative Models, Representation Learning

---

## 1. Abstract

This repository contains a PyTorch implementation of a **Vector Quantized Variational Autoencoder (VQ-VAE)**, a powerful generative model that combines the strengths of autoencoders with the ability to learn discrete latent representations. Unlike traditional Variational Autoencoders (VAEs), which suffer from "posterior collapse" and blurry reconstructions due to continuous Gaussian priors, VQ-VAE learns a discrete codebook of latent embeddings. This allows for high-fidelity image reconstruction and subsequent high-quality generation when paired with an autoregressive prior (e.g., PixelCNN or Transformer) over the discrete latents.

This implementation focuses on modularity and reproducibility, targeting the **CIFAR-10** dataset (scalable to ImageNet) and features Deep Residual Convolutional Networks for both the Encoder and Decoder.

## 2. Methodology

### 2.1. Vector Quantized Variational Autoencoder

The VQ-VAE consists of three main components:
1.  **Encoder** $E(x)$: Maps the input image $x$ to a continuous latent variable $z_e(x)$.
2.  **Vector Quantizer**: Maps $z_e(x)$ to the nearest embedding vector $e_k$ from a discrete codebook $E = \{e_1, ..., e_K\}$.
3.  **Decoder** $D(z_q)$: Reconstructs the image $\hat{x}$ from the quantized latent $z_q(x)$.

#### 2.2. Loss Function

The objective function optimizes three terms:

$$ L = \underbrace{|| x - D(z_q(x)) ||_2^2}_{\text{Reconstruction Loss}} + \underbrace{||sg[z_e(x)] - e ||_2^2}_{\text{Codebook Loss}} + \beta \underbrace{||z_e(x) - sg[e]||_2^2}_{\text{Commitment Loss}} $$

Where:
- $sg[\cdot]$ denotes the stop-gradient operator.
- The **Reconstruction Loss** ensures high fidelity given the quantized latent.
- The **Codebook Loss** updates the dictionary vectors to move towards the encoder outputs.
- The **Commitment Loss** ensures the encoder outputs stay close to the chosen codebook vectors, controlled by $\beta$.

Gradient backpropagation through the non-differentiable quantization step is achieved using the **Straight-Through Estimator (STE)**, where $\nabla_z L = \nabla_{z_q} L$.

### 3. Architecture Details

- **Residual Blocks**: Used in both Encoder and Decoder to allow deep architectures without vanishing gradients.
- **Strided Convolutions**: Downsampling in Encoder (spatial compression not pooling).
- **Transposed Convolutions**: Upsampling in Decoder.

## 4. Project Structure

The project is structured as a compliant Python package:

```text
.
├── configs/            # Configuration files
├── src/
│   ├── dataset.py      # CIFAR-10 DataModule with normalization
│   ├── model.py        # VQ-VAE Implementation (Encoder, Decoder, VQ)
│   ├── trainer.py      # Training loop implementation
│   └── utils.py        # Visualization utilities
├── train.py            # Main entry point
├── requirements.txt    # Vulnerability-free dependencies
└── README.md           # Research documentation
```

## 5. Getting Started

### Prerequisites

Clone the repository and set up a virtual environment (recommended):

```bash
# Create a virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # On macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Training

To train the model on CIFAR-10 with default hyperparameters:

```bash
# Using the venv python directly
./venv/bin/python train.py --epochs 50 --batch_size 128
```

### Results

The training script will output reconstruction grids to the `results/` directory every epoch.
- `epoch_N_orig.png`: Batch of original images.
- `epoch_N_recon.png`: Corresponding reconstructions.

## 6. Future Work / Extensions

- **PixelCNN Prior**: Train an autoregressive model on the learned discrete latents $z_q$ to sample new unique images.
- **Hierarchical VQ-VAE-2**: Implement multi-scale latents for higher resolution (e.g., 256x256 or 1024x1024).

## 7. References

- Van den Oord, A., et al. "Neural Discrete Representation Learning." *Advances in Neural Information Processing Systems*, 2017.
- Razavi, A., et al. "Generating Diverse High-Fidelity Images with VQ-VAE-2." *NeurIPS*, 2019.
