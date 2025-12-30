import torch
import matplotlib.pyplot as plt
import numpy as np
import os

def show_img_grid(images, title="Images", save_path=None):
    """
    Plot a grid of images (Tensor batch).
    images: (B, C, H, W), normalized [-1, 1]
    """
    images = images.cpu().detach()
    # Denormalize
    images = (images * 0.5) + 0.5
    images = torch.clamp(images, 0, 1)
    
    grid_img = np.transpose(images.numpy(), (0, 2, 3, 1))
    
    # Take first 8 images
    if grid_img.shape[0] > 8:
        grid_img = grid_img[:8]
    
    fig, axes = plt.subplots(1, len(grid_img), figsize=(12, 4))
    fig.suptitle(title)
    
    for i, ax in enumerate(axes):
        ax.imshow(grid_img[i])
        ax.axis('off')
        
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
