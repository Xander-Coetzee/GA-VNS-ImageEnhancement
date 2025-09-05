
# evaluation.py
import numpy as np
from skimage.metrics import structural_similarity as ssim

def calculate_mse(image_a, image_b):
    """Calculates the Mean Squared Error between two images."""
    err = np.sum((image_a.astype("float") - image_b.astype("float")) ** 2)
    err /= float(image_a.shape[0] * image_a.shape[1])
    return err

def calculate_psnr(image_a, image_b):
    """Calculates the Peak Signal-to-Noise Ratio between two images."""
    mse = calculate_mse(image_a, image_b)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def calculate_ssim(image_a, image_b):
    """Calculates the Structural Similarity Index between two images."""
    return ssim(image_a, image_b, data_range=image_a.max() - image_a.min())
