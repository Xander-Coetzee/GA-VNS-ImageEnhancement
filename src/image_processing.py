
# image_processing.py
import cv2
import numpy as np

def gamma_correction(image, gamma=1.0):
    """Applies gamma correction to an image."""
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def gaussian_blur(image, kernel_size=(5, 5)):
    """Applies Gaussian blur to an image."""
    # kernel size must be odd
    ksize = (kernel_size[0] if kernel_size[0] % 2 == 1 else kernel_size[0] + 1, 
             kernel_size[1] if kernel_size[1] % 2 == 1 else kernel_size[1] + 1)
    return cv2.GaussianBlur(image, ksize, 0)

def unsharp_masking(image, kernel_size=(5, 5), strength=1.5):
    """Applies unsharp masking to an image."""
    gaussian = gaussian_blur(image, kernel_size)
    return cv2.addWeighted(image, 1.0 + strength, gaussian, -strength, 0)

def histogram_equalization(image):
    """Applies histogram equalization to an image."""
    return cv2.equalizeHist(image)

def contrast_stretching(image):
    """Applies contrast stretching to an image."""
    # Create a new image with float64 data type to avoid overflow
    stretched_image = image.astype(np.float64)

    # Get the minimum and maximum pixel values
    min_val = np.min(stretched_image)
    max_val = np.max(stretched_image)

    # Stretch the pixel values to the full 0-255 range
    stretched_image = 255 * (stretched_image - min_val) / (max_val - min_val)

    # Convert the image back to uint8
    return stretched_image.astype(np.uint8)
