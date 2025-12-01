import cv2
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import pywt
import scipy.fft
import pandas as pd
from pathlib import Path


def load_images(root) -> list[np.ndarray]:
    # collect all .tif files under `root`, recursively, then sort for stable order
    paths = sorted(
        [*Path(root).rglob("*.tif")],
        key=lambda p: p.as_posix().casefold()
    )
    # read each file with cv2 (preserve bit depth/channels) and keep only successful reads
    return [img for p in paths if (img := cv2.imread(str(p), cv2.IMREAD_UNCHANGED)) is not None]

def grayscale(image) -> np.ndarray:
    return image[:,:,1]

def normalize(image) -> np.ndarray:
    return image / np.max(image)

def extract_patches(image, patch_size):
    h, w = image.shape
    patches = []
    for i in range(h // patch_size):
        for j in range(w // patch_size):
            patches.append(image[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size])
    return patches

def display_image(image) -> None:
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def display_grid_of_images(images) -> None:
    grid_size = int(np.ceil(np.sqrt(len(images))))
    plt.figure(figsize=(12, 12))
    for i in range(len(images)):
        plt.subplot(grid_size, grid_size, i + 1)
        plt.imshow(images[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Discrete Cosine Transform, you must normalize and grayscale the image before passing it to this function
def fft_dct(image):
    dct_image = scipy.fft.dctn(image, type=2, norm='ortho')
    return dct_image

#NOT THE FUNCTION TO CALL MEXICAN HAT, USE mexhat_transform
def mexican_hat_function(size=21, sigma=3.0):
    x = np.linspace(-size//2, size//2, size)
    y = np.linspace(-size//2, size//2, size)
    X, Y = np.meshgrid(x, y)
    r2 = X**2 + Y**2
    kernel = (1 - r2 / (2*sigma**2)) * np.exp(-r2 / (2*sigma**2))
    return kernel / (kernel.sum() if kernel.sum() != 0 else 1.0)

# Function to apply Mexican Hat transform to an image
def mexhat_transform(image):
    size = 21
    sigma = 3.0
    kernel = mexican_hat_function(size, sigma)
    transformed = scipy.ndimage.convolve(image, kernel, mode='reflect')
    return transformed

def gaussian_transform(image):
    sigma = 2.0
    gaussian_blur = scipy.ndimage.gaussian_filter(image, sigma=sigma)
    return scipy.ndimage.gaussian_laplace(gaussian_blur, sigma=sigma)

def fft_transform(image):
    fft_image = scipy.fft.fft2(image)
    mag = np.abs(scipy.fft.fftshift(fft_image))
    mag = np.log1p(mag)
    mag = (mag - mag.min()) / (mag.max() - mag.min())      
    return mag