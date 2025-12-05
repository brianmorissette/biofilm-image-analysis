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

def preprocess(
    image, 
    clip_limit=2.0, 
    tile_size=(8, 8), 
    blur_ksize=(5, 5)):
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    enhanced_image = clahe.apply(gray_image)
    normalized_image = cv2.normalize(
        src=enhanced_image, 
        dst=None, 
        alpha=0, 
        beta=255, 
        norm_type=cv2.NORM_MINMAX, 
        dtype=cv2.CV_8U)
    preprocessed_image = cv2.GaussianBlur(normalized_image, blur_ksize, 0)
    return gray_image, enhanced_image, normalized_image, preprocessed_image

def otsu_thresholding(image):
    optimal_threshold, otsu_mask = cv2.threshold(
        src=image, 
        thresh=0,           
        maxval=255,      
        type=cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    biofilm_pixels = np.count_nonzero(otsu_mask)
    total_pixels = otsu_mask.size
    percent_coverage = (biofilm_pixels / total_pixels) * 100
    return percent_coverage, otsu_mask, optimal_threshold

def iterative_threshold(image):
    iteration_count = 0
    current_threshold = 127.0
    last_threshold = -1.0
    tolerance = 0.5
    while abs(current_threshold - last_threshold) > tolerance:
        iteration_count += 1
        last_threshold = current_threshold
        background_pixels = image[image <= current_threshold]
        foreground_pixels = image[image > current_threshold]
        if background_pixels.size == 0:
            mean_bg = 0.0
        else:
            mean_bg = np.mean(background_pixels)
        if foreground_pixels.size == 0:
            mean_fg = 255.0
        else:
            mean_fg = np.mean(foreground_pixels)
        current_threshold = (mean_bg + mean_fg) / 2.0
    return int(round(current_threshold)), iteration_count

def biofilm_area(image, pixel_side_length_um=1.13):
    biofilm_pixels = np.count_nonzero(image)
    area_per_pixel_um2 = pixel_side_length_um ** 2
    total_area_um2 = biofilm_pixels * area_per_pixel_um2
    total_area_mm2 = total_area_um2 / 1_000_000
    
    return {
        'biofilm_pixel_count': biofilm_pixels,
        'area_per_pixel_um2': area_per_pixel_um2,
        'total_area_um2': total_area_um2,
        'total_area_mm2': total_area_mm2
    }

def analyze_image(image, image_id, pixel_side_length_um=1.13):
    gray, enhanced, normalized, blurred = preprocess(image)

    otsu_cov, otsu_mask, otsu_thresh = otsu_thresholding(blurred)

    iter_thresh, iter_iters = iterative_threshold(blurred)
    _, iterative_mask = cv2.threshold(blurred, iter_thresh, 255, cv2.THRESH_BINARY)

    otsu_area = biofilm_area(otsu_mask, pixel_side_length_um)
    iterative_area = biofilm_area(iterative_mask, pixel_side_length_um)

    return {
        "image_id": image_id,
        "otsu_threshold": otsu_thresh,
        "otsu_percent_coverage": otsu_cov,
        "otsu_pixel_count": otsu_area["biofilm_pixel_count"],
        "otsu_area_mm2": otsu_area["total_area_mm2"],
        "iter_threshold": iter_thresh,
        "iter_iterations": iter_iters,
        "iter_percent_coverage": np.count_nonzero(iterative_mask) / iterative_mask.size * 100,
        "iter_pixel_count": iterative_area["biofilm_pixel_count"],
        "iter_area_mm2": iterative_area["total_area_mm2"],
    }

    # Discrete Cosine Transform, you must normalize and grayscale the image before passing it to this function
def fft_dct(image):
    dct_image = scipy.fft.dctn(image, type=2, norm='ortho')
    return dct_image

#NOT THE FUNCTION TO CALL MEXICAN HAT, USE mexhat_transform
def mexican_hat_function(size, sigma):
    x = np.linspace(-size//2, size//2, size)
    y = np.linspace(-size//2, size//2, size)
    X, Y = np.meshgrid(x, y)
    r2 = X**2 + Y**2
    kernel = (1 - r2 / (2*sigma**2)) * np.exp(-r2 / (2*sigma**2))
    return kernel / (kernel.sum() if kernel.sum() != 0 else 1.0)

# Function to apply Mexican Hat transform to an image
def mexhat_transform(image):
    size = 21
    sigma = 12
    kernel = mexican_hat_function(size, sigma)
    transformed = scipy.ndimage.convolve(image, kernel, mode='reflect')
    return transformed

# Function to apply Gaussian transform to an image
def gaussian_transform(image):
    sigma = 12
    gaussian_blur = scipy.ndimage.gaussian_filter(image, sigma=sigma)
    return scipy.ndimage.gaussian_laplace(gaussian_blur, sigma=sigma)

# Function to apply FFT transform to an image
def fft_transform(image):
    fft_image = scipy.fft.fft2(image)
    mag = np.abs(scipy.fft.fftshift(fft_image))
    mag = np.log1p(mag)
    mag = (mag - mag.min()) / (mag.max() - mag.min())      
    return mag