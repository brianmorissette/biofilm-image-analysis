import cv2
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
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

def extract_patches(images, patch_size):
    patches = []
    for image in images:
        h, w = image.shape
        for i in range(h // patch_size):
            for j in range(w // patch_size):
                patch = image[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
                patches.append(patch)
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