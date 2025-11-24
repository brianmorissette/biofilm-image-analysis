import cv2
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
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


def fft_dct(image):
    dct_image = scipy.fft.dct(image)
    return dct_image