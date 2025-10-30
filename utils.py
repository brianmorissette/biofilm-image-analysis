import cv2
import glob
import os
import matplotlib.pyplot as plt
import numpy as np


def load_images(path):
    image_files = [os.path.join(path, filename) for filename in os.listdir(path)]
    return [cv2.imread(file, cv2.IMREAD_UNCHANGED) for file in image_files]

def grayscale(image):
    return image[:,:,1]

def normalize(image):
    return image / np.max(image)

def display_image(image):
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def display_grid(images):
    grid_size = int(np.ceil(np.sqrt(len(images))))
    plt.figure(figsize=(12, 12))
    for i in range(len(images)):
        plt.subplot(grid_size, grid_size, i + 1)
        plt.imshow(images[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

biofilm_images = [grayscale(image) for image in load_images('biofilm_data/biofilm')]
release_cells_images = [grayscale(image) for image in load_images('biofilm_data/release_cells')]

print(f"Loaded {len(biofilm_images)} biofilm images from 'biofilm_data/biofilm'")
print(f"Loaded {len(release_cells_images)} release cells images from 'biofilm_data/release_cells'")
print(f"Biofilm image shape: {biofilm_images[0].shape}")
print(f"Release cells image shape: {release_cells_images[0].shape}")
display_image(biofilm_images[0])
display_image(release_cells_images[0])
display_grid(biofilm_images)
display_grid(release_cells_images)

