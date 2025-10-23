#!/usr/bin/env python3
"""
Image Processing Module for Biofilm Image Analysis Pipeline

This module handles image visualization and grayscale conversion.
"""

import os
import matplotlib.pyplot as plt


def display_example_images(biofilm_raw_images, release_cell_raw_images):
    """Display example raw images."""
    print("Displaying example images...")
    
    plt.figure(figsize=(12,4))

    plt.subplot(121)
    plt.imshow(biofilm_raw_images[0])
    plt.title('Example Biofilm Raw Image')

    plt.subplot(122)
    plt.imshow(release_cell_raw_images[0])
    plt.title('Example Release Cell Raw Image')

    plt.tight_layout()
    
    # Save plot instead of displaying
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'example_raw_images.png'), dpi=300, bbox_inches='tight')
    print(f"Saved example raw images to {output_dir}/example_raw_images.png")
    plt.close()


def convert_to_grayscale(biofilm_raw_images, release_cell_raw_images):
    """Convert images to grayscale by taking green channel."""
    print("Converting images to grayscale...")
    
    # Convert biofilm images to grayscale by taking green channel
    biofilm_gray_images = []
    for img in biofilm_raw_images:
        gray_img = img[:,:,1]
        biofilm_gray_images.append(gray_img)

    # Convert release cell images to grayscale by taking green channel
    release_cell_gray_images = []
    for img in release_cell_raw_images:
        gray_img = img[:,:,1]
        release_cell_gray_images.append(gray_img)

    # Display example grayscale images
    plt.figure(figsize=(12,4))

    plt.subplot(121)
    plt.imshow(biofilm_gray_images[0], cmap='gray')
    plt.title('Example Biofilm Grayscale Image')
    plt.colorbar()

    plt.subplot(122)
    plt.imshow(release_cell_gray_images[0], cmap='gray')
    plt.title('Example Release Cell Grayscale Image')
    plt.colorbar()

    plt.tight_layout()
    
    # Save plot instead of displaying
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'example_grayscale_images.png'), dpi=300, bbox_inches='tight')
    print(f"Saved example grayscale images to {output_dir}/example_grayscale_images.png")
    plt.close()
    
    return biofilm_gray_images, release_cell_gray_images

