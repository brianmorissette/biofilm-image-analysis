#!/usr/bin/env python3
"""
Data Loading Module for Biofilm Image Analysis Pipeline

This module handles loading biofilm and release cell images from data folders
and checking their color channels.
"""

import os
import glob
import tifffile


def load_data():
    """Load biofilm and release cell images from data folders."""
    print("Loading data...")
    
    # Create paths to data folders
    biofilm_path = 'biofilm_data/biofilm'
    release_cells_path = 'biofilm_data/release_cells'

    # Get all tiff files from each folder
    biofilm_files = glob.glob(os.path.join(biofilm_path, '*.tif'))
    release_cell_files = glob.glob(os.path.join(release_cells_path, '*.tif'))

    # Load biofilm images into list
    biofilm_raw_images = []
    for file in biofilm_files:
        img = tifffile.imread(file)
        biofilm_raw_images.append(img)

    # Load release cell images into list  
    release_cell_raw_images = []
    for file in release_cell_files:
        img = tifffile.imread(file)
        release_cell_raw_images.append(img)

    print(f"Loaded {len(biofilm_raw_images)} biofilm images")
    print(f"Loaded {len(release_cell_raw_images)} release cell images")
    
    return biofilm_raw_images, release_cell_raw_images


def check_color_channels(biofilm_raw_images, release_cell_raw_images):
    """Check if there are any red or blue values in any images."""
    print("Checking color channels...")
    
    # Check biofilm images for red and blue values
    print("Summary of biofilm images:")
    red_counts = sum(img[:,:,0].any() for img in biofilm_raw_images)
    blue_counts = sum(img[:,:,2].any() for img in biofilm_raw_images)
    print(f"Images containing red values: {red_counts} out of {len(biofilm_raw_images)}")
    print(f"Images containing blue values: {blue_counts} out of {len(biofilm_raw_images)}")

    print("\nSummary of release cell images:")
    red_counts = sum(img[:,:,0].any() for img in release_cell_raw_images)
    blue_counts = sum(img[:,:,2].any() for img in release_cell_raw_images)
    print(f"Images containing red values: {red_counts} out of {len(release_cell_raw_images)}")
    print(f"Images containing blue values: {blue_counts} out of {len(release_cell_raw_images)}")

