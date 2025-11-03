#!/usr/bin/env python3
"""
CNN Data Loading Module for Biofilm Image Analysis

This module creates a CNN-ready dataset by:
1. Loading paired biofilm and release cell images
2. Extracting 28x28 patches from release cell images (inputs)
3. Using mean pixel values from biofilm images as labels (outputs)
"""

import os
import glob
import numpy as np
import tifffile
import torch
from sklearn.model_selection import train_test_split
from utils import *


def match_image_pairs(biofilm_path, release_cells_path):
    paired_images = []
    for (biofilm_file, release_cell_file) in zip(load_images(biofilm_path), load_images(release_cells_path)):
        paired_images.append((biofilm_file, release_cell_file))
    return paired_images


def load_and_convert_to_grayscale(image_path):
    """
    Load a TIFF image and convert to grayscale by extracting green channel.
    
    Args:
        image_path: Path to the TIFF image file
    
    Returns:
        2D numpy array of grayscale pixel values
    
    Note:
        Assumes images are RGB where green channel (index 1) contains the data
    """
    img = load_images(image_path)
    return grayscale(img)


def extract_patches(image, patch_size=28):
    """
    Extract non-overlapping square patches from an image.
    
    Args:
        image: 2D numpy array (height x width)
        patch_size: Size of square patches (default 28x28 for CNN)
    
    Returns:
        List of 2D numpy arrays, each of shape (patch_size, patch_size)
    
    Example:
        If image is 560x560 and patch_size=28, this creates:
        (560/28) x (560/28) = 20 x 20 = 400 patches
    """
    height, width = image.shape
    patches = []
    
    # Calculate how many complete patches fit in each dimension
    num_patches_vertical = height // patch_size
    num_patches_horizontal = width // patch_size
    
    # Extract patches row by row
    for i in range(num_patches_vertical):
        for j in range(num_patches_horizontal):
            # Calculate patch boundaries
            start_y = i * patch_size
            end_y = start_y + patch_size
            start_x = j * patch_size
            end_x = start_x + patch_size
            
            # Extract patch
            patch = image[start_y:end_y, start_x:end_x]
            patches.append(patch)
    
    return patches


def normalize_data(patches, labels):
    """
    Normalize patches and labels to [0, 1] range for better CNN training.
    
    Args:
        patches: List of 2D numpy arrays (patches)
        labels: List of float values (labels)
    
    Returns:
        Tuple of (normalized_patches, normalized_labels)
    
    Note:
        Normalization helps CNN converge faster and more reliably
    """
    print("Normalizing data to [0, 1] range...")
    
    # Normalize patches: divide by maximum possible pixel value
    # Assuming 16-bit images (0-65535) or 8-bit (0-255)
    max_pixel_value = np.max([np.max(patch) for patch in patches])
    normalized_patches = [patch.astype(np.float32) / max_pixel_value for patch in patches]
    
    # Normalize labels by their maximum value
    max_label_value = np.max(labels)
    normalized_labels = [label / max_label_value for label in labels]
    
    print(f"  Pixel value range: [0, {max_pixel_value}] -> [0.0, 1.0]")
    print(f"  Label range: [0, {max_label_value:.2f}] -> [0.0, 1.0]")
    
    return normalized_patches, normalized_labels


def process_image_pairs(paired_images, patch_size, split_name=""):
    """
    Process a list of image pairs into a dataset.
    
    Args:
        paired_images: List of (biofilm_file, release_cell_file) tuples
        patch_size: Size of square patches (default 28 for 28x28)
        split_name: Name of the split (e.g., "train" or "test") for logging
    
    Returns:
        List of (patch, label) tuples (not yet formatted as tensors)
    """
    dataset = []
    
    for idx, (biofilm_file, release_cell_file) in enumerate(paired_images):
        print(f"\nProcessing {split_name} pair {idx + 1}/{len(paired_images)}:")
        print(f"  Biofilm: {os.path.basename(biofilm_file)}")
        print(f"  Release cell: {os.path.basename(release_cell_file)}")
        
        # Load and convert biofilm image to grayscale
        biofilm_gray = load_and_convert_to_grayscale(biofilm_file)
        print(f"  Biofilm image shape: {biofilm_gray.shape}")
        
        # Compute label (mean pixel value) from biofilm image
        label = compute_biofilm_label(biofilm_gray)
        print(f"  Biofilm mean pixel value (label): {label:.2f}")
        
        # Load and convert release cell image to grayscale
        release_cell_gray = load_and_convert_to_grayscale(release_cell_file)
        print(f"  Release cell image shape: {release_cell_gray.shape}")
        
        # Extract patches from release cell image
        patches = extract_patches(release_cell_gray, patch_size=patch_size)
        print(f"  Extracted {len(patches)} patches of size {patch_size}x{patch_size}")
        
        # Create dataset entries: each patch paired with the same label
        for patch in patches:
            dataset.append((patch, label))
    
    return dataset


def prepare_cnn_dataset(biofilm_path='biofilm_data/biofilm', 
                        release_cells_path='biofilm_data/release_cells',
                        patch_size=28,
                        normalize=True,
                        train_split=0.8,
                        random_state=42):
    """
    Prepare train and test datasets for CNN training.
    
    This is the main function that orchestrates the entire data preparation pipeline:
    1. Match biofilm and release cell image pairs
    2. Split pairs into train/test (80/20 by default)
    3. Load and convert images to grayscale
    4. Compute labels (mean pixel values) from biofilm images
    5. Extract 28x28 patches from release cell images
    6. Create (input, label) pairs where each patch gets the same label
    7. Format as 3D tensors for CNN input
    
    Args:
        biofilm_path: Path to folder containing biofilm images
        release_cells_path: Path to folder containing release cell images
        patch_size: Size of square patches (default 28 for 28x28)
        normalize: Whether to normalize data to [0, 1] range
        train_split: Fraction of image pairs to use for training (default 0.8)
        random_state: Random seed for reproducible splits (default 42)
    
    Returns:
        Tuple of (train_dataset, test_dataset) where each is a list of 
        (input_tensor, label) pairs:
        - input_tensor: PyTorch tensor of shape (1, 28, 28) - channels x height x width
        - label: Float value (mean pixel value of corresponding biofilm image)
    
    Example:
        With 6 image pairs and patches of 28x28:
        - If each release cell image is 560x560, we get 20x20 = 400 patches per image
        - Train dataset: 5 pairs * 400 = 2000 examples (80%)
        - Test dataset: 1 pair * 400 = 400 examples (20%)
    """
    print("=" * 60)
    print("PREPARING CNN DATASET WITH TRAIN/TEST SPLIT")
    print("=" * 60)
    
    # Step 1: Match image pairs by XY identifier
    paired_images = match_image_pairs(biofilm_path, release_cells_path)
    
    if len(paired_images) == 0:
        raise ValueError("No matched image pairs found!")
    
    # Step 2: Split image pairs into train and test
    if len(paired_images) == 1:
        print("\nWarning: Only 1 image pair found. Using it for both train and test.")
        train_pairs = paired_images
        test_pairs = paired_images
    else:
        train_pairs, test_pairs = train_test_split(
            paired_images, 
            train_size=train_split, 
            random_state=random_state
        )
    
    print(f"\nSplit into:")
    print(f"  Train: {len(train_pairs)} image pairs ({len(train_pairs)/len(paired_images)*100:.0f}%)")
    print(f"  Test: {len(test_pairs)} image pairs ({len(test_pairs)/len(paired_images)*100:.0f}%)")
    
    # Step 3: Process train and test pairs separately
    print(f"\n{'='*60}")
    print("PROCESSING TRAIN PAIRS")
    print(f"{'='*60}")
    train_dataset = process_image_pairs(train_pairs, patch_size, split_name="train")
    print(f"\nTrain dataset size: {len(train_dataset)} examples")
    
    print(f"\n{'='*60}")
    print("PROCESSING TEST PAIRS")
    print(f"{'='*60}")
    test_dataset = process_image_pairs(test_pairs, patch_size, split_name="test")
    print(f"\nTest dataset size: {len(test_dataset)} examples")
    
    # Step 4: Separate patches and labels for normalization
    # IMPORTANT: Use ONLY train data statistics for normalization to avoid data leakage
    train_patches = [item[0] for item in train_dataset]
    train_labels = [item[1] for item in train_dataset]
    test_patches = [item[0] for item in test_dataset]
    test_labels = [item[1] for item in test_dataset]
    
    # Step 5: Normalize if requested (using train statistics for both sets)
    if normalize:
        print("\n" + "="*60)
        print("NORMALIZING DATA (using train statistics)")
        print("="*60)
        
        # Calculate normalization parameters from train data only
        max_pixel_value = np.max([np.max(patch) for patch in train_patches])
        max_label_value = np.max(train_labels)
        
        # Apply normalization to both train and test
        train_patches = [patch.astype(np.float32) / max_pixel_value for patch in train_patches]
        train_labels = [label / max_label_value for label in train_labels]
        test_patches = [patch.astype(np.float32) / max_pixel_value for patch in test_patches]
        test_labels = [label / max_label_value for label in test_labels]
        
        print(f"  Train pixel value range: [0, {max_pixel_value}] -> [0.0, 1.0]")
        print(f"  Train label range: [0, {max_label_value:.2f}] -> [0.0, 1.0]")
        print(f"  Applied same normalization to test data")
    
    # Step 6: Format as PyTorch tensors (channels, height, width) for CNN
    print("\n" + "="*60)
    print("FORMATTING AS PYTORCH TENSORS")
    print("="*60)
    
    def format_as_tensors(patches, labels):
        """Convert patches and labels to PyTorch tensors"""
        formatted = []
        for patch, label in zip(patches, labels):
            # Add channel dimension: (28, 28) -> (1, 28, 28)
            patch_with_channel = np.expand_dims(patch, axis=0).astype(np.float32)
            input_tensor = torch.from_numpy(patch_with_channel)
            label_float = float(label)
            formatted.append((input_tensor, label_float))
        return formatted
    
    train_formatted = format_as_tensors(train_patches, train_labels)
    test_formatted = format_as_tensors(test_patches, test_labels)
    
    print(f"  Input tensor shape: {train_formatted[0][0].shape} (channels, height, width)")
    print(f"  Label type: {type(train_formatted[0][1])}")
    
    # Step 7: Display summary statistics
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"TRAIN:")
    print(f"  Examples: {len(train_formatted)}")
    print(f"  Label range: [{min(train_labels):.4f}, {max(train_labels):.4f}]")
    print(f"  Unique labels: {len(set(train_labels))}")
    print(f"\nTEST:")
    print(f"  Examples: {len(test_formatted)}")
    print(f"  Label range: [{min(test_labels):.4f}, {max(test_labels):.4f}]")
    print(f"  Unique labels: {len(set(test_labels))}")
    print("=" * 60)
    
    return train_formatted, test_formatted


# Example usage
if __name__ == "__main__":
    # Create the CNN train and test datasets
    train_dataset, test_dataset = prepare_cnn_dataset(
        biofilm_path='biofilm_data/biofilm',
        release_cells_path='biofilm_data/release_cells',
        patch_size=28,
        normalize=True,
        train_split=0.8,
        random_state=42
    )
    
    print(f"\nDatasets are ready! You can now use them for CNN training.")
    print(f"Example usage in training:")
    print(f"  for input_tensor, label in train_dataset:")
    print(f"      # input_tensor shape: (1, 28, 28)")
    print(f"      # label: float value")
    print(f"      # ... feed into your CNN ...")

