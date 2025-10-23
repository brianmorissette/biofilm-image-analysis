#!/usr/bin/env python3
"""
Feature Extraction Module for Biofilm Image Analysis Pipeline

This module handles creating dataframes from images, calculating biofilm averages,
and preparing features for machine learning.
"""

import pandas as pd
import numpy as np


def create_dataframes(release_cell_gray_images):
    """Convert release cell images to pandas dataframes."""
    print("Creating dataframes from release cell images...")
    
    # Convert each release cell image to a pandas dataframe
    release_cell_dfs = []
    for img in release_cell_gray_images:
        df = pd.DataFrame(img)
        release_cell_dfs.append(df)

    print("Head of first release cell dataframe:")
    print(release_cell_dfs[0].head())
    
    return release_cell_dfs


def calculate_biofilm_averages(biofilm_gray_images):
    """Calculate average pixel value for each biofilm image."""
    print("Calculating biofilm average pixel values...")
    
    # Calculate average pixel value for each biofilm image
    biofilm_y = []
    for img in biofilm_gray_images:
        avg_pixel = np.mean(img)
        biofilm_y.append(avg_pixel)

    biofilm_y = np.array(biofilm_y)
    print(f"Biofilm average pixel values: {biofilm_y}")
    
    return biofilm_y


def prepare_features(release_cell_dfs, biofilm_y):
    """Format X and y values for Random Forest."""
    print("Preparing features for Random Forest...")
    
    X = [df.values.flatten() for df in release_cell_dfs]   # list of 1D arrays
    X = np.array(X)                                        # shape (n_samples, 250000)
    y = biofilm_y                                          # shape (n_samples,)
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")
    
    return X, y

