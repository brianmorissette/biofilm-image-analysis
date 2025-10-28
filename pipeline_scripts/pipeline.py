#!/usr/bin/env python3
"""
Biofilm Image Analysis Pipeline
Modular version with separate components

This script orchestrates the biofilm image analysis pipeline by importing
and coordinating functions from separate modules:
- data_loading: Load and check biofilm/release cell images
- image_processing: Visualize and convert images to grayscale
- feature_extraction: Create dataframes and prepare features
- model_prediction: Train Random Forest model and evaluate
"""

# Import from custom modules
from data_loading import load_data, check_color_channels
from image_processing import display_example_images, convert_to_grayscale
from feature_extraction import create_dataframes, calculate_biofilm_averages, prepare_features
from model_prediction import train_random_forest


def main():
    """Main pipeline execution that orchestrates all modules."""
    print("Starting Biofilm Image Analysis Pipeline")
    print("=" * 50)
    
    # 1. Load data
    biofilm_raw_images, release_cell_raw_images = load_data()
    
    # 2. Display example images
    display_example_images(biofilm_raw_images, release_cell_raw_images)
    
    # 3. Check color channels
    check_color_channels(biofilm_raw_images, release_cell_raw_images)
    
    # 4. Convert to grayscale
    biofilm_gray_images, release_cell_gray_images = convert_to_grayscale(
        biofilm_raw_images, release_cell_raw_images
    )
    
    # 5. Create dataframes
    release_cell_dfs = create_dataframes(release_cell_gray_images)
    
    # 6. Calculate biofilm averages
    biofilm_y = calculate_biofilm_averages(biofilm_gray_images)
    
    # 7. Prepare features
    X, y = prepare_features(release_cell_dfs, biofilm_y)
    
    # 8. Train Random Forest model
    rf, X_test, y_test, y_pred = train_random_forest(X, y)
    
    print("\nPipeline completed successfully!")
    print("=" * 50)


if __name__ == "__main__":
    main()
