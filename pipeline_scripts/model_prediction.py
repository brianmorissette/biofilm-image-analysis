#!/usr/bin/env python3
"""
Model Prediction Module for Biofilm Image Analysis Pipeline

This module handles Random Forest model training and evaluation for predicting
biofilm density from release cell images.
"""

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


def train_random_forest(X, y):
    """Train Random Forest model and evaluate performance."""
    print("Training Random Forest model...")
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")

    # Create Random Forest model and train
    rf = RandomForestRegressor(
        n_estimators=100, 
        random_state=42,
    )

    rf.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf.predict(X_test)

    # Compute MSE by hand
    mse = 0
    for y_true, y_predicted in zip(y_test, y_pred):
        mse += (y_true - y_predicted) ** 2
    mse = mse / len(y_test)

    print(f"MSE: {mse}")
    
    # Additional metrics
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"MAE: {mae}")
    print(f"R² Score: {r2}")
    
    return rf, X_test, y_test, y_pred

