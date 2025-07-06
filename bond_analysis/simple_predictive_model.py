#!/usr/bin/env python3
"""
Simple Predictive Modeling for Bond Trading
===========================================

A simplified version of predictive modeling for bond trade ratings.
This script demonstrates basic machine learning techniques for predicting
bond trade success based on various bond characteristics.

Author: AI Assistant
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data(data_path='bond_trading_data.csv'):
    """Load and prepare data for modeling"""
    
    # Load data
    df = pd.read_csv(data_path)
    print(f"Loaded data with shape: {df.shape}")
    
    # Select features for modeling
    numerical_features = [
        'duration', 'yield', 'base_coupon', 'price', 'ytm', 'ytw',
        'price_to_par', 'coupon_yield_spread', 'ytm_ytw_spread',
        'moodys_numeric', 'sp_numeric', 'fitch_numeric'
    ]
    
    # Create feature matrix
    X = df[numerical_features].copy()
    
    # Handle missing values
    X = X.fillna(X.mean())
    
    # Target variable
    y = df['trade_rating']
    
    print(f"Prepared {X.shape[1]} features for modeling")
    print(f"Target variable range: {y.min():.1f} - {y.max():.1f}")
    
    return X, y

def train_and_evaluate_models(X, y):
    """Train and evaluate multiple models"""
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'model': model,
            'scaler': scaler,
            'mse': mse,
            'r2': r2,
            'predictions': y_pred
        }
        
        print(f"{name} - MSE: {mse:.4f}, R²: {r2:.4f}")
    
    return results, X_test, y_test

def plot_results(results, X_test, y_test):
    """Plot model results"""
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Model comparison
    models = list(results.keys())
    r2_scores = [results[model]['r2'] for model in models]
    mse_scores = [results[model]['mse'] for model in models]
    
    # R² comparison
    axes[0].bar(models, r2_scores, color=['skyblue', 'lightcoral'])
    axes[0].set_title('R² Score Comparison')
    axes[0].set_ylabel('R² Score')
    axes[0].tick_params(axis='x', rotation=45)
    
    # MSE comparison
    axes[1].bar(models, mse_scores, color=['lightgreen', 'gold'])
    axes[1].set_title('Mean Squared Error Comparison')
    axes[1].set_ylabel('MSE')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Predicted vs Actual for best model
    best_model_name = max(results.keys(), key=lambda x: results[x]['r2'])
    best_model = results[best_model_name]
    
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, best_model['predictions'], alpha=0.6, color='steelblue')
    plt.plot([y_test.min(), y_test.max()], 
            [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Trade Rating')
    plt.ylabel('Predicted Trade Rating')
    plt.title(f'Predicted vs Actual - {best_model_name}')
    plt.grid(True, alpha=0.3)
    
    # Add R² score to plot
    plt.text(0.05, 0.95, f'R² = {best_model["r2"]:.3f}', 
            transform=plt.gca().transAxes, 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('predicted_vs_actual.png', dpi=300, bbox_inches='tight')
    plt.show()

def predict_new_trade(results, trade_data=None):
    """Predict trade rating for a new bond trade"""
    
    if trade_data is None:
        # Example trade data
        trade_data = {
            'duration': 5.5,
            'yield': 4.2,
            'base_coupon': 4.0,
            'price': 98.5,
            'ytm': 4.3,
            'ytw': 4.4,
            'price_to_par': 0.985,
            'coupon_yield_spread': -0.2,
            'ytm_ytw_spread': 0.1,
            'moodys_numeric': 15,  # A-
            'sp_numeric': 16,      # A
            'fitch_numeric': 15    # A-
        }
    
    # Get best model
    best_model_name = max(results.keys(), key=lambda x: results[x]['r2'])
    best_result = results[best_model_name]
    
    # Prepare features
    X, _ = load_and_prepare_data()
    feature_names = X.columns.tolist()
    
    # Create feature vector
    new_trade_features = []
    for col in feature_names:
        if col in trade_data:
            new_trade_features.append(trade_data[col])
        else:
            new_trade_features.append(0)
    
    # Scale features
    new_trade_scaled = best_result['scaler'].transform([new_trade_features])
    
    # Make prediction
    prediction = best_result['model'].predict(new_trade_scaled)[0]
    
    print(f"\nPrediction for new bond trade:")
    print(f"Model used: {best_model_name}")
    print(f"Predicted Trade Rating: {prediction:.2f}")
    print(f"Model R² Score: {best_result['r2']:.3f}")
    
    return prediction

def main():
    """Main function"""
    
    print("=" * 60)
    print("SIMPLE PREDICTIVE MODELING FOR BOND TRADING")
    print("=" * 60)
    
    # Load and prepare data
    print("\n1. Loading and preparing data...")
    X, y = load_and_prepare_data()
    
    # Train and evaluate models
    print("\n2. Training and evaluating models...")
    results, X_test, y_test = train_and_evaluate_models(X, y)
    
    # Plot results
    print("\n3. Generating visualizations...")
    plot_results(results, X_test, y_test)
    
    # Predict new trade
    print("\n4. Making prediction for new trade...")
    prediction = predict_new_trade(results)
    
    # Summary
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)
    
    best_model_name = max(results.keys(), key=lambda x: results[x]['r2'])
    best_result = results[best_model_name]
    
    print(f"Best performing model: {best_model_name}")
    print(f"R² Score: {best_result['r2']:.3f}")
    print(f"Mean Squared Error: {best_result['mse']:.4f}")
    
    print("\nGenerated files:")
    print("  - model_comparison.png")
    print("  - predicted_vs_actual.png")
    
    print("\nNext steps:")
    print("  1. Review the model performance")
    print("  2. Try different feature combinations")
    print("  3. Experiment with hyperparameter tuning")
    print("  4. Consider ensemble methods")

if __name__ == "__main__":
    main() 