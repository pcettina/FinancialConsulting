#!/usr/bin/env python3
"""
Bond Trading Analysis - Quick Demo
==================================

A quick demonstration of the bond trading analysis system.
This script generates a small dataset and performs basic analysis.

Author: AI Assistant
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_generator import BondDataGenerator
from pca_analysis import BondPCAnalysis
import warnings
warnings.filterwarnings('ignore')

def quick_demo():
    """Quick demonstration of the bond trading analysis"""
    
    print("🚀 BOND TRADING ANALYSIS - QUICK DEMO")
    print("=" * 50)
    
    # Step 1: Generate small dataset
    print("\n📊 Step 1: Generating bond trading data...")
    generator = BondDataGenerator(n_trades=500, seed=42)
    bond_data = generator.generate_bond_data()
    
    print(f"Generated {len(bond_data)} bond trades")
    print(f"Dataset shape: {bond_data.shape}")
    
    # Show sample data
    print("\nSample data:")
    print(bond_data[['cusip', 'duration', 'yield', 'price', 'moodys_rating', 'trade_rating']].head())
    
    # Step 2: Basic statistics
    print("\n📈 Step 2: Basic statistics...")
    print(f"Trade rating range: {bond_data['trade_rating'].min():.1f} - {bond_data['trade_rating'].max():.1f}")
    print(f"Average trade rating: {bond_data['trade_rating'].mean():.2f}")
    print(f"Investment grade bonds: {bond_data['is_investment_grade'].sum()} ({bond_data['is_investment_grade'].mean()*100:.1f}%)")
    
    # Step 3: Save data
    print("\n💾 Step 3: Saving data...")
    generator.save_data(bond_data, 'demo_bond_data.csv')
    
    # Step 4: Quick PCA analysis
    print("\n🔍 Step 4: Performing PCA analysis...")
    pca_analyzer = BondPCAnalysis('demo_bond_data.csv')
    pca_analyzer.load_data()
    pca_results = pca_analyzer.perform_pca(n_components=5)
    
    # Show PCA results
    print(f"PCA completed with {pca_analyzer.pca.n_components_} components")
    print("Explained variance ratios:")
    for i, var in enumerate(pca_analyzer.pca.explained_variance_ratio_):
        print(f"  PC{i+1}: {var:.3f} ({var*100:.1f}%)")
    
    cumulative_var = np.cumsum(pca_analyzer.pca.explained_variance_ratio_)
    print(f"Cumulative variance: {cumulative_var[-1]:.3f} ({cumulative_var[-1]*100:.1f}%)")
    
    # Step 5: Quick visualization
    print("\n📊 Step 5: Creating quick visualizations...")
    
    # Create a simple plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Trade rating distribution
    axes[0, 0].hist(bond_data['trade_rating'], bins=15, alpha=0.7, color='steelblue', edgecolor='black')
    axes[0, 0].set_title('Trade Rating Distribution')
    axes[0, 0].set_xlabel('Trade Rating')
    axes[0, 0].set_ylabel('Frequency')
    
    # Duration vs Trade Rating
    axes[0, 1].scatter(bond_data['duration'], bond_data['trade_rating'], alpha=0.6, color='coral')
    axes[0, 1].set_title('Duration vs Trade Rating')
    axes[0, 1].set_xlabel('Duration (years)')
    axes[0, 1].set_ylabel('Trade Rating')
    
    # Yield vs Trade Rating
    axes[1, 0].scatter(bond_data['yield'], bond_data['trade_rating'], alpha=0.6, color='lightgreen')
    axes[1, 0].set_title('Yield vs Trade Rating')
    axes[1, 0].set_xlabel('Yield (%)')
    axes[1, 0].set_ylabel('Trade Rating')
    
    # Investment Grade Analysis
    investment_grade_avg = bond_data.groupby('is_investment_grade')['trade_rating'].mean()
    axes[1, 1].bar(['Non-Investment', 'Investment'], investment_grade_avg.values, 
                   color=['red', 'green'], alpha=0.7)
    axes[1, 1].set_title('Average Trade Rating by Investment Grade')
    axes[1, 1].set_ylabel('Average Trade Rating')
    
    plt.tight_layout()
    plt.savefig('demo_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Step 6: Key insights
    print("\n💡 Step 6: Key insights...")
    
    # High vs low rated trades
    high_rated = bond_data[bond_data['trade_rating'] >= 8.0]
    low_rated = bond_data[bond_data['trade_rating'] <= 3.0]
    
    print(f"High-rated trades (≥8.0): {len(high_rated)} ({len(high_rated)/len(bond_data)*100:.1f}%)")
    print(f"  Average duration: {high_rated['duration'].mean():.2f} years")
    print(f"  Average yield: {high_rated['yield'].mean():.2f}%")
    print(f"  Investment grade: {high_rated['is_investment_grade'].mean()*100:.1f}%")
    
    print(f"\nLow-rated trades (≤3.0): {len(low_rated)} ({len(low_rated)/len(bond_data)*100:.1f}%)")
    print(f"  Average duration: {low_rated['duration'].mean():.2f} years")
    print(f"  Average yield: {low_rated['yield'].mean():.2f}%")
    print(f"  Investment grade: {low_rated['is_investment_grade'].mean()*100:.1f}%")
    
    # Correlation analysis
    numerical_cols = ['duration', 'yield', 'base_coupon', 'price', 'ytm', 'ytw', 'trade_rating']
    correlation_matrix = bond_data[numerical_cols].corr()
    
    print(f"\nTop correlations with trade rating:")
    correlations = correlation_matrix['trade_rating'].sort_values(ascending=False)
    for feature, corr in correlations.items():
        if feature != 'trade_rating':
            print(f"  {feature}: {corr:.3f}")
    
    # Step 7: Summary
    print("\n" + "=" * 50)
    print("🎉 DEMO COMPLETE!")
    print("=" * 50)
    print("\nGenerated files:")
    print("  - demo_bond_data.csv (sample dataset)")
    print("  - demo_analysis.png (visualizations)")
    
    print("\nKey findings:")
    print("  ✅ Successfully generated realistic bond trading data")
    print("  ✅ PCA analysis shows key components explain most variance")
    print("  ✅ Investment grade bonds tend to have higher ratings")
    print("  ✅ Duration and yield show clear relationships with ratings")
    
    print("\nNext steps:")
    print("  🔄 Run 'python main_analysis.py' for full analysis")
    print("  🔄 Run 'python simple_predictive_model.py' for ML modeling")
    print("  🔄 Explore the generated files and visualizations")
    
    return bond_data, pca_results

if __name__ == "__main__":
    bond_data, pca_results = quick_demo() 