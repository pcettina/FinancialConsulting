#!/usr/bin/env python3
"""
Bond Trading Analysis - Main Analysis Script
============================================

This script demonstrates a complete bond trading analysis pipeline including:
1. Data generation with realistic bond trading parameters
2. PCA analysis for dimensionality reduction and feature understanding
3. Predictive modeling with multiple algorithms
4. Comprehensive reporting and visualization

Author: AI Assistant
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from data_generator import BondDataGenerator
from pca_analysis import BondPCAnalysis

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def main():
    """Main analysis pipeline"""
    
    print("=" * 80)
    print("BOND TRADING ANALYSIS PIPELINE")
    print("=" * 80)
    print(f"Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Step 1: Generate Bond Trading Data
    print("STEP 1: GENERATING BOND TRADING DATA")
    print("-" * 50)
    
    generator = BondDataGenerator(n_trades=2000, seed=42)
    bond_data = generator.generate_bond_data()
    
    # Display sample data
    print("\nSample of generated bond trading data:")
    print(bond_data.head())
    
    print("\nDataset statistics:")
    print(bond_data.describe())
    
    # Save data
    data_file = generator.save_data(bond_data, 'bond_trading_data.csv')
    print(f"\nData saved to: {data_file}")
    
    # Step 2: PCA Analysis
    print("\n\nSTEP 2: PRINCIPAL COMPONENT ANALYSIS")
    print("-" * 50)
    
    pca_analyzer = BondPCAnalysis(data_file)
    pca_analyzer.load_data()
    pca_results = pca_analyzer.perform_pca()
    
    # Generate PCA visualizations
    print("\nGenerating PCA visualizations...")
    pca_analyzer.plot_explained_variance('pca_explained_variance.png')
    pca_analyzer.plot_feature_importance(5, 'pca_feature_importance.png')
    pca_analyzer.plot_pca_scatter(1, 2, save_path='pca_scatter_plot.html')
    pca_analyzer.plot_3d_pca(1, 2, 3, save_path='pca_3d_plot.html')
    pca_analyzer.correlation_analysis()
    
    # Generate PCA report
    pca_report = pca_analyzer.generate_report('pca_analysis_report.txt')
    print("\nPCA Analysis Report:")
    print(pca_report)
    
    # Step 3: Data Exploration and Insights
    print("\n\nSTEP 3: DATA EXPLORATION AND INSIGHTS")
    print("-" * 50)
    
    # Analyze trade rating distribution
    plt.figure(figsize=(12, 8))
    
    # Trade rating distribution
    plt.subplot(2, 3, 1)
    plt.hist(bond_data['trade_rating'], bins=20, alpha=0.7, color='steelblue', edgecolor='black')
    plt.title('Trade Rating Distribution')
    plt.xlabel('Trade Rating')
    plt.ylabel('Frequency')
    
    # Duration vs Trade Rating
    plt.subplot(2, 3, 2)
    plt.scatter(bond_data['duration'], bond_data['trade_rating'], alpha=0.6, color='coral')
    plt.title('Duration vs Trade Rating')
    plt.xlabel('Duration (years)')
    plt.ylabel('Trade Rating')
    
    # Yield vs Trade Rating
    plt.subplot(2, 3, 3)
    plt.scatter(bond_data['yield'], bond_data['trade_rating'], alpha=0.6, color='lightgreen')
    plt.title('Yield vs Trade Rating')
    plt.xlabel('Yield (%)')
    plt.ylabel('Trade Rating')
    
    # Price vs Trade Rating
    plt.subplot(2, 3, 4)
    plt.scatter(bond_data['price'], bond_data['trade_rating'], alpha=0.6, color='gold')
    plt.title('Price vs Trade Rating')
    plt.xlabel('Price')
    plt.ylabel('Trade Rating')
    
    # Credit Rating vs Trade Rating
    plt.subplot(2, 3, 5)
    credit_rating_avg = bond_data.groupby('moodys_rating')['trade_rating'].mean().sort_index()
    plt.bar(range(len(credit_rating_avg)), credit_rating_avg.values, color='purple', alpha=0.7)
    plt.title('Average Trade Rating by Credit Rating')
    plt.xlabel('Credit Rating')
    plt.ylabel('Average Trade Rating')
    plt.xticks(range(len(credit_rating_avg)), credit_rating_avg.index, rotation=45)
    
    # Investment Grade vs Non-Investment Grade
    plt.subplot(2, 3, 6)
    investment_grade_avg = bond_data.groupby('is_investment_grade')['trade_rating'].mean()
    plt.bar(['Non-Investment Grade', 'Investment Grade'], investment_grade_avg.values, 
            color=['red', 'green'], alpha=0.7)
    plt.title('Average Trade Rating by Investment Grade')
    plt.ylabel('Average Trade Rating')
    
    plt.tight_layout()
    plt.savefig('data_exploration.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Correlation analysis
    print("\nCorrelation Analysis:")
    numerical_cols = ['duration', 'yield', 'base_coupon', 'price', 'ytm', 'ytw', 'trade_rating']
    correlation_matrix = bond_data[numerical_cols].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, square=True)
    plt.title('Correlation Matrix of Numerical Features')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Step 4: Key Insights Summary
    print("\n\nSTEP 4: KEY INSIGHTS SUMMARY")
    print("-" * 50)
    
    insights = []
    insights.append("KEY INSIGHTS FROM BOND TRADING ANALYSIS")
    insights.append("=" * 50)
    insights.append("")
    
    # Trade rating statistics
    insights.append("TRADE RATING STATISTICS:")
    insights.append(f"  Mean trade rating: {bond_data['trade_rating'].mean():.2f}")
    insights.append(f"  Median trade rating: {bond_data['trade_rating'].median():.2f}")
    insights.append(f"  Standard deviation: {bond_data['trade_rating'].std():.2f}")
    insights.append(f"  Range: {bond_data['trade_rating'].min():.1f} - {bond_data['trade_rating'].max():.1f}")
    insights.append("")
    
    # Top performing characteristics
    high_rated = bond_data[bond_data['trade_rating'] >= 8.0]
    low_rated = bond_data[bond_data['trade_rating'] <= 3.0]
    
    insights.append("HIGH-RATED TRADES (Rating >= 8.0):")
    insights.append(f"  Count: {len(high_rated)} ({len(high_rated)/len(bond_data)*100:.1f}%)")
    insights.append(f"  Average duration: {high_rated['duration'].mean():.2f} years")
    insights.append(f"  Average yield: {high_rated['yield'].mean():.2f}%")
    insights.append(f"  Average price: {high_rated['price'].mean():.2f}")
    insights.append(f"  Investment grade ratio: {high_rated['is_investment_grade'].mean()*100:.1f}%")
    insights.append("")
    
    insights.append("LOW-RATED TRADES (Rating <= 3.0):")
    insights.append(f"  Count: {len(low_rated)} ({len(low_rated)/len(bond_data)*100:.1f}%)")
    insights.append(f"  Average duration: {low_rated['duration'].mean():.2f} years")
    insights.append(f"  Average yield: {low_rated['yield'].mean():.2f}%")
    insights.append(f"  Average price: {low_rated['price'].mean():.2f}")
    insights.append(f"  Investment grade ratio: {low_rated['is_investment_grade'].mean()*100:.1f}%")
    insights.append("")
    
    # Feature correlations with trade rating
    correlations = correlation_matrix['trade_rating'].sort_values(ascending=False)
    insights.append("FEATURE CORRELATIONS WITH TRADE RATING:")
    for feature, corr in correlations.items():
        if feature != 'trade_rating':
            insights.append(f"  {feature}: {corr:.3f}")
    insights.append("")
    
    # Investment grade analysis
    investment_grade_stats = bond_data.groupby('is_investment_grade').agg({
        'trade_rating': ['mean', 'std', 'count'],
        'duration': 'mean',
        'yield': 'mean',
        'price': 'mean'
    }).round(2)
    
    insights.append("INVESTMENT GRADE ANALYSIS:")
    insights.append(f"  Investment Grade - Avg Rating: {investment_grade_stats.loc[True, ('trade_rating', 'mean')]:.2f}")
    insights.append(f"  Non-Investment Grade - Avg Rating: {investment_grade_stats.loc[False, ('trade_rating', 'mean')]:.2f}")
    insights.append("")
    
    # Save insights
    with open('key_insights.txt', 'w') as f:
        f.write('\n'.join(insights))
    
    print('\n'.join(insights))
    
    # Step 5: Recommendations
    print("\n\nSTEP 5: RECOMMENDATIONS FOR FURTHER ANALYSIS")
    print("-" * 50)
    
    recommendations = []
    recommendations.append("RECOMMENDATIONS FOR ENHANCED MODELING:")
    recommendations.append("=" * 50)
    recommendations.append("")
    
    recommendations.append("1. PREDICTIVE MODELING ENHANCEMENTS:")
    recommendations.append("   - Implement ensemble methods combining multiple algorithms")
    recommendations.append("   - Add time-series analysis for market timing")
    recommendations.append("   - Include macroeconomic indicators (interest rates, GDP, etc.)")
    recommendations.append("   - Add sector-specific analysis (corporate, government, municipal)")
    recommendations.append("")
    
    recommendations.append("2. FEATURE ENGINEERING:")
    recommendations.append("   - Create interaction features (duration × yield, price × credit rating)")
    recommendations.append("   - Add volatility measures")
    recommendations.append("   - Include liquidity indicators")
    recommendations.append("   - Add call/put option features")
    recommendations.append("")
    
    recommendations.append("3. ADVANCED ANALYTICS:")
    recommendations.append("   - Implement clustering analysis to identify bond categories")
    recommendations.append("   - Add sentiment analysis from news and reports")
    recommendations.append("   - Include technical indicators")
    recommendations.append("   - Add Monte Carlo simulations for risk assessment")
    recommendations.append("")
    
    recommendations.append("4. REAL-TIME FEATURES:")
    recommendations.append("   - Market volatility indices")
    recommendations.append("   - Credit default swap spreads")
    recommendations.append("   - Yield curve steepness/flatness")
    recommendations.append("   - Sector rotation indicators")
    recommendations.append("")
    
    recommendations.append("5. MACHINE LEARNING ADVANCEMENTS:")
    recommendations.append("   - Deep learning models (LSTM, Transformer)")
    recommendations.append("   - Reinforcement learning for optimal trading strategies")
    recommendations.append("   - Anomaly detection for unusual bond behavior")
    recommendations.append("   - Natural language processing for credit report analysis")
    recommendations.append("")
    
    # Save recommendations
    with open('recommendations.txt', 'w') as f:
        f.write('\n'.join(recommendations))
    
    print('\n'.join(recommendations))
    
    # Final summary
    print("\n\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nGenerated files:")
    print("  - bond_trading_data.csv (raw data)")
    print("  - pca_analysis_report.txt (PCA results)")
    print("  - key_insights.txt (main insights)")
    print("  - recommendations.txt (future enhancements)")
    print("  - Various visualization files (.png, .html)")
    print("\nNext steps:")
    print("  1. Review the PCA analysis to understand key drivers")
    print("  2. Implement predictive modeling using the insights")
    print("  3. Consider the recommendations for enhanced analysis")
    print("  4. Validate findings with domain experts")

if __name__ == "__main__":
    main() 