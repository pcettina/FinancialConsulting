# Simple Visualization System
"""
Simple visualization system for testing basic plotting capabilities.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf

def create_sample_stock_data():
    """Create sample stock data for visualization testing"""
    print("Creating sample stock data...")
    
    # Generate sample dates
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    
    # Create sample data for AAPL, GOOGL, MSFT
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    data = {}
    
    for i, symbol in enumerate(symbols):
        # Generate realistic price movements
        np.random.seed(42 + i)  # Different seed for each symbol
        base_price = 100 + i * 50  # Different base prices
        
        # Generate daily returns
        daily_returns = np.random.normal(0.001, 0.02, len(dates))
        prices = base_price * np.exp(np.cumsum(daily_returns))
        
        # Create DataFrame
        df = pd.DataFrame({
            'Date': dates,
            'Open': prices * (1 + np.random.normal(0, 0.01, len(dates))),
            'High': prices * (1 + np.abs(np.random.normal(0, 0.02, len(dates)))),
            'Low': prices * (1 - np.abs(np.random.normal(0, 0.02, len(dates)))),
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, len(dates))
        })
        
        # Calculate some basic technical indicators
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['RSI'] = calculate_rsi(df['Close'])
        
        df.set_index('Date', inplace=True)
        data[symbol] = df
        
        print(f"Created data for {symbol}: {len(df)} data points")
    
    return data

def calculate_rsi(prices, period=14):
    """Calculate RSI technical indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def plot_price_comparison(data):
    """Plot price comparison for all symbols"""
    print("Creating price comparison chart...")
    
    plt.figure(figsize=(15, 8))
    
    for symbol, df in data.items():
        # Normalize prices to start at 100 for comparison
        normalized_prices = df['Close'] / df['Close'].iloc[0] * 100
        plt.plot(df.index, normalized_prices, label=symbol, linewidth=2)
    
    plt.title('Stock Price Performance Comparison (Normalized)', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Normalized Price (Base = 100)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('results/price_comparison.png', dpi=300, bbox_inches='tight')
    print("Price comparison chart saved to results/price_comparison.png")
    plt.show()

def plot_technical_indicators(data, symbol='AAPL'):
    """Plot technical indicators for a specific symbol"""
    print(f"Creating technical indicators chart for {symbol}...")
    
    if symbol not in data:
        print(f"Symbol {symbol} not found in data")
        return
    
    df = data[symbol]
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
    
    # Price and moving averages
    ax1.plot(df.index, df['Close'], label='Close Price', linewidth=2)
    ax1.plot(df.index, df['SMA_20'], label='SMA 20', alpha=0.7)
    ax1.plot(df.index, df['SMA_50'], label='SMA 50', alpha=0.7)
    ax1.set_title(f'{symbol} - Price and Moving Averages', fontsize=14)
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Volume
    ax2.bar(df.index, df['Volume'], alpha=0.6, color='blue')
    ax2.set_title(f'{symbol} - Trading Volume', fontsize=14)
    ax2.set_ylabel('Volume', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # RSI
    ax3.plot(df.index, df['RSI'], label='RSI', color='purple', linewidth=2)
    ax3.axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Overbought (70)')
    ax3.axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Oversold (30)')
    ax3.set_title(f'{symbol} - Relative Strength Index (RSI)', fontsize=14)
    ax3.set_ylabel('RSI', fontsize=12)
    ax3.set_xlabel('Date', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'results/{symbol}_technical_indicators.png', dpi=300, bbox_inches='tight')
    print(f"Technical indicators chart saved to results/{symbol}_technical_indicators.png")
    plt.show()

def plot_correlation_matrix(data):
    """Plot correlation matrix between symbols"""
    print("Creating correlation matrix...")
    
    # Create price DataFrame
    price_data = pd.DataFrame()
    for symbol, df in data.items():
        price_data[symbol] = df['Close']
    
    # Calculate correlation matrix
    correlation_matrix = price_data.corr()
    
    plt.figure(figsize=(10, 8))
    plt.imshow(correlation_matrix, cmap='RdBu', vmin=-1, vmax=1)
    plt.colorbar(label='Correlation Coefficient')
    
    # Add correlation values to the plot
    for i in range(len(correlation_matrix.columns)):
        for j in range(len(correlation_matrix.columns)):
            plt.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}', 
                    ha='center', va='center', fontsize=12, fontweight='bold')
    
    plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, fontsize=12)
    plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, fontsize=12)
    plt.title('Stock Price Correlation Matrix', fontsize=16)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('results/correlation_matrix.png', dpi=300, bbox_inches='tight')
    print("Correlation matrix saved to results/correlation_matrix.png")
    plt.show()

def plot_volatility_analysis(data):
    """Plot volatility analysis for all symbols"""
    print("Creating volatility analysis...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Calculate daily returns and volatility
    volatilities = {}
    for symbol, df in data.items():
        returns = df['Close'].pct_change().dropna()
        volatilities[symbol] = returns.std() * np.sqrt(252) * 100  # Annualized volatility
        
        # Plot daily returns
        ax1.plot(returns.index, returns, label=symbol, alpha=0.7)
    
    ax1.set_title('Daily Returns', fontsize=14)
    ax1.set_ylabel('Daily Return', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot volatility comparison
    symbols = list(volatilities.keys())
    vol_values = list(volatilities.values())
    
    bars = ax2.bar(symbols, vol_values, color=['blue', 'orange', 'green'])
    ax2.set_title('Annualized Volatility Comparison', fontsize=14)
    ax2.set_ylabel('Volatility (%)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, vol_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('results/volatility_analysis.png', dpi=300, bbox_inches='tight')
    print("Volatility analysis saved to results/volatility_analysis.png")
    plt.show()

def create_summary_statistics(data):
    """Create and display summary statistics"""
    print("Creating summary statistics...")
    
    summary_stats = {}
    
    for symbol, df in data.items():
        stats = {
            'Mean Price': df['Close'].mean(),
            'Std Price': df['Close'].std(),
            'Min Price': df['Close'].min(),
            'Max Price': df['Close'].max(),
            'Total Return': (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100,
            'Annualized Volatility': df['Close'].pct_change().std() * np.sqrt(252) * 100,
            'Sharpe Ratio': (df['Close'].pct_change().mean() / df['Close'].pct_change().std()) * np.sqrt(252)
        }
        summary_stats[symbol] = stats
    
    # Create summary table
    summary_df = pd.DataFrame(summary_stats).T
    print("\nSummary Statistics:")
    print(summary_df.round(2))
    
    # Save to CSV
    summary_df.to_csv('results/summary_statistics.csv')
    print("Summary statistics saved to results/summary_statistics.csv")
    
    return summary_stats

def main():
    """Main function to test visualization system"""
    print("="*60)
    print("SIMPLE VISUALIZATION SYSTEM TEST")
    print("="*60)
    
    # Create sample data
    data = create_sample_stock_data()
    
    # Create various plots
    print("\n" + "="*40)
    print("CREATING VISUALIZATIONS")
    print("="*40)
    
    # 1. Price comparison
    plot_price_comparison(data)
    
    # 2. Technical indicators for AAPL
    plot_technical_indicators(data, 'AAPL')
    
    # 3. Correlation matrix
    plot_correlation_matrix(data)
    
    # 4. Volatility analysis
    plot_volatility_analysis(data)
    
    # 5. Summary statistics
    summary_stats = create_summary_statistics(data)
    
    print("\n" + "="*40)
    print("VISUALIZATION TEST COMPLETE")
    print("="*40)
    print("All charts saved to results/ directory")
    print("Check the generated PNG files to see the visualizations")
    print("Summary statistics saved to results/summary_statistics.csv")

if __name__ == "__main__":
    main() 