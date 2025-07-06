# Test Visualization Integration with GPFA System
"""
Comprehensive test script demonstrating visualization integration with the GPFA prediction system.
"""

import sys
import os
import time
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Any

# Import visualization components
from simple_visualization import create_sample_stock_data, calculate_rsi
from viz_integration import SimpleVizIntegration

# Try to import GPFA components
try:
    from realtime_gpfa_predictor import RealtimeGPFAPredictor
    from real_data_integration import RealDataIntegration
    GPFA_AVAILABLE = True
    print("✓ GPFA components available")
except ImportError as e:
    GPFA_AVAILABLE = False
    print(f"⚠ GPFA components not available: {e}")
    print("Running in visualization-only mode")

class GPFAVisualizationTest:
    """
    Comprehensive test class for GPFA visualization integration
    """
    
    def __init__(self, symbols: List[str]):
        """
        Initialize test system
        
        Args:
            symbols: List of stock symbols to test
        """
        self.symbols = symbols
        self.viz_integration = SimpleVizIntegration(symbols)
        
        # Initialize GPFA components if available
        self.gpfa_predictor = None
        self.data_integration = None
        
        if GPFA_AVAILABLE:
            try:
                self.gpfa_predictor = RealtimeGPFAPredictor(symbols)
                self.data_integration = RealDataIntegration(symbols)
                print("✓ GPFA components initialized successfully")
            except Exception as e:
                print(f"⚠ Could not initialize GPFA components: {e}")
        
        # Test data storage
        self.test_data = {}
        self.prediction_results = {}
        self.visualization_results = {}
        
        print(f"GPFAVisualizationTest initialized for {len(symbols)} symbols")
    
    def test_data_generation(self):
        """Test data generation and preparation"""
        print("\n" + "="*50)
        print("TESTING DATA GENERATION")
        print("="*50)
        
        # Generate sample data
        print("1. Generating sample stock data...")
        self.test_data = create_sample_stock_data()
        
        for symbol, df in self.test_data.items():
            print(f"   {symbol}: {len(df)} data points")
            print(f"   Date range: {df.index[0].date()} to {df.index[-1].date()}")
            print(f"   Price range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")
        
        # Test technical indicators
        print("\n2. Testing technical indicators...")
        for symbol in self.symbols:
            if symbol in self.test_data:
                df = self.test_data[symbol]
                
                # Calculate additional indicators
                df['EMA_12'] = df['Close'].ewm(span=12).mean()
                df['EMA_26'] = df['Close'].ewm(span=26).mean()
                df['MACD'] = df['EMA_12'] - df['EMA_26']
                df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
                df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
                
                print(f"   {symbol}: RSI range {df['RSI'].min():.1f} - {df['RSI'].max():.1f}")
                print(f"   {symbol}: MACD range {df['MACD'].min():.3f} - {df['MACD'].max():.3f}")
        
        print("✓ Data generation test completed")
        return True
    
    def test_prediction_simulation(self):
        """Test prediction simulation"""
        print("\n" + "="*50)
        print("TESTING PREDICTION SIMULATION")
        print("="*50)
        
        if not self.test_data:
            print("No test data available. Running data generation first...")
            self.test_data_generation()
        
        # Simulate predictions
        print("1. Simulating predictions...")
        predictions = {}
        accuracy_metrics = {}
        
        for symbol in self.symbols:
            if symbol in self.test_data:
                df = self.test_data[symbol]
                
                # Create realistic predictions
                last_price = df['Close'].iloc[-1]
                prediction_horizon = 10
                
                # Generate prediction with some randomness
                np.random.seed(42 + self.symbols.index(symbol))
                price_changes = np.random.normal(0.001, 0.02, prediction_horizon)
                predicted_prices = [last_price]
                
                for change in price_changes:
                    predicted_prices.append(predicted_prices[-1] * (1 + change))
                
                predicted_prices = np.array(predicted_prices[1:])  # Remove initial price
                prediction_times = pd.date_range(
                    start=df.index[-1] + timedelta(days=1),
                    periods=prediction_horizon,
                    freq='D'
                )
                
                # Calculate confidence intervals
                confidence_level = 0.95
                std_dev = np.std(df['Close'].pct_change().dropna()) * last_price
                confidence_interval = 1.96 * std_dev  # 95% confidence
                
                predictions[symbol] = {
                    'predicted_prices': predicted_prices,
                    'prediction_times': prediction_times,
                    'confidence_intervals': [
                        predicted_prices - confidence_interval,
                        predicted_prices + confidence_interval
                    ],
                    'prediction_horizon': prediction_horizon,
                    'model_confidence': np.random.uniform(0.7, 0.95),
                    'last_actual_price': last_price
                }
                
                # Calculate accuracy metrics (simulated)
                accuracy_metrics[symbol] = {
                    'accuracy': np.random.uniform(75, 90),
                    'mae': np.random.uniform(1, 5),
                    'rmse': np.random.uniform(2, 6),
                    'mape': np.random.uniform(1, 3),
                    'prediction_std': std_dev,
                    'confidence_level': confidence_level
                }
                
                print(f"   {symbol}: Predicted {prediction_horizon} days ahead")
                print(f"   {symbol}: Price range ${predicted_prices.min():.2f} - ${predicted_prices.max():.2f}")
                print(f"   {symbol}: Confidence interval ±${confidence_interval:.2f}")
        
        self.prediction_results = {
            'predictions': predictions,
            'accuracy_metrics': accuracy_metrics,
            'timestamp': datetime.now()
        }
        
        print("✓ Prediction simulation completed")
        return True
    
    def test_visualization_creation(self):
        """Test visualization creation"""
        print("\n" + "="*50)
        print("TESTING VISUALIZATION CREATION")
        print("="*50)
        
        if not self.test_data:
            print("No test data available. Running data generation first...")
            self.test_data_generation()
        
        if not self.prediction_results:
            print("No prediction results available. Running prediction simulation first...")
            self.test_prediction_simulation()
        
        # Create different types of visualizations
        print("1. Creating basic dashboard...")
        basic_dashboard = self.viz_integration.create_dashboard(
            self.test_data, 
            self.prediction_results['predictions']
        )
        self.viz_integration.save_dashboard(basic_dashboard, "test_basic_dashboard.html")
        
        print("2. Creating advanced dashboard...")
        advanced_dashboard = self.create_advanced_dashboard()
        self.save_plotly_figure(advanced_dashboard, "test_advanced_dashboard.html")
        
        print("3. Creating prediction analysis...")
        prediction_analysis = self.create_prediction_analysis()
        self.save_plotly_figure(prediction_analysis, "test_prediction_analysis.html")
        
        print("4. Creating market overview...")
        market_overview = self.create_market_overview()
        self.save_plotly_figure(market_overview, "test_market_overview.html")
        
        print("✓ Visualization creation completed")
        return True
    
    def create_advanced_dashboard(self) -> go.Figure:
        """Create advanced dashboard with multiple components"""
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Price Predictions with Confidence', 'Technical Indicators',
                'Prediction Accuracy Metrics', 'Market Correlation Matrix',
                'Volatility Analysis', 'Model Performance Summary'
            ],
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "heatmap"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # 1. Price Predictions with Confidence Intervals
        for i, symbol in enumerate(self.symbols):
            if symbol in self.test_data and symbol in self.prediction_results['predictions']:
                df = self.test_data[symbol]
                pred_data = self.prediction_results['predictions'][symbol]
                
                # Actual prices
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['Close'],
                        mode='lines',
                        name=f'{symbol} Actual',
                        line=dict(color=self.viz_integration.colors[i % len(self.viz_integration.colors)]),
                        showlegend=True
                    ),
                    row=1, col=1
                )
                
                # Predicted prices
                fig.add_trace(
                    go.Scatter(
                        x=pred_data['prediction_times'],
                        y=pred_data['predicted_prices'],
                        mode='lines+markers',
                        name=f'{symbol} Predicted',
                        line=dict(color=self.viz_integration.colors[i % len(self.viz_integration.colors)], dash='dash'),
                        marker=dict(size=4),
                        showlegend=False
                    ),
                    row=1, col=1
                )
                
                # Confidence intervals
                ci = pred_data['confidence_intervals']
                fig.add_trace(
                    go.Scatter(
                        x=pred_data['prediction_times'],
                        y=ci[1],  # Upper bound
                        mode='lines',
                        line=dict(width=0),
                        showlegend=False,
                        fillcolor='rgba(0,100,80,0.2)',
                        fill='tonexty'
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=pred_data['prediction_times'],
                        y=ci[0],  # Lower bound
                        mode='lines',
                        line=dict(width=0),
                        showlegend=False,
                        fillcolor='rgba(0,100,80,0.2)',
                        fill='tonexty'
                    ),
                    row=1, col=1
                )
        
        # 2. Technical Indicators
        for i, symbol in enumerate(self.symbols):
            if symbol in self.test_data:
                df = self.test_data[symbol]
                if 'MACD' in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=df['MACD'],
                            mode='lines',
                            name=f'{symbol} MACD',
                            line=dict(color=self.viz_integration.colors[i % len(self.viz_integration.colors)]),
                            showlegend=False
                        ),
                        row=1, col=2
                    )
        
        # 3. Prediction Accuracy Metrics
        if self.prediction_results['accuracy_metrics']:
            metrics = self.prediction_results['accuracy_metrics']
            symbols = list(metrics.keys())
            accuracies = [metrics[symbol]['accuracy'] for symbol in symbols]
            
            fig.add_trace(
                go.Bar(
                    x=symbols,
                    y=accuracies,
                    name='Accuracy %',
                    marker_color=self.viz_integration.colors[:len(symbols)]
                ),
                row=2, col=1
            )
        
        # 4. Market Correlation Matrix
        price_data = pd.DataFrame()
        for symbol in self.symbols:
            if symbol in self.test_data:
                price_data[symbol] = self.test_data[symbol]['Close']
        
        if len(price_data.columns) > 1:
            correlation_matrix = price_data.corr()
            fig.add_trace(
                go.Heatmap(
                    z=correlation_matrix.values,
                    x=correlation_matrix.columns,
                    y=correlation_matrix.columns,
                    colorscale='RdBu',
                    zmid=0
                ),
                row=2, col=2
            )
        
        # 5. Volatility Analysis
        volatilities = []
        for symbol in self.symbols:
            if symbol in self.test_data:
                df = self.test_data[symbol]
                returns = df['Close'].pct_change().dropna()
                volatilities.append(returns.std() * np.sqrt(252) * 100)
        
        fig.add_trace(
            go.Bar(
                x=self.symbols,
                y=volatilities,
                name='Annualized Volatility (%)',
                marker_color='orange'
            ),
            row=3, col=1
        )
        
        # 6. Model Performance Summary
        if self.prediction_results['accuracy_metrics']:
            metrics = self.prediction_results['accuracy_metrics']
            avg_accuracy = np.mean([metrics[symbol]['accuracy'] for symbol in self.symbols])
            avg_mae = np.mean([metrics[symbol]['mae'] for symbol in self.symbols])
            
            performance_metrics = ['Avg Accuracy', 'Avg MAE', 'Active Symbols']
            performance_values = [avg_accuracy, avg_mae, len(self.symbols)]
            
            fig.add_trace(
                go.Bar(
                    x=performance_metrics,
                    y=performance_values,
                    name='Performance Metrics',
                    marker_color='purple'
                ),
                row=3, col=2
            )
        
        fig.update_layout(
            title='Advanced GPFA Trading Dashboard',
            height=1200,
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig
    
    def create_prediction_analysis(self) -> go.Figure:
        """Create detailed prediction analysis"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Prediction vs Actual Comparison', 'Prediction Errors Distribution', 
                           'Confidence Interval Analysis', 'Prediction Horizon Performance'],
            specs=[[{"type": "scatter"}, {"type": "histogram"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # 1. Prediction vs Actual Comparison
        for symbol in self.symbols:
            if symbol in self.prediction_results['predictions']:
                pred_data = self.prediction_results['predictions'][symbol]
                actual_price = pred_data['last_actual_price']
                predicted_price = pred_data['predicted_prices'][0]
                
                fig.add_trace(
                    go.Scatter(
                        x=[actual_price],
                        y=[predicted_price],
                        mode='markers',
                        name=f'{symbol} Prediction',
                        marker=dict(
                            color=self.viz_integration.colors[self.symbols.index(symbol) % len(self.viz_integration.colors)],
                            size=10
                        )
                    ),
                    row=1, col=1
                )
        
        # 2. Prediction Errors Distribution
        all_errors = []
        for symbol in self.symbols:
            if symbol in self.prediction_results['predictions']:
                pred_data = self.prediction_results['predictions'][symbol]
                actual_price = pred_data['last_actual_price']
                predicted_price = pred_data['predicted_prices'][0]
                error = (predicted_price - actual_price) / actual_price * 100
                all_errors.append(error)
        
        if all_errors:
            fig.add_trace(
                go.Histogram(
                    x=all_errors,
                    name='Prediction Errors (%)',
                    nbinsx=10,
                    marker_color='lightblue'
                ),
                row=1, col=2
            )
        
        # 3. Confidence Interval Analysis
        for symbol in self.symbols:
            if symbol in self.prediction_results['predictions']:
                pred_data = self.prediction_results['predictions'][symbol]
                ci = pred_data['confidence_intervals']
                times = pred_data['prediction_times']
                
                fig.add_trace(
                    go.Scatter(
                        x=times,
                        y=ci[1],  # Upper bound
                        mode='lines',
                        name=f'{symbol} Upper CI',
                        line=dict(color=self.viz_integration.colors[self.symbols.index(symbol) % len(self.viz_integration.colors)]),
                        showlegend=False
                    ),
                    row=2, col=1
                )
        
        # 4. Prediction Horizon Performance
        horizons = ['1 day', '2 days', '5 days', '10 days']
        horizon_accuracies = [85, 82, 78, 75]  # Example values
        
        fig.add_trace(
            go.Bar(
                x=horizons,
                y=horizon_accuracies,
                name='Horizon Accuracy',
                marker_color='green'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title='GPFA Prediction Analysis',
            height=800,
            showlegend=True
        )
        
        return fig
    
    def create_market_overview(self) -> go.Figure:
        """Create market overview dashboard"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Price Performance Comparison', 'Volume Analysis',
                           'Risk Metrics', 'Market Trends'],
            specs=[[{"type": "scatter"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # 1. Price Performance Comparison
        for i, symbol in enumerate(self.symbols):
            if symbol in self.test_data:
                df = self.test_data[symbol]
                normalized_prices = df['Close'] / df['Close'].iloc[0] * 100
                
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=normalized_prices,
                        mode='lines',
                        name=symbol,
                        line=dict(color=self.viz_integration.colors[i % len(self.viz_integration.colors)])
                    ),
                    row=1, col=1
                )
        
        # 2. Volume Analysis
        volumes = []
        for symbol in self.symbols:
            if symbol in self.test_data:
                volumes.append(self.test_data[symbol]['Volume'].mean())
        
        fig.add_trace(
            go.Bar(
                x=self.symbols,
                y=volumes,
                name='Average Volume',
                marker_color=self.viz_integration.colors[:len(self.symbols)]
            ),
            row=1, col=2
        )
        
        # 3. Risk Metrics
        risk_metrics = ['Sharpe Ratio', 'Max Drawdown', 'VaR (95%)', 'Beta']
        risk_values = [1.2, -0.15, -0.02, 1.1]  # Example values
        
        fig.add_trace(
            go.Bar(
                x=risk_metrics,
                y=risk_values,
                name='Risk Metrics',
                marker_color='red'
            ),
            row=2, col=1
        )
        
        # 4. Market Trends
        for i, symbol in enumerate(self.symbols):
            if symbol in self.test_data:
                df = self.test_data[symbol]
                if 'SMA_20' in df.columns and 'SMA_50' in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=df['SMA_20'],
                            mode='lines',
                            name=f'{symbol} SMA20',
                            line=dict(color=self.viz_integration.colors[i % len(self.viz_integration.colors)])
                        ),
                        row=2, col=2
                    )
        
        fig.update_layout(
            title='Market Overview Dashboard',
            height=800,
            showlegend=True
        )
        
        return fig
    
    def save_plotly_figure(self, fig: go.Figure, filename: str):
        """Save Plotly figure as HTML"""
        fig.write_html(f"results/{filename}")
        print(f"   Saved: results/{filename}")
    
    def run_comprehensive_test(self):
        """Run comprehensive test of all components"""
        print("="*60)
        print("COMPREHENSIVE GPFA VISUALIZATION INTEGRATION TEST")
        print("="*60)
        
        try:
            # Test data generation
            if not self.test_data_generation():
                print("❌ Data generation test failed")
                return False
            
            # Test prediction simulation
            if not self.test_prediction_simulation():
                print("❌ Prediction simulation test failed")
                return False
            
            # Test visualization creation
            if not self.test_visualization_creation():
                print("❌ Visualization creation test failed")
                return False
            
            print("\n" + "="*60)
            print("✅ ALL TESTS COMPLETED SUCCESSFULLY")
            print("="*60)
            print("\nGenerated files:")
            print("- results/test_basic_dashboard.html")
            print("- results/test_advanced_dashboard.html")
            print("- results/test_prediction_analysis.html")
            print("- results/test_market_overview.html")
            print("\nOpen these files in your browser to view the visualizations")
            
            return True
            
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            return False

def main():
    """Main function to run comprehensive test"""
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    
    # Create test instance
    test_system = GPFAVisualizationTest(symbols)
    
    # Run comprehensive test
    success = test_system.run_comprehensive_test()
    
    if success:
        print("\n🎯 Visualization system is ready for integration with your GPFA model!")
        print("\nNext steps:")
        print("1. Install visualization requirements: pip install -r visualization_requirements.txt")
        print("2. Integrate with your actual GPFA prediction model")
        print("3. Set up real-time data feeds")
        print("4. Deploy interactive dashboards")
    else:
        print("\n❌ Test failed. Please check the error messages above.")

if __name__ == "__main__":
    main() 