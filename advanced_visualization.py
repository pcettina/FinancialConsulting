# Advanced Visualization System
"""
Advanced visualization system for real-time GPFA prediction data and modeling.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class GPFAVisualizer:
    """
    Advanced visualization system for GPFA predictions and market data
    """
    
    def __init__(self, symbols: List[str]):
        """
        Initialize GPFA visualizer
        
        Args:
            symbols: List of stock symbols to visualize
        """
        self.symbols = symbols
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        # Set up matplotlib style
        plt.style.use('seaborn-v0_8')
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        
        print(f"Initialized GPFAVisualizer for {len(symbols)} symbols")
    
    def create_real_time_dashboard(self, data: Dict[str, pd.DataFrame], 
                                 predictions: Optional[Dict] = None,
                                 accuracy_metrics: Optional[Dict] = None) -> go.Figure:
        """
        Create comprehensive real-time dashboard
        
        Args:
            data: Dictionary of market data for each symbol
            predictions: Optional predictions data
            accuracy_metrics: Optional accuracy metrics
            
        Returns:
            Plotly figure object
        """
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Real-Time Price Predictions', 'Technical Indicators',
                'Prediction Accuracy', 'Market Correlation',
                'Volatility Analysis', 'GPFA Model Performance'
            ],
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "heatmap"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # 1. Real-Time Price Predictions
        for i, symbol in enumerate(self.symbols):
            if symbol in data:
                df = data[symbol]
                
                # Actual prices
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['Close'],
                        mode='lines',
                        name=f'{symbol} Actual',
                        line=dict(color=self.colors[i % len(self.colors)]),
                        showlegend=True
                    ),
                    row=1, col=1
                )
                
                # Predictions if available
                if predictions and symbol in predictions:
                    pred_data = predictions[symbol]
                    if 'predicted_prices' in pred_data:
                        pred_prices = pred_data['predicted_prices']
                        pred_times = pred_data.get('prediction_times', df.index[-len(pred_prices):])
                        
                        fig.add_trace(
                            go.Scatter(
                                x=pred_times,
                                y=pred_prices,
                                mode='lines+markers',
                                name=f'{symbol} Predicted',
                                line=dict(color=self.colors[i % len(self.colors)], dash='dash'),
                                marker=dict(size=4),
                                showlegend=False
                            ),
                            row=1, col=1
                        )
        
        # 2. Technical Indicators
        for i, symbol in enumerate(self.symbols):
            if symbol in data:
                df = data[symbol]
                if 'RSI' in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=df['RSI'],
                            mode='lines',
                            name=f'{symbol} RSI',
                            line=dict(color=self.colors[i % len(self.colors)]),
                            showlegend=False
                        ),
                        row=1, col=2
                    )
        
        # Add RSI reference lines
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=2)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=2)
        
        # 3. Prediction Accuracy
        if accuracy_metrics:
            symbols_acc = list(accuracy_metrics.keys())
            accuracies = [accuracy_metrics[symbol].get('accuracy', 0) for symbol in symbols_acc]
            
            fig.add_trace(
                go.Bar(
                    x=symbols_acc,
                    y=accuracies,
                    name='Accuracy %',
                    marker_color=self.colors[:len(symbols_acc)]
                ),
                row=2, col=1
            )
        
        # 4. Market Correlation
        price_data = pd.DataFrame()
        for symbol in self.symbols:
            if symbol in data:
                price_data[symbol] = data[symbol]['Close']
        
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
            if symbol in data:
                returns = data[symbol]['Close'].pct_change().dropna()
                volatilities.append(returns.std() * np.sqrt(252) * 100)
        
        fig.add_trace(
            go.Bar(
                x=self.symbols,
                y=volatilities,
                name='Volatility (%)',
                marker_color='orange'
            ),
            row=3, col=1
        )
        
        # 6. GPFA Model Performance (example metrics)
        model_metrics = ['Latent Factors', 'Convergence', 'Prediction Horizon']
        metric_values = [3, 95.2, 5]  # Example values
        
        fig.add_trace(
            go.Bar(
                x=model_metrics,
                y=metric_values,
                name='Model Metrics',
                marker_color='purple'
            ),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            title='Real-Time GPFA Trading Dashboard',
            height=1200,
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig
    
    def create_prediction_analysis(self, data: Dict[str, pd.DataFrame], 
                                 predictions: Dict) -> go.Figure:
        """
        Create detailed prediction analysis visualization
        
        Args:
            data: Dictionary of market data
            predictions: Dictionary of predictions
            
        Returns:
            Plotly figure object
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Prediction vs Actual', 'Prediction Errors', 
                           'Confidence Intervals', 'Prediction Horizon Performance'],
            specs=[[{"type": "scatter"}, {"type": "histogram"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # 1. Prediction vs Actual
        for symbol in self.symbols:
            if symbol in data and symbol in predictions:
                df = data[symbol]
                pred_data = predictions[symbol]
                
                if 'predicted_prices' in pred_data:
                    pred_prices = pred_data['predicted_prices']
                    pred_times = pred_data.get('prediction_times', df.index[-len(pred_prices):])
                    
                    # Actual prices for comparison period
                    actual_prices = df.loc[pred_times, 'Close']
                    
                    fig.add_trace(
                        go.Scatter(
                            x=actual_prices,
                            y=pred_prices,
                            mode='markers',
                            name=f'{symbol} Predictions',
                            marker=dict(color=self.colors[self.symbols.index(symbol) % len(self.colors)])
                        ),
                        row=1, col=1
                    )
        
        # Add perfect prediction line
        min_val = min([data[symbol]['Close'].min() for symbol in self.symbols if symbol in data])
        max_val = max([data[symbol]['Close'].max() for symbol in self.symbols if symbol in data])
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='black', dash='dash'),
                showlegend=False
            ),
            row=1, col=1
        )
        
        # 2. Prediction Errors
        all_errors = []
        for symbol in self.symbols:
            if symbol in data and symbol in predictions:
                df = data[symbol]
                pred_data = predictions[symbol]
                
                if 'predicted_prices' in pred_data:
                    pred_prices = pred_data['predicted_prices']
                    pred_times = pred_data.get('prediction_times', df.index[-len(pred_prices):])
                    actual_prices = df.loc[pred_times, 'Close']
                    
                    errors = (pred_prices - actual_prices) / actual_prices * 100
                    all_errors.extend(errors)
        
        if all_errors:
            fig.add_trace(
                go.Histogram(
                    x=all_errors,
                    name='Prediction Errors (%)',
                    nbinsx=20,
                    marker_color='lightblue'
                ),
                row=1, col=2
            )
        
        # 3. Confidence Intervals
        for symbol in self.symbols:
            if symbol in data and symbol in predictions:
                df = data[symbol]
                pred_data = predictions[symbol]
                
                if 'confidence_intervals' in pred_data:
                    ci = pred_data['confidence_intervals']
                    pred_times = pred_data.get('prediction_times', df.index[-len(ci[0]):])
                    
                    fig.add_trace(
                        go.Scatter(
                            x=pred_times,
                            y=ci[1],  # Upper bound
                            mode='lines',
                            line=dict(width=0),
                            showlegend=False,
                            fillcolor='rgba(0,100,80,0.2)',
                            fill='tonexty'
                        ),
                        row=2, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=pred_times,
                            y=ci[0],  # Lower bound
                            mode='lines',
                            line=dict(width=0),
                            showlegend=False,
                            fillcolor='rgba(0,100,80,0.2)',
                            fill='tonexty'
                        ),
                        row=2, col=1
                    )
        
        # 4. Prediction Horizon Performance
        horizons = ['1h', '2h', '5h', '10h', '20h']
        horizon_accuracies = [85, 82, 78, 75, 70]  # Example values
        
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
    
    def save_dashboard_html(self, fig: go.Figure, filename: str):
        """
        Save dashboard as interactive HTML file
        
        Args:
            fig: Plotly figure object
            filename: Output filename
        """
        fig.write_html(f"results/{filename}")
        print(f"Dashboard saved to results/{filename}")

def create_sample_gpfa_data():
    """Create sample GPFA data for testing"""
    return {
        'latent_factors': [
            np.random.randn(100).cumsum(),
            np.random.randn(100).cumsum() * 0.5,
            np.random.randn(100).cumsum() * 0.3
        ],
        'factor_loadings': np.random.randn(3, 3),
        'convergence_history': np.random.randn(50).cumsum(),
        'prediction_uncertainty': np.random.uniform(0.1, 0.3, 20)
    }

def main():
    """Main function to test advanced visualization system"""
    print("="*60)
    print("ADVANCED GPFA VISUALIZATION SYSTEM TEST")
    print("="*60)
    
    # Import the simple visualization to get sample data
    from simple_visualization import create_sample_stock_data
    
    # Create sample data
    data = create_sample_stock_data()
    
    # Initialize visualizer
    visualizer = GPFAVisualizer(['AAPL', 'GOOGL', 'MSFT'])
    
    # Create sample predictions
    predictions = {}
    for symbol in ['AAPL', 'GOOGL', 'MSFT']:
        if symbol in data:
            df = data[symbol]
            predictions[symbol] = {
                'predicted_prices': df['Close'].iloc[-10:].values * (1 + np.random.normal(0, 0.02, 10)),
                'prediction_times': df.index[-10:],
                'confidence_intervals': [
                    df['Close'].iloc[-10:].values * 0.95,
                    df['Close'].iloc[-10:].values * 1.05
                ]
            }
    
    # Create sample accuracy metrics
    accuracy_metrics = {
        'AAPL': {'accuracy': 85.5, 'mae': 2.3, 'rmse': 3.1},
        'GOOGL': {'accuracy': 82.1, 'mae': 3.2, 'rmse': 4.1},
        'MSFT': {'accuracy': 88.3, 'mae': 1.9, 'rmse': 2.5}
    }
    
    print("\nCreating advanced visualizations...")
    
    # 1. Real-time dashboard
    dashboard_fig = visualizer.create_real_time_dashboard(data, predictions, accuracy_metrics)
    visualizer.save_dashboard_html(dashboard_fig, 'realtime_dashboard.html')
    
    # 2. Prediction analysis
    prediction_fig = visualizer.create_prediction_analysis(data, predictions)
    visualizer.save_dashboard_html(prediction_fig, 'prediction_analysis.html')
    
    print("\n" + "="*40)
    print("ADVANCED VISUALIZATION TEST COMPLETE")
    print("="*40)
    print("Interactive dashboards saved to results/ directory:")
    print("- realtime_dashboard.html")
    print("- prediction_analysis.html")
    print("\nOpen these files in your browser for interactive visualizations")

if __name__ == "__main__":
    main() 