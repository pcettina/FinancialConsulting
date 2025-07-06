# Simple Visualization Integration
"""
Simple integration system for GPFA predictions and visualization.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List

class SimpleVizIntegration:
    """
    Simple visualization integration for GPFA predictions
    """
    
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        print(f"SimpleVizIntegration initialized for {len(symbols)} symbols")
    
    def create_sample_data(self):
        """Create sample data for testing"""
        from simple_visualization import create_sample_stock_data
        return create_sample_stock_data()
    
    def create_sample_predictions(self, data):
        """Create sample predictions"""
        predictions = {}
        for symbol in self.symbols:
            if symbol in data:
                df = data[symbol]
                pred_prices = df['Close'].iloc[-10:].values * (1 + np.random.normal(0, 0.02, 10))
                pred_times = df.index[-10:]
                
                predictions[symbol] = {
                    'predicted_prices': pred_prices,
                    'prediction_times': pred_times,
                    'confidence_intervals': [
                        pred_prices * 0.95,
                        pred_prices * 1.05
                    ]
                }
        return predictions
    
    def create_dashboard(self, data, predictions=None):
        """Create simple dashboard"""
        if predictions is None:
            predictions = self.create_sample_predictions(data)
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Price Predictions', 'Technical Indicators', 
                           'Prediction Accuracy', 'Market Overview'],
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "heatmap"}]]
        )
        
        # Price predictions
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
                        line=dict(color=self.colors[i % len(self.colors)])
                    ),
                    row=1, col=1
                )
                
                # Predictions
                if symbol in predictions:
                    pred_data = predictions[symbol]
                    fig.add_trace(
                        go.Scatter(
                            x=pred_data['prediction_times'],
                            y=pred_data['predicted_prices'],
                            mode='lines+markers',
                            name=f'{symbol} Predicted',
                            line=dict(color=self.colors[i % len(self.colors)], dash='dash')
                        ),
                        row=1, col=1
                    )
        
        # Technical indicators
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
                            line=dict(color=self.colors[i % len(self.colors)])
                        ),
                        row=1, col=2
                    )
        
        # Add RSI reference lines
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=2)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=2)
        
        # Prediction accuracy (sample)
        accuracies = [85, 82, 88]
        fig.add_trace(
            go.Bar(
                x=self.symbols,
                y=accuracies,
                name='Accuracy %',
                marker_color=self.colors[:len(self.symbols)]
            ),
            row=2, col=1
        )
        
        # Market correlation
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
        
        fig.update_layout(
            title='GPFA Trading Dashboard',
            height=800,
            showlegend=True
        )
        
        return fig
    
    def save_dashboard(self, fig, filename):
        """Save dashboard as HTML"""
        fig.write_html(f"results/{filename}")
        print(f"Dashboard saved to results/{filename}")

def main():
    """Test the simple visualization integration"""
    print("="*50)
    print("SIMPLE VISUALIZATION INTEGRATION TEST")
    print("="*50)
    
    # Initialize
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    viz = SimpleVizIntegration(symbols)
    
    # Create sample data
    print("Creating sample data...")
    data = viz.create_sample_data()
    
    # Create dashboard
    print("Creating dashboard...")
    dashboard = viz.create_dashboard(data)
    
    # Save dashboard
    viz.save_dashboard(dashboard, "simple_integration_dashboard.html")
    
    print("\nTest completed successfully!")
    print("Open results/simple_integration_dashboard.html in your browser")

if __name__ == "__main__":
    main() 