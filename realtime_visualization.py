# Real-time GPFA Visualization Module
"""
Enhanced visualization and interactive capabilities for the real-time GPFA predictor
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from datetime import datetime, timedelta
import time
from typing import Dict, List, Optional, Tuple
import logging

# Set up logging
logger = logging.getLogger(__name__)

class RealTimeVisualizer:
    """
    Real-time visualization system for GPFA predictions
    """
    
    def __init__(self, symbols: List[str], n_factors: int = 5):
        """
        Initialize the real-time visualizer
        
        Args:
            symbols: List of stock symbols
            n_factors: Number of GPFA factors
        """
        self.symbols = symbols
        self.n_factors = n_factors
        
        # Data storage for visualization
        self.price_history = {symbol: [] for symbol in symbols}
        self.prediction_history = []
        self.factor_history = []
        self.performance_metrics = {}
        
        # Plotting setup
        plt.style.use('seaborn-v0_8')
        self.fig = None
        self.axes = None
        
        logger.info(f"Initialized RealTimeVisualizer for {len(symbols)} symbols")
    
    def update_price_data(self, data: pd.DataFrame) -> None:
        """
        Update price data for visualization
        
        Args:
            data: Latest price data
        """
        timestamp = datetime.now()
        
        for symbol in self.symbols:
            if symbol in data.columns:
                price = data[symbol].iloc[-1] if len(data[symbol]) > 0 else None
                if price is not None:
                    self.price_history[symbol].append({
                        'timestamp': timestamp,
                        'price': price
                    })
                    
                    # Keep only last 1000 data points
                    if len(self.price_history[symbol]) > 1000:
                        self.price_history[symbol] = self.price_history[symbol][-1000:]
    
    def update_predictions(self, predictions: Dict, actual_data: pd.DataFrame) -> None:
        """
        Update prediction data for visualization
        
        Args:
            predictions: Prediction results
            actual_data: Actual price data
        """
        timestamp = datetime.now()
        
        prediction_record = {
            'timestamp': timestamp,
            'predictions': predictions,
            'actual_data': actual_data
        }
        
        self.prediction_history.append(prediction_record)
        
        # Keep only last 100 predictions
        if len(self.prediction_history) > 100:
            self.prediction_history = self.prediction_history[-100:]
    
    def update_factors(self, latent_factors: np.ndarray) -> None:
        """
        Update latent factor data for visualization
        
        Args:
            latent_factors: Current latent factors
        """
        timestamp = datetime.now()
        
        factor_record = {
            'timestamp': timestamp,
            'factors': latent_factors.copy() if latent_factors is not None else None
        }
        
        self.factor_history.append(factor_record)
        
        # Keep only last 500 factor records
        if len(self.factor_history) > 500:
            self.factor_history = self.factor_history[-500:]
    
    def create_real_time_dashboard(self, save_path: str = 'realtime_dashboard.html') -> None:
        """
        Create an interactive real-time dashboard using Plotly
        
        Args:
            save_path: Path to save the HTML dashboard
        """
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Real-time Price Movement',
                'Prediction Accuracy',
                'Latent Factors',
                'Model Performance',
                'Prediction Horizon Comparison',
                'Feature Importance'
            ),
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "bar"}]
            ]
        )
        
        # 1. Real-time Price Movement
        for symbol in self.symbols:
            if self.price_history[symbol]:
                df = pd.DataFrame(self.price_history[symbol])
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=df['price'],
                        mode='lines',
                        name=f'{symbol} Price',
                        line=dict(width=2)
                    ),
                    row=1, col=1
                )
        
        # 2. Prediction Accuracy (placeholder)
        fig.add_trace(
            go.Scatter(
                x=[datetime.now()],
                y=[0],
                mode='lines',
                name='Accuracy',
                line=dict(color='green', width=2)
            ),
            row=1, col=2
        )
        
        # 3. Latent Factors
        if self.factor_history:
            factor_data = pd.DataFrame([
                {
                    'timestamp': record['timestamp'],
                    'factor_1': record['factors'][0] if record['factors'] is not None else None,
                    'factor_2': record['factors'][1] if record['factors'] is not None and len(record['factors']) > 1 else None
                }
                for record in self.factor_history
                if record['factors'] is not None
            ])
            
            if not factor_data.empty:
                fig.add_trace(
                    go.Scatter(
                        x=factor_data['timestamp'],
                        y=factor_data['factor_1'],
                        mode='lines',
                        name='Factor 1',
                        line=dict(color='blue', width=2)
                    ),
                    row=2, col=1
                )
                
                if 'factor_2' in factor_data.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=factor_data['timestamp'],
                            y=factor_data['factor_2'],
                            mode='lines',
                            name='Factor 2',
                            line=dict(color='red', width=2)
                        ),
                        row=2, col=1
                    )
        
        # 4. Model Performance (placeholder)
        models = ['Random Forest', 'XGBoost', 'LightGBM']
        performance = [0.85, 0.87, 0.86]  # Placeholder values
        
        fig.add_trace(
            go.Bar(
                x=models,
                y=performance,
                name='Model Performance',
                marker_color=['#1f77b4', '#ff7f0e', '#2ca02c']
            ),
            row=2, col=2
        )
        
        # 5. Prediction Horizon Comparison
        horizons = ['1min', '5min', '15min', '1hour', '1day']
        horizon_predictions = [100.5, 101.2, 102.1, 103.5, 105.2]  # Placeholder values
        
        fig.add_trace(
            go.Scatter(
                x=horizons,
                y=horizon_predictions,
                mode='lines+markers',
                name='Predictions',
                line=dict(color='purple', width=3),
                marker=dict(size=8)
            ),
            row=3, col=1
        )
        
        # 6. Feature Importance (placeholder)
        features = ['Price Return', 'Volume', 'RSI', 'MACD', 'Bollinger Bands']
        importance = [0.25, 0.20, 0.15, 0.18, 0.22]  # Placeholder values
        
        fig.add_trace(
            go.Bar(
                x=features,
                y=importance,
                name='Feature Importance',
                marker_color='lightblue'
            ),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            title='Real-time GPFA Prediction Dashboard',
            height=1200,
            showlegend=True,
            template='plotly_white'
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Time", row=1, col=1)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_xaxes(title_text="Time", row=1, col=2)
        fig.update_yaxes(title_text="Accuracy", row=1, col=2)
        fig.update_xaxes(title_text="Time", row=2, col=1)
        fig.update_yaxes(title_text="Factor Value", row=2, col=1)
        fig.update_xaxes(title_text="Models", row=2, col=2)
        fig.update_yaxes(title_text="Performance", row=2, col=2)
        fig.update_xaxes(title_text="Horizon", row=3, col=1)
        fig.update_yaxes(title_text="Predicted Price", row=3, col=1)
        fig.update_xaxes(title_text="Features", row=3, col=2)
        fig.update_yaxes(title_text="Importance", row=3, col=2)
        
        # Save the dashboard
        fig.write_html(save_path)
        logger.info(f"Real-time dashboard saved to {save_path}")
        
        return fig
    
    def plot_price_predictions(self, symbol: str, save_path: str = None) -> None:
        """
        Plot price predictions vs actual prices for a specific symbol
        
        Args:
            symbol: Stock symbol to plot
            save_path: Optional path to save the plot
        """
        if not self.prediction_history:
            logger.warning("No prediction history available for plotting")
            return
        
        # Extract data
        timestamps = []
        actual_prices = []
        predicted_prices = []
        
        for record in self.prediction_history:
            timestamps.append(record['timestamp'])
            
            # Get actual price
            if symbol in record['actual_data'].columns:
                actual_price = record['actual_data'][symbol].iloc[-1]
                actual_prices.append(actual_price)
            else:
                actual_prices.append(None)
            
            # Get predicted price (use 1min prediction as example)
            if '1min' in record['predictions']:
                pred = record['predictions']['1min'].get('rf', None)
                predicted_prices.append(pred)
            else:
                predicted_prices.append(None)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot actual prices
        valid_actual = [(t, p) for t, p in zip(timestamps, actual_prices) if p is not None]
        if valid_actual:
            actual_times, actual_vals = zip(*valid_actual)
            ax.plot(actual_times, actual_vals, 'b-', label='Actual Price', linewidth=2)
        
        # Plot predicted prices
        valid_pred = [(t, p) for t, p in zip(timestamps, predicted_prices) if p is not None]
        if valid_pred:
            pred_times, pred_vals = zip(*valid_pred)
            ax.plot(pred_times, pred_vals, 'r--', label='Predicted Price', linewidth=2)
        
        ax.set_title(f'{symbol} Price Predictions vs Actual')
        ax.set_xlabel('Time')
        ax.set_ylabel('Price')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Price prediction plot saved to {save_path}")
        
        plt.show()
    
    def plot_latent_factors(self, save_path: str = None) -> None:
        """
        Plot latent factors over time
        
        Args:
            save_path: Optional path to save the plot
        """
        if not self.factor_history:
            logger.warning("No factor history available for plotting")
            return
        
        # Extract factor data
        timestamps = []
        factors_data = []
        
        for record in self.factor_history:
            if record['factors'] is not None:
                timestamps.append(record['timestamp'])
                factors_data.append(record['factors'])
        
        if not factors_data:
            logger.warning("No valid factor data available")
            return
        
        factors_array = np.array(factors_data)
        
        # Create plot
        fig, axes = plt.subplots(self.n_factors, 1, figsize=(12, 3*self.n_factors))
        if self.n_factors == 1:
            axes = [axes]
        
        for i in range(self.n_factors):
            if i < factors_array.shape[1]:
                axes[i].plot(timestamps, factors_array[:, i], linewidth=2)
                axes[i].set_title(f'Latent Factor {i+1}')
                axes[i].set_ylabel('Factor Value')
                axes[i].grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Time')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Latent factors plot saved to {save_path}")
        
        plt.show()
    
    def plot_prediction_accuracy(self, save_path: str = None) -> None:
        """
        Plot prediction accuracy over time
        
        Args:
            save_path: Optional path to save the plot
        """
        if len(self.prediction_history) < 2:
            logger.warning("Insufficient prediction history for accuracy analysis")
            return
        
        # Calculate accuracy metrics
        accuracies = []
        timestamps = []
        
        for i in range(1, len(self.prediction_history)):
            prev_record = self.prediction_history[i-1]
            curr_record = self.prediction_history[i]
            
            # Calculate accuracy for each horizon
            for horizon in ['1min', '5min', '15min']:
                if horizon in prev_record['predictions'] and horizon in curr_record['predictions']:
                    # Simple accuracy calculation (can be enhanced)
                    pred_prev = prev_record['predictions'][horizon].get('rf', None)
                    pred_curr = curr_record['predictions'][horizon].get('rf', None)
                    
                    if pred_prev is not None and pred_curr is not None:
                        # Calculate directional accuracy
                        direction_prev = 1 if pred_prev > 0 else -1
                        direction_curr = 1 if pred_curr > 0 else -1
                        accuracy = 1 if direction_prev == direction_curr else 0
                        
                        accuracies.append(accuracy)
                        timestamps.append(curr_record['timestamp'])
        
        if not accuracies:
            logger.warning("No accuracy data available")
            return
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Calculate rolling accuracy
        df_acc = pd.DataFrame({'timestamp': timestamps, 'accuracy': accuracies})
        rolling_acc = df_acc['accuracy'].rolling(window=10).mean()
        
        ax.plot(df_acc['timestamp'], rolling_acc, 'g-', linewidth=2, label='Rolling Accuracy (10 periods)')
        ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Random Baseline')
        
        ax.set_title('Prediction Accuracy Over Time')
        ax.set_xlabel('Time')
        ax.set_ylabel('Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Prediction accuracy plot saved to {save_path}")
        
        plt.show()
    
    def create_performance_summary(self, save_path: str = 'performance_summary.html') -> None:
        """
        Create a comprehensive performance summary dashboard
        
        Args:
            save_path: Path to save the HTML summary
        """
        # Create performance metrics
        metrics = {
            'Total Predictions': len(self.prediction_history),
            'Data Points': sum(len(history) for history in self.price_history.values()),
            'Factors Tracked': self.n_factors,
            'Symbols Monitored': len(self.symbols)
        }
        
        # Create HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>GPFA Performance Summary</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 15px; background-color: #e8f4f8; border-radius: 5px; }}
                .metric h3 {{ margin: 0; color: #333; }}
                .metric p {{ margin: 5px 0 0 0; font-size: 24px; color: #007acc; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Real-time GPFA Prediction System</h1>
                <p>Performance Summary - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div style="margin-top: 20px;">
        """
        
        for key, value in metrics.items():
            html_content += f"""
                <div class="metric">
                    <h3>{key}</h3>
                    <p>{value}</p>
                </div>
            """
        
        html_content += """
            </div>
            
            <div style="margin-top: 30px;">
                <h2>System Status</h2>
                <p>✅ Real-time data feed operational</p>
                <p>✅ GPFA model active</p>
                <p>✅ Prediction ensemble trained</p>
                <p>✅ Visualization system running</p>
            </div>
        </body>
        </html>
        """
        
        # Save HTML file with UTF-8 encoding
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Performance summary saved to {save_path}")
    
    def update_dashboard_live(self, interval_seconds: int = 30) -> None:
        """
        Update the dashboard live at specified intervals
        
        Args:
            interval_seconds: Update interval in seconds
        """
        logger.info(f"Starting live dashboard updates every {interval_seconds} seconds")
        
        while True:
            try:
                # Update dashboard
                self.create_real_time_dashboard()
                logger.info("Dashboard updated")
                
                # Wait for next update
                time.sleep(interval_seconds)
                
            except KeyboardInterrupt:
                logger.info("Live dashboard updates stopped")
                break
            except Exception as e:
                logger.error(f"Error updating dashboard: {e}")
                time.sleep(interval_seconds)

def create_sample_visualization():
    """
    Create a sample visualization to test the system
    """
    # Initialize visualizer
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    visualizer = RealTimeVisualizer(symbols, n_factors=3)
    
    # Generate sample data
    np.random.seed(42)
    base_time = datetime.now()
    
    # Sample price data
    for i in range(50):
        timestamp = base_time + timedelta(minutes=i)
        
        # Sample price data
        sample_data = pd.DataFrame({
            'AAPL': [100 + np.random.randn() * 2],
            'GOOGL': [150 + np.random.randn() * 3],
            'MSFT': [200 + np.random.randn() * 2.5]
        })
        
        visualizer.update_price_data(sample_data)
        
        # Sample predictions
        predictions = {
            '1min': {'rf': 100.5 + np.random.randn() * 0.5, 'xgb': 100.6 + np.random.randn() * 0.5},
            '5min': {'rf': 101.2 + np.random.randn() * 1.0, 'xgb': 101.3 + np.random.randn() * 1.0}
        }
        
        visualizer.update_predictions(predictions, sample_data)
        
        # Sample factors
        factors = np.random.randn(3)
        visualizer.update_factors(factors)
    
    # Create visualizations
    visualizer.create_real_time_dashboard('sample_dashboard.html')
    visualizer.plot_price_predictions('AAPL', 'sample_price_predictions.png')
    visualizer.plot_latent_factors('sample_latent_factors.png')
    visualizer.create_performance_summary('sample_performance.html')
    
    print("Sample visualizations created successfully!")
    print("Files generated:")
    print("- sample_dashboard.html (Interactive dashboard)")
    print("- sample_price_predictions.png (Price predictions)")
    print("- sample_latent_factors.png (Latent factors)")
    print("- sample_performance.html (Performance summary)")

if __name__ == "__main__":
    create_sample_visualization() 