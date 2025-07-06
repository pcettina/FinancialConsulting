# Real-Time Visualization System
"""
Comprehensive visualization system for real-time GPFA prediction data and modeling.
Provides multiple chart types, interactive features, and real-time updates.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class RealTimeVisualizer:
    """
    Real-time visualization system for GPFA predictions and market data
    """
    
    def __init__(self, symbols: List[str]):
        """
        Initialize real-time visualizer
        
        Args:
            symbols: List of stock symbols to visualize
        """
        self.symbols = symbols
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        # Set up matplotlib style
        plt.style.use('seaborn-v0_8')
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        
        print(f"Initialized RealTimeVisualizer for {len(symbols)} symbols")
    
    def create_price_chart(self, data: Dict[str, pd.DataFrame], 
                          predictions: Optional[Dict] = None) -> go.Figure:
        """
        Create interactive price chart with predictions
        
        Args:
            data: Dictionary of price data for each symbol
            predictions: Optional predictions data
            
        Returns:
            Plotly figure object
        """
        fig = make_subplots(
            rows=len(self.symbols), cols=1,
            subplot_titles=self.symbols,
            vertical_spacing=0.05,
            shared_xaxes=True
        )
        
        for i, symbol in enumerate(self.symbols):
            if symbol in data:
                df = data[symbol]
                
                # Add actual price line
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['Close'],
                        mode='lines',
                        name=f'{symbol} Actual',
                        line=dict(color=self.colors[i % len(self.colors)]),
                        showlegend=(i == 0)
                    ),
                    row=i+1, col=1
                )
                
                # Add predictions if available
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
                                showlegend=(i == 0)
                            ),
                            row=i+1, col=1
                        )
                
                # Add confidence intervals if available
                if predictions and symbol in predictions:
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
                            row=i+1, col=1
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
                            row=i+1, col=1
                        )
        
        fig.update_layout(
            title='Real-Time Stock Prices and Predictions',
            xaxis_title='Time',
            yaxis_title='Price ($)',
            height=300 * len(self.symbols),
            hovermode='x unified'
        )
        
        return fig
    
    def create_technical_indicators_chart(self, data: Dict[str, pd.DataFrame]) -> go.Figure:
        """
        Create technical indicators chart
        
        Args:
            data: Dictionary of price data with technical indicators
            
        Returns:
            Plotly figure object
        """
        fig = make_subplots(
            rows=len(self.symbols), cols=2,
            subplot_titles=[f'{symbol} - Price & Volume' for symbol in self.symbols] + 
                          [f'{symbol} - Technical Indicators' for symbol in self.symbols],
            specs=[[{"secondary_y": True}, {"secondary_y": False}] for _ in self.symbols]
        )
        
        for i, symbol in enumerate(self.symbols):
            if symbol in data:
                df = data[symbol]
                
                # Price and volume subplot
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['Close'],
                        mode='lines',
                        name=f'{symbol} Price',
                        line=dict(color=self.colors[i % len(self.colors)])
                    ),
                    row=i+1, col=1, secondary_y=False
                )
                
                fig.add_trace(
                    go.Bar(
                        x=df.index,
                        y=df['Volume'],
                        name=f'{symbol} Volume',
                        opacity=0.3,
                        marker_color=self.colors[i % len(self.colors)]
                    ),
                    row=i+1, col=1, secondary_y=True
                )
                
                # Technical indicators subplot
                if 'SMA_20' in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=df['SMA_20'],
                            mode='lines',
                            name=f'{symbol} SMA 20',
                            line=dict(color='orange')
                        ),
                        row=i+1, col=2
                    )
                
                if 'RSI' in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=df['RSI'],
                            mode='lines',
                            name=f'{symbol} RSI',
                            line=dict(color='purple')
                        ),
                        row=i+1, col=2
                    )
                    
                    # Add RSI overbought/oversold lines
                    fig.add_hline(y=70, line_dash="dash", line_color="red", row=i+1, col=2)
                    fig.add_hline(y=30, line_dash="dash", line_color="green", row=i+1, col=2)
        
        fig.update_layout(
            title='Technical Indicators Analysis',
            height=300 * len(self.symbols),
            showlegend=True
        )
        
        return fig
    
    def create_prediction_accuracy_chart(self, accuracy_data: Dict) -> go.Figure:
        """
        Create prediction accuracy visualization
        
        Args:
            accuracy_data: Dictionary containing accuracy metrics
            
        Returns:
            Plotly figure object
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Prediction Accuracy by Symbol', 'Error Distribution', 
                           'Accuracy Over Time', 'Model Performance Metrics'],
            specs=[[{"type": "bar"}, {"type": "histogram"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # Accuracy by symbol
        symbols = list(accuracy_data.keys())
        accuracies = [accuracy_data[symbol].get('accuracy', 0) for symbol in symbols]
        
        fig.add_trace(
            go.Bar(
                x=symbols,
                y=accuracies,
                name='Accuracy %',
                marker_color=self.colors[:len(symbols)]
            ),
            row=1, col=1
        )
        
        # Error distribution
        all_errors = []
        for symbol in symbols:
            if 'errors' in accuracy_data[symbol]:
                all_errors.extend(accuracy_data[symbol]['errors'])
        
        if all_errors:
            fig.add_trace(
                go.Histogram(
                    x=all_errors,
                    name='Error Distribution',
                    nbinsx=20,
                    marker_color='lightblue'
                ),
                row=1, col=2
            )
        
        # Accuracy over time
        for symbol in symbols:
            if 'accuracy_over_time' in accuracy_data[symbol]:
                time_data = accuracy_data[symbol]['accuracy_over_time']
                fig.add_trace(
                    go.Scatter(
                        x=list(time_data.keys()),
                        y=list(time_data.values()),
                        mode='lines+markers',
                        name=f'{symbol} Accuracy',
                        line=dict(color=self.colors[symbols.index(symbol) % len(self.colors)])
                    ),
                    row=2, col=1
                )
        
        # Performance metrics
        metrics = ['MAE', 'RMSE', 'MAPE']
        metric_values = []
        for metric in metrics:
            avg_value = np.mean([accuracy_data[symbol].get(metric.lower(), 0) 
                               for symbol in symbols])
            metric_values.append(avg_value)
        
        fig.add_trace(
            go.Bar(
                x=metrics,
                y=metric_values,
                name='Average Metrics',
                marker_color='lightgreen'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title='Prediction Accuracy Analysis',
            height=800,
            showlegend=True
        )
        
        return fig
    
    def create_market_overview_dashboard(self, data: Dict[str, pd.DataFrame]) -> go.Figure:
        """
        Create comprehensive market overview dashboard
        
        Args:
            data: Dictionary of market data for all symbols
            
        Returns:
            Plotly figure object
        """
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Price Performance Comparison', 'Volume Analysis',
                'Volatility Comparison', 'Correlation Matrix',
                'Market Trends', 'Risk Metrics'
            ],
            specs=[[{"type": "scatter"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "heatmap"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # Price performance comparison
        for i, symbol in enumerate(self.symbols):
            if symbol in data:
                df = data[symbol]
                normalized_prices = df['Close'] / df['Close'].iloc[0] * 100
                
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=normalized_prices,
                        mode='lines',
                        name=symbol,
                        line=dict(color=self.colors[i % len(self.colors)])
                    ),
                    row=1, col=1
                )
        
        # Volume analysis
        volumes = []
        for symbol in self.symbols:
            if symbol in data:
                volumes.append(data[symbol]['Volume'].mean())
        
        fig.add_trace(
            go.Bar(
                x=self.symbols,
                y=volumes,
                name='Average Volume',
                marker_color=self.colors[:len(self.symbols)]
            ),
            row=1, col=2
        )
        
        # Volatility comparison
        volatilities = []
        for symbol in self.symbols:
            if symbol in data:
                returns = data[symbol]['Close'].pct_change().dropna()
                volatilities.append(returns.std() * np.sqrt(252) * 100)  # Annualized
        
        fig.add_trace(
            go.Bar(
                x=self.symbols,
                y=volatilities,
                name='Volatility (%)',
                marker_color='orange'
            ),
            row=2, col=1
        )
        
        # Correlation matrix
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
        
        # Market trends
        for i, symbol in enumerate(self.symbols):
            if symbol in data:
                df = data[symbol]
                if 'SMA_20' in df.columns and 'SMA_50' in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=df['SMA_20'],
                            mode='lines',
                            name=f'{symbol} SMA20',
                            line=dict(color=self.colors[i % len(self.colors)])
                        ),
                        row=3, col=1
                    )
        
        # Risk metrics
        risk_metrics = ['Sharpe Ratio', 'Max Drawdown', 'VaR (95%)']
        risk_values = [1.2, -0.15, -0.02]  # Example values
        
        fig.add_trace(
            go.Bar(
                x=risk_metrics,
                y=risk_values,
                name='Risk Metrics',
                marker_color='red'
            ),
            row=3, col=2
        )
        
        fig.update_layout(
            title='Market Overview Dashboard',
            height=1200,
            showlegend=True
        )
        
        return fig
    
    def create_gpfa_analysis_chart(self, gpfa_data: Dict) -> go.Figure:
        """
        Create GPFA-specific analysis charts
        
        Args:
            gpfa_data: Dictionary containing GPFA analysis results
            
        Returns:
            Plotly figure object
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Latent Factors', 'Factor Loadings', 
                           'Prediction Horizon Performance', 'Model Convergence'],
            specs=[[{"type": "scatter"}, {"type": "heatmap"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # Latent factors
        if 'latent_factors' in gpfa_data:
            factors = gpfa_data['latent_factors']
            for i, factor in enumerate(factors):
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(factor))),
                        y=factor,
                        mode='lines',
                        name=f'Factor {i+1}',
                        line=dict(color=self.colors[i % len(self.colors)])
                    ),
                    row=1, col=1
                )
        
        # Factor loadings
        if 'factor_loadings' in gpfa_data:
            loadings = gpfa_data['factor_loadings']
            fig.add_trace(
                go.Heatmap(
                    z=loadings,
                    x=[f'Factor {i+1}' for i in range(loadings.shape[1])],
                    y=self.symbols,
                    colorscale='RdBu',
                    zmid=0
                ),
                row=1, col=2
            )
        
        # Prediction horizon performance
        if 'horizon_performance' in gpfa_data:
            horizons = list(gpfa_data['horizon_performance'].keys())
            accuracies = list(gpfa_data['horizon_performance'].values())
            
            fig.add_trace(
                go.Scatter(
                    x=horizons,
                    y=accuracies,
                    mode='lines+markers',
                    name='Horizon Accuracy',
                    line=dict(color='green')
                ),
                row=2, col=1
            )
        
        # Model convergence
        if 'convergence_history' in gpfa_data:
            convergence = gpfa_data['convergence_history']
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(convergence))),
                    y=convergence,
                    mode='lines',
                    name='Convergence',
                    line=dict(color='purple')
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title='GPFA Analysis Dashboard',
            height=800,
            showlegend=True
        )
        
        return fig
    
    def create_streamlit_dashboard(self, data: Dict[str, pd.DataFrame], 
                                 predictions: Optional[Dict] = None,
                                 accuracy_data: Optional[Dict] = None):
        """
        Create Streamlit dashboard for interactive visualization
        
        Args:
            data: Dictionary of market data
            predictions: Optional predictions data
            accuracy_data: Optional accuracy data
        """
        st.set_page_config(page_title="GPFA Trading Dashboard", layout="wide")
        
        st.title("🚀 Real-Time GPFA Trading Dashboard")
        
        # Sidebar controls
        st.sidebar.header("Dashboard Controls")
        selected_symbols = st.sidebar.multiselect(
            "Select Symbols",
            self.symbols,
            default=self.symbols
        )
        
        chart_type = st.sidebar.selectbox(
            "Chart Type",
            ["Price Chart", "Technical Indicators", "Market Overview", 
             "Prediction Accuracy", "GPFA Analysis"]
        )
        
        # Main content
        if chart_type == "Price Chart":
            fig = self.create_price_chart(data, predictions)
            st.plotly_chart(fig, use_container_width=True)
            
        elif chart_type == "Technical Indicators":
            fig = self.create_technical_indicators_chart(data)
            st.plotly_chart(fig, use_container_width=True)
            
        elif chart_type == "Market Overview":
            fig = self.create_market_overview_dashboard(data)
            st.plotly_chart(fig, use_container_width=True)
            
        elif chart_type == "Prediction Accuracy" and accuracy_data:
            fig = self.create_prediction_accuracy_chart(accuracy_data)
            st.plotly_chart(fig, use_container_width=True)
            
        elif chart_type == "GPFA Analysis":
            # Example GPFA data
            gpfa_data = {
                'latent_factors': [np.random.randn(100) for _ in range(3)],
                'factor_loadings': np.random.randn(len(self.symbols), 3),
                'horizon_performance': {f'{i}h': np.random.uniform(0.7, 0.95) for i in [1, 2, 5, 10, 20]},
                'convergence_history': np.random.randn(50).cumsum()
            }
            fig = self.create_gpfa_analysis_chart(gpfa_data)
            st.plotly_chart(fig, use_container_width=True)
        
        # Metrics display
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Symbols", len(self.symbols))
        
        with col2:
            st.metric("Data Points", sum(len(data[symbol]) for symbol in selected_symbols if symbol in data))
        
        with col3:
            if predictions:
                st.metric("Predictions Made", len(predictions))
            else:
                st.metric("Predictions Made", 0)
        
        with col4:
            if accuracy_data:
                avg_accuracy = np.mean([accuracy_data[symbol].get('accuracy', 0) 
                                      for symbol in selected_symbols if symbol in accuracy_data])
                st.metric("Avg Accuracy", f"{avg_accuracy:.2f}%")
            else:
                st.metric("Avg Accuracy", "N/A")

def create_sample_data():
    """Create sample data for testing visualization"""
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    
    data = {}
    for i, symbol in enumerate(['AAPL', 'GOOGL', 'MSFT']):
        # Generate realistic price data
        np.random.seed(42 + i)
        base_price = 100 + i * 50
        returns = np.random.normal(0.001, 0.02, len(dates))
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Add technical indicators
        df = pd.DataFrame({
            'Date': dates,
            'Open': prices * (1 + np.random.normal(0, 0.01, len(dates))),
            'High': prices * (1 + np.abs(np.random.normal(0, 0.02, len(dates)))),
            'Low': prices * (1 - np.abs(np.random.normal(0, 0.02, len(dates)))),
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, len(dates))
        })
        
        # Calculate technical indicators
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['RSI'] = calculate_rsi(df['Close'])
        
        df.set_index('Date', inplace=True)
        data[symbol] = df
    
    return data

def calculate_rsi(prices, period=14):
    """Calculate RSI technical indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def main():
    """Main function to test visualization system"""
    print("🚀 Testing Real-Time Visualization System")
    
    # Create sample data
    data = create_sample_data()
    
    # Initialize visualizer
    visualizer = RealTimeVisualizer(['AAPL', 'GOOGL', 'MSFT'])
    
    # Create sample predictions
    predictions = {
        'AAPL': {
            'predicted_prices': data['AAPL']['Close'].iloc[-10:].values * 1.02,
            'prediction_times': data['AAPL'].index[-10:],
            'confidence_intervals': [
                data['AAPL']['Close'].iloc[-10:].values * 0.98,
                data['AAPL']['Close'].iloc[-10:].values * 1.06
            ]
        }
    }
    
    # Create sample accuracy data
    accuracy_data = {
        'AAPL': {
            'accuracy': 85.5,
            'mae': 2.3,
            'rmse': 3.1,
            'mape': 1.2,
            'errors': np.random.normal(0, 2, 100),
            'accuracy_over_time': {f'Day {i}': 80 + np.random.uniform(0, 10) for i in range(1, 31)}
        },
        'GOOGL': {
            'accuracy': 82.1,
            'mae': 3.2,
            'rmse': 4.1,
            'mape': 1.8,
            'errors': np.random.normal(0, 3, 100),
            'accuracy_over_time': {f'Day {i}': 75 + np.random.uniform(0, 15) for i in range(1, 31)}
        },
        'MSFT': {
            'accuracy': 88.3,
            'mae': 1.9,
            'rmse': 2.5,
            'mape': 0.9,
            'errors': np.random.normal(0, 1.5, 100),
            'accuracy_over_time': {f'Day {i}': 85 + np.random.uniform(0, 8) for i in range(1, 31)}
        }
    }
    
    # Test different chart types
    print("\n📊 Creating visualization charts...")
    
    # Price chart
    price_fig = visualizer.create_price_chart(data, predictions)
    price_fig.write_html("results/price_chart.html")
    print("✅ Price chart saved to results/price_chart.html")
    
    # Technical indicators
    tech_fig = visualizer.create_technical_indicators_chart(data)
    tech_fig.write_html("results/technical_indicators.html")
    print("✅ Technical indicators chart saved to results/technical_indicators.html")
    
    # Market overview
    market_fig = visualizer.create_market_overview_dashboard(data)
    market_fig.write_html("results/market_overview.html")
    print("✅ Market overview dashboard saved to results/market_overview.html")
    
    # Prediction accuracy
    accuracy_fig = visualizer.create_prediction_accuracy_chart(accuracy_data)
    accuracy_fig.write_html("results/prediction_accuracy.html")
    print("✅ Prediction accuracy chart saved to results/prediction_accuracy.html")
    
    # GPFA analysis
    gpfa_data = {
        'latent_factors': [np.random.randn(100) for _ in range(3)],
        'factor_loadings': np.random.randn(3, 3),
        'horizon_performance': {f'{i}h': np.random.uniform(0.7, 0.95) for i in [1, 2, 5, 10, 20]},
        'convergence_history': np.random.randn(50).cumsum()
    }
    gpfa_fig = visualizer.create_gpfa_analysis_chart(gpfa_data)
    gpfa_fig.write_html("results/gpfa_analysis.html")
    print("✅ GPFA analysis chart saved to results/gpfa_analysis.html")
    
    print("\n🎯 Visualization system ready for integration!")
    print("📁 All charts saved to results/ directory")
    print("🌐 Open the HTML files in your browser to view interactive charts")

if __name__ == "__main__":
    main() 