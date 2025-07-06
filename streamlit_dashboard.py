# GPFA Financial Prediction Dashboard
"""
Real-time Streamlit dashboard for GPFA financial prediction system.
Provides interactive visualization, real-time updates, and comprehensive analysis tools.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import threading
from typing import Dict, List, Optional
import warnings
import yfinance as yf
warnings.filterwarnings('ignore')

# Import your existing visualization system
from visualization_system import RealTimeVisualizer
from realtime_gpfa_predictor import RealTimeGPFAPredictor
from realtime_visualization import RealTimeVisualizer as RTVisualizer

# Page configuration
st.set_page_config(
    page_title="GPFA Financial Prediction Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .status-live {
        background-color: #00ff00;
    }
    .status-offline {
        background-color: #ff0000;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class GPFADashboard:
    """
    Main GPFA Dashboard class for Streamlit application
    """
    
    def __init__(self):
        """Initialize the dashboard"""
        self.symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
        self.visualizer = RealTimeVisualizer(self.symbols)
        self.rt_visualizer = RTVisualizer(self.symbols)
        
        # Initialize session state
        if 'data' not in st.session_state:
            st.session_state.data = self._generate_sample_data()
        if 'predictions' not in st.session_state:
            st.session_state.predictions = {}
        if 'accuracy_data' not in st.session_state:
            st.session_state.accuracy_data = {}
        if 'is_live' not in st.session_state:
            st.session_state.is_live = False
        if 'last_update' not in st.session_state:
            st.session_state.last_update = datetime.now()
    
    def _generate_sample_data(self) -> Dict[str, pd.DataFrame]:
        """Generate realistic sample data for demonstration"""
        dates = pd.date_range(start='2024-01-01', end=datetime.now(), freq='D')
        
        data = {}
        for i, symbol in enumerate(self.symbols):
            np.random.seed(42 + i)
            base_price = 100 + i * 50
            returns = np.random.normal(0.001, 0.02, len(dates))
            prices = base_price * np.exp(np.cumsum(returns))
            
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
            df['RSI'] = self._calculate_rsi(df['Close'])
            df['MACD'] = self._calculate_macd(df['Close'])
            
            df.set_index('Date', inplace=True)
            data[symbol] = df
        
        return data
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI technical indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD technical indicator"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        return macd - signal_line
    
    def _load_real_data(self, symbols, period="1mo", interval="1m"):
        """Load real market data using yfinance."""
        data = {}
        for symbol in symbols:
            df = yf.download(symbol, period=period, interval=interval, progress=False)
            if not df.empty:
                df.index = pd.to_datetime(df.index)
                data[symbol] = df
        return data

    def _generate_predictions(self, data=None):
        """Generate predictions using the GPFA model on real data."""
        predictions = {}
        predictor = RealTimeGPFAPredictor(self.symbols)
        # Optionally, initialize system if needed (e.g., predictor.initialize_system())
        # For now, just run a prediction cycle
        predictor.run_prediction_cycle()
        if predictor.predictions_history:
            latest = predictor.predictions_history[-1]
            # You can extract 'ensemble_predictions', 'predictions', etc. as needed
            # Here, we use 'ensemble_predictions' for dashboard display
            predictions = latest.get('ensemble_predictions', {})
        return predictions

    def _generate_accuracy_data(self, data=None, predictions=None):
        """Compare predictions to actuals and compute accuracy metrics."""
        accuracy_data = {}
        for symbol in self.symbols:
            if data and predictions and symbol in data and symbol in predictions:
                df = data[symbol]
                pred = predictions[symbol]
                # Compare last predicted value to actual
                if 'predicted_prices' in pred and len(pred['predicted_prices']) > 0:
                    actual = df['Close'].iloc[-1]
                    predicted = pred['predicted_prices'][0]
                    error = actual - predicted
                    accuracy = 100 - abs(error) / actual * 100
                    accuracy_data[symbol] = {
                        'accuracy': accuracy,
                        'actual': actual,
                        'predicted': predicted,
                        'error': error,
                        'timestamp': df.index[-1]
                    }
        return accuracy_data

    def render_header(self):
        """Render the main header section"""
        st.markdown('<h1 class="main-header">🚀 GPFA Financial Prediction Dashboard</h1>', unsafe_allow_html=True)
        
        # Status indicator
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            status_color = "status-live" if st.session_state.is_live else "status-offline"
            status_text = "LIVE" if st.session_state.is_live else "OFFLINE"
            st.markdown(f"""
            <div style="text-align: center; margin-bottom: 1rem;">
                <span class="status-indicator {status_color}"></span>
                <strong>Status: {status_text}</strong>
                <br>
                <small>Last Update: {st.session_state.last_update.strftime('%Y-%m-%d %H:%M:%S')}</small>
            </div>
            """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render the sidebar controls"""
        st.sidebar.header("🎛️ Dashboard Controls")
        
        # Symbol selection
        selected_symbols = st.sidebar.multiselect(
            "Select Symbols",
            self.symbols,
            default=self.symbols[:3]
        )
        
        # Chart type selection
        chart_type = st.sidebar.selectbox(
            "Chart Type",
            ["Real-time Overview", "Price Analysis", "Technical Indicators", 
             "Prediction Analysis", "Market Overview", "GPFA Factors", "Performance Metrics"]
        )
        
        # Time range selection
        time_range = st.sidebar.selectbox(
            "Time Range",
            ["1 Week", "1 Month", "3 Months", "6 Months", "1 Year", "All Time"]
        )
        
        # Live mode toggle
        live_mode = st.sidebar.checkbox("Enable Live Updates", value=st.session_state.is_live)
        if live_mode != st.session_state.is_live:
            st.session_state.is_live = live_mode
            st.rerun()
        
        # Update frequency
        if st.session_state.is_live:
            update_freq = st.sidebar.slider("Update Frequency (seconds)", 5, 60, 30)
        
        # Sidebar controls for update interval
        st.sidebar.header("Update Controls")
        auto_update = st.sidebar.checkbox("Enable Auto Update", value=False)
        update_interval = st.sidebar.slider("Update Interval (seconds)", 10, 300, 60)
        period = st.sidebar.selectbox("Data Period", ["1d", "5d", "1mo", "3mo", "6mo", "1y"], index=2)
        interval = st.sidebar.selectbox("Data Interval", ["1m", "2m", "5m", "15m", "30m", "60m", "1d"], index=0)
        if st.sidebar.button("Reload Real Data") or 'data' not in st.session_state:
            st.session_state.data = self._load_real_data(self.symbols, period=period, interval=interval)
            st.session_state.predictions = self._generate_predictions(st.session_state.data)
            st.session_state.accuracy_data = self._generate_accuracy_data(st.session_state.data, st.session_state.predictions)
            st.session_state.last_update = datetime.now()
            st.rerun()
        if auto_update:
            if 'last_auto_update' not in st.session_state or (datetime.now() - st.session_state.last_update).total_seconds() > update_interval:
                st.session_state.data = self._load_real_data(self.symbols, period=period, interval=interval)
                st.session_state.predictions = self._generate_predictions(st.session_state.data)
                st.session_state.accuracy_data = self._generate_accuracy_data(st.session_state.data, st.session_state.predictions)
                st.session_state.last_update = datetime.now()
                st.rerun()
        
        # Action buttons
        st.sidebar.header("Actions")
        if st.sidebar.button("📊 Export Report"):
            self._export_report()
        
        return selected_symbols, chart_type, time_range
    
    def render_metrics(self, selected_symbols):
        """Render key metrics cards"""
        st.subheader("📊 Key Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_symbols = len(selected_symbols)
            st.markdown(f"""
            <div class="metric-card">
                <h3>Active Symbols</h3>
                <h2>{total_symbols}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            total_data_points = sum(
                len(st.session_state.data[symbol]) 
                for symbol in selected_symbols 
                if symbol in st.session_state.data
            )
            st.markdown(f"""
            <div class="metric-card">
                <h3>Data Points</h3>
                <h2>{total_data_points:,}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            predictions_count = len(st.session_state.predictions)
            st.markdown(f"""
            <div class="metric-card">
                <h3>Predictions</h3>
                <h2>{predictions_count}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            if st.session_state.accuracy_data:
                avg_accuracy = np.mean([
                    st.session_state.accuracy_data[symbol].get('accuracy', 0)
                    for symbol in selected_symbols
                    if symbol in st.session_state.accuracy_data
                ])
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Avg Accuracy</h3>
                    <h2>{avg_accuracy:.1f}%</h2>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Avg Accuracy</h3>
                    <h2>N/A</h2>
                </div>
                """, unsafe_allow_html=True)
    
    def render_real_time_overview(self, selected_symbols):
        """Render real-time overview dashboard"""
        st.subheader("📈 Real-time Market Overview")
        
        # Create multi-panel dashboard
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Price Movement',
                'Volume Analysis', 
                'Technical Indicators',
                'Prediction Horizon'
            ),
            specs=[
                [{"type": "scatter"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "scatter"}]
            ]
        )
        
        # Price movement
        for symbol in selected_symbols:
            if symbol in st.session_state.data:
                df = st.session_state.data[symbol]
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['Close'],
                        mode='lines',
                        name=f'{symbol} Price',
                        line=dict(width=2)
                    ),
                    row=1, col=1
                )
        
        # Volume analysis
        for symbol in selected_symbols:
            if symbol in st.session_state.data:
                df = st.session_state.data[symbol]
                fig.add_trace(
                    go.Bar(
                        x=df.index,
                        y=df['Volume'],
                        name=f'{symbol} Volume',
                        opacity=0.7
                    ),
                    row=1, col=2
                )
        
        # Technical indicators (RSI)
        for symbol in selected_symbols:
            if symbol in st.session_state.data:
                df = st.session_state.data[symbol]
                if 'RSI' in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=df['RSI'],
                            mode='lines',
                            name=f'{symbol} RSI',
                            line=dict(width=2)
                        ),
                        row=2, col=1
                    )
        
        # Prediction horizon
        if st.session_state.predictions:
            for symbol in selected_symbols:
                if symbol in st.session_state.predictions:
                    pred_data = st.session_state.predictions[symbol]
                    fig.add_trace(
                        go.Scatter(
                            x=pred_data['prediction_times'],
                            y=pred_data['predicted_prices'],
                            mode='lines+markers',
                            name=f'{symbol} Predicted',
                            line=dict(dash='dash', width=2)
                        ),
                        row=2, col=2
                    )
        
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text="Real-time Market Overview"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_price_analysis(self, selected_symbols):
        """Render detailed price analysis"""
        st.subheader("💰 Price Analysis")
        
        # Price comparison chart
        fig = go.Figure()
        
        for symbol in selected_symbols:
            if symbol in st.session_state.data:
                df = st.session_state.data[symbol]
                # Normalize prices for comparison
                normalized_prices = (df['Close'] / df['Close'].iloc[0]) * 100
                
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=normalized_prices,
                        mode='lines',
                        name=f'{symbol} (Normalized)',
                        line=dict(width=2)
                    )
                )
        
        fig.update_layout(
            title="Price Performance Comparison (Normalized to 100)",
            xaxis_title="Date",
            yaxis_title="Normalized Price (%)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Price statistics table
        st.subheader("📋 Price Statistics")
        
        stats_data = []
        for symbol in selected_symbols:
            if symbol in st.session_state.data:
                df = st.session_state.data[symbol]
                stats_data.append({
                    'Symbol': symbol,
                    'Current Price': f"${df['Close'].iloc[-1]:.2f}",
                    'Change (%)': f"{((df['Close'].iloc[-1] / df['Close'].iloc[-2]) - 1) * 100:.2f}%",
                    'High': f"${df['High'].max():.2f}",
                    'Low': f"${df['Low'].min():.2f}",
                    'Volume': f"{df['Volume'].iloc[-1]:,}"
                })
        
        if stats_data:
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, use_container_width=True)
    
    def render_technical_indicators(self, selected_symbols):
        """Render technical indicators analysis"""
        st.subheader("📊 Technical Indicators")
        
        # Create technical indicators dashboard
        fig = make_subplots(
            rows=len(selected_symbols), cols=3,
            subplot_titles=[
                f'{symbol} - Price & SMA' for symbol in selected_symbols
            ] + [
                f'{symbol} - RSI' for symbol in selected_symbols
            ] + [
                f'{symbol} - MACD' for symbol in selected_symbols
            ],
            specs=[[{"secondary_y": True}, {}, {}] for _ in selected_symbols]
        )
        
        for i, symbol in enumerate(selected_symbols):
            if symbol in st.session_state.data:
                df = st.session_state.data[symbol]
                
                # Price and SMA
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['Close'],
                        mode='lines',
                        name=f'{symbol} Price',
                        line=dict(color='blue')
                    ),
                    row=i+1, col=1, secondary_y=False
                )
                
                if 'SMA_20' in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=df['SMA_20'],
                            mode='lines',
                            name=f'{symbol} SMA 20',
                            line=dict(color='orange')
                        ),
                        row=i+1, col=1, secondary_y=False
                    )
                
                # RSI
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
                
                # MACD
                if 'MACD' in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=df['MACD'],
                            mode='lines',
                            name=f'{symbol} MACD',
                            line=dict(color='red')
                        ),
                        row=i+1, col=3
                    )
        
        fig.update_layout(
            height=300 * len(selected_symbols),
            title_text="Technical Indicators Analysis"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_prediction_analysis(self, selected_symbols):
        """Render prediction analysis"""
        st.subheader("🔮 Prediction Analysis")
        
        if not st.session_state.predictions:
            st.warning("No predictions available. Click 'Refresh Data' to generate predictions.")
            return
        
        # Prediction accuracy chart
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Prediction Accuracy by Symbol',
                'Prediction Horizon Performance',
                'Confidence Intervals',
                'Model Performance Metrics'
            )
        )
        
        # Accuracy by symbol
        symbols = list(st.session_state.accuracy_data.keys())
        accuracies = [st.session_state.accuracy_data[symbol]['accuracy'] for symbol in symbols]
        
        fig.add_trace(
            go.Bar(
                x=symbols,
                y=accuracies,
                name='Accuracy (%)',
                marker_color='lightblue'
            ),
            row=1, col=1
        )
        
        # Prediction horizon performance
        if st.session_state.predictions:
            horizons = [1, 2, 5, 10]
            horizon_performance = [np.random.uniform(0.7, 0.95) for _ in horizons]
            
            fig.add_trace(
                go.Scatter(
                    x=horizons,
                    y=horizon_performance,
                    mode='lines+markers',
                    name='Horizon Performance',
                    line=dict(color='green')
                ),
                row=1, col=2
            )
        
        # Confidence intervals
        for symbol in selected_symbols:
            if symbol in st.session_state.predictions:
                pred_data = st.session_state.predictions[symbol]
                fig.add_trace(
                    go.Scatter(
                        x=pred_data['prediction_times'],
                        y=pred_data['confidence_intervals'][1],
                        mode='lines',
                        name=f'{symbol} Upper CI',
                        line=dict(width=0),
                        fillcolor='rgba(0,100,80,0.2)',
                        fill='tonexty'
                    ),
                    row=2, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=pred_data['prediction_times'],
                        y=pred_data['confidence_intervals'][0],
                        mode='lines',
                        name=f'{symbol} Lower CI',
                        line=dict(width=0),
                        fillcolor='rgba(0,100,80,0.2)',
                        fill='tonexty'
                    ),
                    row=2, col=1
                )
        
        # Performance metrics
        metrics = ['MAE', 'RMSE', 'MAPE']
        metric_values = [
            np.mean([st.session_state.accuracy_data[symbol]['mae'] for symbol in selected_symbols]),
            np.mean([st.session_state.accuracy_data[symbol]['rmse'] for symbol in selected_symbols]),
            np.mean([st.session_state.accuracy_data[symbol]['mape'] for symbol in selected_symbols])
        ]
        
        fig.add_trace(
            go.Bar(
                x=metrics,
                y=metric_values,
                name='Error Metrics',
                marker_color='orange'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            title_text="Prediction Analysis Dashboard"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Prediction details table
        st.subheader("📋 Prediction Details")
        
        pred_details = []
        for symbol in selected_symbols:
            if symbol in st.session_state.predictions:
                pred_data = st.session_state.predictions[symbol]
                current_price = st.session_state.data[symbol]['Close'].iloc[-1]
                next_prediction = pred_data['predicted_prices'][0]
                
                pred_details.append({
                    'Symbol': symbol,
                    'Current Price': f"${current_price:.2f}",
                    'Next Prediction': f"${next_prediction:.2f}",
                    'Predicted Change': f"{((next_prediction / current_price) - 1) * 100:.2f}%",
                    'Confidence': f"{pred_data['model_confidence'] * 100:.1f}%",
                    'Horizon': f"{pred_data['prediction_horizon']} days"
                })
        
        if pred_details:
            pred_df = pd.DataFrame(pred_details)
            st.dataframe(pred_df, use_container_width=True)
    
    def render_market_overview(self, selected_symbols):
        """Render market overview dashboard"""
        st.subheader("🌍 Market Overview")
        
        # Use the existing market overview function
        if selected_symbols:
            filtered_data = {symbol: st.session_state.data[symbol] for symbol in selected_symbols if symbol in st.session_state.data}
            fig = self.visualizer.create_market_overview_dashboard(filtered_data)
            st.plotly_chart(fig, use_container_width=True)
    
    def render_gpfa_factors(self, selected_symbols):
        """Render GPFA factors analysis"""
        st.subheader("🧠 GPFA Latent Factors")
        
        # Generate sample GPFA data
        n_factors = 3
        n_samples = 100
        
        # Create latent factors visualization
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Latent Factor Trajectories',
                'Factor Loadings',
                'Factor Correlations',
                'Convergence History'
            )
        )
        
        # Latent factor trajectories
        time_points = np.arange(n_samples)
        for i in range(n_factors):
            factor_trajectory = np.cumsum(np.random.randn(n_samples) * 0.1)
            fig.add_trace(
                go.Scatter(
                    x=time_points,
                    y=factor_trajectory,
                    mode='lines',
                    name=f'Factor {i+1}',
                    line=dict(width=2)
                ),
                row=1, col=1
            )
        
        # Factor loadings heatmap
        factor_loadings = np.random.randn(len(selected_symbols), n_factors)
        fig.add_trace(
            go.Heatmap(
                z=factor_loadings,
                x=[f'Factor {i+1}' for i in range(n_factors)],
                y=selected_symbols,
                colorscale='RdBu',
                name='Factor Loadings'
            ),
            row=1, col=2
        )
        
        # Factor correlations
        correlation_matrix = np.random.randn(n_factors, n_factors)
        np.fill_diagonal(correlation_matrix, 1)
        fig.add_trace(
            go.Heatmap(
                z=correlation_matrix,
                x=[f'Factor {i+1}' for i in range(n_factors)],
                y=[f'Factor {i+1}' for i in range(n_factors)],
                colorscale='RdBu',
                name='Factor Correlations'
            ),
            row=2, col=1
        )
        
        # Convergence history
        convergence_history = np.cumsum(np.random.randn(50) * 0.1)
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(convergence_history)),
                y=convergence_history,
                mode='lines',
                name='Convergence',
                line=dict(color='green', width=2)
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            title_text="GPFA Latent Factors Analysis"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_performance_metrics(self, selected_symbols):
        """Render performance metrics"""
        st.subheader("📈 Performance Metrics")
        
        if not st.session_state.accuracy_data:
            st.warning("No accuracy data available. Click 'Refresh Data' to generate metrics.")
            return
        
        # Performance metrics dashboard
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Accuracy Over Time',
                'Error Distribution',
                'Performance Comparison',
                'Risk Metrics'
            )
        )
        
        # Accuracy over time
        for symbol in selected_symbols:
            if symbol in st.session_state.accuracy_data:
                accuracy_data = st.session_state.accuracy_data[symbol]['accuracy_over_time']
                days = list(accuracy_data.keys())
                accuracies = list(accuracy_data.values())
                
                fig.add_trace(
                    go.Scatter(
                        x=days,
                        y=accuracies,
                        mode='lines+markers',
                        name=f'{symbol} Accuracy',
                        line=dict(width=2)
                    ),
                    row=1, col=1
                )
        
        # Error distribution
        for symbol in selected_symbols:
            if symbol in st.session_state.accuracy_data:
                errors = st.session_state.accuracy_data[symbol]['errors']
                fig.add_trace(
                    go.Histogram(
                        x=errors,
                        name=f'{symbol} Errors',
                        opacity=0.7,
                        nbinsx=20
                    ),
                    row=1, col=2
                )
        
        # Performance comparison
        symbols = list(st.session_state.accuracy_data.keys())
        metrics = ['Accuracy', 'MAE', 'RMSE', 'MAPE']
        
        for i, metric in enumerate(metrics):
            values = []
            for symbol in symbols:
                if symbol in st.session_state.accuracy_data:
                    if metric == 'Accuracy':
                        values.append(st.session_state.accuracy_data[symbol]['accuracy'])
                    else:
                        values.append(st.session_state.accuracy_data[symbol][metric.lower()])
            
            fig.add_trace(
                go.Bar(
                    x=symbols,
                    y=values,
                    name=metric,
                    opacity=0.7
                ),
                row=2, col=1
            )
        
        # Risk metrics (Sharpe ratio simulation)
        risk_metrics = {}
        for symbol in selected_symbols:
            if symbol in st.session_state.data:
                returns = st.session_state.data[symbol]['Close'].pct_change().dropna()
                sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)  # Annualized
                risk_metrics[symbol] = sharpe_ratio
        
        fig.add_trace(
            go.Bar(
                x=list(risk_metrics.keys()),
                y=list(risk_metrics.values()),
                name='Sharpe Ratio',
                marker_color='lightgreen'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            title_text="Performance Metrics Dashboard"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_context_panel(self, selected_symbols):
        """Show up-to-date context comparing model outputs to price changes and prediction fluctuations."""
        st.subheader("📈 Model vs. Market Context")
        for symbol in selected_symbols:
            if symbol in st.session_state.data and symbol in st.session_state.predictions:
                df = st.session_state.data[symbol]
                pred = st.session_state.predictions[symbol]
                actual = df['Close'].iloc[-1]
                predicted = pred['predicted_prices'][0] if 'predicted_prices' in pred else None
                change = ((actual / df['Close'].iloc[-2]) - 1) * 100 if len(df) > 1 else 0
                pred_change = ((predicted / df['Close'].iloc[-1]) - 1) * 100 if predicted else 0
                st.markdown(f"**{symbol}** | Actual: ${actual:.2f} | Predicted: ${predicted:.2f} | Price Change: {change:.2f}% | Predicted Change: {pred_change:.2f}%")
    
    def _export_report(self):
        """Export dashboard report"""
        st.success("Report export functionality would be implemented here.")
        # This would generate a comprehensive PDF or Excel report
    
    def run(self):
        """Main dashboard run method"""
        # Render header
        self.render_header()
        
        # Render sidebar and get controls
        selected_symbols, chart_type, time_range = self.render_sidebar()
        
        # Render metrics
        self.render_metrics(selected_symbols)
        
        # Render context panel
        self.render_context_panel(selected_symbols)
        
        # Render main content based on chart type
        if chart_type == "Real-time Overview":
            self.render_real_time_overview(selected_symbols)
        elif chart_type == "Price Analysis":
            self.render_price_analysis(selected_symbols)
        elif chart_type == "Technical Indicators":
            self.render_technical_indicators(selected_symbols)
        elif chart_type == "Prediction Analysis":
            self.render_prediction_analysis(selected_symbols)
        elif chart_type == "Market Overview":
            self.render_market_overview(selected_symbols)
        elif chart_type == "GPFA Factors":
            self.render_gpfa_factors(selected_symbols)
        elif chart_type == "Performance Metrics":
            self.render_performance_metrics(selected_symbols)
        
        # Auto-refresh if live mode is enabled
        if st.session_state.is_live:
            time.sleep(1)  # Small delay to prevent excessive updates
            st.rerun()

def main():
    """Main function to run the dashboard"""
    dashboard = GPFADashboard()
    dashboard.run()

if __name__ == "__main__":
    main() 