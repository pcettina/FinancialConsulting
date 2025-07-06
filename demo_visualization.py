# GPFA Visualization Demo
"""
Demonstration script showing how to use the visualization system with your GPFA model.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import our visualization components
from simple_visualization import create_sample_stock_data, calculate_rsi
from viz_integration import SimpleVizIntegration

def demo_basic_workflow():
    """Demonstrate the basic visualization workflow"""
    print("="*60)
    print("GPFA VISUALIZATION DEMO - BASIC WORKFLOW")
    print("="*60)
    
    # Step 1: Get your data (replace with your actual data source)
    print("1. Loading market data...")
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    market_data = create_sample_stock_data()
    
    for symbol in symbols:
        print(f"   {symbol}: {len(market_data[symbol])} data points")
    
    # Step 2: Run your GPFA predictions (replace with your actual model)
    print("\n2. Running GPFA predictions...")
    predictions = {}
    
    for symbol in symbols:
        df = market_data[symbol]
        
        # Simulate GPFA prediction output
        # Replace this with your actual GPFA model call
        last_price = df['Close'].iloc[-1]
        prediction_horizon = 10
        
        # Generate realistic predictions (replace with your model output)
        np.random.seed(42 + symbols.index(symbol))
        price_changes = np.random.normal(0.001, 0.02, prediction_horizon)
        predicted_prices = [last_price]
        
        for change in price_changes:
            predicted_prices.append(predicted_prices[-1] * (1 + change))
        
        predicted_prices = np.array(predicted_prices[1:])
        prediction_times = pd.date_range(
            start=df.index[-1] + timedelta(days=1),
            periods=prediction_horizon,
            freq='D'
        )
        
        # Calculate confidence intervals (your model should provide these)
        std_dev = np.std(df['Close'].pct_change().dropna()) * last_price
        confidence_interval = 1.96 * std_dev
        
        predictions[symbol] = {
            'predicted_prices': predicted_prices,
            'prediction_times': prediction_times,
            'confidence_intervals': [
                predicted_prices - confidence_interval,
                predicted_prices + confidence_interval
            ],
            'model_confidence': np.random.uniform(0.7, 0.95),
            'prediction_horizon': prediction_horizon
        }
        
        print(f"   {symbol}: Predicted {prediction_horizon} days ahead")
        print(f"   {symbol}: Price range ${predicted_prices.min():.2f} - ${predicted_prices.max():.2f}")
    
    # Step 3: Create visualizations
    print("\n3. Creating visualizations...")
    
    # Initialize visualization system
    viz = SimpleVizIntegration(symbols)
    
    # Create basic dashboard
    basic_dashboard = viz.create_dashboard(market_data, predictions)
    viz.save_dashboard(basic_dashboard, "demo_basic_dashboard.html")
    print("   ✓ Basic dashboard created")
    
    # Create advanced dashboard
    advanced_dashboard = create_advanced_demo_dashboard(market_data, predictions)
    save_plotly_figure(advanced_dashboard, "demo_advanced_dashboard.html")
    print("   ✓ Advanced dashboard created")
    
    # Create prediction analysis
    prediction_analysis = create_prediction_analysis_demo(market_data, predictions)
    save_plotly_figure(prediction_analysis, "demo_prediction_analysis.html")
    print("   ✓ Prediction analysis created")
    
    print("\n" + "="*40)
    print("DEMO COMPLETED SUCCESSFULLY")
    print("="*40)
    print("Generated files:")
    print("- results/demo_basic_dashboard.html")
    print("- results/demo_advanced_dashboard.html")
    print("- results/demo_prediction_analysis.html")
    print("\nOpen these files in your browser to see the visualizations!")

def create_advanced_demo_dashboard(market_data, predictions):
    """Create an advanced demo dashboard"""
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            'GPFA Price Predictions', 'Technical Indicators',
            'Prediction Confidence', 'Market Correlation',
            'Volatility Analysis', 'Model Performance'
        ],
        specs=[[{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "bar"}, {"type": "heatmap"}],
               [{"type": "scatter"}, {"type": "bar"}]]
    )
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    symbols = list(market_data.keys())
    
    # 1. GPFA Price Predictions
    for i, symbol in enumerate(symbols):
        df = market_data[symbol]
        
        # Actual prices
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['Close'],
                mode='lines',
                name=f'{symbol} Actual',
                line=dict(color=colors[i % len(colors)]),
                showlegend=True
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
                    name=f'{symbol} GPFA Predicted',
                    line=dict(color=colors[i % len(colors)], dash='dash'),
                    marker=dict(size=4),
                    showlegend=False
                ),
                row=1, col=1
            )
    
    # 2. Technical Indicators
    for i, symbol in enumerate(symbols):
        df = market_data[symbol]
        if 'RSI' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['RSI'],
                    mode='lines',
                    name=f'{symbol} RSI',
                    line=dict(color=colors[i % len(colors)]),
                    showlegend=False
                ),
                row=1, col=2
            )
    
    # Add RSI reference lines
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=2)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=2)
    
    # 3. Prediction Confidence
    confidences = []
    for symbol in symbols:
        if symbol in predictions:
            confidences.append(predictions[symbol]['model_confidence'] * 100)
        else:
            confidences.append(0)
    
    fig.add_trace(
        go.Bar(
            x=symbols,
            y=confidences,
            name='Model Confidence %',
            marker_color=colors[:len(symbols)]
        ),
        row=2, col=1
    )
    
    # 4. Market Correlation
    price_data = pd.DataFrame()
    for symbol in symbols:
        price_data[symbol] = market_data[symbol]['Close']
    
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
    for symbol in symbols:
        df = market_data[symbol]
        returns = df['Close'].pct_change().dropna()
        volatilities.append(returns.std() * np.sqrt(252) * 100)
    
    fig.add_trace(
        go.Bar(
            x=symbols,
            y=volatilities,
            name='Annualized Volatility %',
            marker_color='orange'
        ),
        row=3, col=1
    )
    
    # 6. Model Performance
    performance_metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    performance_values = [85.2, 82.1, 88.3, 85.1]  # Example values
    
    fig.add_trace(
        go.Bar(
            x=performance_metrics,
            y=performance_values,
            name='Model Performance',
            marker_color='purple'
        ),
        row=3, col=2
    )
    
    fig.update_layout(
        title='GPFA Advanced Trading Dashboard - Demo',
        height=1200,
        showlegend=True,
        hovermode='x unified'
    )
    
    return fig

def create_prediction_analysis_demo(market_data, predictions):
    """Create prediction analysis demo"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['GPFA Predictions vs Actual', 'Prediction Errors', 
                       'Confidence Intervals', 'Prediction Horizon Analysis'],
        specs=[[{"type": "scatter"}, {"type": "histogram"}],
               [{"type": "scatter"}, {"type": "bar"}]]
    )
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    symbols = list(market_data.keys())
    
    # 1. Predictions vs Actual
    for i, symbol in enumerate(symbols):
        if symbol in predictions:
            pred_data = predictions[symbol]
            actual_price = market_data[symbol]['Close'].iloc[-1]
            predicted_price = pred_data['predicted_prices'][0]
            
            fig.add_trace(
                go.Scatter(
                    x=[actual_price],
                    y=[predicted_price],
                    mode='markers',
                    name=f'{symbol} Prediction',
                    marker=dict(
                        color=colors[i % len(colors)],
                        size=12
                    )
                ),
                row=1, col=1
            )
    
    # Add perfect prediction line
    min_price = min([market_data[symbol]['Close'].min() for symbol in symbols])
    max_price = max([market_data[symbol]['Close'].max() for symbol in symbols])
    
    fig.add_trace(
        go.Scatter(
            x=[min_price, max_price],
            y=[min_price, max_price],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='black', dash='dash'),
            showlegend=False
        ),
        row=1, col=1
    )
    
    # 2. Prediction Errors
    all_errors = []
    for symbol in symbols:
        if symbol in predictions:
            pred_data = predictions[symbol]
            actual_price = market_data[symbol]['Close'].iloc[-1]
            predicted_price = pred_data['predicted_prices'][0]
            error = (predicted_price - actual_price) / actual_price * 100
            all_errors.append(error)
    
    if all_errors:
        fig.add_trace(
            go.Histogram(
                x=all_errors,
                name='Prediction Errors %',
                nbinsx=10,
                marker_color='lightblue'
            ),
            row=1, col=2
        )
    
    # 3. Confidence Intervals
    for i, symbol in enumerate(symbols):
        if symbol in predictions:
            pred_data = predictions[symbol]
            ci = pred_data['confidence_intervals']
            times = pred_data['prediction_times']
            
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=ci[1],  # Upper bound
                    mode='lines',
                    name=f'{symbol} Upper CI',
                    line=dict(color=colors[i % len(colors)]),
                    showlegend=False
                ),
                row=2, col=1
            )
    
    # 4. Prediction Horizon Analysis
    horizons = ['1 day', '2 days', '5 days', '10 days']
    horizon_accuracies = [87, 84, 80, 76]  # Example values
    
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
        title='GPFA Prediction Analysis - Demo',
        height=800,
        showlegend=True
    )
    
    return fig

def save_plotly_figure(fig, filename):
    """Save Plotly figure as HTML"""
    fig.write_html(f"results/{filename}")
    print(f"   Saved: results/{filename}")

def main():
    """Main demo function"""
    print("🎯 GPFA VISUALIZATION SYSTEM DEMO")
    print("This demo shows how to use the visualization system with your GPFA model")
    
    # Run the basic workflow demo
    demo_basic_workflow()
    
    print("\n" + "="*60)
    print("DEMO COMPLETE!")
    print("="*60)
    print("You now have:")
    print("✅ Working visualization system")
    print("✅ Sample dashboards to review")
    print("✅ Ready to connect with your GPFA model")
    print()
    print("Next: Open the generated HTML files in your browser to see the visualizations!")

if __name__ == "__main__":
    main() 