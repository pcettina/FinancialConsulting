# 🚀 GPFA Financial Prediction Dashboard

A comprehensive, real-time Streamlit dashboard for visualizing GPFA (Generalized Probabilistic Factor Analysis) financial predictions with interactive charts, live updates, and professional-grade analytics.

## ✨ Features

### 📊 **Real-time Visualization**
- Live price charts with predictions
- Interactive technical indicators
- Real-time market overview
- Dynamic prediction accuracy tracking

### 🎯 **Advanced Analytics**
- GPFA latent factor analysis
- Prediction confidence intervals
- Performance metrics dashboard
- Risk analysis and Sharpe ratios

### 🔄 **Interactive Controls**
- Symbol selection and filtering
- Time range controls
- Chart type switching
- Live update toggles

### 📈 **Professional Dashboard**
- Multi-panel layouts
- Responsive design
- Export capabilities
- Real-time status indicators

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Start

1. **Clone or navigate to your project directory**
   ```bash
   cd FinancialConsulting
   ```

2. **Install dependencies**
   ```bash
   pip install -r streamlit_requirements.txt
   ```

3. **Launch the dashboard**
   ```bash
   python run_dashboard.py
   ```

4. **Open your browser**
   Navigate to: `http://localhost:8501`

## 🎮 Usage

### Dashboard Navigation

#### **Main Sections**
1. **Real-time Overview** - Live market data and predictions
2. **Price Analysis** - Detailed price charts and statistics
3. **Technical Indicators** - RSI, MACD, SMA analysis
4. **Prediction Analysis** - GPFA prediction accuracy and confidence
5. **Market Overview** - Comprehensive market dashboard
6. **GPFA Factors** - Latent factor analysis
7. **Performance Metrics** - Model performance tracking

#### **Sidebar Controls**
- **Symbol Selection**: Choose which stocks to display
- **Chart Type**: Switch between different visualization types
- **Time Range**: Select data time period
- **Live Updates**: Toggle real-time data updates
- **Refresh Data**: Manually update predictions and data

### Key Features

#### **Real-time Updates**
- Enable live mode for continuous data updates
- Configurable update frequency (5-60 seconds)
- Live status indicators

#### **Interactive Charts**
- Zoom, pan, and hover functionality
- Multi-panel dashboard layouts
- Export chart images
- Customizable time ranges

#### **Prediction Analysis**
- Actual vs predicted price comparisons
- Confidence intervals visualization
- Prediction horizon performance
- Model accuracy metrics

## 📁 File Structure

```
FinancialConsulting/
├── streamlit_dashboard.py          # Main dashboard application
├── run_dashboard.py               # Dashboard launcher
├── streamlit_requirements.txt     # Dashboard dependencies
├── visualization_system.py        # Core visualization components
├── realtime_gpfa_predictor.py     # GPFA prediction engine
├── realtime_visualization.py      # Real-time visualization tools
└── results/                       # Generated charts and reports
    ├── *.html                     # Interactive dashboards
    ├── *.png                      # Static charts
    └── *.csv                      # Data exports
```

## 🔧 Configuration

### Environment Variables
```bash
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=localhost
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
```

### Customization Options

#### **Adding New Symbols**
Edit `streamlit_dashboard.py` and modify the `symbols` list in the `GPFADashboard.__init__()` method:

```python
self.symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'YOUR_SYMBOL']
```

#### **Customizing Charts**
Modify chart functions in the dashboard class to add new visualizations or change existing ones.

#### **Integration with Real Data**
Replace the sample data generation with your actual data sources:

```python
# Replace _generate_sample_data() with your data source
def _load_real_data(self):
    # Your data loading logic here
    pass
```

## 📊 Dashboard Components

### **Real-time Overview**
- Price movement charts
- Volume analysis
- Technical indicators
- Prediction horizons

### **Price Analysis**
- Normalized price comparisons
- Price statistics table
- Change percentage tracking
- High/low analysis

### **Technical Indicators**
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- SMA (Simple Moving Averages)
- Volume analysis

### **Prediction Analysis**
- Prediction accuracy by symbol
- Confidence intervals
- Horizon performance
- Error metrics (MAE, RMSE, MAPE)

### **GPFA Factors**
- Latent factor trajectories
- Factor loadings heatmap
- Factor correlations
- Convergence history

### **Performance Metrics**
- Accuracy over time
- Error distributions
- Performance comparisons
- Risk metrics (Sharpe ratios)

## 🚀 Advanced Features

### **Live Data Integration**
The dashboard is designed to integrate with real-time data sources:

```python
# Example: Integrate with your GPFA predictor
from realtime_gpfa_predictor import RealTimeGPFAPredictor

predictor = RealTimeGPFAPredictor()
predictions = predictor.get_latest_predictions()
```

### **Custom Visualizations**
Add new chart types by extending the dashboard class:

```python
def render_custom_chart(self, selected_symbols):
    """Add your custom visualization here"""
    st.subheader("Custom Chart")
    # Your chart logic here
```

### **Data Export**
Export dashboard data and charts:

```python
# Export functionality is built into the dashboard
# Use the "Export Report" button in the sidebar
```

## 🔍 Troubleshooting

### Common Issues

#### **Dashboard won't start**
1. Check if Streamlit is installed: `pip install streamlit`
2. Verify Python version: `python --version`
3. Check port availability: Try different port in `run_dashboard.py`

#### **Charts not loading**
1. Verify Plotly installation: `pip install plotly`
2. Check browser console for JavaScript errors
3. Ensure data files exist in the results directory

#### **Performance issues**
1. Reduce update frequency in sidebar
2. Limit number of selected symbols
3. Use shorter time ranges for large datasets

### Debug Mode
Run with debug information:

```bash
streamlit run streamlit_dashboard.py --logger.level=debug
```

## 📈 Performance Optimization

### **For Large Datasets**
- Use data sampling for historical data
- Implement data caching
- Optimize chart rendering with Plotly

### **For Real-time Updates**
- Use efficient data structures
- Implement incremental updates
- Consider WebSocket connections for live data

## 🔐 Security Considerations

- Dashboard runs on localhost by default
- No sensitive data is exposed
- Use HTTPS for production deployments
- Implement authentication for multi-user environments

## 🚀 Production Deployment

### **Using Streamlit Cloud**
1. Push code to GitHub
2. Connect repository to Streamlit Cloud
3. Deploy automatically

### **Using Docker**
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_dashboard.py"]
```

### **Using Traditional Web Server**
- Use nginx as reverse proxy
- Run Streamlit as a service
- Implement SSL certificates

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the code comments
3. Check Streamlit documentation: https://docs.streamlit.io/

## 🎯 Next Steps

### **Immediate Enhancements**
- [ ] Add more technical indicators
- [ ] Implement data caching
- [ ] Add user authentication
- [ ] Create mobile-responsive design

### **Future Features**
- [ ] Real-time news integration
- [ ] Portfolio tracking
- [ ] Alert system
- [ ] Advanced backtesting tools

---

**🎉 Congratulations!** You now have a professional-grade, real-time financial prediction dashboard that provides comprehensive visualization of your GPFA model's predictions with interactive features and live updates. 