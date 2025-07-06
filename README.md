# Real-Time GPFA Financial Prediction System

## Overview

This project implements a comprehensive real-time Gaussian Process Factor Analysis (GPFA) system for financial market prediction. The system combines advanced machine learning techniques with real-time data processing to provide multi-horizon price predictions with uncertainty quantification.

## 🚀 Key Features

- **Real-time Data Processing**: Live market data ingestion and processing
- **Multi-Horizon Predictions**: 1min, 5min, 15min, 1hour, and 1day forecasts
- **Ensemble Learning**: RandomForest, XGBoost, and LightGBM models
- **Uncertainty Quantification**: Confidence intervals and prediction uncertainty
- **Advanced Feature Engineering**: 40+ technical indicators and price action features
- **Dynamic Market Regimes**: Bull, bear, sideways, and volatile market simulation
- **Real-time Visualization**: Interactive dashboards and performance monitoring
- **Robust Error Handling**: Comprehensive error recovery and model retraining

## 📁 Project Structure

```
FinancialConsulting/
├── realtime_gpfa_predictor.py      # Main prediction system
├── enhanced_test_system.py         # Comprehensive testing framework
├── realtime_visualization.py       # Visualization and dashboard components
├── custom_gpfa.py                  # Custom GPFA implementation
├── gpfa_stock_data_generator.py    # Stock data generation utilities
├── test_realtime_gpfa.py          # Basic testing script
├── REALTIME_GPFA_IMPLEMENTATION_GUIDE.md  # Implementation guide
├── realtime_prediction_requirements.txt    # Dependencies
└── bond_analysis/                  # Bond trading analysis components
    ├── clustering_analysis.html
    ├── data_generator.py
    ├── demo_analysis.png
    ├── demo_bond_data.csv
    ├── demo.py
    ├── enhanced_analysis_report.txt
    ├── enhanced_analysis.py
    ├── feature_contributions_heatmap.png
    ├── main_analysis.py
    ├── pca_analysis.py
    ├── predictive_modeling.py
    ├── README.md
    ├── realistic_bond_data.csv
    ├── requirements.txt
    └── simple_predictive_model.py
```

## 🏗️ System Architecture

### Core Components

#### 1. **RealTimeDataFeed**
- Manages real-time market data ingestion
- Maintains data buffers for multiple symbols
- Handles data updates and synchronization

#### 2. **RealTimeGPFA**
- Implements Gaussian Process Factor Analysis
- Extracts latent factors from price data
- Provides factor predictions and price reconstruction

#### 3. **PredictionEnsemble**
- Manages multiple ML models (RandomForest, XGBoost, LightGBM)
- Handles model training, retraining, and prediction
- Provides uncertainty quantification
- Supports multiple prediction horizons

#### 4. **RealTimeGPFAPredictor**
- Main orchestrator class
- Coordinates all system components
- Manages prediction cycles and real-time loops
- Handles system initialization and monitoring

#### 5. **RealTimeVisualizer**
- Creates interactive dashboards
- Generates performance reports
- Provides real-time visualization updates

### Feature Engineering

The system generates **40+ features** including:

#### Technical Indicators
- **RSI** (Relative Strength Index) with overbought/oversold signals
- **MACD** (Moving Average Convergence Divergence) with signal line and histogram
- **Bollinger Bands** with width and position indicators
- **Stochastic Oscillator** with %K and %D lines
- **Williams %R** momentum indicator

#### Price Action Features
- **Price Range**: High-Low range as percentage of close
- **Body Size**: Open-Close difference as percentage
- **Upper/Lower Shadows**: Wick sizes relative to body
- **Moving Averages**: 5 and 20-period averages
- **Standard Deviation**: 5-period rolling volatility

#### Market Regime Features
- **Trend Components**: Directional bias indicators
- **Volatility Measures**: Dynamic volatility scaling
- **Correlation Factors**: Cross-asset relationships

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Dependencies
```bash
pip install -r realtime_prediction_requirements.txt
```

### Key Dependencies
- `numpy` - Numerical computing
- `pandas` - Data manipulation
- `scikit-learn` - Machine learning algorithms
- `xgboost` - Gradient boosting
- `lightgbm` - Light gradient boosting
- `matplotlib` - Plotting
- `seaborn` - Statistical visualization
- `plotly` - Interactive plots
- `ta` - Technical analysis indicators

## 🚀 Quick Start

### 1. Basic Usage
```python
from realtime_gpfa_predictor import RealTimeGPFAPredictor

# Initialize predictor
symbols = ['AAPL', 'GOOGL', 'MSFT']
predictor = RealTimeGPFAPredictor(symbols, n_factors=5)

# Run prediction test
predictor.run_prediction_test(duration_minutes=10)
```

### 2. Enhanced Testing
```python
from enhanced_test_system import EnhancedTestRunner

# Run comprehensive test
test_runner = EnhancedTestRunner(['AAPL', 'GOOGL', 'MSFT'], test_duration_minutes=15)
test_runner.run_enhanced_test()
```

### 3. Real-time Operation
```python
# Start real-time prediction loop
predictor.start_real_time_loop(interval_seconds=60)
```

## 📊 Model Performance

### Current Status
- **RandomForest**: ✅ Fully functional with optimized hyperparameters
- **XGBoost**: 🔄 Training improvements implemented
- **LightGBM**: 🔄 Enhanced configuration applied
- **Ensemble**: ✅ Working with uncertainty quantification

### Performance Metrics
- **Feature Count**: 40+ engineered features
- **Prediction Horizons**: 5 timeframes (1min to 1day)
- **Model Diversity**: 3 different algorithms
- **Data Quality**: Robust validation and cleaning
- **Real-time Processing**: <3 seconds per prediction cycle

## 🔧 Configuration

### Model Hyperparameters

#### RandomForest
```python
RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
```

#### XGBoost
```python
XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    enable_categorical=False,
    missing=0
)
```

#### LightGBM
```python
LGBMRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1,
    force_col_wise=True
)
```

### Data Requirements
- **Minimum Data**: 100-1200 samples depending on horizon
- **Update Frequency**: Configurable (default: 60 seconds)
- **Symbols**: Support for multiple assets
- **Data Quality**: Automatic validation and cleaning

## 📈 Visualization

### Generated Reports
- **Price Movements**: Historical vs predicted price charts
- **Prediction Accuracy**: Model performance over time
- **Prediction Comparison**: Multi-model ensemble analysis
- **Performance Summary**: HTML dashboard with metrics

### Real-time Dashboards
- **Live Predictions**: Current forecast values
- **Uncertainty Bands**: Confidence intervals
- **Model Status**: Training and prediction status
- **Performance Metrics**: Real-time accuracy tracking

## 🔄 Market Regime Simulation

The enhanced test system includes realistic market simulation:

### Regime Types
- **Bull Market**: Positive trends, moderate volatility
- **Bear Market**: Negative trends, high volatility  
- **Sideways Market**: No trend, low volatility
- **Volatile Market**: Random trends, high volatility

### Dynamic Features
- **Sector Correlations**: Tech, retail, auto sector factors
- **Market-wide Factors**: Systemic market movements
- **Individual Noise**: Stock-specific variations
- **Momentum Effects**: Trend following behavior
- **Mean Reversion**: Price stabilization mechanisms

## 🛡️ Error Handling

### Robust Features
- **Data Validation**: Comprehensive feature quality checks
- **Model Recovery**: Automatic retraining and reinitialization
- **Graceful Degradation**: System continues with available models
- **Error Logging**: Detailed error tracking and reporting

### Recovery Mechanisms
- **Model Reinitialization**: Automatic model recreation on failure
- **Data Cleaning**: Infinite value and NaN handling
- **Feature Fallbacks**: Safe default values for failed calculations
- **Training Validation**: Model fitting verification

## 📋 Usage Examples

### Example 1: Basic Prediction Test
```python
# Simple 5-minute test
predictor = RealTimeGPFAPredictor(['AAPL'])
predictor.run_prediction_test(duration_minutes=5)
```

### Example 2: Multi-Symbol Real-time
```python
# Real-time prediction for multiple symbols
symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
predictor = RealTimeGPFAPredictor(symbols)
predictor.start_real_time_loop(interval_seconds=30)
```

### Example 3: Enhanced Testing
```python
# Comprehensive testing with visualization
test_runner = EnhancedTestRunner(
    symbols=['AAPL', 'GOOGL', 'MSFT'],
    test_duration_minutes=20
)
test_runner.run_enhanced_test()
```

## 🔍 Monitoring and Debugging

### Logging
- **INFO**: System status and predictions
- **WARNING**: Model training issues
- **ERROR**: System failures and recovery attempts
- **DEBUG**: Detailed processing information

### Performance Monitoring
- **Prediction Latency**: Time per prediction cycle
- **Model Status**: Training and prediction success rates
- **Data Quality**: Feature validation results
- **Memory Usage**: System resource monitoring

## 🚧 Known Issues and Limitations

### Current Limitations
- **XGBoost/LightGBM**: Some training issues under investigation
- **Long Horizons**: 1hour and 1day predictions require more data
- **Real Data**: Currently using simulated data for testing

### Planned Improvements
- **Model Optimization**: Further hyperparameter tuning
- **Feature Selection**: Automated feature importance analysis
- **Real Data Integration**: Live market data feeds
- **Performance Optimization**: Reduced prediction latency

## 🤝 Contributing

### Development Guidelines
1. **Code Style**: Follow PEP 8 standards
2. **Documentation**: Add docstrings for all functions
3. **Testing**: Include unit tests for new features
4. **Error Handling**: Implement robust error recovery

### Testing
```bash
# Run basic tests
python test_realtime_gpfa.py

# Run enhanced tests
python enhanced_test_system.py

# Run specific components
python -m pytest tests/
```

## 📄 License

This project is for educational and research purposes. Please ensure compliance with relevant financial regulations when using for trading purposes.

## 📞 Support

For questions or issues:
1. Check the implementation guide: `REALTIME_GPFA_IMPLEMENTATION_GUIDE.md`
2. Review the test outputs and logs
3. Examine the visualization reports for insights

## 🔮 Future Roadmap

### Phase 1: System Stabilization ✅
- [x] Core GPFA implementation
- [x] Basic prediction ensemble
- [x] Real-time data processing
- [x] Visualization framework

### Phase 2: Model Enhancement 🔄
- [x] Enhanced feature engineering
- [x] Improved model hyperparameters
- [x] Better error handling
- [ ] Advanced model selection
- [ ] Automated hyperparameter tuning

### Phase 3: Production Readiness 📋
- [ ] Real market data integration
- [ ] Performance optimization
- [ ] Advanced monitoring
- [ ] Deployment automation

### Phase 4: Advanced Features 📋
- [ ] Multi-asset correlation modeling
- [ ] Risk management integration
- [ ] Backtesting framework
- [ ] Strategy optimization

---

**Note**: This system is designed for educational and research purposes. Always validate predictions and use appropriate risk management when applying to real trading scenarios. 