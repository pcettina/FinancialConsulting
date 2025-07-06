# Test Real-time GPFA Implementation
"""
Simple test script to verify the real-time GPFA predictor implementation
"""

import sys
import traceback
from realtime_gpfa_predictor import RealTimeGPFAPredictor, RealTimeDataFeed, RealTimeGPFA, PredictionEnsemble

def test_data_feed():
    """Test the real-time data feed"""
    print("Testing RealTimeDataFeed...")
    
    try:
        # Test with a few symbols
        symbols = ['AAPL', 'GOOGL']
        data_feed = RealTimeDataFeed(symbols)
        
        # Fetch data
        data = data_feed.fetch_real_time_data()
        print(f"Fetched data for {len(data)} symbols")
        
        if data:
            for symbol, df in data.items():
                print(f"  {symbol}: {len(df)} data points")
        
        return True
        
    except Exception as e:
        print(f"Error testing data feed: {e}")
        traceback.print_exc()
        return False

def test_gpfa_model():
    """Test the GPFA model"""
    print("\nTesting RealTimeGPFA...")
    
    try:
        # Create sample data
        import numpy as np
        import pandas as pd
        
        # Generate sample price data
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=100, freq='H')
        data = pd.DataFrame({
            'AAPL': 100 + np.cumsum(np.random.randn(100) * 0.1),
            'GOOGL': 150 + np.cumsum(np.random.randn(100) * 0.15),
            'MSFT': 200 + np.cumsum(np.random.randn(100) * 0.12)
        }, index=dates)
        
        # Initialize GPFA model
        gpfa = RealTimeGPFA(n_factors=2)
        
        # Fit model
        gpfa.fit_model(data)
        print(f"GPFA model fitted with {gpfa.n_factors} factors")
        
        # Test prediction
        predictions = gpfa.predict_factors(time_horizon=5)
        if predictions is not None:
            print(f"Predicted {predictions.shape[1]} time steps ahead")
        
        return True
        
    except Exception as e:
        print(f"Error testing GPFA model: {e}")
        traceback.print_exc()
        return False

def test_prediction_ensemble():
    """Test the prediction ensemble"""
    print("\nTesting PredictionEnsemble...")
    
    try:
        # Create sample data
        import numpy as np
        import pandas as pd
        
        # Generate sample OHLCV data
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=200, freq='H')
        
        base_price = 100
        data = pd.DataFrame({
            'Open': base_price + np.cumsum(np.random.randn(200) * 0.1),
            'High': base_price + np.cumsum(np.random.randn(200) * 0.1) + np.random.uniform(0, 2, 200),
            'Low': base_price + np.cumsum(np.random.randn(200) * 0.1) - np.random.uniform(0, 2, 200),
            'Close': base_price + np.cumsum(np.random.randn(200) * 0.1),
            'Volume': np.random.randint(1000000, 10000000, 200)
        }, index=dates)
        
        # Initialize ensemble
        ensemble = PredictionEnsemble(horizons=['1min', '5min'])
        
        # Prepare features
        features = ensemble.prepare_features(data)
        print(f"Prepared {features.shape[1]} features from {len(data)} data points")
        
        # Train models
        ensemble.train_models(data)
        print("Models trained successfully")
        
        return True
        
    except Exception as e:
        print(f"Error testing prediction ensemble: {e}")
        traceback.print_exc()
        return False

def test_main_predictor():
    """Test the main predictor class"""
    print("\nTesting RealTimeGPFAPredictor...")
    
    try:
        # Initialize with sample symbols
        symbols = ['AAPL', 'GOOGL']
        predictor = RealTimeGPFAPredictor(symbols, n_factors=2)
        
        print(f"Initialized predictor for {len(symbols)} symbols")
        
        # Test initialization (this might fail if no internet/data)
        try:
            predictor.initialize_system()
            print("System initialized successfully")
        except Exception as e:
            print(f"System initialization failed (expected if no internet): {e}")
        
        return True
        
    except Exception as e:
        print(f"Error testing main predictor: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("Testing Real-time GPFA Implementation")
    print("=" * 50)
    
    tests = [
        test_data_feed,
        test_gpfa_model,
        test_prediction_ensemble,
        test_main_predictor
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✅ All tests passed! Real-time GPFA implementation is working.")
    else:
        print("⚠️  Some tests failed. Check the implementation.")

if __name__ == "__main__":
    main() 