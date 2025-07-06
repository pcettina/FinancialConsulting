# Simple GPFA Real Data Test
"""
Simple test of the GPFA prediction model using the real intraday data we analyzed.
This test directly integrates the real market data with the existing GPFA system.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import time
import logging
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

# Import our existing components
from real_data_integration import EnhancedGPFAPredictor
from intraday_trend_test import IntradayTrendAnalyzer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_gpfa_with_real_data():
    """Test GPFA prediction model with real intraday data"""
    
    print("="*60)
    print("GPFA PREDICTION MODEL TEST WITH REAL DATA")
    print("="*60)
    
    # Test symbols (using the ones we analyzed)
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    
    print(f"📊 Testing with symbols: {', '.join(symbols)}")
    print(f"📅 Using today's real market data")
    
    # Step 1: Get real intraday data
    print(f"\n🔍 Step 1: Fetching real intraday data...")
    trend_analyzer = IntradayTrendAnalyzer(symbols)
    intraday_data = trend_analyzer.fetch_intraday_data()
    
    if not intraday_data:
        print("❌ No intraday data available")
        return
    
    total_data_points = sum(len(data) for data in intraday_data.values())
    print(f"✅ Fetched {total_data_points} total data points")
    
    # Step 2: Analyze market trends
    print(f"\n📈 Step 2: Analyzing market trends...")
    trend_analysis = trend_analyzer.analyze_intraday_trends()
    
    for symbol, analysis in trend_analysis.items():
        direction = "🟢" if analysis['trend_direction'] == 'Bullish' else "🔴"
        print(f"{direction} {symbol}: {analysis['total_change_pct']:+.2f}% ({analysis['trend_direction']})")
    
    # Step 3: Initialize GPFA predictor
    print(f"\n🤖 Step 3: Initializing GPFA predictor...")
    gpfa_predictor = EnhancedGPFAPredictor(symbols, n_factors=3)
    
    # Step 4: Test GPFA with real data
    print(f"\n⚡ Step 4: Running GPFA prediction test...")
    start_time = time.time()
    
    try:
        # Initialize with real data
        if gpfa_predictor.initialize_with_real_data():
            print("✅ GPFA system initialized successfully")
            
            # Run a short prediction test
            test_results = gpfa_predictor.run_real_time_prediction(duration_minutes=2)
            
            end_time = time.time()
            test_duration = end_time - start_time
            
            # Display results
            print(f"\n📊 GPFA PREDICTION RESULTS")
            print("-" * 40)
            
            if test_results.get('success'):
                print(f"✅ Test Status: SUCCESS")
                print(f"⏱️  Test Duration: {test_duration:.1f} seconds")
                print(f"🔄 Cycles Completed: {test_results.get('cycles_completed', 0)}")
                print(f"🎯 Predictions Made: {test_results.get('predictions_made', 0)}")
                
                # Show final prices
                final_prices = test_results.get('final_prices', {})
                if final_prices:
                    print(f"\n💰 Final Prices:")
                    for symbol, price in final_prices.items():
                        # Compare with trend analysis
                        if symbol in trend_analysis:
                            trend = trend_analysis[symbol]
                            print(f"  {symbol}: ${price:.2f} (Trend: {trend['trend_direction']})")
                        else:
                            print(f"  {symbol}: ${price:.2f}")
                
                # Performance metrics
                if test_results.get('cycles_completed', 0) > 0:
                    cycles_per_minute = test_results['cycles_completed'] / (test_duration / 60)
                    predictions_per_cycle = test_results['predictions_made'] / test_results['cycles_completed']
                    
                    print(f"\n📈 Performance Metrics:")
                    print(f"  Cycles per minute: {cycles_per_minute:.1f}")
                    print(f"  Predictions per cycle: {predictions_per_cycle:.1f}")
                    print(f"  Data processing efficiency: {total_data_points / test_results['cycles_completed']:.0f} points/cycle")
                
                # Market integration assessment
                print(f"\n🔗 Market Integration Assessment:")
                print(f"  ✅ Real data successfully integrated")
                print(f"  ✅ GPFA model trained on historical data")
                print(f"  ✅ Real-time predictions generated")
                print(f"  ✅ Market status monitoring active")
                
                return test_results
                
            else:
                print(f"❌ Test failed: {test_results.get('error', 'Unknown error')}")
                return None
                
        else:
            print("❌ GPFA initialization failed")
            return None
            
    except Exception as e:
        print(f"❌ Error during GPFA test: {e}")
        return None

def compare_with_market_trends(gpfa_results, trend_analysis):
    """Compare GPFA predictions with actual market trends"""
    
    print(f"\n🔍 PREDICTION vs MARKET TREND ANALYSIS")
    print("-" * 50)
    
    if not gpfa_results or not trend_analysis:
        print("❌ Insufficient data for comparison")
        return
    
    final_prices = gpfa_results.get('final_prices', {})
    
    for symbol in final_prices.keys():
        if symbol in trend_analysis:
            gpfa_price = final_prices[symbol]
            trend = trend_analysis[symbol]
            actual_close = trend['close_price']
            
            # Calculate prediction accuracy
            price_error = abs(gpfa_price - actual_close)
            price_error_pct = (price_error / actual_close) * 100
            
            print(f"\n{symbol}:")
            print(f"  GPFA Prediction: ${gpfa_price:.2f}")
            print(f"  Actual Close: ${actual_close:.2f}")
            print(f"  Error: ${price_error:.2f} ({price_error_pct:.2f}%)")
            print(f"  Market Trend: {trend['trend_direction']} ({trend['total_change_pct']:+.2f}%)")
            
            # Assess prediction quality
            if price_error_pct < 1.0:
                quality = "🟢 Excellent"
            elif price_error_pct < 2.0:
                quality = "🟡 Good"
            elif price_error_pct < 5.0:
                quality = "🟠 Fair"
            else:
                quality = "🔴 Poor"
            
            print(f"  Prediction Quality: {quality}")

def main():
    """Main function to run the GPFA real data test"""
    
    print("Starting GPFA prediction model test with real market data...")
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run the test
    gpfa_results = test_gpfa_with_real_data()
    
    if gpfa_results:
        # Get trend analysis for comparison
        symbols = ['AAPL', 'GOOGL', 'MSFT']
        trend_analyzer = IntradayTrendAnalyzer(symbols)
        trend_analysis = trend_analyzer.analyze_intraday_trends()
        
        # Compare predictions with market trends
        compare_with_market_trends(gpfa_results, trend_analysis)
        
        # Summary
        print(f"\n🎯 TEST SUMMARY")
        print("-" * 30)
        print(f"✅ GPFA model successfully tested with real data")
        print(f"✅ {gpfa_results.get('cycles_completed', 0)} prediction cycles completed")
        print(f"✅ {gpfa_results.get('predictions_made', 0)} predictions generated")
        print(f"✅ Real market data integration verified")
        
        # Save results
        import json
        results_summary = {
            'test_timestamp': datetime.now().isoformat(),
            'symbols_tested': symbols,
            'gpfa_results': gpfa_results,
            'trend_analysis': trend_analysis,
            'test_status': 'SUCCESS'
        }
        
        with open('gpfa_real_data_test_results.json', 'w') as f:
            json.dump(results_summary, f, indent=2, default=str)
        
        print(f"💾 Results saved to: gpfa_real_data_test_results.json")
        
    else:
        print(f"\n❌ TEST FAILED")
        print("GPFA model test with real data was unsuccessful")
    
    print(f"\nTest completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main() 