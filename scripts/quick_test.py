# Quick Test Script
"""
Quick test to demonstrate the real data integration cycle configuration
"""

from real_data_integration import EnhancedGPFAPredictor
import time

def test_cycle_configuration():
    """Test the cycle configuration"""
    print("=== Real Data Integration Cycle Configuration ===")
    
    # Configuration
    symbols = ['AAPL', 'GOOGL']
    duration_minutes = 2  # Shorter test
    cycle_interval = 30   # seconds
    
    # Calculate expected cycles
    expected_cycles = (duration_minutes * 60) // cycle_interval
    
    print(f"Symbols: {symbols}")
    print(f"Duration: {duration_minutes} minutes")
    print(f"Cycle interval: {cycle_interval} seconds")
    print(f"Expected cycles: {expected_cycles}")
    print(f"Total runtime: {duration_minutes} minutes")
    
    # Initialize predictor
    print("\nInitializing EnhancedGPFAPredictor...")
    predictor = EnhancedGPFAPredictor(symbols, n_factors=3)
    
    # Run test
    print(f"\nRunning {duration_minutes}-minute test...")
    start_time = time.time()
    
    try:
        results = predictor.run_real_time_prediction(duration_minutes=duration_minutes)
        
        end_time = time.time()
        actual_duration = (end_time - start_time) / 60
        
        print(f"\n=== Test Results ===")
        print(f"Success: {results.get('success', False)}")
        print(f"Expected cycles: {expected_cycles}")
        print(f"Actual cycles: {results.get('cycles_completed', 0)}")
        print(f"Predictions made: {results.get('predictions_made', 0)}")
        print(f"Actual duration: {actual_duration:.1f} minutes")
        
        if results.get('final_prices'):
            print("\nFinal prices:")
            for symbol, price in results['final_prices'].items():
                print(f"  {symbol}: ${price:.2f}")
        
        if not results.get('success'):
            print(f"Error: {results.get('error', 'Unknown error')}")
            
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"\nTest failed with exception: {e}")

if __name__ == "__main__":
    test_cycle_configuration() 