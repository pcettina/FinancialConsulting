# Market Open Testing Script
"""
Automated script for market open testing and live trading validation.
Run this script at 9:30 AM EST when markets open.
"""

import json
import time
from datetime import datetime
from market_hours_testing import MarketHoursTester

def run_market_open_test():
    """Run comprehensive market open test"""
    
    print("MARKET OPEN TESTING")
    print("="*50)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize tester
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    tester = MarketHoursTester(symbols, n_factors=3)
    
    # Check market status
    market_status = tester.check_market_status()
    print(f"Market Status: {market_status['time_status']}")
    
    if not market_status['is_market_hours']:
        print("Markets not yet open - waiting for market open...")
        return
    
    # Run market hours test
    print("Markets are open - starting live trading test...")
    results = tester.run_market_hours_test(duration_minutes=30)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'market_open_test_{timestamp}.json'
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Market open test completed")
    print(f"Results saved to: {filename}")
    
    return results

if __name__ == "__main__":
    run_market_open_test()
