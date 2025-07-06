# Intraday Trend Test
"""
Test the real data integration system with intraday data from earlier in the day
to analyze day trends and market movements from the past trading session.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import time
import logging
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IntradayTrendAnalyzer:
    """
    Analyzer for intraday trends and market movements
    """
    
    def __init__(self, symbols: List[str]):
        """
        Initialize intraday trend analyzer
        
        Args:
            symbols: List of stock symbols to analyze
        """
        self.symbols = symbols
        self.intraday_data = {}
        self.trend_analysis = {}
        
        logger.info(f"Initialized IntradayTrendAnalyzer for {len(symbols)} symbols")
    
    def fetch_intraday_data(self, date: str = None) -> Dict[str, pd.DataFrame]:
        """
        Fetch intraday data for a specific date (default: today)
        
        Args:
            date: Date string in YYYY-MM-DD format (default: today)
            
        Returns:
            Dictionary of intraday DataFrames for each symbol
        """
        if date is None:
            # Get today's date
            date = datetime.now().strftime('%Y-%m-%d')
        
        logger.info(f"Fetching intraday data for {date}")
        
        for symbol in self.symbols:
            try:
                ticker = yf.Ticker(symbol)
                
                # Fetch intraday data (1-minute intervals for the specified date)
                # Note: yfinance doesn't support specific date ranges for intraday data
                # So we'll get recent intraday data and filter by date
                intraday_data = ticker.history(period='5d', interval='1m')
                
                if not intraday_data.empty:
                    # Filter for the specified date
                    intraday_data['Date'] = intraday_data.index.date
                    target_date = pd.to_datetime(date).date()
                    filtered_data = intraday_data[intraday_data['Date'] == target_date]
                    
                    if not filtered_data.empty:
                        self.intraday_data[symbol] = filtered_data
                        logger.info(f"✓ {symbol}: {len(filtered_data)} intraday data points for {date}")
                    else:
                        logger.warning(f"✗ No intraday data for {symbol} on {date}")
                else:
                    logger.warning(f"✗ No intraday data available for {symbol}")
                    
            except Exception as e:
                logger.error(f"Error fetching intraday data for {symbol}: {e}")
                continue
        
        return self.intraday_data
    
    def analyze_intraday_trends(self) -> Dict[str, Dict]:
        """
        Analyze intraday trends for each symbol
        
        Returns:
            Dictionary with trend analysis for each symbol
        """
        logger.info("Analyzing intraday trends...")
        
        for symbol, data in self.intraday_data.items():
            try:
                if data.empty:
                    continue
                
                # Calculate trend metrics
                open_price = data['Open'].iloc[0]
                close_price = data['Close'].iloc[-1]
                high_price = data['High'].max()
                low_price = data['Low'].min()
                
                # Price changes
                total_change = close_price - open_price
                total_change_pct = (total_change / open_price) * 100
                high_low_range = high_price - low_price
                high_low_range_pct = (high_low_range / open_price) * 100
                
                # Volume analysis
                avg_volume = data['Volume'].mean()
                total_volume = data['Volume'].sum()
                
                # Volatility (standard deviation of returns)
                returns = data['Close'].pct_change().dropna()
                volatility = returns.std() * 100
                
                # Trend direction
                if total_change > 0:
                    trend_direction = "Bullish"
                elif total_change < 0:
                    trend_direction = "Bearish"
                else:
                    trend_direction = "Sideways"
                
                # Intraday momentum
                # Calculate momentum as the difference between first and last hour
                if len(data) >= 60:  # At least 60 minutes of data
                    first_hour_close = data['Close'].iloc[59]  # 60th minute
                    last_hour_close = data['Close'].iloc[-1]
                    momentum = last_hour_close - first_hour_close
                    momentum_pct = (momentum / first_hour_close) * 100
                else:
                    momentum = 0
                    momentum_pct = 0
                
                # Store analysis
                self.trend_analysis[symbol] = {
                    'open_price': open_price,
                    'close_price': close_price,
                    'high_price': high_price,
                    'low_price': low_price,
                    'total_change': total_change,
                    'total_change_pct': total_change_pct,
                    'high_low_range': high_low_range,
                    'high_low_range_pct': high_low_range_pct,
                    'avg_volume': avg_volume,
                    'total_volume': total_volume,
                    'volatility': volatility,
                    'trend_direction': trend_direction,
                    'momentum': momentum,
                    'momentum_pct': momentum_pct,
                    'data_points': len(data)
                }
                
                logger.info(f"✓ {symbol} analysis completed: {trend_direction} ({total_change_pct:.2f}%)")
                
            except Exception as e:
                logger.error(f"Error analyzing trends for {symbol}: {e}")
                continue
        
        return self.trend_analysis
    
    def get_market_session_summary(self) -> Dict:
        """
        Get summary of the market session
        
        Returns:
            Market session summary
        """
        if not self.trend_analysis:
            return {}
        
        # Aggregate statistics
        total_symbols = len(self.trend_analysis)
        bullish_symbols = sum(1 for analysis in self.trend_analysis.values() 
                             if analysis['trend_direction'] == 'Bullish')
        bearish_symbols = sum(1 for analysis in self.trend_analysis.values() 
                             if analysis['trend_direction'] == 'Bearish')
        
        avg_change_pct = np.mean([analysis['total_change_pct'] 
                                 for analysis in self.trend_analysis.values()])
        avg_volatility = np.mean([analysis['volatility'] 
                                 for analysis in self.trend_analysis.values()])
        
        # Find best and worst performers
        best_performer = max(self.trend_analysis.items(), 
                           key=lambda x: x[1]['total_change_pct'])
        worst_performer = min(self.trend_analysis.items(), 
                            key=lambda x: x[1]['total_change_pct'])
        
        return {
            'total_symbols': total_symbols,
            'bullish_symbols': bullish_symbols,
            'bearish_symbols': bearish_symbols,
            'sideways_symbols': total_symbols - bullish_symbols - bearish_symbols,
            'avg_change_pct': avg_change_pct,
            'avg_volatility': avg_volatility,
            'best_performer': {
                'symbol': best_performer[0],
                'change_pct': best_performer[1]['total_change_pct']
            },
            'worst_performer': {
                'symbol': worst_performer[0],
                'change_pct': worst_performer[1]['total_change_pct']
            }
        }

class IntradayGPFATester:
    """
    Test GPFA system with intraday data
    """
    
    def __init__(self, symbols: List[str], n_factors: int = 3):
        """
        Initialize intraday GPFA tester
        
        Args:
            symbols: List of stock symbols
            n_factors: Number of GPFA factors
        """
        self.symbols = symbols
        self.n_factors = n_factors
        self.trend_analyzer = IntradayTrendAnalyzer(symbols)
        
        logger.info(f"Initialized IntradayGPFATester for {len(symbols)} symbols")
    
    def run_intraday_test(self, date: str = None, duration_minutes: int = 5) -> Dict:
        """
        Run intraday trend test
        
        Args:
            date: Date to test (default: today)
            duration_minutes: Test duration in minutes
            
        Returns:
            Test results
        """
        try:
            logger.info(f"Starting intraday trend test for {date or 'today'}")
            
            # Fetch intraday data
            intraday_data = self.trend_analyzer.fetch_intraday_data(date)
            
            if not intraday_data:
                return {'success': False, 'error': 'No intraday data available'}
            
            # Analyze trends
            trend_analysis = self.trend_analyzer.analyze_intraday_trends()
            
            # Get market session summary
            session_summary = self.trend_analyzer.get_market_session_summary()
            
            # Simulate prediction cycles using intraday data
            prediction_results = self._simulate_predictions_with_intraday_data(
                intraday_data, duration_minutes
            )
            
            # Compile results
            results = {
                'success': True,
                'date_tested': date or datetime.now().strftime('%Y-%m-%d'),
                'duration_minutes': duration_minutes,
                'symbols_tested': self.symbols,
                'intraday_data_points': sum(len(data) for data in intraday_data.values()),
                'trend_analysis': trend_analysis,
                'session_summary': session_summary,
                'prediction_results': prediction_results
            }
            
            logger.info("✓ Intraday trend test completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Error in intraday test: {e}")
            return {'success': False, 'error': str(e)}
    
    def _simulate_predictions_with_intraday_data(self, intraday_data: Dict[str, pd.DataFrame], 
                                               duration_minutes: int) -> Dict:
        """
        Simulate predictions using intraday data
        
        Args:
            intraday_data: Intraday data for each symbol
            duration_minutes: Test duration
            
        Returns:
            Prediction simulation results
        """
        try:
            # Find the minimum data length across all symbols
            min_data_length = min(len(data) for data in intraday_data.values())
            
            # Calculate how many prediction cycles we can simulate
            cycle_interval = 30  # seconds
            cycles_per_minute = 60 // cycle_interval
            total_cycles = duration_minutes * cycles_per_minute
            
            # Limit cycles to available data
            max_cycles = min(total_cycles, min_data_length // 5)  # Use every 5th data point
            
            logger.info(f"Simulating {max_cycles} prediction cycles with intraday data")
            
            prediction_history = []
            
            for cycle in range(max_cycles):
                cycle_data = {}
                
                for symbol, data in intraday_data.items():
                    # Get data point for this cycle
                    data_index = cycle * 5  # Use every 5th data point
                    if data_index < len(data):
                        current_data = data.iloc[data_index]
                        
                        cycle_data[symbol] = {
                            'timestamp': current_data.name,
                            'open': current_data['Open'],
                            'high': current_data['High'],
                            'low': current_data['Low'],
                            'close': current_data['Close'],
                            'volume': current_data['Volume']
                        }
                
                if cycle_data:
                    prediction_history.append({
                        'cycle': cycle + 1,
                        'timestamp': datetime.now(),
                        'data': cycle_data
                    })
            
            return {
                'cycles_simulated': len(prediction_history),
                'prediction_history': prediction_history,
                'data_points_used': max_cycles * 5
            }
            
        except Exception as e:
            logger.error(f"Error simulating predictions: {e}")
            return {'cycles_simulated': 0, 'error': str(e)}

def display_intraday_results(results: Dict):
    """Display intraday test results"""
    print("\n" + "="*60)
    print("INTRADAY TREND TEST RESULTS")
    print("="*60)
    
    if not results.get('success'):
        print(f"❌ Test failed: {results.get('error', 'Unknown error')}")
        return
    
    print(f"📅 Date Tested: {results['date_tested']}")
    print(f"⏱️  Duration: {results['duration_minutes']} minutes")
    print(f"📊 Symbols: {', '.join(results['symbols_tested'])}")
    print(f"📈 Data Points: {results['intraday_data_points']}")
    
    # Display trend analysis
    print(f"\n📊 INTRADAY TREND ANALYSIS")
    print("-" * 40)
    
    for symbol, analysis in results['trend_analysis'].items():
        direction_emoji = "🟢" if analysis['trend_direction'] == 'Bullish' else "🔴" if analysis['trend_direction'] == 'Bearish' else "🟡"
        print(f"{direction_emoji} {symbol}:")
        print(f"   Open: ${analysis['open_price']:.2f} | Close: ${analysis['close_price']:.2f}")
        print(f"   Change: {analysis['total_change_pct']:+.2f}% ({analysis['trend_direction']})")
        print(f"   Range: ${analysis['high_low_range']:.2f} ({analysis['high_low_range_pct']:.2f}%)")
        print(f"   Volatility: {analysis['volatility']:.2f}%")
        print(f"   Volume: {analysis['total_volume']:,.0f}")
        print()
    
    # Display session summary
    session = results['session_summary']
    print(f"📈 MARKET SESSION SUMMARY")
    print("-" * 40)
    print(f"Total Symbols: {session['total_symbols']}")
    print(f"Bullish: {session['bullish_symbols']} | Bearish: {session['bearish_symbols']} | Sideways: {session['sideways_symbols']}")
    print(f"Average Change: {session['avg_change_pct']:+.2f}%")
    print(f"Average Volatility: {session['avg_volatility']:.2f}%")
    print(f"Best Performer: {session['best_performer']['symbol']} ({session['best_performer']['change_pct']:+.2f}%)")
    print(f"Worst Performer: {session['worst_performer']['symbol']} ({session['worst_performer']['change_pct']:+.2f}%)")
    
    # Display prediction simulation
    pred_results = results['prediction_results']
    print(f"\n🤖 PREDICTION SIMULATION")
    print("-" * 40)
    print(f"Cycles Simulated: {pred_results['cycles_simulated']}")
    print(f"Data Points Used: {pred_results.get('data_points_used', 0)}")

def main():
    """Main function to run intraday trend test"""
    # Test symbols
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    
    # Initialize tester
    tester = IntradayGPFATester(symbols, n_factors=3)
    
    # Run test with today's data
    print("Testing intraday trends from today's market session...")
    results = tester.run_intraday_test(duration_minutes=3)
    
    # Display results
    display_intraday_results(results)
    
    # Try with yesterday's data if today's data is limited
    if not results.get('success') or results['intraday_data_points'] < 100:
        print("\nTrying with yesterday's data...")
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        results = tester.run_intraday_test(date=yesterday, duration_minutes=3)
        display_intraday_results(results)

if __name__ == "__main__":
    main() 