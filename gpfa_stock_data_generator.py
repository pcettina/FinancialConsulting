import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Parameters
N_STOCKS = 100
N_DAYS = 10
MINUTES_PER_DAY = 390  # 9:30 to 16:00
START_DATE = datetime(2024, 7, 1)
TICKERS = [f'STK{i:03d}' for i in range(1, N_STOCKS + 1)]

# Market open/close times
MARKET_OPEN = timedelta(hours=9, minutes=30)
MARKET_CLOSE = timedelta(hours=16, minutes=0)

# Geometric Brownian motion parameters
MU = 0.0002  # drift per minute
SIGMA = 0.002  # volatility per minute

np.random.seed(42)

def generate_trading_minutes(start_date, n_days):
    """Generate a list of trading minute datetimes for n_days starting from start_date."""
    trading_minutes = []
    for day in range(n_days):
        date = start_date + timedelta(days=day)
        day_start = datetime(date.year, date.month, date.day, 9, 30)
        for minute in range(MINUTES_PER_DAY):
            trading_minutes.append(day_start + timedelta(minutes=minute))
    return trading_minutes

def simulate_stock_prices(n_stocks, n_days, minutes_per_day):
    """Simulate stock prices for n_stocks over n_days at 1-minute intervals."""
    trading_minutes = generate_trading_minutes(START_DATE, n_days)
    n_total = len(trading_minutes)
    data = []
    for ticker in TICKERS:
        # Randomize starting price
        start_price = np.random.uniform(20, 200)
        # Simulate log returns
        log_returns = np.random.normal(MU, SIGMA, n_total)
        log_prices = np.log(start_price) + np.cumsum(log_returns)
        prices = np.exp(log_prices)
        data.append(pd.DataFrame({
            'datetime': trading_minutes,
            'ticker': ticker,
            'price': prices
        }))
    df = pd.concat(data, ignore_index=True)
    return df

def main():
    print("Generating phony stock price data...")
    df = simulate_stock_prices(N_STOCKS, N_DAYS, MINUTES_PER_DAY)
    print(f"Generated {len(df)} rows for {N_STOCKS} stocks over {N_DAYS} days.")
    print(df.head())
    df.to_csv('stock_prices.csv', index=False)
    print("Saved to stock_prices.csv")

if __name__ == "__main__":
    main() 