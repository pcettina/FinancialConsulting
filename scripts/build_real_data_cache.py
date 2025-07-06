import os
import yfinance as yf
import pandas as pd
from datetime import datetime

# Symbols to track
SYMBOLS = [
    'AAPL', 'MSFT', 'GOOGL', 'PLTR', 'TSLA', 'LMT', 'AMZN', 'NVDA', 'JPM'
]

# Data cache directory
CACHE_DIR = 'real_data_cache'
os.makedirs(CACHE_DIR, exist_ok=True)

# Date range (max available)
START_DATE = '2015-01-01'
END_DATE = datetime.today().strftime('%Y-%m-%d')

for symbol in SYMBOLS:
    print(f"Downloading {symbol}...")
    df = yf.download(symbol, start=START_DATE, end=END_DATE, progress=False)
    if not df.empty:
        df.reset_index(inplace=True)
        out_path = os.path.join(CACHE_DIR, f"{symbol}.csv")
        df.to_csv(out_path, index=False)
        print(f"Saved {symbol} to {out_path}")
    else:
        print(f"No data for {symbol}")
print("Done.") 