# data.py
import ccxt
import pandas as pd
from datetime import datetime

def fetch_okx_data(symbol='TON/USDT', timeframe='1d', limit=1500):
    okx = ccxt.okx()
    okx.load_markets()

    # Fetch OHLCV: [timestamp, open, high, low, close, volume]
    ohlcv = okx.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

if __name__ == "__main__":
    df = fetch_okx_data(symbol='BTC/USDT', timeframe='1d')
    print(df.tail())