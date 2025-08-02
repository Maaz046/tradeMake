# labeling.py
import pandas as pd
import numpy as np

def label_data(df: pd.DataFrame, forward_days=1, pos_thresh=0.01, neg_thresh=-0.01) -> pd.DataFrame:
    df = df.copy()

    # Compute forward return
    df['future_return'] = df['close'].pct_change(periods=forward_days).shift(-forward_days)

    # Drop any rows where return can't be computed
    df = df.dropna(subset=['future_return'])

    # Label using np.select to avoid NaN issues
    conditions = [
        df['future_return'] < neg_thresh,          # Sell
        (df['future_return'] >= neg_thresh) & (df['future_return'] <= pos_thresh),  # Hold
        df['future_return'] > pos_thresh           # Buy
    ]
    choices = [0, 1, 2]  # 0 = Sell, 1 = Hold, 2 = Buy
    df['label'] = np.select(conditions, choices, default=1)

    # Sanity check
    df = df[df['label'].isin([0, 1, 2])]  # remove any -1 from np.select default

    print(df[['future_return', 'label']].tail(10))
    print("Unique labels:", df['label'].unique())

    return df

if __name__ == "__main__":
    from data import fetch_okx_data
    from features import add_technical_indicators
    
    df = fetch_okx_data(symbol='TON/USDT', timeframe='1d')
    df = add_technical_indicators(df)
    df = label_data(df)
    print(df[['close', 'future_return', 'label']].tail(10))