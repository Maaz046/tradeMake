import pandas as pd
import numpy as np

def label_data(df: pd.DataFrame, forward_days=2) -> pd.DataFrame:
    df = df.copy()

    # Calculate ATR
    df['high_low'] = df['high'] - df['low']
    df['high_close'] = np.abs(df['high'] - df['close'].shift(1))
    df['low_close'] = np.abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
    df['atr'] = df['tr'].rolling(window=14).mean()

    # Compute forward return
    df['future_return'] = df['close'].pct_change(periods=forward_days).shift(-forward_days)

    # Drop any rows where return or ATR can't be computed
    df = df.dropna(subset=['future_return', 'atr'])

    # Dynamic thresholds based on ATR
    df['pos_thresh'] = df['atr'] * 0.15
    df['neg_thresh'] = df['atr'] * -0.15

    # Logging ATR and thresholds
    print("\n🔍 ATR & Thresholds (last 10 rows):")
    print(df[['atr', 'pos_thresh', 'neg_thresh']].tail(10))

    # Label using np.select to avoid NaN issues
    conditions = [
        df['future_return'] < df['neg_thresh'],          # Sell
        (df['future_return'] >= df['neg_thresh']) & (df['future_return'] <= df['pos_thresh']),  # Hold
        df['future_return'] > df['pos_thresh']           # Buy
    ]
    choices = [0, 1, 2]  # 0 = Sell, 1 = Hold, 2 = Buy
    df['label'] = np.select(conditions, choices, default=1)

    # Log future_return samples and counts
    print("\n📈 Sample future_return (last 10 rows):")
    print(df[['future_return', 'label']].tail(10))

    print("\n🔢 Label counts:")
    print(df['label'].value_counts())

    # Count threshold breaches
    num_buy = (df['future_return'] > df['pos_thresh']).sum()
    num_sell = (df['future_return'] < df['neg_thresh']).sum()
    num_hold = ((df['future_return'] >= df['neg_thresh']) & (df['future_return'] <= df['pos_thresh'])).sum()
    print(f"\n📊 Threshold breaches — BUY: {num_buy}, SELL: {num_sell}, HOLD: {num_hold}")

    # Compute holding duration and return only for labeled trades (non-Hold)
    trade_indices = df[df['label'] != 1].index

    durations = []
    returns = []
    for i in trade_indices:
        entry_idx = df.index.get_loc(i)
        entry_price = df.loc[i, 'close']
        entry_label = df.loc[i, 'label']

        for j in range(entry_idx + 1, len(df)):
            exit_i = df.index[j]
            exit_price = df.loc[exit_i, 'close']
            duration = (
                (df.loc[exit_i, 'timestamp'] - df.loc[i, 'timestamp']).total_seconds() / 86400
                if 'timestamp' in df.columns else float(j - entry_idx)
            )
            ret = (exit_price - entry_price) / entry_price if entry_price else 0

            if entry_label == 2 and ret < 0:  # Buy but it went down
                durations.append(duration)
                returns.append(ret)
                break
            elif entry_label == 0 and ret > 0:  # Sell but it went up
                durations.append(duration)
                returns.append(ret)
                break
            elif duration >= forward_days:
                durations.append(duration)
                returns.append(ret)
                break

    # Ensure durations are stored as floats (in days)
    durations = [
        d.total_seconds() / 86400 if isinstance(d, pd.Timedelta) else float(d)
        for d in durations
    ]
    df_trades = pd.DataFrame({'duration': durations, 'return': returns})
    print(f"\n⏱️ Avg Duration: {np.mean(durations):.2f} days")
    print(f"💰 Avg Return: {np.mean(returns):.4f}")

    # Final label integrity check
    df = df[df['label'].isin([0, 1, 2])]
    print("\n✅ Final Label Distribution:", df['label'].value_counts(normalize=True).to_dict())

    return df, df_trades

if __name__ == "__main__":
    from data import fetch_okx_data
    from features import add_technical_indicators
    
    df = fetch_okx_data(symbol='TON/USDT', timeframe='1d')
    df = add_technical_indicators(df)
    df = label_data(df, forward_days=2)
    print(df[['close', 'future_return', 'label']].tail(10))