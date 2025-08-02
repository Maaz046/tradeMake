import ta
import pandas as pd

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']

    # Trend
    df['sma_10'] = ta.trend.SMAIndicator(close, 10).sma_indicator().shift(1)
    df['ema_10'] = ta.trend.EMAIndicator(close, 10).ema_indicator().shift(1)
    df['ema_20'] = ta.trend.EMAIndicator(close, 20).ema_indicator().shift(1)
    df['ema_50'] = ta.trend.EMAIndicator(close, 50).ema_indicator().shift(1)
    df['adx'] = ta.trend.ADXIndicator(high, low, close).adx().shift(1)

    # Momentum
    df['rsi'] = ta.momentum.RSIIndicator(close).rsi().shift(1)
    macd = ta.trend.MACD(close)
    df['macd'] = macd.macd().shift(1)
    df['macd_signal'] = macd.macd_signal().shift(1)
    df['macd_diff'] = macd.macd_diff().shift(1)
    df['roc'] = ta.momentum.ROCIndicator(close).roc().shift(1)
    df['cci'] = ta.trend.CCIIndicator(high, low, close).cci().shift(1)
    df['stoch_rsi'] = ta.momentum.StochRSIIndicator(close).stochrsi().shift(1)

    # Volatility
    boll = ta.volatility.BollingerBands(close)
    df['bb_upper'] = boll.bollinger_hband().shift(1)
    df['bb_lower'] = boll.bollinger_lband().shift(1)
    df['bb_width'] = df['bb_upper'] - df['bb_lower']
    df['atr'] = ta.volatility.AverageTrueRange(high, low, close).average_true_range().shift(1)
    df['volatility_10'] = close.pct_change().rolling(10).std().shift(1)

    # Volume
    df['vol_sma_10'] = volume.rolling(10).mean().shift(1)
    df['vol_delta'] = volume.pct_change().shift(1)
    df['obv'] = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume().shift(1)

    # Candle shape
    df['candle_wick'] = ((high - low) / (close - low + 1e-6)).shift(1)
    df['body_pct'] = ((close - low) / (high - low + 1e-6)).shift(1)

    df.dropna(inplace=True)
    return df

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['day_of_week'] = df.index.dayofweek  # 0 = Monday, ..., 6 = Sunday
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    return df

def add_signal_timing_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Days since last BUY (label == 2)
    last_buy_idx = None
    days_since_buy = []
    for i, label in enumerate(df['label']):
        if label == 2:
            last_buy_idx = i
        days_since_buy.append(i - last_buy_idx if last_buy_idx is not None else -1)
    df['days_since_buy'] = days_since_buy

    # Days since last SELL (label == 0)
    last_sell_idx = None
    days_since_sell = []
    for i, label in enumerate(df['label']):
        if label == 0:
            last_sell_idx = i
        days_since_sell.append(i - last_sell_idx if last_sell_idx is not None else -1)
    df['days_since_sell'] = days_since_sell

    return df