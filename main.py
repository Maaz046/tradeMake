# main.py

from data import fetch_okx_data
from features import (
    add_technical_indicators,
    add_time_features
)
from labeling import label_data
from model import train_model
from signals import generate_signals
from backtest import run_backtest
from utils import add_signal_timing_features

def run_pipeline():
    print("running main")
    model_path = 'xgb_model.pkl'
    
    # 1. Fetch historical OHLCV data
    df = fetch_okx_data(symbol='TON/USDT', timeframe='1d')

    # 2. Add technical indicators
    df = add_technical_indicators(df)

    # 3. Add time-based features
    df = add_time_features(df)

    # 4. Label the data with buy/hold/sell
    df_labeled = label_data(df)

    # 5. Add signal timing context features (after labeling)
    df_labeled = add_signal_timing_features(df_labeled)

    # 6. Train the model and select top features
    print("🔁 Training model...")
    model, top_features = train_model(df_labeled, model_path=model_path)
    print("✅ Model training completed")

    # 7. Generate predictions and trading signals
    df_signals = generate_signals(df_labeled, model_path=model_path, feature_names=top_features)

    # 8. Run backtest
    run_backtest(df_signals)

if __name__ == "__main__":
    run_pipeline()