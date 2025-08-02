# signals.py
import pandas as pd
import joblib
from data import fetch_okx_data
from features import add_technical_indicators

def generate_signals(df: pd.DataFrame, model_path='xgb_model.pkl', feature_names=None) -> pd.DataFrame:
    df = df.copy()

    # Load model
    model = joblib.load(model_path)

    if feature_names is None:
        raise ValueError("feature_names must be provided to match training set.")

    # Drop any rows with NaNs in the feature set to avoid leakage-related artifacts
    df = df.dropna(subset=feature_names)

    # Predict
    X = df[feature_names]
    df['predicted_label'] = model.predict(X)

    # Signals for vectorbt
    df['entry_signal'] = df['predicted_label'] == 2
    df['exit_signal'] = df['predicted_label'] == 0

    return df

if __name__ == "__main__":
    df = fetch_okx_data(symbol='TON/USDT', timeframe='1d')
    df = add_technical_indicators(df)

    # Add any additional feature steps here if needed for testing
    # For example:
    # df = add_time_features(df)
    # df = add_signal_timing_features(df)

    feature_names = [...]  # specify your feature columns here

    df = generate_signals(df, feature_names=feature_names)
    print(df[['close', 'predicted_label', 'entry_signal', 'exit_signal']].tail(10))