# utils.py
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from typing import List, Tuple
from sklearn.metrics import classification_report, accuracy_score
import vectorbt as vbt
import pandas as pd
from IPython.display import display, HTML

def get_top_features(model: XGBClassifier, feature_names: List[str], percentile: float = 0.1) -> List[str]:
    """
    Select top features based on feature importance percentile threshold.

    Args:
        model: Trained XGBClassifier model.
        feature_names: List of all feature column names.
        percentile: Float between 0–1. e.g., 0.1 keeps top 90% of features.

    Returns:
        List of top feature names.
    """
    importance_scores = model.feature_importances_

    # Print all feature importances
    print("\n🔍 Feature Importances:")
    for f, imp in zip(feature_names, importance_scores):
        print(f"{f}: {imp:.5f}")

    cutoff = np.percentile(importance_scores, 100 * (1 - percentile))  # top x%
    top_indices = np.where(importance_scores > cutoff)[0]
    top_features = [feature_names[i] for i in top_indices]

    # Fallback if none pass the threshold
    if len(top_features) == 0:
        print("⚠️ No features passed the threshold. Falling back to top 5 by importance.")
        top_indices = np.argsort(importance_scores)[::-1][:5]
        top_features = [feature_names[i] for i in top_indices]
        print(f"Top features selected: {top_features}")
    else:
        print(f"✅ {len(top_features)} features selected above threshold.")

    return top_features

def summarize_performance(y_true, y_pred, title=""):
    from collections import Counter
    print("Predicted label distribution:", Counter(y_pred))
    report = classification_report(y_true, y_pred, output_dict=True)
    print(f"\n==== {title} ====")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print("F1 Scores:")
    for label, scores in report.items():
        if label in ['0', '1', '2']:
            print(f"  Class {label}: F1 = {scores['f1-score']:.4f}")


def display_trade_log(pf, max_rows=10):
    trades_df = pf.trades.records_readable.copy()

    # Select relevant columns
    trades_df = trades_df[[
        'Entry Timestamp', 'Avg Entry Price',
        'Exit Timestamp', 'Avg Exit Price',
        'PnL', 'Return'
    ]]

    # Color rows based on PnL
    def highlight(row):
        color = 'background-color: lightgreen' if row['PnL'] > 0 else 'background-color: salmon'
        return [color] * len(row)

    styled_table = trades_df.tail(max_rows).style.apply(highlight, axis=1).format({
        'Avg Entry Price': '{:.4f}',
        'Avg Exit Price': '{:.4f}',
        'PnL': '{:.2f}',
        'Return': '{:.2%}'
    })

    # Display nicely in notebook or web view
    try:
        from IPython.display import display
        display(styled_table)
    except ImportError:
        print(trades_df.tail(max_rows).to_string(index=False))

def add_signal_timing_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Days since last BUY (label 2)
    last_buy = None
    since_buy = []
    for i, val in enumerate(df['label']):
        if val == 2:
            last_buy = i
        since_buy.append(i - last_buy if last_buy is not None else -1)
    df['days_since_buy'] = since_buy

    # Days since last SELL (label 0)
    last_sell = None
    since_sell = []
    for i, val in enumerate(df['label']):
        if val == 0:
            last_sell = i
        since_sell.append(i - last_sell if last_sell is not None else -1)
    df['days_since_sell'] = since_sell

    return df

def confidence_filter(preds, probs, threshold=0.6):
    """
    Applies confidence thresholding.
    Replaces predictions with HOLD (1) if max class probability is below threshold.
    """
    filtered_preds = []
    for pred, prob in zip(preds, probs):
        if max(prob) >= threshold:
            filtered_preds.append(pred)
        else:
            filtered_preds.append(1)
    return np.array(filtered_preds)

def cooldown_filter(preds, timestamps, cooldown_days=3):
    """
    Prevents placing new trades (labels 0 or 2) within cooldown_days of the last one.
    Converts such trades to HOLD (1) if too close to previous trade.
    Also returns an audit log for visualization.
    """
    last_trade_time = None
    filtered_preds = []
    audit_log = []

    for pred, ts in zip(preds, timestamps):
        original_pred = pred
        if pred in [0, 2]:
            if last_trade_time is None or (ts - last_trade_time).days >= cooldown_days:
                filtered_preds.append(pred)
                last_trade_time = ts
            else:
                filtered_preds.append(1)  # HOLD due to cooldown
        else:
            filtered_preds.append(pred)
        audit_log.append((ts, original_pred, filtered_preds[-1]))  # Always append actual value used

    print(f"🧊 Cooldown filter applied — {filtered_preds} trades converted to HOLD")
    return np.array(filtered_preds), audit_log

def print_cooldown_summary(cooldown_log):
    print("🧊 Cooldown Filter Summary:")
    print(f"{'Idx':<5}{'Timestamp':<15}{'Raw':<10}{'Filtered':<10}{'Changed':<8}")
    for i, (ts, raw, filt) in enumerate(cooldown_log):
        if raw != filt:
            print(f"{i:<5}{ts.strftime('%Y-%m-%d'):<15}{raw:<10}{filt:<10}✅")

def directional_proximity_filter(preds, close_prices, support, resistance, tolerance=0.03):
    """
    Prevents trades if current price is not close to support/resistance.
    BUY (2) only if close is within tolerance of support.
    SELL (0) only if close is within tolerance of resistance.
    """
    filtered_preds = []
    for pred, price, sup, res in zip(preds, close_prices, support, resistance):
        if pred == 2:  # BUY
            if price <= sup * (1 + tolerance):
                filtered_preds.append(pred)
            else:
                filtered_preds.append(1)  # HOLD
        elif pred == 0:  # SELL
            if price >= res * (1 - tolerance):
                filtered_preds.append(pred)
            else:
                filtered_preds.append(1)  # HOLD
        else:
            filtered_preds.append(pred)
    print("📏 Directional proximity filter applied")
    return np.array(filtered_preds)

def get_price_dict(df: pd.DataFrame, start_idx: int = 0) -> dict:
    """
    Returns a dictionary of sliced close, high, low prices starting from split index.
    """
    return {
        "close": df["close"].iloc[start_idx:],
        "high": df["high"].iloc[start_idx:],
        "low": df["low"].iloc[start_idx:]
    }

def get_split_index(df_len: int) -> int:
    # Currently hardcoded to 80-20 split
    return int(df_len * 0.8)

def get_support_resistance(data: pd.DataFrame, window: int = 10) -> Tuple[pd.Series, pd.Series]:
    """
    Calculates rolling support (min low) and resistance (max high) levels.

    Args:
        prices: DataFrame with 'high' and 'low' columns.
        window: Rolling window size in days.

    Returns:
        Tuple of (support_series, resistance_series)
    """
    split_index = get_split_index(len(data))
    price_dict = get_price_dict(data,split_index)  # Extract close, high, low
    support = price_dict["low"].rolling(window=window, min_periods=1).min()
    resistance = price_dict["high"].rolling(window=window, min_periods=1).max()
    return support, resistance, price_dict["close"]