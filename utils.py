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
    Converts such trades to HOLD (1).
    """
    last_trade_time = None
    filtered_preds = []
    for pred, ts in zip(preds, timestamps):
        if pred in [0, 2]:
            if last_trade_time is None or (ts - last_trade_time).days >= cooldown_days:
                filtered_preds.append(pred)
                last_trade_time = ts
            else:
                filtered_preds.append(1)  # HOLD due to cooldown
        else:
            filtered_preds.append(pred)
    print(f"🧊 Cooldown filter applied — {filtered_preds} trades converted to HOLD")
    return np.array(filtered_preds)