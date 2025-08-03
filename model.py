# model.py
import pandas as pd
from xgboost import XGBClassifier,plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import numpy as np
import matplotlib.pyplot as plt
from utils import get_top_features, summarize_performance
from collections import Counter
from inference import run_inference


def train_model(df: pd.DataFrame, feature_cols=None, model_path='xgb_model.pkl'):
    df = df.copy()

    if feature_cols is None:
        feature_cols = [col for col in df.columns if col not in [
            'open', 'high', 'low', 'close', 'volume', 'future_return', 'label']]

    X = df[feature_cols]
    y = df['label']

    # === Sanity check: shuffle labels if activated
    import random
    sanity_check = False  # Set to True to activate label shuffling
    if sanity_check:
        print("⚠️ Sanity check active: shuffling labels...")
        y = y.sample(frac=1.0, random_state=42).reset_index(drop=True)
        X = X.reset_index(drop=True)

    # ❗ Filter rows with any NaN in features or label
    mask = (~X.isna().any(axis=1)) & (~y.isna())
    X = X[mask]
    y = y[mask]

    # Debug print
    print("Training set shape:", X.shape)
    print("Unique training labels:", np.unique(y))

    print("✅ Reached before train/test split")
    # Train/test split
    split_index = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    print("✅ Reached after train/test split")

    from collections import Counter
    print("Train Label Distribution:", Counter(y_train))
    print("Test Label Distribution:", Counter(y_test))

    # Final check
    assert not X.isna().any().any(), "NaNs still present in X"
    assert not y.isna().any(), "NaNs still present in y"

    if X.empty or y.empty:
        print("❌ Training aborted: X or y is empty after preprocessing.")
        return None, []

    # Train/test split
    split_index = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    from xgboost import XGBClassifier
    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=1.0,
        objective='multi:softmax',
        num_class=3,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )

    print("✅ Training model with all features")
    # === 1. Train on all features
    model.fit(X_train, y_train)
    print("✅ Predicting...")
    preds_all, _ = run_inference(model, X_test)

    # === 2. Evaluate performance on all features
    try:
        print("Initial Classification Report (all features):\n", classification_report(y_test, preds_all))
    except ValueError as e:
        print("⚠️ Skipping classification report due to error:", e)
        print("y_test values:", np.unique(y_test))
        print("predictions:", np.unique(preds_all))
    print("✅ Summarizing performance BEFORE tuning")
    summarize_performance(y_test, preds_all, "Before Indicator Tuning")

    # === 3. Plot feature importance
    # plot_importance(model, max_num_features=15)
    # plt.tight_layout()
    # plt.show()

    # === 4. Select top features using importance percentile
    top_features = get_top_features(model, X_train.columns.tolist(), percentile=0.7)
    print("Top features selected:", top_features)

    # === 5. Retrain model using only top features
    model.fit(X_train[top_features], y_train)
    preds_top, _ = run_inference(model, X_test[top_features])

    # === 6. Evaluate performance after tuning
    try:
        print("Final Classification Report (top features):\n", classification_report(y_test, preds_top))    
    except ValueError as e:
        print("⚠️ Skipping classification report (top features) due to error:", e)
        print("y_test values:", np.unique(y_test))
        print("predictions:", np.unique(preds_top))
    print("✅ Summarizing performance AFTER tuning")
    summarize_performance(y_test, preds_top, "After Indicator Tuning")

    # === 7. Save model
    joblib.dump(model, model_path)

    # Return the trained model, list of top features, and test sets for further evaluation
    return model, top_features, X_test, y_test

if __name__ == "__main__":
    from data import fetch_okx_data
    from features import add_technical_indicators
    from labeling import label_data

    df = fetch_okx_data(symbol='TON/USDT', timeframe='1d')
    df = add_technical_indicators(df)
    df = label_data(df)

    model = train_model(df)