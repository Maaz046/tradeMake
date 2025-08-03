# inference.py

from visualization import plot_prediction_confidences
from utils import confidence_filter, cooldown_filter

"""
This module handles inference logic for the trading model,
including prediction, probability generation, and signal filtering.
"""
import numpy as np

def predict_labels(model, X):
    """Predicts class labels."""
    return model.predict(X)

def predict_probs(model, X):
    """Returns class probabilities for each input."""
    return model.predict_proba(X)

def apply_filters(preds, probs, timestamps, threshold=0.6):
    """
    Applies all filters sequentially: confidence and cooldown.
    Logs which labels were changed due to cooldown.
    """
    preds = confidence_filter(preds, probs, threshold)
    preds, cooldown_log = cooldown_filter(preds, timestamps)

    print("🧊 Cooldown Filter Summary:")
    print(f"{'Idx':<5}{'Timestamp':<15}{'Raw':<10}{'Filtered':<10}{'Changed':<8}")
    for i, (ts, raw, filt) in enumerate(cooldown_log):
        changed = "✅" if raw != filt else ""
        if changed:  # Show only changed rows
            print(f"{i:<5}{ts.strftime('%Y-%m-%d'):<15}{raw:<10}{filt:<10}{changed:<8}")

    return preds

def run_raw_inference(model, X):
    """Returns raw predictions and probabilities without filtering."""
    return predict_labels(model, X), predict_probs(model, X)

def run_inference(model, X, timestamps, threshold=0.6, apply_filter=True):
    """
    Runs inference on input features X.

    Parameters:
        model: Trained classifier.
        X: Feature matrix.
        threshold: Probability threshold for signal filtering.
        apply_filter: If True, filters low-confidence predictions.

    Returns:
        Tuple of (final predictions, prediction probabilities)
    """
    raw_preds = predict_labels(model, X)
    probs = predict_probs(model, X)
    # Uncomment if needed
    # plot_prediction_confidences(probs, raw_preds, threshold)

    if apply_filter:
        final_preds = apply_filters(raw_preds, probs, timestamps, threshold)
    else:
        final_preds = raw_preds

    return final_preds, probs