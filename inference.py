# inference.py

from visualization import plot_prediction_confidences
from utils import confidence_filter, cooldown_filter, directional_proximity_filter, print_cooldown_summary, get_price_dict, get_support_resistance

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

def apply_filters(preds, probs, timestamps, df, threshold=0.6):
    """
    Applies all filters sequentially: confidence and cooldown.
    Logs which labels were changed due to cooldown.
    """
    preds = confidence_filter(preds, probs, threshold)
    preds, cooldown_log = cooldown_filter(preds, timestamps)

    support, resistance, close = get_support_resistance(df)
    directional_proximity_filter(preds, close, support, resistance)

    print_cooldown_summary(cooldown_log)

    return preds

def run_raw_inference(model, X):
    """Returns raw predictions and probabilities without filtering."""
    return predict_labels(model, X), predict_probs(model, X)

def run_inference(model, X, timestamps, df, threshold=0.6, apply_filter=True):
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
        final_preds = apply_filters(raw_preds, probs, timestamps, df, threshold)
    else:
        final_preds = raw_preds

    return final_preds, probs