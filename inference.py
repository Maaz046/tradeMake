# inference.py
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

def apply_filters(preds, probs, threshold=0.6):
    """
    Applies confidence thresholding.
    Returns HOLD (1) if confidence is too low.
    """
    filtered_preds = []
    for pred, prob in zip(preds, probs):
        # If the highest class probability meets the threshold, keep prediction
        if max(prob) >= threshold:
            filtered_preds.append(pred)
        else:
            filtered_preds.append(1)  # HOLD as fallback
    return np.array(filtered_preds)

def run_raw_inference(model, X):
    """Returns raw predictions and probabilities without filtering."""
    return predict_labels(model, X), predict_probs(model, X)

def run_inference(model, X, threshold=0.6, apply_filter=True):
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

    if apply_filter:
        final_preds = apply_filters(raw_preds, probs, threshold)
    else:
        final_preds = raw_preds

    return final_preds, probs