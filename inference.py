# inference.py
import numpy as np

def predict_labels(model, X):
    return model.predict(X)

def predict_probs(model, X):
    return model.predict_proba(X)

def apply_filters(preds, probs, threshold=0.6):
    filtered_preds = []
    for pred, prob in zip(preds, probs):
        if max(prob) >= threshold:
            filtered_preds.append(pred)
        else:
            filtered_preds.append(1)  # default to HOLD
    return np.array(filtered_preds)

def run_inference(model, X, threshold=0.6):
    raw_preds = predict_labels(model, X)
    probs = predict_probs(model, X)
    filtered_preds = apply_filters(raw_preds, probs, threshold)
    return filtered_preds, probs