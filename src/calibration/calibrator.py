"""
Probability Calibration and Confidence Assessment for Deepfake Detection.

Features:
  - Temperature Scaling
  - Expected Calibration Error (ECE) & Brier Score
  - Categorization into REAL, UNCERTAIN, and DEEPFAKE
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Tuple, Dict, Any


REAL_THRESHOLD = 0.35
FAKE_THRESHOLD = 0.65


class Calibrator:
    """
    Platt / Temperature probability calibrator.
    Maps uncalibrated scores to well-calibrated probabilities.
    """
    def __init__(self, temperature: float = 1.0):
        self.temperature = float(temperature)

    def fit_temperature(self, uncalibrated_probs: np.ndarray, labels: np.ndarray):
        """Find temperature T using cross-entropy minimization."""
        eps = 1e-6
        clipped = np.clip(uncalibrated_probs, eps, 1 - eps)
        logits = np.log(clipped / (1 - clipped))

        # Grid search over T in [0.2, 5.0]
        best_t, best_loss = 1.0, float('inf')
        for t in np.linspace(0.2, 5.0, 100):
            scaled_probs = 1.0 / (1.0 + np.exp(-logits / t))
            loss = -np.mean(labels * np.log(scaled_probs + eps) + (1 - labels) * np.log(1 - scaled_probs + eps))
            if loss < best_loss:
                best_loss = loss
                best_t = t

        self.temperature = float(best_t)
        return self.temperature

    def calibrate(self, prob: float) -> float:
        """Calibrate a single probability value."""
        eps = 1e-6
        p = np.clip(prob, eps, 1 - eps)
        logit = np.log(p / (1 - p))
        calibrated = 1.0 / (1.0 + np.exp(-logit / self.temperature))
        return float(np.clip(calibrated, 0.0, 1.0))

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({'temperature': self.temperature}, f)

    def load(self, path: str):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.temperature = float(data.get('temperature', 1.0))


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE)."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(probs)
    if total == 0:
        return 0.0

    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if np.sum(mask) > 0:
            bin_acc = np.mean(labels[mask])
            bin_conf = np.mean(probs[mask])
            ece += (np.sum(mask) / total) * abs(bin_acc - bin_conf)

    return float(ece)


def compute_brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Compute Brier Score: mean((prob - label)^2). Lower is better."""
    return float(np.mean((probs - labels) ** 2))


def classify_decision(fake_probability: float) -> Dict[str, Any]:
    """
    Apply calibrated decision logic:
      - fake_prob < 0.35  -> REAL
      - 0.35 <= fake_prob <= 0.65 -> UNCERTAIN
      - fake_prob > 0.65  -> DEEPFAKE
    """
    if fake_probability > FAKE_THRESHOLD:
        pred = "DEEPFAKE"
    elif fake_probability < REAL_THRESHOLD:
        pred = "REAL"
    else:
        pred = "UNCERTAIN"

    # Confidence is distance from maximum uncertainty (0.5)
    confidence_val = float(abs(fake_probability - 0.5) * 2.0)

    if confidence_val >= 0.70:
        level = "HIGH"
    elif confidence_val >= 0.35:
        level = "MEDIUM"
    else:
        level = "LOW"

    return {
        "prediction": pred,
        "confidence_score": round(confidence_val, 4),
        "confidence_level": level,
        "fake_probability": round(fake_probability, 4),
        "real_probability": round(1.0 - fake_probability, 4),
    }
