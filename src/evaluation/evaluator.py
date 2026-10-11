"""
Evaluation utilities for federated healthcare risk prediction:
- Comprehensive classification metrics (Accuracy, Precision, Recall, F1, AUROC, AUPRC, Brier, ECE)
- Reliability & Calibration curve generation
- Privacy-Utility-Explainability Trade-off evaluator
"""
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
)
from sklearn.calibration import calibration_curve


def compute_expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute Expected Calibration Error (ECE):
    Weighted average difference between predicted confidence and observed accuracy across bins.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total_samples = len(y_true)

    for i in range(n_bins):
        in_bin = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        bin_count = np.sum(in_bin)
        if bin_count > 0:
            bin_acc = np.mean(y_true[in_bin])
            bin_conf = np.mean(y_prob[in_bin])
            ece += (bin_count / total_samples) * np.abs(bin_acc - bin_conf)

    return float(ece)


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute comprehensive clinical predictive evaluation metrics.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if y_prob is not None:
        y_prob = np.asarray(y_prob, dtype=float)
        try:
            metrics['auroc'] = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            metrics['auroc'] = 0.5

        try:
            metrics['auprc'] = float(average_precision_score(y_true, y_prob))
        except ValueError:
            metrics['auprc'] = float(np.mean(y_true))

        metrics['brier'] = float(brier_score_loss(y_true, y_prob))
        metrics['ece'] = compute_expected_calibration_error(y_true, y_prob)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics['specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    metrics['npv'] = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0

    return metrics


def get_calibration_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute calibration curve prob_true and prob_pred."""
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')
    return prob_true, prob_pred
