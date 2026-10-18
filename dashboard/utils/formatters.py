"""
Clinical Metric Formatters
Translates technical ML metrics into plain English for clinical audiences.
"""
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class MetricExplanation:
    """Structured explanation of a metric."""
    label: str
    value: str
    interpretation: str
    clinical_meaning: str
    benchmark: str
    is_good: bool
    color: str  # success, warning, danger, info


# ===== AUROC =====
def explain_auroc(value: float) -> MetricExplanation:
    """Explain AUROC in clinical terms."""
    if value >= 0.80:
        interp, color = "Excellent", "success"
    elif value >= 0.70:
        interp, color = "Good", "success"
    elif value >= 0.60:
        interp, color = "Fair", "warning"
    else:
        interp, color = "Poor", "danger"
    
    return MetricExplanation(
        label="AUROC (Area Under ROC Curve)",
        value=f"{value:.3f}",
        interpretation=interp,
        clinical_meaning=f"Of 100 random patient pairs (one who died, one who survived), "
                         f"the model correctly ranks the higher-risk patient {value:.0%} of the time.",
        benchmark="Random=0.50 | Fair=0.60 | Good=0.70 | Excellent=0.80+",
        is_good=value >= 0.70,
        color=color
    )


# ===== AUPRC =====
def explain_auprc(value: float, prevalence: float = 0.10) -> MetricExplanation:
    """Explain AUPRC in clinical terms."""
    baseline = prevalence
    improvement = (value - baseline) / baseline * 100 if baseline > 0 else 0
    
    if value >= 0.60:
        interp, color = "Excellent", "success"
    elif value >= 0.40:
        interp, color = "Good", "success"
    elif value >= 0.25:
        interp, color = "Fair", "warning"
    else:
        interp, color = "Poor", "danger"
    
    return MetricExplanation(
        label="AUPRC (Area Under Precision-Recall Curve)",
        value=f"{value:.3f}",
        interpretation=interp,
        clinical_meaning=f"Focuses on the minority class (mortality ~{prevalence:.0%}). "
                         f"Baseline random classifier would score {baseline:.2f}. "
                         f"This model is {improvement:+.0f}% better than random.",
        benchmark=f"Random={baseline:.2f} | Fair=0.25 | Good=0.40 | Excellent=0.60+",
        is_good=value >= 0.40,
        color=color
    )


# ===== ACCURACY =====
def explain_accuracy(value: float, prevalence: float = 0.10) -> MetricExplanation:
    """Explain accuracy with class imbalance context."""
    naive_accuracy = max(prevalence, 1 - prevalence)
    
    if value >= 0.90:
        interp, color = "Excellent", "success"
    elif value >= 0.80:
        interp, color = "Good", "success"
    elif value >= 0.70:
        interp, color = "Fair", "warning"
    else:
        interp, color = "Poor", "danger"
    
    return MetricExplanation(
        label="Accuracy",
        value=f"{value:.1%}",
        interpretation=interp,
        clinical_meaning=f"Overall correct predictions. Naive 'always predict survival' "
                         f"would achieve {naive_accuracy:.1%} accuracy. "
                         f"This model adds {value - naive_accuracy:+.1%} value.",
        benchmark=f"Naive baseline={naive_accuracy:.1%} | Good=80%+ | Excellent=90%+",
        is_good=value > naive_accuracy + 0.05,
        color=color
    )


# ===== F1 SCORE =====
def explain_f1(value: float) -> MetricExplanation:
    """Explain F1 score."""
    if value >= 0.70:
        interp, color = "Excellent", "success"
    elif value >= 0.50:
        interp, color = "Good", "success"
    elif value >= 0.35:
        interp, color = "Fair", "warning"
    else:
        interp, color = "Poor", "danger"
    
    return MetricExplanation(
        label="F1 Score",
        value=f"{value:.3f}",
        interpretation=interp,
        clinical_meaning="Harmonic mean of precision and recall. "
                         "Balances false alarms vs missed cases. "
                         "Critical for mortality where both errors have costs.",
        benchmark="Poor<0.35 | Fair=0.35-0.50 | Good=0.50-0.70 | Excellent>0.70",
        is_good=value >= 0.50,
        color=color
    )


# ===== BRIER SCORE =====
def explain_brier(value: float) -> MetricExplanation:
    """Explain Brier score (lower is better)."""
    if value <= 0.10:
        interp, color = "Excellent", "success"
    elif value <= 0.15:
        interp, color = "Good", "success"
    elif value <= 0.20:
        interp, color = "Fair", "warning"
    else:
        interp, color = "Poor", "danger"
    
    return MetricExplanation(
        label="Brier Score (Lower=Better)",
        value=f"{value:.4f}",
        interpretation=interp,
        clinical_meaning="Mean squared difference between predicted probabilities "
                         "and actual outcomes. 0 = perfect calibration. "
                         "Measures both discrimination AND calibration.",
        benchmark="Excellent<0.10 | Good=0.10-0.15 | Fair=0.15-0.20 | Poor>0.20",
        is_good=value <= 0.15,
        color=color
    )


# ===== ECE =====
def explain_ece(value: float) -> MetricExplanation:
    """Explain Expected Calibration Error."""
    if value <= 0.02:
        interp, color = "Well Calibrated", "success"
    elif value <= 0.05:
        interp, color = "Acceptable", "success"
    elif value <= 0.10:
        interp, color = "Moderate Miscalibration", "warning"
    else:
        interp, color = "Poor Calibration", "danger"
    
    return MetricExplanation(
        label="ECE (Expected Calibration Error)",
        value=f"{value:.4f}",
        interpretation=interp,
        clinical_meaning="Average difference between predicted risk and observed frequency. "
                         "If ECE=0.05, a patient given 30% risk actually has 25-35% risk. "
                         "Critical for clinical decision thresholds.",
        benchmark="Well Calibrated<0.02 | Acceptable<0.05 | Moderate<0.10 | Poor>0.10",
        is_good=value <= 0.05,
        color=color
    )


# ===== PRIVACY EPSILON =====
def explain_epsilon(eps: float) -> MetricExplanation:
    """Explain differential privacy epsilon."""
    if eps >= 10:
        risk, color = "Low Risk", "success"
    elif eps >= 5:
        risk, color = "Moderate Risk", "warning"
    elif eps >= 1:
        risk, color = "High Risk", "warning"
    else:
        risk, color = "Very High Risk", "danger"
    
    return MetricExplanation(
        label="Privacy Budget (ε)",
        value=f"{eps:.1f}",
        interpretation=risk,
        clinical_meaning=f"Lower ε = stronger privacy. At ε={eps:.1f}, an attacker's confidence "
                         f"in whether a specific patient was in the training data changes by at most "
                         f"{100*(1-1/(2.718**eps)):.0f}%. "
                         f"Clinical data typically requires ε=1-10.",
        benchmark="High privacy<1 | Balanced=1-5 | Practical=5-10 | Low privacy>10",
        is_good=eps >= 5,
        color=color
    )


# ===== SPEARMAN RHO =====
def explain_spearman_rho(rho: float) -> MetricExplanation:
    """Explain cross-hospital consistency."""
    if rho >= 0.80:
        interp, color = "Strong Agreement", "success"
    elif rho >= 0.60:
        interp, color = "Moderate Agreement", "success"
    elif rho >= 0.40:
        interp, color = "Weak Agreement", "warning"
    else:
        interp, color = "Poor Agreement", "danger"
    
    return MetricExplanation(
        label="Spearman ρ (Rank Correlation)",
        value=f"{rho:.3f}",
        interpretation=interp,
        clinical_meaning=f"Hospitals rank feature importance {rho:.0%} similarly. "
                         f"High ρ means the model learns generalizable clinical patterns, "
                         f"not hospital-specific artifacts.",
        benchmark="Poor<0.40 | Weak=0.40-0.60 | Moderate=0.60-0.80 | Strong>0.80",
        is_good=rho >= 0.60,
        color=color
    )


# ===== JACCARD =====
def explain_jaccard(jac: float, k: int = 5) -> MetricExplanation:
    """Explain top-k feature overlap."""
    if jac >= 0.50:
        interp, color = "High Overlap", "success"
    elif jac >= 0.30:
        interp, color = "Moderate Overlap", "success"
    elif jac >= 0.15:
        interp, color = "Low Overlap", "warning"
    else:
        interp, color = "Minimal Overlap", "danger"
    
    return MetricExplanation(
        label=f"Top-{k} Jaccard Similarity",
        value=f"{jac:.3f}",
        interpretation=interp,
        clinical_meaning=f"Of the top {k} features each hospital considers most important, "
                         f"{jac:.0%} are the same features. "
                         f"Low overlap suggests different patient populations drive different risk factors.",
        benchmark="Minimal<0.15 | Low=0.15-0.30 | Moderate=0.30-0.50 | High>0.50",
        is_good=jac >= 0.30,
        color=color
    )


# ===== SURROGATE FIDELITY =====
def explain_fidelity(fidelity: float) -> MetricExplanation:
    """Explain surrogate model fidelity."""
    if fidelity >= 0.90:
        interp, color = "Excellent", "success"
    elif fidelity >= 0.80:
        interp, color = "Good", "success"
    elif fidelity >= 0.70:
        interp, color = "Acceptable", "warning"
    else:
        interp, color = "Poor", "danger"
    
    return MetricExplanation(
        label="Surrogate Fidelity",
        value=f"{fidelity:.1%}",
        interpretation=interp,
        clinical_meaning=f"The simple decision tree matches the complex federated model's "
                         f"predictions {fidelity:.0%} of the time. "
                         f"Clinicians can trust the IF-THEN rules as faithful approximations.",
        benchmark="Poor<70% | Acceptable=70-80% | Good=80-90% | Excellent>90%",
        is_good=fidelity >= 0.80,
        color=color
    )


# ===== MODEL COMPARISON =====
def compare_settings(federated_auroc: float, centralized_auroc: float, local_auroc: float) -> Dict[str, Any]:
    """Generate comparison insights."""
    gain_vs_central = federated_auroc - centralized_auroc
    gain_vs_local = federated_auroc - local_auroc
    
    insights = []
    
    if gain_vs_central > 0:
        insights.append({
            "type": "success",
            "title": "Federated beats Centralized!",
            "text": f"Federated learning (+{gain_vs_central:.3f} AUROC) outperforms centralized training. "
                    f"This happens because each hospital learns different decision boundaries "
                    f"from their local population, and combining them acts like ensemble learning."
        })
    elif gain_vs_central > -0.02:
        insights.append({
            "type": "info",
            "title": "Near-Parity with Centralized",
            "text": f"Federated is within {abs(gain_vs_central):.3f} AUROC of centralized — "
                    f"achieving comparable performance WITHOUT sharing raw data."
        })
    else:
        insights.append({
            "type": "warning",
            "title": "Gap to Centralized",
            "text": f"Federated trails centralized by {abs(gain_vs_central):.3f} AUROC. "
                    f"Consider more clients, different aggregation, or more local data."
        })
    
    if gain_vs_local > 0.05:
        insights.append({
            "type": "success",
            "title": "Significant Gain Over Local-Only",
            "text": f"Collaboration adds +{gain_vs_local:.3f} AUROC vs hospitals training alone. "
                    f"Each hospital benefits from others' data patterns."
        })
    elif gain_vs_local > 0:
        insights.append({
            "type": "info",
            "title": "Modest Gain Over Local-Only",
            "text": f"Federated improves on local-only by {gain_vs_local:.3f} AUROC."
        })
    
    return {
        "federated_auroc": federated_auroc,
        "centralized_auroc": centralized_auroc,
        "local_auroc": local_auroc,
        "gain_vs_central": gain_vs_central,
        "gain_vs_local": gain_vs_local,
        "insights": insights
    }


# ===== FORMATTER REGISTRY =====
FORMATTERS = {
    "auroc": explain_auroc,
    "auprc": explain_auprc,
    "accuracy": explain_accuracy,
    "f1": explain_f1,
    "brier": explain_brier,
    "ece": explain_ece,
    "epsilon": explain_epsilon,
    "spearman_rho": explain_spearman_rho,
    "jaccard": explain_jaccard,
    "fidelity": explain_fidelity,
}


def format_metric(metric_name: str, value: float, **kwargs) -> MetricExplanation:
    """Get formatted explanation for any metric."""
    formatter = FORMATTERS.get(metric_name)
    if formatter:
        return formatter(value, **kwargs)
    return MetricExplanation(
        label=metric_name.replace("_", " ").title(),
        value=f"{value:.4f}",
        interpretation="Unknown",
        clinical_meaning="No explanation available.",
        benchmark="N/A",
        is_good=False,
        color="gray"
    )