"""
Rule card components for clinical decision rules.
"""
import streamlit as st
from typing import Dict, Any, List
import re


def parse_clinical_rules(text: str) -> List[Dict[str, Any]]:
    """Parse clinical rules text into structured list."""
    rules = []
    lines = text.strip().split("\n")
    current_rule = {}
    for line in lines:
        line = line.strip()
        if line.startswith("Rule "):
            if current_rule:
                rules.append(current_rule)
            current_rule = {"text": line}
        elif line.startswith("IF"):
            current_rule["condition"] = line
            # Extract risk and support
            risk_match = re.search(r"Risk:\s*([\d.]+)%", line)
            support_match = re.search(r"Support:\s*(\d+)", line)
            pred_match = re.search(r"Mortality\s*=\s*(\d+)", line)
            if risk_match:
                current_rule["risk"] = float(risk_match.group(1))
            if support_match:
                current_rule["support"] = int(support_match.group(1))
            if pred_match:
                current_rule["prediction"] = int(pred_match.group(1))
    if current_rule:
        rules.append(current_rule)
    return rules


def rule_card(rule: Dict[str, Any], index: int):
    """Display a clinical rule as an expandable card."""
    risk = rule.get("risk", 0)
    support = rule.get("support", 0)
    prediction = rule.get("prediction", 0)
    condition = rule.get("condition", "")
    text = rule.get("text", "")
    
    # Color based on risk
    if risk >= 50:
        risk_color = "🔴"
        risk_label = "Critical Risk"
    elif risk >= 30:
        risk_color = "🟠"
        risk_label = "High Risk"
    elif risk >= 10:
        risk_color = "🟡"
        risk_label = "Moderate Risk"
    else:
        risk_color = "🟢"
        risk_label = "Low Risk"
    
    pred_label = "Mortality" if prediction == 1 else "Survival"
    
    with st.expander(f"{risk_color} Rule {index+1}: {pred_label} | Risk: {risk:.1f}% | Support: {support} patients"):
        st.markdown(f"**Condition:** `{condition}`")
        st.markdown(f"**Prediction:** {pred_label} (Class {prediction})")
        st.markdown(f"**Mortality Risk:** {risk:.1f}%")
        st.markdown(f"**Support:** {support} patients")
        st.markdown(f"**Risk Level:** {risk_label}")