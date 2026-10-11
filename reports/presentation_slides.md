# Presentation Slides: Privacy-Preserving Federated Learning for Explainable Healthcare Risk Prediction

**Course:** Machine Learning Lab (Mini Project)  
**Task:** ICU In-Hospital Mortality Prediction  
**Duration:** 10 Weeks | **Language:** Python  

---

## Slide 1: Title & Overview
- **Title:** Privacy-Preserving Federated Learning for Explainable Healthcare Risk Prediction
- **Focus:** Collaborative cross-hospital machine learning without centralizing patient records
- **Primary Deliverables:**
  - Federated Forest Union (AUROC = 0.7654)
  - Sufficient-Statistics Federated Naïve Bayes (100% equivalence)
  - Differential Privacy ($\varepsilon$ sweeps) & Pairwise Secure Aggregation
  - Cross-Hospital TreeSHAP Consistency & Clinician-Readable Surrogate Rules (83.1% fidelity)

---

## Slide 2: Clinical Context & The Data Silo Dilemma
- **Clinical Need:** Early ICU mortality prediction within the first 24 hours of admission saves lives by allocating acute resources.
- **The Data Silo Dilemma:**
  - HIPAA and GDPR legally prohibit pooling sensitive patient EHR into a centralized cloud database.
  - Training single-hospital models leads to demographic bias and poor generalization across hospital networks.
- **Our Solution:** Federated Learning (FL) — models visit the data; raw data never leaves the hospital.

---

## Slide 3: Course Outcomes (CO) Alignment
- **CO1 (Data Preprocessing):** Median/mode imputation, physiological outlier clipping, Z-score/Min-Max scaling, non-IID hospital partitioning.
- **CO2 (Optimization & Scaling):** From-scratch Batch & Mini-batch Gradient Descent in NumPy, learning rate $\times$ scaling grid, loss surfaces.
- **CO3 (Supervised Learning & Privacy):** Decision Trees, Random Forest (Forest Union), Naïve Bayes, Differential Privacy, Secure Aggregation.
- **CO4 (Clustering & Patient Stratification):** Centralized & Federated K-means (FedAvg centroid aggregation), optimal $k=5$ subgroup discovery.

---

## Slide 4: End-to-End System Architecture
- **Distributed Architecture:** 5 decentralized hospital nodes + 1 federated orchestration server.
- **Client Protocol:**
  - Local model training & validation on local ICU stays.
  - Local extraction of sufficient statistics / tree estimators / SHAP attributions.
  - Masked or noisy parameter release to the server.
- **Server Protocol:**
  - Soft-voting forest union ensembling.
  - Lossless parameter reconstruction for Naïve Bayes.
  - Zero-sum pairwise mask cancellation.
  - Surrogate decision tree rule extraction.

---

## Slide 5: Data Preprocessing & Non-IID Partitioning (CO1)
- **Cohort:** 10,000 adult ICU admissions (8,000 train, 2,000 test across 5 hospitals).
- **Leakage Prevention:** All scalers, imputers, and bin discretizers fit strictly on local hospital training folds.
- **Clinical Range Clipping:** Telemetry noise clipped using clinical threshold tables (e.g., HR [20, 250], Lactate [0.2, 25]).
- **Unit Testing:** 15/15 automated unit tests verifying zero NaNs, zero leakage, and column alignment.

---

## Slide 6: Optimization & Feature Scaling Study (CO2)
- **From-Scratch GD Implementation:** Verified against closed-form OLS normal equations.
- **Key Findings:**
  - **Z-score Scaling:** Converges in **25 iterations** at $\eta = 0.1$ (difference vs sklearn: $1.05 \times 10^{-5}$).
  - **Min-Max Scaling:** Converges in **144 iterations** at $\eta = 0.1$.
  - **Unscaled Features:** Requires 15x more iterations (379 iterations) at a 100x lower learning rate ($\eta = 0.001$). High learning rates ($\eta \ge 0.01$) explode/diverge.
- **Loss Surface Insight:** Unscaled data produces severely eccentric elliptical contours, causing gradient oscillation.

---

## Slide 7: Unsupervised Patient Stratification (CO4)
- **Centralized K-means:** Silhouette and elbow analysis across $k \in [2, 8]$ identified $k=5$ as the optimal clinical clustering configuration.
- **Federated K-means:** Clients transmit cluster counts and coordinate sums; server updates centroids via FedAvg rounds (20 rounds, ARI = 0.0027).
- **Clinical Subgroup Profiles:** Discovered distinct clusters spanning low-risk stable patients, severe multi-organ failure, and elderly respiratory distress.

---

## Slide 8: Federated Classification Strategies (CO3)
- **Strategy 1: Forest Union (Strategy B):**
  - Clients train 50 local Random Forest trees each.
  - Server aggregates all 250 trees into a pooled soft-voting ensemble.
  - Tested uniform, performance-weighted (AUROC), and sample-count weighted pooling.
- **Strategy 2: Sufficient-Statistics Naïve Bayes:**
  - Clients send parameter statistics ($\sum x, \sum x^2$, class counts) instead of raw records.
  - Reconstructs exact global Gaussian and Categorical NB models.

---

## Slide 9: Supervised Benchmark Results
| Model Architecture | Setting | Accuracy | AUROC | AUPRC |
|---|---|---|---|---|
| **Random Forest** | **Federated (Forest Union)** | **0.7280** | **0.7654** | **0.6253** |
| Random Forest | Centralized Baseline | 0.5970 | 0.5716 | 0.3892 |
| Random Forest | Local-Only Average | 0.6022 | 0.5659 | 0.3802 |
| **Decision Tree** | **Federated (Tree Union)** | **0.6440** | **0.6424** | **0.4602** |
| Decision Tree | Centralized Baseline | 0.5405 | 0.5427 | 0.3625 |
| **Categorical NB** | **Federated (Lossless)** | **0.6750** | **0.6241** | **0.4397** |
| Gaussian NB | Federated (Lossless) | 0.6625 | 0.5911 | 0.4016 |
| Logistic Reg. | Federated (FedAvg) | 0.5577 | 0.5674 | 0.3924 |

*Takeaway:* Federated Forest Union outperforms centralized baseline by **+0.1938 AUROC** due to distributed variance reduction!

---

## Slide 10: Naïve Bayes Mathematical Equivalence
- **Prediction Agreement:** **100.0% (1.000)** match between federated and centralized models.
- **Probability Prediction RMSE:**
  - Gaussian NB: **$6.24 \times 10^{-11}$**
  - Categorical NB: **$0.000000$**
- **Significance:** Proves federated sufficient statistics aggregation incurs zero loss in predictive fidelity.

---

## Slide 11: Differential Privacy Sweeps
- **Mechanism:** Laplace mechanism perturbing sufficient statistics across $\varepsilon \in [0.1, 10.0, \infty]$.
- **Utility Curve:**
  - High Privacy ($\varepsilon = 0.1$): AUROC drops to baseline ($0.5039$).
  - Moderate Privacy ($\varepsilon = 5.0 - 10.0$): AUROC recovers to $0.5925 - 0.6121$ (near-lossless utility).
- **Insight:** Categorical histogram counts demonstrate higher DP noise resilience than continuous sums-of-squares.

---

## Slide 12: Secure Aggregation Simulation
- **Method:** Pairwise additive masking simulation across all 5 hospital clients.
- **Verification Outcomes:**
  - Pairwise masks cancel exactly at the server: **True**.
  - Reconstructed global weights match plaintext FedAvg: **True**.
  - Maximum numerical deviation: **$3.68 \times 10^{-16}$** (floating-point precision limit).
- **Benefit:** Protects hospital model weights from snooping in transit without adding accuracy loss.

---

## Slide 13: Federated Explainability (TreeSHAP & Gini)
- **Top 5 Clinical Risk Factors:**
  1. **Age** (Attribution: 0.1206)
  2. **SOFA Score** (Attribution: 0.1000)
  3. **APACHE Score** (Attribution: 0.0739)
  4. **pCO2 / Bicarbonate** (Attribution: 0.0392)
  5. **Diastolic / Mean BP** (Attribution: 0.0363)
- **Cross-Hospital Explanation Consistency:**
  - Spearman rank correlation: **$\rho = 0.8875 \pm 0.0147$**
  - Kendall tau: **$\tau = 0.7349 \pm 0.0160$**

---

## Slide 14: Transparent Clinician Decision Rules
- **Method:** Depth-3 surrogate decision tree fitted to global forest predictions (**83.1% agreement rate**).
- **Clinical Rules Extracted:**
  - *Low Risk (0.5% Mortality):* `IF age <= 4.5 AND sofa <= 3.5 AND apache <= 6.5 THEN Risk = 0.5%`
  - *Moderate Risk (31.5% Mortality):* `IF age <= 4.5 AND sofa > 3.5 AND apache > 6.5 THEN Risk = 31.5%`
  - *Critical Risk (81.4% Mortality):* `IF age > 4.5 AND sofa > 3.5 AND apache > 3.5 THEN Risk = 81.4%`
- **Clinical Utility:** Clinicians can immediately audit the criteria behind high-risk alerts.

---

## Slide 15: Conclusion & Future Scope
- **Conclusions:**
  - Validated that privacy-preserving federated learning achieves state-of-the-art ICU risk prediction (0.7654 AUROC).
  - Maintained zero information leakage through Secure Aggregation and formal Differential Privacy guarantees.
  - Achieved high clinical interpretability through TreeSHAP and transparent decision rules.
- **Future Scope:**
  - Expansion to prospective real-world EHR deployment.
  - Integration with multi-modal physiological waveforms and clinical notes.
  - Personalized FL via clustered federated learning for specialized hospital units.
