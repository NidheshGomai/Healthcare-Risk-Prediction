# Implementation Plan
## Privacy-Preserving Federated Learning for Explainable Healthcare Risk Prediction

**Duration:** 10 weeks | **Language:** Python | **Mapped COs:** CO1 (Preprocessing), CO2 (Optimization), CO3 (Classification), CO4 (Clustering)

---

## 0. Plan at a Glance

| Phase | Weeks | Goal | Key Deliverable |
|---|---|---|---|
| 0 | Pre-week 1 | Setup and data access | Repo, environment, data access request |
| 1 | 1–2 | Data and preprocessing | Reproducible preprocessing pipeline + client partitions |
| 2 | 3 | Centralized baselines | Benchmark table (DT, RF, NB, Linear) |
| 3 | 4 | Feature scaling study (CO2) | Convergence plots and analysis |
| 4 | 5 | K-means and FL skeleton (CO4) | Working FedAvg-style loop + clustering |
| 5 | 6–7 | Federated DT/RF, NB, privacy | Federated models with DP / secure aggregation |
| 6 | 8 | Explainability and evaluation | Metrics, SHAP/LIME results, trade-off analysis |
| 7 | 9–10 | Documentation | Final report, slides, demo notebook |

---

## Phase 0: Project Setup (before Week 1)

### Step 0.1 – Request data access early
- MIMIC-III/IV on PhysioNet requires **credentialed access** (CITI "Data or Specimens Only Research" training + signed DUA). This can take days to weeks.
- **Start the request on day 1.** In parallel, prepare fallbacks:
  - eICU Collaborative Research Database (also credentialed, but has a **public demo subset**)
  - MIMIC-IV Demo (open, ~100 patients; good for pipeline debugging only)
  - Synthetic EHR (e.g., Synthea) for open development

> **Recommendation:** eICU is a strong choice because it contains a `hospitalid` field, giving a **natural non-IID, multi-hospital split** instead of an artificial one.

### Step 0.2 – Repository and environment
```
fl-healthcare/
├── data/                  # raw/ and processed/ (never commit raw data)
├── notebooks/             # EDA, analysis, final figures
├── src/
│   ├── data/              # loading, cohort extraction, partitioning
│   ├── preprocessing/     # imputers, scalers, discretizers, feature engineering
│   ├── models/            # centralized + federated models
│   ├── federated/         # clients, server, strategies, clustering
│   ├── privacy/           # DP and secure aggregation
│   ├── explain/           # SHAP/LIME, consistency metrics
│   └── evaluation/        # metrics, plots
├── configs/               # YAML experiment configs
├── results/               # metrics, figures, logs
├── requirements.txt
└── README.md
```
- Create a virtual environment and pin versions: `scikit-learn, xgboost/lightgbm, numpy, pandas, flwr, torch, opacus, shap, lime, matplotlib, seaborn`.
- Set global random seeds and use YAML configs so every experiment is reproducible.
- Use Git from day 1; add `data/` to `.gitignore` (patient data must never be committed).

### Step 0.3 – Define the prediction task (freeze this in week 1)
- **Primary target:** in-hospital mortality (binary) *or* 30-day readmission.
- **Cohort:** adult ICU stays, first stay per patient, fixed observation window (e.g., first 24 h) to avoid label leakage.
- **Class imbalance:** mortality is typically 8–15%, so plan for AUPRC, class weights, and stratified splits.

**Exit criteria:** access requested, repo created, target and cohort definition written down.

---

## Phase 1: Data and Preprocessing (Weeks 1–2) — CO1

### Step 1.1 – Literature review (Week 1, ~2–3 days)
Cover four themes and keep short notes (1 paragraph each) for the final report:
1. FL in healthcare (FedAvg, FedProx)
2. Federated tree models (federated XGBoost, tree-ensemble aggregation)
3. DP and secure aggregation in FL
4. Explainability under FL (federated SHAP, feature-importance aggregation)

### Step 1.2 – Cohort extraction and exploratory analysis
- Extract features: demographics, first-24h vitals, labs, comorbidity flags, and the label.
- EDA: missingness per feature and per hospital, class balance, feature distributions, outliers, and **between-hospital distribution shift**.

### Step 1.3 – Build the preprocessing pipeline (scikit-learn `Pipeline` / `ColumnTransformer`)

| Component | Method | Notes |
|---|---|---|
| Missing values | Median/mode imputation; add **missingness indicator** columns | Missingness is often clinically informative (a lab not ordered) |
| Outliers | Clip to physiologically plausible ranges / IQR | Use clinical range tables |
| Normalization | Min-Max and Z-score (both kept as options) | Needed for the scaling study |
| Discretization | Equal-width, quantile, and clinical bins (e.g., age groups) | Required for Naïve Bayes (categorical) |
| Feature engineering | Risk scores (e.g., SOFA-like components), temporal features (min/max/mean/slope of vitals), comorbidity counts | |
| Encoding | One-hot or ordinal for categoricals | Trees: ordinal; NB: one-hot/binned |

**Critical federated rule:** all preprocessing parameters (means, medians, bin edges) must be **fit on each client's local training data only**. Compare two options and document the difference:
- **Local-only statistics** (strictly private, but clients may be inconsistent)
- **Federated statistics** (clients send aggregated mean/variance/counts, ideally with secure aggregation or DP noise, so every client uses shared scalers/bins)

### Step 1.4 – Create client partitions (non-IID simulation)
- Preferred: one client per hospital (natural split in eICU).
- Otherwise simulate with:
  - **Label skew:** Dirichlet partition (α = 0.1, 0.5, 1.0, 10)
  - **Feature skew:** split by age group, admission type, or care unit
  - **Quantity skew:** unequal client sizes
- Hold out a **global test set** (stratified) and a **per-client test set**. Never use the global test set for tuning.
- Use 5–10 clients as the default; vary this later (e.g., 3, 5, 10, 20).

### Step 1.5 – Validate
- Unit tests: no NaNs after pipeline, no train/test leakage, consistent column order across clients.
- Log a summary table: samples, positive rate, missing rate per client.

**Exit criteria:** `preprocess.py` runs end-to-end; partitions saved; EDA notebook complete.

---

## Phase 2: Centralized Baselines (Week 3) — CO3

Establish the performance **upper bound** before introducing federation.

### Step 2.1 – Models
| Model | Configuration |
|---|---|
| Decision Tree | `max_depth` ∈ {3,5,7,10}, `min_samples_leaf`, `class_weight='balanced'`; tune with CV |
| Random Forest | `n_estimators`, `max_depth`, `max_features`; optional XGBoost/LightGBM as a performance reference |
| Naïve Bayes | `GaussianNB` (scaled features) and `CategoricalNB` (discretized features) |
| Linear baseline | See Phase 3 |

### Step 2.2 – Evaluation
- Metrics: Accuracy, Precision, Recall, F1, AUROC, AUPRC, Brier score.
- Add calibration curves and a confusion matrix at a clinically chosen threshold.
- Report mean ± std over 5 random seeds.

### Step 2.3 – Local-only baseline
Train each hospital on its own data only and evaluate. This gives the **lower bound**, and FL should land between local-only and centralized.

**Exit criteria:** a baseline results table (centralized vs. local-only) saved to `results/baselines.csv`.

---

## Phase 3: Feature Scaling and Gradient Descent Study (Week 4) — CO2

### Step 3.1 – Choose the linear task
Linear Regression predicts continuous output, so decide one of:
- **(Recommended)** Regress a continuous target (e.g., ICU length of stay) for the scaling study, **and** use the linear probability model / logistic regression as the risk classifier for comparison.
- Or use linear regression on the binary label as a risk score (simple, but state this limitation).

### Step 3.2 – Implement gradient descent from scratch (NumPy)
- Batch GD and mini-batch SGD; MSE loss; configurable learning rate and iterations.
- Verify against `sklearn.linear_model.LinearRegression` (closed-form solution) to confirm correctness.

### Step 3.3 – Experiment design
Run a grid over:
- Scaling: **None, Min-Max, Z-score**
- Learning rate: 1e-4 … 1e0 (log scale)
- Setting: centralized and federated (FedAvg of linear weights)

Record: loss vs. iteration/round, iterations to reach a tolerance, divergence (yes/no), final MSE/R².

### Step 3.4 – Analysis outputs
- Convergence curves per scaling method (one figure per learning rate).
- Table: "rounds/iterations to convergence" by scaling × learning rate.
- Loss-surface contour plot on two features, showing why unscaled data creates elongated contours and zig-zag paths.
- Federated twist: compare **local scaling per client** vs. **globally shared scaling statistics** and their effect on convergence under non-IID data.

**Exit criteria:** convergence figures and a short written conclusion (which scaling is most stable and why).

---

## Phase 4: K-means and Federated Skeleton (Week 5) — CO4

### Step 4.1 – Centralized K-means
- Cluster patients on standardized features; choose *k* via elbow method and silhouette score.
- Profile clusters (mean risk, key feature differences) for **patient subgroup discovery**.

### Step 4.2 – Federated K-means (patient clustering)
- Server initializes centroids (k-means++ on a shared/public or synthetic sample, or random).
- Each round: clients assign local points → send **per-cluster sums and counts** → server updates centroids.
- Compare to centralized K-means (centroid distance, inertia, ARI).

### Step 4.3 – Client clustering for personalization (Option B)
- Represent each hospital by a vector (e.g., label rate, feature means, or model-weight/importance vector).
- Cluster hospitals with K-means; train **one federated model per cluster**.
- Compare: global FL vs. clustered FL vs. local-only on per-client test sets.

> **Decision rule:** implement subgroup discovery (Step 4.2) first, since it is cheap. Do client clustering (4.3) if time allows; it is the stronger personalization contribution.

### Step 4.4 – FL skeleton with Flower
- Implement `Client` (local fit/evaluate) and a `Server` strategy; run in Flower **simulation mode** (single machine).
- Validate the pipeline first with the **linear model (FedAvg)** since weights average cleanly.
- Log per-round metrics: global AUROC, per-client AUROC, communication size.

**Exit criteria:** FedAvg runs for N rounds on all clients; K-means results and cluster profiles available.

---

## Phase 5: Federated Decision Trees, Naïve Bayes, and Privacy (Weeks 6–7) — CO3

### Step 5.1 – Federated tree strategy (pick one primary, one optional)

Trees cannot be averaged like weights, so choose an aggregation strategy:

| Strategy | How it works | Pros / Cons |
|---|---|---|
| **A. Federated XGBoost (Flower)** | Histogram-based boosting or tree bagging across clients | Closest to production; uses an existing, tested implementation |
| **B. Forest union (one-shot)** | Each client trains a local RF; the server pools trees (optionally weighted by client size/performance) | Simple, one round, easy to explain; less accurate under strong non-IID |
| **C. Federated histogram trees** | Clients send gradient/count histograms per feature bin; server picks the best split | Mirrors centralized tree-building; the histograms are the privacy-sensitive object |

**Recommended path:** implement **B** first (fast, working end-to-end), then **A or C** as the "advanced" version. Compare communication rounds vs. accuracy.

### Step 5.2 – Federated Naïve Bayes
- Naïve Bayes is naturally federated because its parameters are **sufficient statistics**.
- Clients send class counts, and per-feature counts (categorical) or sums/sums-of-squares (Gaussian). The server adds them to obtain a model **identical to centralized NB** (a useful sanity check).
- Add DP noise to the counts in the private variant.

### Step 5.3 – Privacy mechanisms (implement at least one fully, the other as a comparison)

**Option 1 – Differential Privacy**
- *Linear model:* DP-SGD with Opacus (per-sample clipping + Gaussian noise); this requires the linear model in PyTorch (`nn.Linear`).
- *Naïve Bayes / histograms:* add Laplace/Gaussian noise to released counts or histograms.
- *Trees:* add noise to split selection (exponential mechanism) or leaf values.
- Track ε (at fixed δ ≈ 1/N) using the Opacus accountant. Sweep ε ∈ {0.5, 1, 2, 5, 10, ∞}.

**Option 2 – Secure Aggregation**
- Use Flower's secure aggregation if available, or simulate with pairwise additive masking (clients add masks that cancel when summed).
- Note that secure aggregation protects individual updates but **does not by itself give a formal DP guarantee**; state this in the report.

### Step 5.4 – Experiments in this phase
Run the full grid and save results to CSV:
- Models: DT/RF, NB, Linear
- Settings: centralized, local-only, FL (no privacy), FL + DP, FL + secure aggregation
- Heterogeneity: Dirichlet α ∈ {0.1, 0.5, 10}
- Clients: 3, 5, 10
- Optionally compare with FedProx for the non-IID case

**Exit criteria:** federated DT/RF, NB, and linear models all run with and without privacy; ε–utility results logged.

---

## Phase 6: Explainability and Evaluation (Week 8)

### Step 6.1 – Explanation methods
| Level | Method |
|---|---|
| Global | Tree feature importance (impurity and permutation), SHAP summary (TreeSHAP) |
| Local | SHAP force/waterfall for individual patients, LIME comparison |
| Inherent | Decision paths / extracted rules from a shallow tree (clinician-readable) |
| Baselines | Linear coefficients; NB log-likelihood ratios per feature |

### Step 6.2 – Federated explanation aggregation
- Compute SHAP/importance **locally at each client** (data never leaves) and aggregate **only the summary** (mean |SHAP| per feature) at the server.
- Report which explanation quantities are shared and consider adding DP noise to them.

### Step 6.3 – Explainability metrics (from the PDR)
- **Consistency across clients:** Spearman/Kendall rank correlation of feature importances between hospitals; top-k overlap (Jaccard).
- **Fidelity:** how well a surrogate (e.g., depth-3 tree or LIME model) reproduces the full model's predictions (R² / agreement).
- **Stability:** change in explanations under small input perturbation and across random seeds / FL rounds.
- **Clinical sanity check:** do top features align with known risk factors (age, lactate, GCS, urine output, etc.)?

### Step 6.4 – Consolidated evaluation
- Predictive: Accuracy, Precision, Recall, F1, AUROC, AUPRC, Brier + calibration plots.
- Privacy: ε vs. AUROC curve.
- Federated: performance vs. communication rounds; effect of α (non-IID); effect of number of clients.
- Statistical rigor: 5 seeds, report mean ± std; use paired tests or confidence intervals for key comparisons.

### Step 6.5 – Privacy–utility–explainability trade-off
Produce one summary table and one figure that jointly show how ε changes accuracy **and** explanation stability, which is the central contribution of the project.

**Exit criteria:** all figures/tables generated by scripts (not manually), saved in `results/`.

---

## Phase 7: Documentation and Presentation (Weeks 9–10)

### Step 7.1 – Final report structure
1. Introduction and problem statement
2. Related work (from the Phase 1 notes)
3. Methodology (architecture diagram, preprocessing, models, privacy)
4. Experimental setup (dataset, cohort, partitions, hyperparameters)
5. Results (baselines, scaling study, clustering, FL, privacy, explainability)
6. Discussion (trade-offs, limitations, ethical considerations)
7. Conclusion and future work
8. Reproducibility appendix (commands, configs, seeds)

### Step 7.2 – Course outcome mapping (include as a table in the report)
| CO | Evidence in the project |
|---|---|
| CO1 | Phase 1 preprocessing pipeline (imputation, scaling, discretization, feature engineering) |
| CO2 | Phase 3 gradient descent and scaling study |
| CO3 | Phases 2 and 5: Decision Tree/RF and Naïve Bayes, centralized and federated |
| CO4 | Phase 4 K-means (patient and client clustering) |

### Step 7.3 – Deliverables checklist
- [ ] Clean, runnable repository with README and one-command reproduction
- [ ] Demo notebook (Colab-friendly) using the open/synthetic dataset
- [ ] Final report (PDF)
- [ ] Slide deck (10–15 slides) and a short live demo
- [ ] Results folder with all figures and CSVs

---

## Risk Register

| Risk | Likelihood | Mitigation |
|---|---|---|
| MIMIC/eICU access delayed | High | Request on day 1; develop on demo/synthetic data; keep the code dataset-agnostic |
| Federated trees underperform with non-IID data | Medium | Add client clustering, weighting, or federated XGBoost; report honestly |
| DP noise destroys utility on small clients | Medium | Use larger ε range, fewer/larger clients, clip tuning; report the curve rather than a single point |
| Class imbalance hides poor performance | Medium | Use AUPRC, class weights, threshold tuning |
| Data leakage in preprocessing | Medium | Fit all transforms on training folds only; add leakage unit tests |
| Scope creep (HE, vertical FL, RL) | Low | These are explicitly out of scope; stick to the PDR |
| Compute limits in Colab | Low | Use Flower simulation, small models, subsample for SHAP |

---

## Weekly Checklist (Quick Reference)

- **Week 1:** Data access requested, repo set up, literature review, cohort defined, EDA started
- **Week 2:** Preprocessing pipeline finished, client partitions created, leakage tests passing
- **Week 3:** Centralized DT/RF/NB baselines and local-only baselines with metrics
- **Week 4:** From-scratch gradient descent; scaling × learning-rate experiments; convergence plots
- **Week 5:** Centralized and federated K-means; FL skeleton (Flower) with FedAvg linear model
- **Week 6:** Federated forest/XGBoost and federated NB working without privacy
- **Week 7:** DP and/or secure aggregation added; ε sweeps; non-IID and client-count experiments
- **Week 8:** SHAP/LIME, federated explanation aggregation, explainability metrics, final evaluation
- **Week 9:** Report draft, figures finalized, code cleanup
- **Week 10:** Final report, slides, demo rehearsal, submission

---

## Suggested Minimum Viable Project (if time runs short)

If the schedule slips, protect these in priority order:
1. Preprocessing pipeline + centralized baselines (CO1, CO3)
2. Scaling study on gradient descent (CO2)
3. Federated RF (forest union) + federated Naïve Bayes
4. One privacy mechanism (DP on NB/linear is the quickest to demonstrate)
5. Feature-importance + SHAP explanation consistency
6. K-means subgroup analysis (CO4); client clustering is the first item to cut
