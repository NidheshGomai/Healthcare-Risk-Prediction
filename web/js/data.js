/*
 * All numbers shown on the site, transcribed from reports/final_report.md
 * and results/phase6_clinical_rules.txt. Edit here to update the page.
 */
window.PROJECT_DATA = {
  dataset: {
    hospitals: 5,
    train: 8000,
    test: 2000,
    features: 38,
    mortalityRate: 0.10,
    windowHours: 24,
    perHospitalMin: 1553,
    perHospitalMax: 1672
  },

  settings: [
    { key: "centralized", label: "Centralized" },
    { key: "local", label: "Local-only" },
    { key: "federated", label: "Federated" }
  ],

  // accuracy / auroc / auprc / brier per model and setting
  models: [
    {
      name: "Random Forest",
      federatedMode: "Forest Union",
      centralized: { accuracy: 0.5970, auroc: 0.5716, auprc: 0.3892, brier: 0.2382 },
      local:       { accuracy: 0.6022, auroc: 0.5659, auprc: 0.3802, brier: 0.2410 },
      federated:   { accuracy: 0.7280, auroc: 0.7654, auprc: 0.6253, brier: 0.1882 }
    },
    {
      name: "Decision Tree",
      federatedMode: "Tree Union",
      centralized: { accuracy: 0.5405, auroc: 0.5427, auprc: 0.3625, brier: 0.2497 },
      local:       { accuracy: 0.5297, auroc: 0.5314, auprc: 0.3374, brier: 0.2541 },
      federated:   { accuracy: 0.6440, auroc: 0.6424, auprc: 0.4602, brier: 0.2215 }
    },
    {
      name: "Gaussian NB",
      federatedMode: "Sufficient statistics",
      centralized: { accuracy: 0.6610, auroc: 0.5718, auprc: 0.3848, brier: 0.2198 },
      local:       { accuracy: 0.6487, auroc: 0.5464, auprc: 0.3767, brier: 0.2245 },
      federated:   { accuracy: 0.6625, auroc: 0.5911, auprc: 0.4016, brier: 0.2198 }
    },
    {
      name: "Categorical NB",
      federatedMode: "Histogram sums",
      centralized: { accuracy: 0.6620, auroc: 0.5709, auprc: 0.3902, brier: 0.2210 },
      local:       { accuracy: 0.6263, auroc: 0.5390, auprc: 0.3691, brier: 0.2280 },
      federated:   { accuracy: 0.6750, auroc: 0.6241, auprc: 0.4397, brier: 0.2185 }
    },
    {
      name: "Logistic Reg.",
      federatedMode: "FedAvg, 10 rounds",
      centralized: { accuracy: 0.5675, auroc: 0.5818, auprc: 0.3872, brier: 0.2443 },
      local: null,
      federated:   { accuracy: 0.5577, auroc: 0.5674, auprc: 0.3924, brier: 0.2460 }
    }
  ],

  forestWeighting: [
    { weighting: "sample_count", auroc: 0.7654 },
    { weighting: "uniform", auroc: 0.7646 },
    { weighting: "performance", auroc: 0.7645 }
  ],

  nbEquivalence: {
    agreement: 1.0,
    gaussianRmse: "6.24 × 10⁻¹¹",
    categoricalRmse: "0"
  },

  // Laplace-mechanism DP sweep; epsilon = null means no noise (∞)
  dpSweep: [
    { epsilon: 0.1,  gaussian: 0.5125, gaussianStd: 0.0082, categorical: 0.5039, categoricalStd: 0.0160, gap: 0.0670,  rho: -0.1161, rhoStd: 0.1264 },
    { epsilon: 0.5,  gaussian: 0.5333, gaussianStd: 0.0133, categorical: 0.5163, categoricalStd: 0.0168, gap: 0.0546,  rho: -0.1077, rhoStd: 0.1314 },
    { epsilon: 1.0,  gaussian: 0.5340, gaussianStd: 0.0140, categorical: 0.5301, categoricalStd: 0.0142, gap: 0.0407,  rho: -0.0811, rhoStd: 0.1199 },
    { epsilon: 2.0,  gaussian: 0.5321, gaussianStd: 0.0194, categorical: 0.5515, categoricalStd: 0.0124, gap: 0.0194,  rho: -0.0296, rhoStd: 0.1313 },
    { epsilon: 5.0,  gaussian: 0.5330, gaussianStd: 0.0174, categorical: 0.5925, categoricalStd: 0.0065, gap: -0.0216, rho: 0.1319,  rhoStd: 0.1057 },
    { epsilon: 10.0, gaussian: 0.5443, gaussianStd: 0.0122, categorical: 0.6121, categoricalStd: 0.0029, gap: -0.0412, rho: 0.3968,  rhoStd: 0.0581 },
    { epsilon: null, gaussian: 0.5911, gaussianStd: 0,      categorical: 0.6241, categoricalStd: 0,      gap: -0.0532, rho: 1.0,     rhoStd: 0 }
  ],

  secureAgg: {
    cancellationVerified: true,
    matchesFedAvg: true,
    maxDeviation: "3.68 × 10⁻¹⁶"
  },

  featureImportance: [
    { feature: "Age", key: "age", value: 0.1206, note: "Baseline vulnerability" },
    { feature: "SOFA score", key: "sofa_score", value: 0.1000, note: "Multi-organ dysfunction" },
    { feature: "APACHE score", key: "apache_score", value: 0.0739, note: "Acute physiological acuity" },
    { feature: "Bicarbonate", key: "bicarbonate", value: 0.0392, note: "Acid-base balance" },
    { feature: "Diastolic BP", key: "diastolic_bp", value: 0.0363, note: "Hemodynamic shock" },
    { feature: "Mean BP", key: "mean_bp", value: 0.0312, note: "Hemodynamic shock" },
    { feature: "pCO₂", key: "pco2", value: 0.0287, note: "Respiratory failure" }
  ],

  consistency: [
    { label: "Spearman ρ", mean: 0.8875, std: 0.0147 },
    { label: "Kendall τ", mean: 0.7349, std: 0.0160 },
    { label: "Top-5 Jaccard", mean: 0.3452, std: 0.1336 },
    { label: "Top-10 Jaccard", mean: 0.4458, std: 0.1025 }
  ],

  surrogate: { fidelity: 0.831, kappa: 0.5093, depth: 3 },

  // Depth-3 surrogate tree rules (thresholds on quantile-bin values)
  rules: [
    { id: 1, age: "<=4.5", sofa: "<=3.5", apache: "<=6.5", predict: 0, risk: 0.5 },
    { id: 2, age: "<=4.5", sofa: "<=3.5", apache: ">6.5",  predict: 0, risk: 7.0 },
    { id: 3, age: "<=4.5", sofa: ">3.5",  apache: "<=6.5", predict: 0, risk: 10.6 },
    { id: 4, age: "<=4.5", sofa: ">3.5",  apache: ">6.5",  predict: 0, risk: 31.5 },
    { id: 5, age: ">4.5",  sofa: "<=3.5", apache: "<=4.5", predict: 0, risk: 10.4 },
    { id: 6, age: ">4.5",  sofa: "<=3.5", apache: ">4.5",  predict: 0, risk: 34.9 },
    { id: 7, age: ">4.5",  sofa: ">3.5",  apache: "<=3.5", predict: 1, risk: 52.9 },
    { id: 8, age: ">4.5",  sofa: ">3.5",  apache: ">3.5",  predict: 1, risk: 81.4 }
  ],

  scaling: [
    { method: "Z-score", lr: "0.1", mse: 0.2164, iterations: 25 },
    { method: "Min-Max", lr: "0.1", mse: 0.2165, iterations: 144 },
    { method: "Unscaled", lr: "0.001", mse: 0.2172, iterations: 379 }
  ],

  kmeans: { k: 5, silhouette: 0.0444, rounds: 20, ari: 0.0027, centroidDistance: 7.50 },

  courseOutcomes: [
    { co: "CO1", focus: "Data preprocessing", work: "Imputation, clinical range clipping, Z-score / Min-Max scaling, quantile discretization, non-IID hospital partitioning", evidence: "src/preprocessing/pipeline.py" },
    { co: "CO2", focus: "Optimization and feature scaling", work: "From-scratch batch GD and mini-batch SGD, scaling × learning-rate grid, closed-form validation", evidence: "src/experiments/run_scaling.py" },
    { co: "CO3", focus: "Classification and privacy", work: "Federated Forest Union, sufficient-statistics Naïve Bayes, Laplace DP sweep, secure aggregation", evidence: "src/federated/, src/privacy/" },
    { co: "CO4", focus: "Clustering", work: "Centralized and federated K-means, elbow / silhouette k selection, subgroup profiles", evidence: "src/experiments/run_federated.py" }
  ],

  reproduce: [
    "python -m pytest tests/ -v",
    "python -m src.data.preprocess --config configs/preprocess.yaml --use-demo",
    "python -m src.experiments.run_baselines --config configs/baselines.yaml",
    "python -m src.experiments.run_scaling --config configs/scaling.yaml",
    "python -m src.experiments.run_federated --config configs/federated.yaml --phase both",
    "python -m src.experiments.run_phase5 --config configs/phase5.yaml",
    "python -m src.experiments.run_phase6 --config configs/phase6.yaml"
  ]
};
