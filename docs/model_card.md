# Model card — AI Balance Buddy synthetic prototype

**Version:** 2.0 Sprint 1 (development branch)

**Intended use:** Educational demonstration of clustering, classification, explainability, and reproducible ML benchmarking.

**Training data:** 400 synthetically generated student-like profiles; no real participants.

**Target construction:** K-Means clusters ranked by a heuristic dependency index. Target is not an independently measured diagnosis or validated outcome.

**Models:** XGBoost in the live dashboard; Logistic Regression, Random Forest, SVM and XGBoost in research comparison.

**Outputs:** Exploratory three-cohort probabilities and a heuristic 0–100 balance index.

**Explainability:** TreeSHAP local attributions to XGBoost output. Values are model-specific, not causal.

**Evaluation:** Stratified cross-validation of synthetic pseudo-label reproduction. No claims of population validity or clinical utility.

**Known limitations:** Circular pseudo-label validation, simulated correlations, untested subgroup generalization, uncalibrated probabilities, absence of real-world construct validation.

**Prohibited use:** Grading, admissions, disciplinary action, mental-health diagnosis, or automated interventions affecting students.

**Privacy:** Current prototype does not intentionally persist survey responses. Any real-data deployment requires ethics approval and explicit consent.
