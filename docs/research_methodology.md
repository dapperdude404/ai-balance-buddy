# AI Balance Buddy 2.0 — Research methodology (Sprint 1)

## Research construct
Investigate generative-AI usage patterns, independent task initiation, verification, self-regulated learning, and perceived cognitive autonomy. Avoid conflating AI usage frequency with harmful dependence.

## Current evidence
The baseline generates 400 synthetic records; K-Means creates three pseudo-labels, ranked by a heuristic dependency index; XGBoost learns these pseudo-labels. Benchmark accuracy and macro-F1 reflect synthetic pseudo-label reconstruction only. The scores have **not** been validated against real student outcomes.

## Proposed research design
1. Review related work and refine the 16-item survey with expert feedback.
2. Obtain institutional ethics approval before recruitment.
3. Pilot consented surveys; assess item clarity, missingness, and internal consistency only where a coherent scale is theoretically justified.
4. Define outcomes independently of model features and clustering-derived labels.
5. Conduct participant-level stratified splitting and nested tuning where practical; reserve a final untouched test set.
6. Compare Logistic Regression, Random Forest, RBF SVM, and XGBoost with macro-F1, class-wise recall, calibration, confidence intervals, and error analysis.
7. Examine explanation fidelity/stability and subgroup performance where sample sizes permit.
8. Test coaching utility with preregistered outcomes, not anecdotal changes in model scores.

## Threats to validity
Synthetic generation assumptions, pseudo-label circularity, survey self-report bias, sample selection, confounding, domain shift, class imbalance, and explanation instability. SHAP is associative model attribution, not a causal explanation.

## Reproducibility
Version the survey contract, record seeds and package versions, preserve splits and training metadata, and never publish identifiable participant records.
