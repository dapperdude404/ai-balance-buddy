<div align="center">

# AI BALANCE BUDDY
### Version 2.0 — Research & Development Edition

**An Explainable Machine Learning Framework for Studying Generative AI Reliance and Cognitive Autonomy in Higher Education**

---

**M.Tech — Artificial Intelligence**  
**Academic Research Prototype | Python · Streamlit · Machine Learning**

[**LIVE APPLICATION**](https://ai-balance-buddy-efjzcwv3ohhckmzbuaapcp.streamlit.app/) &nbsp; | &nbsp; [**RESEARCH METHODOLOGY**](docs/research_methodology.md) &nbsp; | &nbsp; [**MODEL CARD**](docs/model_card.md)

</div>


> **Quick discovery:** Explainable educational AI research prototype using synthetic surveys, Streamlit, XGBoost and SHAP.
>
> **Start here:** [Streamlit demo](https://ai-balance-buddy-efjzcwv3ohhckmzbuaapcp.streamlit.app/) · [Model card](docs/model_card.md) · [Contribution guide](CONTRIBUTING.md) · [Report a bug](https://github.com/dapperdude404/ai-balance-buddy/issues/new?template=bug_report.yml) · [Discovery & social preview](DISCOVERY_PACK.md).
>
> **Scope:** The documented demonstration uses 400 synthetic profiles with pseudo-labels, not validated outcomes or diagnoses for students.

## I. ABSTRACT

AI Balance Buddy is an educational research prototype for exploring patterns of generative-AI use among university students. It combines structured self-report indicators, exploratory clustering, supervised machine-learning benchmarks, and explainable AI to examine how AI-assisted workflows relate to independent task initiation, verification practices, and self-regulated learning.

**Research status:** The present demonstration uses **400 synthetic student profiles** and K-Means-derived pseudo-labels. Its scores are illustrative and have **not** been validated against real-world student outcomes. It is not a clinical or disciplinary assessment tool.

## II. PROJECT AT A GLANCE

| Item | Description |
|:---|:---|
| **Project** | AI Balance Buddy 2.0 |
| **Discipline** | M.Tech — Artificial Intelligence |
| **Research domain** | Explainable AI · Educational AI · Human–AI Interaction |
| **Frontend** | Streamlit |
| **Core ML** | scikit-learn, XGBoost, K-Means |
| **Explainability** | SHAP |
| **Benchmark models** | Logistic Regression, Random Forest, SVM, XGBoost |
| **Data** | Synthetic research demonstration |
| **Deployment** | Streamlit Community Cloud |
| **Development branch** | `develop` — Sprint 1 work |
| **Production branch** | `main` — existing live application |

## III. SYSTEM OVERVIEW

```text
               AI BALANCE BUDDY
                       |
            Structured Student Survey
                       |
            Schema Validation (v2)
                       |
            Feature Encoding
                       |
         +-------------+-------------+
         |                           |
    Research Pipeline          Live Assessment
         |                           |
    K-Means Profiles             XGBoost Model
         |                           |
    Four-Model Benchmark         SHAP Explanations
         |                           |
    Research Lab                  Coaching Guidance
         +-------------+-------------+
                       |
             Streamlit Interface
```

## IV. KEY FEATURES

**Student assessment.** A structured questionnaire explores AI-assisted activities, verification habits, independent work, and perceived reliance.

**Explainable predictions.** SHAP highlights the model features associated with an individual synthetic-model prediction; these attributions are not causal evidence.

**Research laboratory.** Compare Logistic Regression, Random Forest, SVM, and XGBoost with reproducible cross-validation on synthetic pseudo-labels.

**Research documentation.** A documented methodology, model card, and explicit limitations support reproducible academic work.

**Responsive interface.** A dark-mode Streamlit dashboard with separate research, methodology, and privacy pages on the development branch.

## V. INSTALLATION AND EXECUTION

**Requirements:** Python 3.11 or 3.12.

```bash
git clone https://github.com/dapperdude404/ai-balance-buddy.git
cd ai-balance-buddy
git checkout develop
python -m venv .venv
```

Activate the environment:

```bash
# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

Install and run:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
streamlit run balance_app.py
```

To execute the synthetic model comparison:

```bash
python -m ml.model_comparison
```

## VI. DEPLOYMENT

The public production demonstration is available at:

**[Open AI Balance Buddy](https://ai-balance-buddy-efjzcwv3ohhckmzbuaapcp.streamlit.app/)**

Streamlit Cloud deployment configuration:

| Field | Value |
|:---|:---|
| Repository | `dapperdude404/ai-balance-buddy` |
| Production branch | `main` |
| Entrypoint | `balance_app.py` |

For a **Version 2.0 preview**, create a *separate* Streamlit app using the `develop` branch. Do not point the existing production deployment at `develop` until validation is complete.

See [Deployment Guide](DEPLOYMENT.md) and [Troubleshooting](TROUBLESHOOTING.md).

## VII. RESEARCH DESIGN AND LIMITATIONS

The prototype generates synthetic records, derives exploratory usage clusters, and trains classifiers to reconstruct those cluster-derived labels. Accordingly, high cross-validation performance does not establish the ability to identify real-world AI dependency.

The proposed next research stage involves consent-based pilot recruitment, validated constructs, participant-level evaluation, uncertainty analysis, and explanation-stability studies, subject to institutional ethics approval.

Read the [Research Methodology](docs/research_methodology.md) and [Model Card](docs/model_card.md) for detailed scope and limitations.

## VIII. REPOSITORY ORGANIZATION

```text
ai-balance-buddy/
├── balance_app.py               # Streamlit assessment entrypoint
├── pages/                       # Multipage research interface
├── survey_schema.py             # Versioned survey validation
├── features.py                  # Shared feature encoding
├── train.py                     # Synthetic model training
├── ml/
│   ├── model_comparison.py      # Four-model benchmark
│   └── explainability.py        # SHAP attribution utilities
├── tests/                       # Automated tests
├── docs/
│   ├── research_methodology.md
│   └── model_card.md
├── .github/workflows/           # CI checks
└── requirements.txt
```

## IX. ETHICS AND RESPONSIBLE USE

- No real student dataset is included.
- The demonstration does not intentionally store survey responses.
- AI-use patterns must not be interpreted as diagnoses.
- Do not use outputs for student grading, discipline, or automated high-stakes decisions.
- Real-participant studies require informed consent and appropriate institutional review.

---

<div align="center">

**AI BALANCE BUDDY — EXPLAINABLE AI FOR RESPONSIBLE LEARNING**

*Academic research prototype · Development release 2.0*

</div>
