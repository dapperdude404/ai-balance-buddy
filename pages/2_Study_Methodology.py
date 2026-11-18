import streamlit as st
st.set_page_config(page_title="Methodology | AI Balance Buddy",page_icon="📚",layout="wide")
st.title("📚 Research methodology")
st.markdown("""
### Research objective
Explore associations between generative-AI usage, independent task initiation, verification, and self-regulation among university students.

### Current baseline
- 400 **synthetic** profiles generated with a fixed seed.
- K-Means discovers three synthetic usage groups.
- Groups are ordered using a heuristic dependency index.
- XGBoost learns to reproduce those pseudo-labels.
- SHAP attributes predictions to input features; attribution does not establish causation.

### Proposed validation study
1. Expert review and cognitive interviews to refine the survey.
2. Institutional ethics approval before recruitment.
3. Consent-based pilot assessing question clarity and reliability.
4. Independently defined outcomes and participant-level data splits.
5. Cross-validation, calibration, fairness, and explanation-stability analysis.
6. Evaluation of coaching usefulness and comprehension.

### Limitations
The present balance score is an exploratory index, not a clinical diagnosis or validated student-risk score. Never use it for grading, discipline, or individual screening.
""")
