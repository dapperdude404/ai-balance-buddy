# AI Balance Buddy ⚖️

Explainable machine-learning research demo for university students' generative-AI usage patterns.

## Local setup
Python 3.11 or 3.12:
```bash
pip install -r requirements.txt
python train.py
streamlit run balance_app.py
```

## Streamlit Community Cloud
Train locally, commit generated `xgb_balance_model.json` and `training_report.json`, then deploy `balance_app.py` from branch `main` at https://share.streamlit.io/.

Optional Anthropic API key belongs in Streamlit Secrets, never in Git.

**Important:** Training data is synthetic. Scores and cohorts are illustrative, not validated diagnoses or grounds for grading students.
