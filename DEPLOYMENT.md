# Deployment checklist

1. Clone https://github.com/dapperdude404/ai-balance-buddy
2. Create Python 3.11 or 3.12 environment and run `pip install -r requirements.txt`.
3. Run `pytest -q` and `python train.py` locally. The Streamlit app can also train its synthetic model on startup.
4. Visit https://share.streamlit.io/ and sign in with GitHub.
5. Select repository `dapperdude404/ai-balance-buddy`, branch `main`, main file `balance_app.py`.
6. In Advanced settings select Python 3.11 or 3.12, and optionally set `ANTHROPIC_API_KEY` in Secrets. Do not commit API keys.
7. Click Deploy and wait for the build; check app logs for any dependency errors.

This application uses synthetic research data and does not save student responses. Predictions must not be used as clinical diagnoses or academic disciplinary decisions.
