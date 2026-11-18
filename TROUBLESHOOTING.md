# If AI Balance Buddy does not open

The GitHub repository is source code, not a running website. The Streamlit dashboard is launched separately.

## Deploy
1. Visit https://share.streamlit.io and sign in with GitHub.
2. Click **Create app** and choose **Deploy a public app from GitHub**.
3. Repository: `dapperdude404/ai-balance-buddy`
4. Branch: `main`
5. Main file: `balance_app.py`
6. Select Python 3.11 if prompted, then click **Deploy**.
7. Wait for dependencies and the initial synthetic-model training to finish.

## If the app shows an error
- Open **Manage app → Logs** and read the first traceback.
- Confirm the main file is exactly `balance_app.py`.
- Reboot from **Manage app → Reboot app**.
- Never put API keys in GitHub. The coaching feature works offline without Anthropic credentials.
- Streamlit Cloud may sleep inactive free apps; waking them can take a moment.

The demo uses synthetic research data and is not a clinical diagnostic tool.
