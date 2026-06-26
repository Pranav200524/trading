# Options Trading Bot — Streamlit Dashboard

Self-contained Streamlit app for browsing backtest results, ML S/R zones, and
CPR pivot levels.

## Structure

```
streamlit_app/
├── dashboard.py          # Streamlit app
├── requirements.txt      # pip deps
├── db/
│   └── dashboard.db      # Pre-exported slim DB (committed)
└── README.md
```

## Run locally

```bash
pip3 install -r requirements.txt
streamlit run dashboard.py
```

## Deploy to Streamlit Cloud

1. Push this folder to a GitHub repo
2. In Streamlit Cloud, create a new app:
   - **Repo**: your GitHub repo
   - **Branch**: `main`
   - **Main file path**: `dashboard.py`
3. Done — Streamlit Cloud installs from `requirements.txt`, reads `db/dashboard.db`

## Refreshing the data

The dashboard reads from `db/dashboard.db` — a slim copy of the main trading
database. After running new backtests in the engine, regenerate this file:

```bash
# From the project root
python3 ml/sync_streamlit_app.py
```

Then commit and push:

```bash
git -C streamlit_app add db/dashboard.db
git -C streamlit_app commit -m "Refresh dashboard data"
git -C streamlit_app push
```

Streamlit Cloud will auto-redeploy on push.
