# Volatility Regime Radar

Predicts the probability that the S&P 500 enters a **high-volatility week** (next 5 trading days),
served through a fully automated MLOps pipeline.

## Architecture

```
git push ─▶ GitHub Actions: [lint+test] ─▶ [build+push images → Docker Hub] ─▶ [POST webhook]
                                                                                  │ (ngrok)
                                                                                  ▼
 DagsHub MLflow (model @champion) ─▶ model-fetcher ─▶ volume ─▶ api :8000 ◀── ui :8501
                       ▲                                     local machine (docker compose)
 dvc repro: download ─▶ features ─▶ train/register ── data versioned by DVC on DagsHub
```

## Stack
DVC + DagsHub (data), MLflow on DagsHub (tracking + registry), scikit-learn, FastAPI, Streamlit,
Docker / Compose, GitHub Actions, Docker Hub, ngrok.

## Quick start
```powershell
py -3.11 -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements-train.txt -r requirements-dev.txt; pip install --no-deps -e .
copy .env.example .env   # fill in DagsHub / Docker Hub / webhook values
dvc pull                 # data
dvc repro                # retrain + register (only if something changed)
docker compose pull; docker compose up -d
```
UI: http://localhost:8501 · API docs: http://localhost:8000/docs

## Key design decisions
- **Model is not baked into images.** Images come from Docker Hub, the model from the MLflow
  registry (`@champion` alias) via a one-shot `model-fetcher` container at every deploy.
- **No look-ahead bias.** Features and the high-vol threshold use past data only (tested).
  Chronological split with a purge gap equal to the label horizon.
- **Quality gate.** A new version is promoted to `@champion` only if test ROC-AUC ≥ 0.60.
- **Shared feature code** between training and serving (no training/serving skew).
- **Graceful degradation.** If live market data fails, the API scores the reference window
  shipped with the model.

## Endpoints
`GET /health` · `GET /model-info` · `POST /predict {"source": "live"|"reference"}` ·
`POST /predict/features`
