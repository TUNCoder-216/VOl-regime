"""Streamlit front-end: talks to the API only, never to the model directly."""

import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Volatility Regime Radar", page_icon="📉", layout="wide")
st.title("Volatility Regime Radar")
st.caption("Probability that the S&P 500 enters a high-volatility week (next 5 trading days).")


def api_get(path: str) -> dict:
    response = requests.get(f"{API_URL}{path}", timeout=10)
    response.raise_for_status()
    return response.json()


def api_post(path: str, payload: dict) -> dict:
    response = requests.post(f"{API_URL}{path}", json=payload, timeout=60)
    response.raise_for_status()
    return response.json()


try:
    info = api_get("/model-info")
except requests.RequestException as exc:
    st.error(f"API unavailable or no model loaded: {exc}")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Model", f"{info['name']} v{info['version']}")
c2.metric("Test ROC-AUC", f"{info.get('metrics', {}).get('roc_auc', float('nan')):.3f}")
c3.metric("Trained on data up to", info.get("data_end", "?"))
c4.metric("Algorithm", info.get("candidate", "?"))
st.caption(f"Resolved via {info.get('resolved_by')} • fetched at {info.get('fetched_at')}")

source = st.radio(
    "Market data source",
    ["live", "reference"],
    horizontal=True,
    help="live = latest prices from Yahoo Finance; reference = snapshot shipped with the model",
)

if st.button("Predict", type="primary"):
    with st.spinner("Scoring the latest market window..."):
        try:
            result = api_post("/predict", {"source": source})
        except requests.RequestException as exc:
            st.error(f"Prediction failed: {exc}")
            st.stop()

    probability = result["probability_high_vol"]
    left, right = st.columns([1, 2])
    with left:
        st.metric("P(high-volatility week)", f"{probability:.1%}")
        st.progress(min(max(probability, 0.0), 1.0))
        if result["regime"] == "HIGH":
            st.error("HIGH volatility regime expected")
        else:
            st.success("NORMAL regime expected")
        st.caption(f"As of {result['as_of']} • source: {result['source']}")
    with right:
        history = pd.DataFrame(result["history"]).set_index("dates")
        st.line_chart(history[["rv_21d"]], height=180)
        st.line_chart(history[["vix"]], height=180)
    with st.expander("Feature values used for this prediction"):
        st.json(result["features"])
