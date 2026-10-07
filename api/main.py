"""FastAPI inference service."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from datetime import date, timedelta
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, field_validator

from api.model_store import ModelBundle, load_bundle
from vol_regime.data import fetch_market_data
from vol_regime.features import FEATURE_COLUMNS, compute_features, latest_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("api")

HISTORY_DAYS = 500  # calendar days fetched live: enough for the 252-trading-day features
CHART_POINTS = 120


class PredictRequest(BaseModel):
    source: Literal["live", "reference"] = "live"


class FeaturesRequest(BaseModel):
    features: dict[str, float]

    @field_validator("features")
    @classmethod
    def check_feature_names(cls, value: dict[str, float]) -> dict[str, float]:
        missing = set(FEATURE_COLUMNS) - set(value)
        extra = set(value) - set(FEATURE_COLUMNS)
        if missing or extra:
            raise ValueError(f"missing={sorted(missing)} extra={sorted(extra)}")
        return value


def _regime(probability: float, threshold: float) -> str:
    return "HIGH" if probability >= threshold else "NORMAL"


def _market_window(source: str, bundle: ModelBundle) -> tuple[pd.DataFrame, str]:
    if source == "live":
        try:
            today = date.today()
            raw = fetch_market_data(
                start=str(today - timedelta(days=HISTORY_DAYS)),
                end=str(today + timedelta(days=1)),
            )
            return raw, "live"
        except Exception as exc:  # noqa: BLE001 - degrade gracefully to the reference window
            logger.warning("Live market data failed: %s", exc)
            if bundle.reference is None:
                raise HTTPException(502, "Live data unavailable and no reference window") from exc
            return bundle.reference, "reference (live fetch failed)"
    if bundle.reference is None:
        raise HTTPException(404, "This model has no reference window")
    return bundle.reference, "reference"


def create_app(bundle: ModelBundle | None = None, model_dir: Path | None = None) -> FastAPI:
    model_dir = model_dir or Path(os.getenv("MODEL_DIR", "/models"))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.bundle is None:
            try:
                app.state.bundle = load_bundle(model_dir)
            except Exception:  # noqa: BLE001 - keep the API up; /health reports the state
                logger.exception("Failed to load model from %s", model_dir)
        if app.state.bundle is None:
            logger.warning("No model loaded from %s", model_dir)
        else:
            meta = app.state.bundle.metadata
            logger.info("Loaded model %s v%s", meta.get("name"), meta.get("version"))
        yield

    app = FastAPI(title="Volatility Regime API", version="1.0.0", lifespan=lifespan)
    app.state.bundle = bundle

    def get_bundle(request: Request) -> ModelBundle:
        current = request.app.state.bundle
        if current is None:
            raise HTTPException(503, "No model loaded. Run the model-fetcher first.")
        return current

    @app.get("/health")
    def health(request: Request) -> dict:
        return {"status": "ok", "model_loaded": request.app.state.bundle is not None}

    @app.get("/model-info")
    def model_info(request: Request) -> dict:
        return get_bundle(request).metadata

    @app.post("/predict")
    def predict(payload: PredictRequest, request: Request) -> dict:
        current = get_bundle(request)
        raw, source_used = _market_window(payload.source, current)
        as_of, x = latest_features(raw)
        probability = float(current.predict_proba(x)[0])
        threshold = float(current.metadata.get("decision_threshold", 0.5))
        history = compute_features(raw).dropna().tail(CHART_POINTS)
        return {
            "as_of": as_of.date().isoformat(),
            "source": source_used,
            "probability_high_vol": round(probability, 4),
            "regime": _regime(probability, threshold),
            "decision_threshold": threshold,
            "model_version": current.metadata.get("version"),
            "features": {k: float(v) for k, v in x.iloc[0].items()},
            "history": {
                "dates": [d.date().isoformat() for d in history.index],
                "rv_21d": history["rv_21d"].round(4).tolist(),
                "vix": history["vix"].round(2).tolist(),
            },
        }

    @app.post("/predict/features")
    def predict_features(payload: FeaturesRequest, request: Request) -> dict:
        current = get_bundle(request)
        x = pd.DataFrame([payload.features])[FEATURE_COLUMNS]
        probability = float(current.predict_proba(x)[0])
        threshold = float(current.metadata.get("decision_threshold", 0.5))
        return {
            "probability_high_vol": round(probability, 4),
            "regime": _regime(probability, threshold),
            "model_version": current.metadata.get("version"),
        }

    return app


app = create_app()
