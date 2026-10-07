import pytest
from fastapi.testclient import TestClient

import api.main as api_main
from vol_regime.features import FEATURE_COLUMNS


@pytest.fixture
def client(bundle):
    with TestClient(api_main.create_app(bundle=bundle)) as test_client:
        yield test_client


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}


def test_model_info(client):
    body = client.get("/model-info").json()
    assert body["version"] == 3
    assert body["alias"] == "champion"


def test_predict_on_reference_window(client):
    response = client.post("/predict", json={"source": "reference"})
    assert response.status_code == 200
    body = response.json()
    assert body["probability_high_vol"] == 0.7
    assert body["regime"] == "HIGH"
    assert body["source"] == "reference"
    assert set(body["features"]) == set(FEATURE_COLUMNS)
    assert len(body["history"]["dates"]) == len(body["history"]["rv_21d"]) <= 120


def test_predict_live(client, monkeypatch, raw_market):
    monkeypatch.setattr(api_main, "fetch_market_data", lambda **_: raw_market)
    body = client.post("/predict", json={"source": "live"}).json()
    assert body["source"] == "live"
    assert body["as_of"] == raw_market.index[-1].date().isoformat()


def test_live_failure_falls_back_to_reference(client, monkeypatch):
    def broken(**_):
        raise ConnectionError("Yahoo is down")

    monkeypatch.setattr(api_main, "fetch_market_data", broken)
    body = client.post("/predict", json={"source": "live"}).json()
    assert body["source"] == "reference (live fetch failed)"


def test_predict_from_features(client):
    payload = {"features": {name: 0.1 for name in FEATURE_COLUMNS}}
    body = client.post("/predict/features", json=payload).json()
    assert body == {"probability_high_vol": 0.7, "regime": "HIGH", "model_version": 3}


def test_missing_feature_is_rejected(client):
    payload = {"features": {name: 0.1 for name in FEATURE_COLUMNS[:-1]}}
    assert client.post("/predict/features", json=payload).status_code == 422


def test_invalid_source_is_rejected(client):
    assert client.post("/predict", json={"source": "tomorrow"}).status_code == 422


def test_no_model_returns_503(tmp_path):
    with TestClient(api_main.create_app(model_dir=tmp_path)) as test_client:
        assert test_client.get("/health").json()["model_loaded"] is False
        assert test_client.get("/model-info").status_code == 503
