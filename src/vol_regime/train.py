"""Training + tracking + registration (DVC stage: train)."""

from __future__ import annotations

import json
import logging
import os
import subprocess

import mlflow
import mlflow.sklearn
import pandas as pd
from dotenv import load_dotenv
from mlflow import MlflowClient
from mlflow.models import infer_signature

from vol_regime.config import DATASET_PATH, METRICS_PATH, REFERENCE_PATH, load_params
from vol_regime.data import load_raw
from vol_regime.modeling import (
    build_candidates,
    evaluate,
    passes_quality_gate,
    select_best,
    time_split,
)

logger = logging.getLogger(__name__)


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    load_dotenv()
    if not os.getenv("MLFLOW_TRACKING_URI"):
        raise SystemExit("MLFLOW_TRACKING_URI is not set. Copy .env.example to .env and fill it.")

    params = load_params()
    fp, sp, tp, rp = params["features"], params["split"], params["train"], params["registry"]

    dataset = pd.read_csv(DATASET_PATH, parse_dates=["date"], index_col="date")
    x_train, x_test, y_train, y_test = time_split(dataset, sp["test_size"], gap=fp["horizon"])

    # Recent raw window shipped with the model: lets the API predict even if live data fails.
    REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    load_raw().tail(rp["reference_rows"]).to_csv(REFERENCE_PATH)

    mlflow.set_experiment(rp["experiment"])
    tags = {
        "git_sha": git_sha(),
        "data_start": str(dataset.index.min().date()),
        "data_end": str(dataset.index.max().date()),
        "test_start": str(x_test.index.min().date()),
        "train_rows": str(len(x_train)),
        "test_rows": str(len(x_test)),
    }

    results = []
    for name, model in build_candidates(tp).items():
        with mlflow.start_run(run_name=name) as run:
            mlflow.set_tags({**tags, "candidate": name})
            mlflow.log_params(
                {
                    "candidate": name,
                    "horizon": fp["horizon"],
                    "threshold_window": fp["threshold_window"],
                    "threshold_quantile": fp["threshold_quantile"],
                    "test_size": sp["test_size"],
                    "decision_threshold": tp["decision_threshold"],
                    **{f"{name}.{k}": v for k, v in tp[name].items()},
                }
            )
            model.fit(x_train, y_train)
            proba = model.predict_proba(x_test)[:, 1]
            metrics = evaluate(y_test.to_numpy(), proba, tp["decision_threshold"])
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                signature=infer_signature(x_test, proba),
                input_example=x_test.head(3),
            )
            mlflow.log_artifact(str(REFERENCE_PATH), artifact_path="reference")
            results.append({"name": name, "run_id": run.info.run_id, "metrics": metrics})
            logger.info("%s: %s", name, {k: round(v, 4) for k, v in metrics.items()})

    best = select_best(results, rp["selection_metric"])
    version = mlflow.register_model(f"runs:/{best['run_id']}/model", rp["model_name"]).version
    client = MlflowClient()
    promoted = passes_quality_gate(best["metrics"], rp["quality_gate"])
    client.set_model_version_tag(
        rp["model_name"], version, "quality_gate", "passed" if promoted else "failed"
    )
    if promoted:
        client.set_registered_model_alias(rp["model_name"], rp["alias"], version)
        logger.info("Promoted %s v%s to @%s", rp["model_name"], version, rp["alias"])
    else:
        logger.warning("v%s failed the quality gate; @%s unchanged", version, rp["alias"])

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(
        json.dumps(
            {**best["metrics"], "model_version": int(version), "promoted": int(promoted)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
