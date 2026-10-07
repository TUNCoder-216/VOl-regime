"""One-shot job: download the serving model from the MLflow registry into MODEL_DIR.

Runs as the `model-fetcher` compose service before the API starts.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

import mlflow
from mlflow import MlflowClient

try:  # local runs read .env; in Docker, compose injects the variables instead
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("model-fetcher")


def resolve_model_version(client, name: str, alias: str):
    """Prefer the alias (explicit promotion); fall back to the highest version number."""
    try:
        return client.get_model_version_by_alias(name, alias), f"alias:{alias}"
    except Exception as exc:  # noqa: BLE001 - any registry error triggers the fallback
        logger.warning("Alias '%s' unavailable (%s); using latest version", alias, exc)
    versions = client.search_model_versions(f"name='{name}'")
    if not versions:
        raise RuntimeError(f"No registered versions for model '{name}'")
    return max(versions, key=lambda v: int(v.version)), "latest-version"


def main() -> int:
    if not os.getenv("MLFLOW_TRACKING_URI"):
        logger.error("MLFLOW_TRACKING_URI is not set")
        return 1
    name = os.getenv("MODEL_NAME", "vol-regime")
    alias = os.getenv("MODEL_ALIAS", "champion")
    model_dir = Path(os.getenv("MODEL_DIR", "/models"))

    client = MlflowClient()
    mv, resolved_by = resolve_model_version(client, name, alias)
    run = client.get_run(mv.run_id)
    logger.info("Resolved %s v%s via %s (run %s)", name, mv.version, resolved_by, mv.run_id)

    staging = model_dir / ".staging"
    shutil.rmtree(staging, ignore_errors=True)
    (staging / "model").mkdir(parents=True)
    mlflow.artifacts.download_artifacts(
        artifact_uri=f"models:/{name}/{mv.version}", dst_path=str(staging / "model")
    )
    try:
        mlflow.artifacts.download_artifacts(
            run_id=mv.run_id, artifact_path="reference", dst_path=str(staging)
        )
    except Exception as exc:  # noqa: BLE001 - reference data is optional
        logger.warning("No reference window for this run (%s)", exc)

    metadata = {
        "name": name,
        "version": int(mv.version),
        "alias": alias,
        "resolved_by": resolved_by,
        "run_id": mv.run_id,
        "candidate": run.data.tags.get("candidate", "unknown"),
        "git_sha": run.data.tags.get("git_sha", "unknown"),
        "data_end": run.data.tags.get("data_end", "unknown"),
        "decision_threshold": float(run.data.params.get("decision_threshold", 0.5)),
        "metrics": run.data.metrics,
        "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (staging / "metadata.json").write_text(json.dumps(metadata, indent=2))

    current = model_dir / "current"
    shutil.rmtree(current, ignore_errors=True)
    staging.rename(current)
    logger.info("=== Serving model ready: %s v%s (%s) ===", name, mv.version, metadata["candidate"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
