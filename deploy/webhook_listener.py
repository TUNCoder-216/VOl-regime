"""Local deployment webhook (stdlib only). Exposed to GitHub Actions through ngrok.

POST /deploy with header X-Webhook-Token -> docker compose pull + up (in a background thread).
Run from the repo root:  python deploy/webhook_listener.py
"""

from __future__ import annotations

import hmac
import json
import logging
import os
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG_FILE = ROOT / "deploy" / "deploy.log"
DEPLOY_LOCK = threading.Lock()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(LOG_FILE, encoding="utf-8")],
)
logger = logging.getLogger("webhook")


def load_env_file(path: Path) -> dict[str, str]:
    """Minimal .env parser (KEY=VALUE, # comments). Existing env vars take precedence."""
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def is_authorized(received: str | None, expected: str) -> bool:
    """Constant-time comparison: does not leak the token through response timing."""
    return bool(received) and bool(expected) and hmac.compare_digest(received, expected)


DEPLOY_STEPS = [
    ["docker", "compose", "pull"],
    ["docker", "compose", "up", "-d", "--force-recreate", "--remove-orphans"],
    ["docker", "compose", "logs", "model-fetcher", "--no-color", "--tail", "15"],
]


def run_deploy(sha: str) -> None:
    with DEPLOY_LOCK:
        logger.info("Deploy started for commit %s", sha)
        for cmd in DEPLOY_STEPS:
            logger.info("$ %s", " ".join(cmd))
            try:
                result = subprocess.run(
                    cmd,
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
            except OSError as exc:
                logger.error("Cannot run %s (%s). Is Docker Desktop running?", cmd[0], exc)
                return
            output = (result.stdout + result.stderr).strip()
            if output:
                logger.info(output)
            if result.returncode != 0:
                logger.error("Deploy FAILED at: %s", " ".join(cmd))
                return
        logger.info("Deploy finished for commit %s", sha)


class WebhookHandler(BaseHTTPRequestHandler):
    token = ""

    def _reply(self, status: int, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802 - http.server naming
        if self.path == "/health":
            self._reply(200, {"status": "ok", "deploying": DEPLOY_LOCK.locked()})
        else:
            self._reply(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802 - http.server naming
        if self.path != "/deploy":
            self._reply(404, {"error": "not found"})
            return
        if not is_authorized(self.headers.get("X-Webhook-Token"), self.token):
            logger.warning("Rejected unauthorized request from %s", self.client_address[0])
            self._reply(401, {"error": "unauthorized"})
            return
        if DEPLOY_LOCK.locked():
            self._reply(409, {"status": "a deploy is already running"})
            return
        length = int(self.headers.get("Content-Length") or 0)
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            payload = {}
        sha = str(payload.get("sha", "unknown"))[:12]
        threading.Thread(target=run_deploy, args=(sha,), daemon=True).start()
        self._reply(202, {"status": "deploy started", "sha": sha})


def main() -> None:
    env = {**load_env_file(ROOT / ".env"), **os.environ}
    token = env.get("WEBHOOK_TOKEN", "")
    if len(token) < 16:
        raise SystemExit("Set a WEBHOOK_TOKEN of at least 16 characters in .env")
    port = int(env.get("WEBHOOK_PORT", "9000"))
    WebhookHandler.token = token
    server = ThreadingHTTPServer(("127.0.0.1", port), WebhookHandler)
    logger.info("Webhook listener on http://127.0.0.1:%d (POST /deploy, GET /health)", port)
    server.serve_forever()


if __name__ == "__main__":
    main()
