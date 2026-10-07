import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "webhook_listener", Path(__file__).parents[1] / "deploy" / "webhook_listener.py"
)
webhook = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(webhook)


def test_token_check():
    assert webhook.is_authorized("s3cret-token-value", "s3cret-token-value")
    assert not webhook.is_authorized("wrong", "s3cret-token-value")
    assert not webhook.is_authorized(None, "s3cret-token-value")
    assert not webhook.is_authorized("", "")


def test_env_file_parsing(tmp_path):
    env = tmp_path / ".env"
    env.write_text('# comment\nWEBHOOK_TOKEN="abc"\nWEBHOOK_PORT=9100\nBROKEN_LINE\n')
    assert webhook.load_env_file(env) == {"WEBHOOK_TOKEN": "abc", "WEBHOOK_PORT": "9100"}
