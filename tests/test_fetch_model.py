from types import SimpleNamespace

import pytest

from api.fetch_model import resolve_model_version


class FakeClient:
    def __init__(self, alias_version=None, versions=()):
        self.alias_version = alias_version
        self.versions = [SimpleNamespace(version=str(v)) for v in versions]

    def get_model_version_by_alias(self, name, alias):
        if self.alias_version is None:
            raise RuntimeError("alias not found")
        return SimpleNamespace(version=str(self.alias_version))

    def search_model_versions(self, query):
        return self.versions


def test_alias_is_preferred():
    mv, how = resolve_model_version(
        FakeClient(alias_version=2, versions=[1, 2, 3]), "m", "champion"
    )
    assert (mv.version, how) == ("2", "alias:champion")


def test_falls_back_to_latest_version():
    mv, how = resolve_model_version(FakeClient(versions=[1, 10, 2]), "m", "champion")
    assert (mv.version, how) == ("10", "latest-version")


def test_no_versions_raises():
    with pytest.raises(RuntimeError, match="No registered versions"):
        resolve_model_version(FakeClient(), "m", "champion")
