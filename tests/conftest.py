"""Explicit data fixtures for portable own-period tests."""

import os

import pytest


@pytest.fixture
def temporary_delphi_data(tmp_path, monkeypatch):
    """Give synthetic tests their own data root, regardless of the user's setting."""
    root = tmp_path / "data" / "own-period"
    root.mkdir(parents=True)
    monkeypatch.setenv("DELPHI_DATA", str(root))
    return root


@pytest.fixture
def require_delphi_files():
    """Skip optional stored-data checks before resolving lazy, unconfigured paths."""
    def require(*paths):
        if not os.environ.get("DELPHI_DATA"):
            pytest.skip("DELPHI_DATA is unset; optional stored data or solver is unavailable")
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            pytest.skip("optional DELPHI_DATA files are unavailable: " + ", ".join(missing))

    return require
