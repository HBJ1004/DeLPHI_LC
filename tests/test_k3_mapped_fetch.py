"""Offline regression tests for mapped MPC-numbered ZTF acquisition."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lc_pipeline.k3.mapped_ztf import FINK_SSO_API_URL, MappedZTFError, fetch_mapped_fink


def _identity_map(path: Path) -> Path:
    objects = []
    for offset in range(170):
        damit_id = 101 + offset
        objects.append(
            {
                "object_id": f"asteroid_{damit_id}",
                "damit_id": damit_id,
                "mpc_number": 2 if offset == 0 else 1000 + offset,
                "held_out_fold": offset % 5,
            }
        )
    path.write_text(
        json.dumps(
            {
                "schema": "delphi.k3-survey-identity-map.v1",
                "identity_table_sha256": "b" * 64,
                "splits_sha256": "c" * 64,
                "objects": objects,
            }
        ),
        encoding="utf-8",
    )
    return path


def _row(*, numeric_identity: int = 2, fid: int = 2, sigma: float = 0.1) -> dict[str, object]:
    return {
        "i:ssnamenr": numeric_identity,
        "sso_name": "Pallas",
        "i:ra": 10.0,
        "i:dec": 20.0,
        "i:jd": 2459000.0,
        "i:magpsf": 18.0,
        "i:sigmapsf": sigma,
        "i:fid": fid,
        "Dhelio": 2.0,
        "Dobs": 1.5,
        "Phase": 20.0,
    }


class _Response:
    def __init__(self, body: bytes, status: int = 200):
        self.body = body
        self.status = status
        self.url = FINK_SSO_API_URL

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self) -> bytes:
        return self.body


def _fetch(tmp_path: Path, monkeypatch, body: bytes):
    identity = _identity_map(tmp_path / "identity-map.json")
    calls: list[object] = []

    def fake_urlopen(request, timeout):
        calls.append(request)
        assert request.full_url == FINK_SSO_API_URL
        assert json.loads(request.data.decode("utf-8"))["n_or_d"] == "2"
        assert timeout == 3.0
        return _Response(body)

    monkeypatch.setattr("lc_pipeline.k3.mapped_ztf.urlopen", fake_urlopen)
    monkeypatch.setattr("lc_pipeline.k3.mapped_ztf.time.sleep", lambda _: None)
    result = fetch_mapped_fink(
        identity_map_path=identity,
        output_directory=tmp_path / "fetch",
        object_ids=["asteroid_101"],
        timeout_seconds=3.0,
    )
    return identity, result, calls


def test_mapped_fetch_uses_mpc_two_for_internal_damit_101_and_current_ztf_endpoint(tmp_path, monkeypatch):
    identity, result, calls = _fetch(tmp_path, monkeypatch, json.dumps([_row()]).encode())
    assert len(calls) == 1
    record = result["objects"][0]
    assert record["object_id"] == "asteroid_101"
    assert record["damit_id"] == 101
    assert record["mpc_number"] == 2
    assert record["status"] == "received_validated"
    assert json.loads(calls[0].data.decode("utf-8"))["n_or_d"] != "101"


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (b"[]", "empty_response"),
        (json.dumps([_row(fid=1)]).encode(), "received_no_retained_rows"),
    ],
)
def test_mapped_fetch_distinguishes_empty_and_no_retained_rows(tmp_path, monkeypatch, body, expected):
    _, result, _ = _fetch(tmp_path, monkeypatch, body)
    assert result["objects"][0]["status"] == expected


def test_mapped_fetch_rejects_numeric_identity_for_wrong_mpc_target(tmp_path, monkeypatch):
    _, result, _ = _fetch(tmp_path, monkeypatch, json.dumps([_row(numeric_identity=101)]).encode())
    record = result["objects"][0]
    assert record["status"] == "invalid_response"
    assert "identity mismatch" in record["error"]


def test_mapped_fetch_resume_accepts_equal_raw_hash_and_rejects_tampering(tmp_path, monkeypatch):
    identity = _identity_map(tmp_path / "identity-map.json")
    body = json.dumps([_row()]).encode()
    calls: list[object] = []
    monkeypatch.setattr(
        "lc_pipeline.k3.mapped_ztf.urlopen",
        lambda request, timeout: (calls.append(request) or _Response(body)),
    )
    monkeypatch.setattr("lc_pipeline.k3.mapped_ztf.time.sleep", lambda _: None)
    kwargs = dict(identity_map_path=identity, output_directory=tmp_path / "fetch", object_ids=["asteroid_101"], timeout_seconds=3.0)
    fetch_mapped_fink(**kwargs)
    fetch_mapped_fink(**kwargs)
    assert len(calls) == 1
    raw = tmp_path / "fetch" / "raw" / "2.json"
    raw.write_text("[]", encoding="utf-8")
    with pytest.raises(MappedZTFError, match="raw response hash mismatch"):
        fetch_mapped_fink(**kwargs)


def test_mapped_fetch_classifies_plain_timeout_as_failed_http(tmp_path, monkeypatch):
    identity = _identity_map(tmp_path / "identity-map.json")
    monkeypatch.setattr("lc_pipeline.k3.mapped_ztf.urlopen", lambda *args, **kwargs: (_ for _ in ()).throw(TimeoutError("timed out")))
    result = fetch_mapped_fink(identity_map_path=identity, output_directory=tmp_path / "fetch", object_ids=["asteroid_101"], timeout_seconds=3.0)
    row = result["objects"][0]
    assert row["status"] == "failed_http"
    assert row["error"] == "timed out"
    assert result["failed_http_count"] == 1
    receipt = json.loads((tmp_path / "fetch" / "receipts" / "2.json").read_text())
    assert receipt == row
    assert not (tmp_path / "fetch" / "raw" / "2.json").exists()
