import io
import json

from repro.fetch_k3_generalization_sources import fetch


def test_small_download_receipt_counts_buffered_bytes_and_strips_signed_query(tmp_path, monkeypatch):
    from repro import fetch_k3_generalization_sources as module

    class Response(io.BytesIO):
        url = "https://public.example/data?signature=temporary"
        headers = {"Date": "Thu, 10 Sep 2026 00:00:00 GMT", "Content-Type": "text/plain"}

    monkeypatch.setattr(module, "SOURCES", {"fixture": {"small.txt": "https://public.example/data"}})
    monkeypatch.setattr(module, "urlopen", lambda *args, **kwargs: Response(b"three bytes are buffered"))
    result = fetch("fixture", tmp_path / "source")
    row = result["files"][0]
    assert row["bytes"] == (tmp_path / "source" / "small.txt").stat().st_size
    assert row["bytes"] > 0
    assert row["final_url"] == "https://public.example/data"
    assert "temporary" not in json.dumps(result)
