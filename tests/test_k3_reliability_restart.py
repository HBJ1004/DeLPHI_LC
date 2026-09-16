"""Black-box interruption check of the runner's real file/budget primitives."""
import subprocess
import sys
import time

from lc_pipeline.k3.reliability_study import unseal


def test_background_launcher_detaches_and_uses_the_snapshot(tmp_path, monkeypatch):
    from lc_pipeline.k3 import reliability_study as study
    calls = []

    class Process:
        pid = 7654321

    def fake_popen(command, **kwargs):
        calls.append((command, kwargs))
        return Process()

    study.seal(tmp_path / "study.json", {"python": study.binding(sys.executable),
                                         "source_snapshot": str(tmp_path / "snapshot")})
    monkeypatch.setattr(study.subprocess, "Popen", fake_popen)
    receipt = study.start_background(tmp_path)
    assert receipt["pid"] == 7654321
    command, kwargs = calls[0]
    assert kwargs["start_new_session"] is True
    assert kwargs["cwd"] == str(tmp_path / "snapshot")
    assert kwargs["env"]["PYTHONPATH"] == str(tmp_path / "snapshot")
    assert command[-2:] == ["--root", str(tmp_path)]
    assert study.read_json(tmp_path / "progress.json")["pid"] == receipt["pid"]


def test_terminated_process_resumes_without_reset_or_duplicate_cells(tmp_path):
    program = """
import sys, time
from pathlib import Path
from lc_pipeline.k3.reliability_study import Budget, exclusive, seal, unseal
root=Path(sys.argv[1])
# Import startup on the mounted dependency tree is not a restart operation.
(root / ('ready-' + sys.argv[2])).write_text('ready')
with exclusive(root):
    budget=Budget(root, 'inference', 300)
    for index in range(10):
        budget.check()
        path=root / f'cell-{index}.json'
        if path.exists():
            assert unseal(path) == {'index': index}
            continue
        seal(path, {'index': index})
        time.sleep(.1)
"""
    def wait_for(path, process, seconds):
        deadline = time.monotonic() + seconds
        while not path.exists():
            assert process.poll() is None
            assert time.monotonic() < deadline
            time.sleep(0.02)

    first = subprocess.Popen([sys.executable, "-c", program, str(tmp_path), "first"])
    second = None
    try:
        wait_for(tmp_path / "ready-first", first, 120)
        wait_for(tmp_path / "cell-1.json", first, 20)
        before = unseal(tmp_path / "budget-inference.json")
        initial_cell = (tmp_path / "cell-0.json").read_bytes()
        first.terminate()
        first.wait(timeout=5)
        second = subprocess.Popen([sys.executable, "-c", program, str(tmp_path), "resume"],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        wait_for(tmp_path / "ready-resume", second, 120)
        _, stderr = second.communicate(timeout=20)
        assert second.returncode == 0, stderr.decode()
        assert unseal(tmp_path / "budget-inference.json") == before
        assert (tmp_path / "cell-0.json").read_bytes() == initial_cell
        assert len(list(tmp_path.glob("cell-*.json"))) == 10
        assert [unseal(tmp_path / f"cell-{i}.json")["index"] for i in range(10)] == list(range(10))
    finally:
        for process in (first, second):
            if process is not None and process.poll() is None:
                process.kill()
                process.wait(timeout=5)
