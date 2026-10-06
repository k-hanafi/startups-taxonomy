"""Paid archive commands exit 2 unless --live is passed."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from wayback_machine.live_gate import require_live

ROOT = Path(__file__).resolve().parents[2]


def _load_script(name: str):
    path = ROOT / "wayback_machine" / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "script",
    ["run_extract", "run_extract_dead", "spike_extract"],
)
def test_paid_extract_exits_without_live(script: str, capsys: pytest.CaptureFixture[str]) -> None:
    module = _load_script(script)
    with pytest.raises(SystemExit) as caught:
        module.main([])
    assert caught.value.code == 2
    assert "Pass --live" in capsys.readouterr().err


def test_require_live_allows_an_explicit_run() -> None:
    require_live(True)


def test_live_extract_reaches_the_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_script("run_extract")

    def stop(**_kwargs: object) -> None:
        raise RuntimeError("runner reached")

    monkeypatch.setattr(module, "run_extract", stop)
    with pytest.raises(RuntimeError, match="runner reached"):
        module.main(["--live"])
