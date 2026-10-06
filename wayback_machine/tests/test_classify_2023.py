from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env.pop("OPENAI_API_KEY", None)
    return subprocess.run(
        args,
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _assert_retired(completed: subprocess.CompletedProcess[str]) -> None:
    combined = completed.stdout + completed.stderr
    assert completed.returncode == 2
    assert "retired batch classifier" in completed.stderr
    assert "python -m two_pass_classifier" in completed.stderr
    assert "ModuleNotFoundError" not in combined


def test_historical_classify_command_exits_without_legacy_package() -> None:
    completed = _run(
        [sys.executable, "-m", "wayback_machine.classify_2023", "run"]
    )
    _assert_retired(completed)


def test_dead_classify_command_exits_without_legacy_package() -> None:
    script = PROJECT_ROOT / "wayback_machine" / "scripts" / "classify_dead.py"
    completed = _run([sys.executable, str(script), "run"])
    _assert_retired(completed)
