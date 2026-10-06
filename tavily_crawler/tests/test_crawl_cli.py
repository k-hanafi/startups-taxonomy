"""The paid crawl stays off unless --live is passed."""

from __future__ import annotations

import pytest

from tavily_crawler.crawl_cli import main


def test_crawl_exits_without_live(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        main([])
    assert caught.value.code == 2
    assert "Pass --live" in capsys.readouterr().err


def test_crawl_live_reaches_the_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    def stop(**_kwargs: object) -> None:
        raise RuntimeError("runner reached")

    monkeypatch.setattr("tavily_crawler.crawl_cli.run_tavily_crawl", stop)
    with pytest.raises(RuntimeError, match="runner reached"):
        main(["--live"])
