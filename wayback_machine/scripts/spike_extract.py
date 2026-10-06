#!/usr/bin/env python3
"""De-risk the paid run: extract a small sample of archive snapshots and report.

Measures the two unknowns before committing budget to ~16k companies:
1. Success rate of Tavily fetching the *Internet Archive* (the Archive throttles
   Tavily's fetches, so this is NOT the same as the live-site success rate).
2. Evidence richness (chars) and the resulting credits/company.

Writes raw responses to outputs/raw/spike_extract.jsonl. Does not touch the main
scrape state, so it is safe to run repeatedly.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from wayback_machine.config import ExtractConfig, estimate_credits  # noqa: E402
from wayback_machine.extract import (  # noqa: E402
    _api_key,
    _evidence_from_response,
    _has_usable_results,
    call_tavily_extract,
)
from wayback_machine.live_gate import add_live_flag, require_live  # noqa: E402
from wayback_machine.paths import SCRAPE_TARGETS_CSV, SPIKE_JSONL  # noqa: E402

csv.field_size_limit(1_000_000_000)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_live_flag(parser)
    parser.add_argument("--targets", type=Path, default=SCRAPE_TARGETS_CSV)
    parser.add_argument("--n", type=int, default=50, help="Companies to sample.")
    parser.add_argument("--output", type=Path, default=SPIKE_JSONL)
    parser.add_argument("--extract-depth", default="basic", choices=["basic", "advanced"])
    args = parser.parse_args(argv)
    require_live(args.live)

    if not args.targets.exists():
        raise SystemExit(f"Targets not found: {args.targets}. Run build_targets.py first.")

    with args.targets.open(encoding="utf-8", newline="") as f:
        targets = list(csv.DictReader(f))[: args.n]
    if not targets:
        raise SystemExit("No targets to sample.")

    cfg = ExtractConfig(extract_depth=args.extract_depth)
    api_key = _api_key()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    ok = empty = failed = 0
    char_lengths: list[int] = []
    started = time.monotonic()
    with args.output.open("w", encoding="utf-8") as out:
        for i, t in enumerate(targets, start=1):
            url = t.get("snapshot_url", "")
            rec: dict[str, object] = {"org_uuid": t.get("org_uuid", ""), "snapshot_url": url}
            try:
                resp = call_tavily_extract(url, cfg, api_key)
            except Exception as exc:  # noqa: BLE001 - record and continue
                failed += 1
                rec.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
                print(f"  [{i}/{len(targets)}] FAIL {t.get('name','')}: {type(exc).__name__}",
                      file=sys.stderr)
                continue
            if not _has_usable_results(resp):
                empty += 1
                rec["status"] = "empty"
            else:
                _, evidence = _evidence_from_response(resp, t.get("homepage_url", ""))
                if evidence:
                    ok += 1
                    char_lengths.append(len(evidence))
                    rec.update({"status": "ok", "evidence_chars": len(evidence)})
                else:
                    empty += 1
                    rec["status"] = "thin"
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            print(f"  [{i}/{len(targets)}] {rec['status']:<6} {t.get('name','')}", file=sys.stderr)

    n = len(targets)
    elapsed = time.monotonic() - started
    avg_chars = sum(char_lengths) / len(char_lengths) if char_lengths else 0
    cph = estimate_credits(ok, extract_depth=cfg.extract_depth) / ok if ok else 0.0
    print("\n=== SPIKE SUMMARY ===", file=sys.stderr)
    print(f"  sampled:        {n}", file=sys.stderr)
    print(f"  ok:             {ok} ({ok / n * 100:.0f}%)", file=sys.stderr)
    print(f"  empty/thin:     {empty}", file=sys.stderr)
    print(f"  failed:         {failed}", file=sys.stderr)
    print(f"  avg evidence:   {avg_chars:,.0f} chars", file=sys.stderr)
    print(
        f"  est credits:    {estimate_credits(ok, extract_depth=cfg.extract_depth):.1f} "
        f"({cph:.2f}/company)",
        file=sys.stderr,
    )
    print(f"  elapsed:        {elapsed:.0f}s ({n / elapsed * 60:.0f}/min)", file=sys.stderr)
    print(f"  raw -> {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
