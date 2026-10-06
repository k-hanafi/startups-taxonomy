# wayback_machine

Reconstruct the **March-2023 (GPT-4 launch) homepages** of our classified
startups from the Internet Archive, clean them the exact same way the live crawl
did, and emit a `classifier_input_2023.csv`. Only `website_evidence` should
differ from the live input. Production classification is
`python -m two_pass_classifier`. `classify_2023.py` exits 2. The batch
classifier that used to score this file was removed.

The evidence-recovery stages are a **self-contained sub-project**. Their cleaner
is vendored in `evidence.py` and guarded by a golden test. Do not merge that
copy with the live cleaner.

## Why Tavily `/extract` (not a raw HTML download)?

We deliberately fetch each archive snapshot through Tavily `/extract`, the same
engine the live site crawl used. Identical fetch + identical cleaning means the
2023 evidence is comparable to today's, so any classification change is real and
not a tooling artifact. Tavily takes URLs only (it cannot read local HTML), and
we already have every snapshot URL for free from the coverage probe — so there is
no separate HTML pre-download step.

## Layout

```
wayback_machine/
  config.py          # all tunables (target date, extract config, rate/budget)
  paths.py           # every path; nothing hard-codes a string
  cohort.py          # vendored column contracts + snapshot-URL builder + filters
  evidence.py        # VENDORED frozen crawler cleaner (golden-tested)
  state.py           # atomic resume state + JSONL healing + completed-ids
  extract.py         # the resumable Tavily /extract engine
  targets.py         # Stage B: coverage_full.csv -> scrape_targets.csv
  classifier_input.py# Stage D: master + 2023 evidence -> classifier_input_2023.csv
  classify_2023.py   # exits 2; the batch classifier was removed
  scripts/           # thin argparse CLIs (run these)
  tests/             # golden cleaner + cohort helpers
  data/              # frozen inputs   (git-ignored)
  outputs/           # generated       (git-ignored)
```

## Commands

Run these from the project root. Paid extract commands exit 2 unless you pass `--live`.

| Command | What it does |
|---------|----------------|
| `python3 wayback_machine/scripts/build_targets.py` | Freeze the March-2023 company list. |
| `python3 wayback_machine/scripts/spike_extract.py --live --n 50` | Try a small paid sample. |
| `python3 wayback_machine/scripts/run_extract.py --live` | Pull archive pages for that list. |
| `python3 wayback_machine/scripts/build_classifier_input_2023.py` | Join archive text to company metadata. |
| `python3 wayback_machine/scripts/run_extract_dead.py --live` | Pull a pre-death page for companies the live crawl missed. |
| `python3 wayback_machine/scripts/build_classifier_input_dead.py` | Join that recovered text to company metadata. |

`python -m wayback_machine.classify_2023` and `python wayback_machine/scripts/classify_dead.py` exit 2. Score the resulting CSV with `python -m two_pass_classifier`.

## Resumability & safety

- **Resume:** re-running `run_extract.py` reads the append-only
  `outputs/raw/snapshots.jsonl` and skips companies already finished — it never
  pays twice. Smoke-test first with `--max-companies 20`.
- **Crash-safe:** state is written atomically; the JSONL is fsynced per row and
  a crash-truncated tail is healed on startup.
- **Interrupt:** Ctrl-C drains cleanly at the next row boundary.
- **Budget:** capped by credit estimate (basic extract bills 1 credit / 5
  successes); raise `--budget-credits` to lift the guardrail.
- **Empty/thin results** are recorded as terminal and skipped on resume. To retry
  them later, delete their lines from `snapshots.jsonl` and re-run.

## The one rule that matters

`evidence.py` must stay behavior-identical to `tavily_crawler/website_evidence.py`.
If the live cleaner ever changes, re-vendor it and run `pytest wayback_machine/tests`.
