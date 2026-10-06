# wayback_machine

Reconstruct the **March-2023 (GPT-4 launch) homepages** of our classified
startups from the Internet Archive, clean them the exact same way the live crawl
did, and emit a `classifier_input_2023.csv` that drops straight into the existing
classifier. Running the unchanged classifier on it lets us diff today vs 2023 and
measure how startup AI messaging shifted.

The evidence-recovery stages are a **self-contained sub-project**. Their cleaner
is vendored in `evidence.py` and guarded by a golden test. The only intentional
bridge to a root application is `classify_2023.py`, which binds an isolated
namespace before importing the unchanged V1 classifier.

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
  classify_2023.py   # Stage E: namespaced bridge to the unchanged V1 classifier
  scripts/           # thin argparse CLIs (run these)
  tests/             # golden cleaner + cohort helpers
  data/              # frozen inputs   (git-ignored)
  outputs/           # generated       (git-ignored)
```

## Run order

All commands run from the project root. Stages C and the recovery probe touch the
network and must run **outside the Cursor sandbox**.

| Stage | Command | Cost |
|-------|---------|------|
| A. Discover (done) | `scripts/probe_coverage.py` → `scripts/summarize_coverage.py` | free |
| B. Freeze targets | `python3 wayback_machine/scripts/build_targets.py` | free |
| Spike (de-risk) | `python3 wayback_machine/scripts/spike_extract.py --n 50` | ~10 credits |
| C. Extract | `python3 wayback_machine/scripts/run_extract.py` | paid |
| D. Build input | `python3 wayback_machine/scripts/build_classifier_input_2023.py` | free |
| E. Classify | `python -m wayback_machine.classify_2023 run` | paid |

Stage B keeps a company only if it is **both** retrievable **and** existed at
GPT-4 launch (`founded_date` ≤ `2023-03`, override with `--founded-cutoff`). The
existence filter is not redundant: the probe picks the capture closest to March
14 within Dec 2022 – Jun 2023, so a company founded mid-2023 could otherwise slip
in via a later snapshot.

The Stage E wrapper sets `CLASSIFY_NS=wayback_2023` before any classifier
imports. Its state and final CSV therefore stay under `outputs/wayback_2023/`
and cannot overwrite the live V1 artifacts.

For the overnight extract, hold sleep with caffeinate (outside the sandbox):

```bash
caffeinate -ims python3 wayback_machine/scripts/run_extract.py
```

## Dead-company recovery

These commands recover a pre-death homepage for companies the live crawl could
not extract. They are frozen. Do not start the paid steps without a new brief.

| Stage | Command | Cost |
|-------|---------|------|
| Cohort | `python3 wayback_machine/scripts/build_not_found_cohort.py` | free |
| Death probe | `python3 wayback_machine/scripts/probe_death_coverage.py` | free |
| Summarize probe | `python3 wayback_machine/scripts/summarize_death_coverage.py` | free |
| Targets | `python3 wayback_machine/scripts/build_targets_dead.py` | free |
| Extract | `python3 wayback_machine/scripts/run_extract_dead.py` | paid |
| Classifier input | `python3 wayback_machine/scripts/build_classifier_input_dead.py` | free |
| Classify | `python3 wayback_machine/scripts/classify_dead.py run` | paid |
| Merge | `python3 wayback_machine/scripts/merge_survivorship.py` | free |

`classify_dead.py` sets `CLASSIFY_NS=wayback_dead` before the legacy classifier
imports, so its CSV stays under `outputs/wayback_dead/`.

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
