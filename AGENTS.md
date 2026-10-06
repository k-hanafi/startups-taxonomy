# AGENTS.md

Briefing for AI coding agents working in this repo. **Read this first** — it
replaces an exhaustive codebase search. It is auto-injected into every chat.

If you change the repo's structure, architecture, data flow, commands, or
status, **update this file in the same change**. See [Maintaining this file](#maintaining-this-file).

Last updated: 2026-10-06 | Active branch: `cursor/portfolio-entry-points-9aae` (README is a command glossary; paid crawl and extract need `--live`)

---

## Project overview

Research codebase for a two-axis taxonomy of AI-native startups (UBC student
research; companion SSRN paper "Prompted to Start"). Every company gets:

- an **AI-native axis** — `ai_native` 0/1 plus a `subclass` (1A–1G AI-native,
  0A/0B/0C not), and
- a **RAD score** (Resource-Adjusted AI Dependency: how dependent/defensible the
  company is vs. foundation-model providers).

The pipeline enriches Crunchbase rows with website evidence, then classifies
them with an LLM. Production classification is `python -m two_pass_classifier`
only. It calls the OpenAI Responses API in two passes. The landed live file
was written by a retired batch classifier. Historical and dead-company classify
commands exit 2.

1. **Live** (built, run): classify companies on today's websites.
2. **Historical / wayback** (infra built, paid extract not run): recover each
   company's **March-2023 (GPT-4 launch)** homepage from the Internet Archive.
   The classify command exits 2.
3. **Survivorship-bias** (extract complete, classify/merge not landed): recover
   **pre-death** snapshots for the ~22k companies Tavily couldn't extract.
   The classify command exits 2. Merge is not landed.

**Core invariant:** production classification reads the source columns in
`two_pass_classifier/input_contract.py`. Each strand is a different way to
produce `website_evidence`. The taxonomy does not change between strands. The
only thing that differs across strands is the evidence.

## Status / roadmap

This snapshot is **frozen**. Do not resume paid crawl, extract, or classify
work from this repo without an explicit new brief.

| Strand | Stage | Status |
|--------|-------|--------|
| Live | crawl → classify → merge | DONE — 44,387 companies classified (`production_classifications.csv`) |
| Historical (wayback) | coverage probe done; infra built | FROZEN — GO verdict (~16k retrievable at Mar-2023); paid extract not run |
| Survivorship-bias | probe done → extract DONE | FROZEN — 19,044 targets covered, 15,714 with evidence in `scrape_processed_dead.csv`; classify/merge not landed |

Eval alignment: `two_pass_classifier` owns every classifier contract used by
`evals` (prompts, schemas, request bodies, formatting, cohort, confidence,
models, defaults, output caps, and normal Responses pricing). The eval package
keeps golden-data research and orchestration only. Existing local eval results
that used the prior prompt fingerprint remain historical until a new paid sweep.

Repo agent skills (committed): **`portfolio-git-messages`**, **`git-commit-batch-plan`**,
**`code-structure`**, **`clean-my-repo`** under **`.cursor/skills/`**. **`.cursor/rules/`**
stays local. Implementation notes that used to live under `.cursor/plans/` are in git history.

## Tech stack

Python ≥3.11 · `openai` (Responses API) · `pandas` · `pydantic` (structured
output) · `tiktoken` (pre-flight cost) · `tenacity` (retries) · `rich` (terminal
UI) · `python-dotenv`. Tavily HTTP API for web crawl/extract (stdlib `urllib`).
Internet Archive CDX API for snapshot discovery. Tests: `pytest`. The
alive-vs-dead dashboard adds `statsmodels` and `scipy` (logistic regression,
proportion tests), installed via the `analysis` extra.

## Architecture & data flow

```
LIVE strand
data/master_csv.csv ──python -m tavily_crawler liveness──▶ website_alive set in place
        └──python -m tavily_crawler crawl──▶ outputs/tavilycrawl/processed/classifier_input.csv
                └──python -m two_pass_classifier──▶ outputs/two_pass_classifier/runs/<run>/classifications.csv

HISTORICAL strand (self-contained recovery; classify command exits 2)
coverage_full.csv ──build_targets.py──▶ scrape_targets.csv
        └──run_extract.py (Tavily /extract on archive URLs)──▶ outputs/raw/snapshots.jsonl
                └──build_classifier_input_2023.py──▶ classifier_input_2023.csv ──▶ python -m wayback_machine.classify_2023 (exits 2)

SURVIVORSHIP strand (frozen; GO = archive crawl matching the live cohort)
classifier_input.csv (empty-evidence rows) ──build_not_found_cohort.py──▶ not_found_cohort.csv
 └──probe_death_coverage.py (death-anchored CDX)──▶ death_coverage.csv
 └──build_targets_dead.py──▶ scrape_targets_dead.csv (if_ snapshot URL + per-company scope)
 └──run_extract_dead.py (Tavily /extract on pre-death snapshot)──▶ scrape_processed_dead.csv
 └──build_classifier_input_dead.py──▶ classifier_input_dead.csv
 └──classify_dead.py (exits 2; batch classifier removed)
 └──merge_survivorship.py──▶ outputs/wayback_dead/survivorship_corrected.csv
 └──build_v1_alive_dead_dashboard.py (evidence-only alive-vs-dead, 4-act survivorship story)──▶ data visualization/01_Presentation_Materials/v1_alive_dead_cohort.html

```

`two_pass_classifier` is the only classifier. A full run needs a matching
10-row smoke. `events.jsonl` is the resume authority. The batch classifier
that wrote `outputs/production_csvs/production_classifications.csv` was removed.

## Repository layout

### Root
| Path | Purpose |
|------|---------|
| `tavily_crawler/` | Live liveness and Tavily crawl application and `python -m tavily_crawler` CLI |
| `two_pass_classifier/` | Production V2 application: immutable manifest, offline cost preview, 10-row smoke gate, async Responses runner, status/resume/retry, confidence, professor exporter |
| `README.md` | Public-facing writeup (taxonomy + pipeline narrative + mermaid diagrams) |
| `LOCAL_SETUP.md` | Local clone, venv, keys, and pytest |
| `pyproject.toml` | Dependencies + pytest config |
| `AGENTS.md` | This file |
| `.cursor/skills/` | Four committed repo skills: `portfolio-git-messages`, `git-commit-batch-plan`, `code-structure`, `clean-my-repo` |

### `tavily_crawler/` (live website enrichment)
| File | Responsibility |
|------|----------------|
| `cli.py` / `__main__.py` | Canonical CLI with `liveness` and `crawl` subcommands |
| `paths.py` | Existing live crawl input and output locations |
| `master_csv.py` | Column contracts, URL validation, and Tavily eligibility mask |
| `website_evidence.py` | Cleans/compacts raw Tavily markdown into evidence text (strips chrome, packs signal-first) |
| `crawl.py` | Cost-controlled Tavily `/crawl` runner for live homepage enrichment (resumable, rate-limited, budget-capped) |
| `crawl_cli.py` | Crawl flags and command adapter |
| `liveness.py` | Parallel homepage probe and `website_alive` updater |

### `two_pass_classifier/` (production classifier)
| File | Responsibility |
|------|----------------|
| `cli.py` / `__main__.py` | Canonical CLI (`build-manifest`, `cost-preview`, `smoke`, `run`, `status`, `resume`, `retry`) with lazy paid-key loading |
| `api_key.py` | Paid-call key load from the environment or `keys/openai.env`. Import does not require a key |
| `README.md` | Beginner run order and load-bearing flags for `python -m two_pass_classifier` |
| `config.py` | Supported models and locked defaults (`gpt-5.6-luna`, Pass A effort `none`, Pass B effort `low`, Pass A `top_logprobs=5`) |
| `schema.py` | Strict family-specific Pydantic contracts with 100-word reasoning and critique limits |
| `prompts/` | Single production prompt source for Pass A/B (moved out of root `prompts/`) |
| `formatter.py` / `request_builder.py` | User message, custom id, and character cap, plus strict Responses request bodies, cache routes, token reservations, request fingerprints |
| `input_contract.py` / `cohort.py` | Stable source/model-visible fields and deterministic PRE-GENAI vs GENAI-ERA assignment |
| `manifest.py` | Evidence-only live+dead JSONL manifest; joins `company_alive` / `website_snapshot_date` at build time |
| `confidence.py` | Offline sampled-token confidence (censored-opponent midpoint) |
| `exporter.py` | Exact 18-column professor CSV (`company_alive` and `website_snapshot_date` after `cohort`) |
| `costing.py` | Offline production token counts and normal Responses price ranges |
| `paths.py` | Manifest/run output locations under `outputs/two_pass_classifier/` |
| `journal.py` | Group-committed JSONL writer, run lock, authoritative resume state from `events.jsonl`, derived CSV/JSON rebuild |
| `rate_control.py` | Dual RPM/TPM admission, adaptive concurrency, and cache-route warming |
| `runner.py` | Coupled Pass A/B orchestration over AsyncOpenAI, retries, graceful shutdown, raw response preservation |
| `workflow.py` / `status.py` | Run IDs, deterministic smoke selection, smoke fingerprint gate, journal-owned resume context, and offline status metrics |

### `wayback_machine/` (historical + survivorship strands)
| File | Responsibility |
|------|----------------|
| `README.md` | Sub-project guide + stage-by-stage run order |
| `config.py` | Historical tunables: target date, CDX rate limits, `ExtractConfig`, budget, death-anchor lookback (`DEATH_LOOKBACK_DAYS`) |
| `paths.py` | All wayback paths |
| `cohort.py` | Vendored column contracts + snapshot-URL builder + retrievable/existence filters |
| `evidence.py` | **VENDORED** frozen copy of `tavily_crawler/website_evidence.py` (golden-tested; must stay behavior-identical) |
| `cdx.py` | Minimal IA CDX client (`to_host` + rate-limited `cdx_get`, freezes all workers on 429); used by the death probe |
| `state.py` | `ExtractState` resume + JSONL tail-healing + completed-ids reconciliation |
| `extract.py` | Resumable, budget-capped Tavily `/extract` engine (historical analogue of `tavily_crawl.py`) |
| `targets.py` | Stage B: `coverage_full.csv` → `scrape_targets.csv` |
| `targets_dead.py` | **(survivorship)** Stage B: `death_coverage.csv` → `scrape_targets_dead.csv` (emits `if_` crawl URL + per-company `select_paths` scope; no founded cutoff) |
| `extract_dead.py` | **(survivorship)** Stage C: resumable, budget-capped Tavily `/extract` over pre-death `if_`/`id_` snapshots; reuses `extract.py`'s reliability harness + failure-reason instrumentation (rate_limited vs no_archive_content); writes to the crawl-era artifact names to preserve resume state |
| `classifier_input.py` | Stage D: master metadata + 2023 evidence → `classifier_input_2023.csv` (reused by the dead strand) |
| `classify_2023.py` | **(historical)** Exits 2. The batch classifier was removed. Production classification is `python -m two_pass_classifier` |

### `wayback_machine/scripts/` — thin CLIs
| File | Purpose |
|------|---------|
| `extract_cohort.py` | Build the frozen wayback cohort from live data |
| `probe_coverage.py` | Stage A: CDX coverage probe at the global Mar-2023 anchor |
| `summarize_coverage.py` | Aggregate `coverage_full.csv` for the dashboard |
| `build_targets.py` | CLI for `targets.py` |
| `spike_extract.py` | Small de-risk extract (~50 companies) before the full run |
| `run_extract.py` | CLI for the paid extract engine |
| `build_classifier_input_2023.py` | CLI for `classifier_input.py` |
| `build_not_found_cohort.py` | **(survivorship)** Build `not_found_cohort.csv` from empty-evidence rows |
| `probe_death_coverage.py` | **(survivorship, active)** Death-anchored CDX probe → `death_coverage.csv` |
| `run_probe_recovery.sh` | Shell helper to resume the recovery probe |
| `summarize_death_coverage.py` | **(survivorship)** Aggregate `death_coverage.csv` into compact JSON |
| `build_targets_dead.py` | **(survivorship)** CLI for `targets_dead.py` |
| `run_extract_dead.py` | **(survivorship, paid)** CLI for the dead-cohort extract engine (`extract_dead.run_extract_dead`); wrap in `caffeinate -ims` outside the sandbox |
| `build_classifier_input_dead.py` | **(survivorship)** CLI: dead evidence → `classifier_input_dead.csv` |
| `classify_dead.py` | **(survivorship)** Exits 2. The batch classifier was removed. Production classification is `python -m two_pass_classifier` |
| `merge_survivorship.py` | **(survivorship)** Stage F: overlay dead verdicts onto `production_classifications.csv`, tag `evidence_source`, write `survivorship_corrected.csv` + before/after summary |
| `summarize_crawl_failures.py` | **(survivorship)** Offline (stdlib-only, no keys) breakdown of `crawl_dead.jsonl` by `failure_reason` (rate_limited / no_archive_content / transient / network / legacy_empty) |

### `evals/` — golden-set eval harness
| Path | Purpose |
|------|---------|
| `dashboard_metrics.py` | Eval dashboard metrics: scored.json/fixture → chart metrics (ECE, reliability bins, selective curves, vs_baseline, Pass B isolating fields, finalist mean±range aggregates, per-config `cost_breakdown` for the cost popover). Real loads recompute production $ from each run's `predictions.jsonl` and scale by the newest valid production manifest, with an explicit offline fallback of 37,746. Also `build_robustness` + `build_run_instance`. No OpenAI import. |
| `tests/fixtures/dashboard/dashboard_mock_runs.json` | Synthetic locked matrix; Pass A metrics identical across efforts within each model (bank-once design); calibration blocks derive from one set of 100 synthetic rows per model (nano seeds the ECE ~0.077 early signal); per-run robustness blocks |
| `instances.py` | Numbered dashboard archive: writes `eval_instance_NN.html` + `index.html` + `instances.json` under `01_Presentation_Materials/eval_instances/`; an instance is identified by the scored runs behind it (same sweep replaces its page; a later sweep still gets a new number); synthetic `--save-instance` previews replace the prior mock. Also owns the run-headline / run-meta text shared with the suite header card. |
| `config.py` | Research-only sampling, scoring, calibration, and robustness settings; the matrix model and effort tuples plus Luna-low defaults are direct aliases of production config |
| `classification.py` | Normal Responses eval orchestration over production-owned Pass A/B request builders; Pass A auto-banks under `evals/runs/pass_a_banks/<model>/`, with production fingerprint checks that reject historical banks |
| `cost_preview.py` | Offline matrix estimates from production request bodies, token counting, pricing, provisional output estimates, and one-attempt cap projections; Pass A is counted once per model |
| `orchestrate.py` | `run-evals` supervisor: always from scratch (rebuild Pass A banks, mint new cell run ids; re-paying intentional). Phase-1 banks (3 parallel), then 9 cells in parallel; each cell scores with `--confidence-from-raw` (writes calibration + `robustness.valid_mass`); dashboard; rich live checklist; `open-dashboard` opens the instance index. |
| `logprob_extract.py` | Thin raw-artifact adapter over production confidence extraction; eval-only valid-mass summaries and run-directory loading remain here |
| `runner.py` | Shared eval mechanics only: golden-row loading, retry policy, completed-ID resume scan, and git provenance; no classifier builder |
| `scoring.py` | End-to-end accuracy axes plus family-conditional subclass and AI-native-only RAD metrics; `--baseline` paired deltas; refuses partial confidence unless `--allow-partial-confidence` |
| `__main__.py` | CLI: `cost-preview` / `run-evals` / `open-dashboard` (paid path); also `bank-pass-a`, `run-classification`, `matrix`, `score`, and `dashboard`; historical runs remain scoreable but cannot be reused as aligned banks |

### Other
| Path | Purpose |
|------|---------|
| `data visualization/01_Presentation_Materials/*.html` | Generated dashboards (`eval_dashboard.html` is overwritten every build) |
| `data visualization/01_Presentation_Materials/eval_instances/` | Kept eval suite builds: `eval_instance_NN.html` pages, `index.html` to browse them, `instances.json` registry |
| `data visualization/02_Analysis_Code/*.py` | Scripts that build those dashboards |
| `data visualization/02_Analysis_Code/survivorship_analysis.py` | Survivor-vs-dead compute on the evidence-only universe: distributions, BH-tested subclass deltas, funding/thin-history/snapshot-age cuts, coverage funnel, 3 logistic models (pure metrics dict; PREVIEW from production if `survivorship_corrected.csv` absent) |
| `data visualization/02_Analysis_Code/build_v1_alive_dead_dashboard.py` | Flagship V1 alive-vs-dead dashboard: 5 corrected base sections + 4-act survivorship story (bias / who dies / why / robustness); writes `v1_alive_dead_cohort.html`; loud PREVIEW banner pre-merge (replaces the retired `build_survivorship_insights_dashboard.py`) |
| `data visualization/02_Analysis_Code/build_v2_alive_dead_dashboard.py` | V2 alive-vs-dead dashboard on the professor CSV (`outputs/two_pass_classifier/production_classifications.csv`); three confidence explainers; consolidated survivorship charts; writes `v2_alive_dead_cohort.html` |
| `data visualization/02_Analysis_Code/build_eval_dashboard.py` | Classifier Eval Suite (flat enterprise SPA, three tabs): Pipeline robustness (checks panel), Model benchmarks (leaderboard + cost-ladder popover + Pareto + latency), Confidence correctness correlation (reliability diagram, per-model ECE, selective curves). Shared filter shell (chips + search) on benchmarks and confidence tabs. Header run-instance card names the run (synthetic on the fixture, run date and time on real loads). Defaults to mock fixture; `--runs`/`--scored` for real runs. Writes a self-contained `eval_dashboard.html` (Plotly inlined from `vendor/plotly-2.35.2.min.js`, no CDN) via `write_dashboard`, which archives real runs to `eval_instances/` automatically (mock builds need `--save-instance`). |
| `data visualization/02_Analysis_Code/vendor/plotly-2.35.2.min.js` | Vendored Plotly for offline/email-safe dashboard HTML (inlined at build time) |
| `tavily_crawler/tests/` | Live enrichment and crawl reliability tests |
| `two_pass_classifier/tests/` | V2 contracts plus async runner, journal, rate-control, retry, resume, lock, and export-gating tests |
| `wayback_machine/tests/` | pytest for wayback (golden cleaner, cohort, state, config, budget, probe) |
| `keys/` | API key env files, e.g. `keys/openai.env` (`OPENAI_API_KEY`). Git-ignored + cursor-ignored. **Never commit.** |
| `data/`, `outputs/`, `wayback_machine/data/`, `wayback_machine/outputs/` | Generated/large data. Git-ignored **and not indexed** — read via terminal/Read, not semantic search. |

## Key data artifacts

| Artifact | What it is |
|----------|-----------|
| `data/master_csv.csv` | 44,387 companies — static Crunchbase metadata + `website_alive`. The base everything joins against. |
| `outputs/tavilycrawl/processed/classifier_input.csv` | master + live `website_evidence`. Live input to the production manifest. |
| `outputs/two_pass_classifier/manifests/manifest_<sha256>.jsonl` | Immutable V2 evidence-only live+dead input; header stores measured source counts and raw source hashes |
| `outputs/two_pass_classifier/runs/<run>/events.jsonl` | Sole V2 resume authority (attempts, Pass A checkpoints, completed companies, raw responses). Derived CSV/JSON must never decide which requests run |
| `outputs/two_pass_classifier/runs/<run>/classifications.csv` | Atomic exact 18-column V2 professor artifact, created only when every manifest row is complete |
| `outputs/production_csvs/production_classifications.csv` | 44,387 classified rows from the retired batch run |
| `wayback_machine/data/coverage_full.csv` | Mar-2023 coverage probe over the 22,032 survivors |
| `wayback_machine/data/not_found_cohort.csv` | ~22,002 companies Tavily couldn't extract (survivorship target) |
| `wayback_machine/data/death_coverage.csv` | Death-anchored probe output (complete: 22,002 rows, 19,044 `ok`) |
| `wayback_machine/data/scrape_targets_dead.csv` | 19,044 dead-cohort extract targets (`if_` snapshot URL + scope); the frozen Stage-C work list |
| `outputs/wayback_dead/survivorship_corrected.csv` | Stage F output: modern dataset with dead verdicts overlaid (survivorship-corrected) |

## Domain model

The production artifact (contracts in `two_pass_classifier/exporter.py`) is exactly 18
analytical columns: `company_id`, `company_name`, `cohort`, `company_alive`,
`website_snapshot_date`, then classification/confidence/reasoning fields.
`company_alive` is evidence-strand yes/no (live vs archive/dead), not the HTTP
probe `website_alive`. Snapshot date is frozen into the immutable manifest at build.
Cohort is PRE-GENAI or GENAI-ERA, split at the GPT-4 launch on 2023-03-14.
The retired batch classifier wrote an 11-field row. That package is gone.


## Development commands

**`pytest` collects with `OPENAI_API_KEY` unset.** Paid commands load a real
key from the environment or `keys/openai.env` and reject placeholders.
`build-manifest`, `cost-preview`, and `status` do not need a key. Paid
commands (`smoke`, `run`, `resume`, `retry`) load the key only after
confirmation. Tavily enrichment still needs `keys/tavily.env`.

```bash
pip install -e ".[dev]"            # install with dev (pytest) extras
pytest                             # all offline test suites
pytest two_pass_classifier/tests -q # contracts, runner, CLI, and artifacts
pytest tavily_crawler/tests wayback_machine/tests

python -m two_pass_classifier build-manifest       # validate and freeze live+dead input
python -m two_pass_classifier cost-preview         # count tokens and price offline
python -m two_pass_classifier smoke                # paid exact 10-row production smoke
python -m two_pass_classifier run                  # paid new full run; matching smoke required
python -m two_pass_classifier status <run_id>      # fully offline progress and usage
python -m two_pass_classifier resume <run_id>      # paid continuation with locked semantics
python -m two_pass_classifier retry <run_id>       # append retry events; prints resume command

python -m tavily_crawler liveness              # set website_alive
python -m tavily_crawler crawl                 # live homepage crawl
python -m wayback_machine.classify_2023        # exits 2; batch classifier removed
python wayback_machine/scripts/classify_dead.py # exits 2; batch classifier removed
# wayback extract order: see wayback_machine/README.md

pytest evals/tests -q                       # full eval harness, no API key required at import
pytest evals/tests/test_dashboard_metrics.py   # dashboard metrics (no OpenAI key)
# Paid matrix (beginner path). Key loads from keys/openai.env automatically.
python -m evals cost-preview                    # per-config + total $ estimate (no API calls)
python -m evals run-evals                       # full matrix from scratch (rebuild banks → 9 cells → score → dashboard)
python -m evals open-dashboard                  # open eval_instances/index.html (newest run at top)
# Lower-level / escape hatches:
python -m evals matrix                          # list locked 9-cell matrix commands
python -m evals run-classification --model gpt-5.4-nano --effort-b low --require-matrix-cell
# later efforts for the same model auto-reuse Pass A (bank at evals/runs/pass_a_banks/<model>/)
# escape: --rerun-pass-a  |  advanced pin: --pass-a-from <run_id>
python -m evals dashboard                       # build eval_dashboard.html from mock matrix (default)
python -m evals dashboard --runs <run_id>...    # real scored.json only (no auto-discovery); auto-archives to eval_instances/
python -m evals dashboard --save-instance       # also keep this mock build as eval_instance_NN.html
python -m evals score <run_id> --confidence-from-raw [--baseline <run_id>]
python -m evals score <run_id> --allow-partial                 # incomplete n_scored only
python -m evals score <run_id> --allow-partial-confidence      # incomplete raw confidence only
python -m evals score <run_id> --confidence-from-raw --allow-missing-confidence  # accuracy even if no row has both {0,1}
# Existing pre-alignment runs are historical and cannot provide reusable Pass A banks.
```

## Conventions & invariants (don't break these)

- **Classifier tunables live in `two_pass_classifier/config.py`; its prompts live only in `two_pass_classifier/prompts/`. Wayback tunables live in `wayback_machine/config.py`.**
- **`evals` must import production classifier contracts from `two_pass_classifier`; it may own research orchestration and metrics, but never duplicate classifier behavior.**
- **V2 `events.jsonl` is the sole resume authority.** Derived JSON and CSV files must never decide which requests run.
- **A V2 full run requires a successful 10-row smoke with the same parent manifest and semantic request fingerprint.** Smoke outputs are never reused as full-run classifications.
- **V2 row and cost counts must come from the immutable manifest, never a hardcoded production population constant.**
- **Identical request prefix** across all requests is what enables prompt caching — keep it byte-stable.
- **Match results by `custom_id`**, never by position (batch order is not guaranteed).
- **`wayback_machine/evidence.py` must stay behavior-identical** to `tavily_crawler/website_evidence.py`. If you change the live cleaner, re-vendor and run `pytest wayback_machine/tests`.
- **Only `website_evidence` may differ** between strands fed to the classifier — that's the whole fair-comparison design.
- **Historical and dead-company classify commands exit 2.** They do not classify. Production classification is `python -m two_pass_classifier`. Do not resume paid crawl or extract.
- **Paid crawl and archive extract exit 2 unless `--live` is passed.** `python -m tavily_crawler crawl`, `run_extract.py`, `run_extract_dead.py`, and `spike_extract.py` spend Tavily credits only with that flag. `build-manifest --live` is a CSV path, not this gate.
- **Network/paid stages run OUTSIDE the Cursor sandbox** (Tavily crawl/extract, CDX probes, OpenAI). Wrap long runs in `caffeinate -ims` and/or `tmux`.
- **CDX is hard-capped at 60 req/min per IP**; exceeding it risks a 1-hour IP ban. Pace via `cdx.py`'s shared limiter; never raise rpm above ~58.
- `data/`, `outputs/`, `keys/` are git-ignored; `data/` & `outputs/` are also not indexed.

## Where to work

| Task | Start here |
|------|-----------|
| Change taxonomy / output fields | `two_pass_classifier/schema.py` + `two_pass_classifier/prompts/` |
| Change the company row shown to the model | `two_pass_classifier/formatter.py` |
| Change execution, resume, retry, or rate control | `two_pass_classifier/runner.py` + `journal.py` + `rate_control.py` + `request_builder.py` |
| Change CLI, smoke gate, cost preview, or status | `two_pass_classifier/cli.py` + `workflow.py` + `costing.py` + `status.py` |
| Change prompt/schema contracts | `two_pass_classifier/prompts/` + `two_pass_classifier/schema.py`; rerun two-pass and affected eval tests |
| Change manifest/export contract | `two_pass_classifier/manifest.py` + `two_pass_classifier/exporter.py` |
| Change evidence cleaning | `tavily_crawler/website_evidence.py` → re-vendor `wayback_machine/evidence.py` → run golden test |
| Live website scraping behavior | `tavily_crawler/crawl.py` |
| Historical archive scraping | `wayback_machine/extract.py` + `scripts/run_extract.py` |
| Survivorship death probe | `wayback_machine/scripts/probe_death_coverage.py` + `wayback_machine/cdx.py` |
| Survivorship extract→merge | `wayback_machine/extract_dead.py` + `scripts/{build_targets_dead,run_extract_dead,build_classifier_input_dead,merge_survivorship}.py` (`classify_dead.py` exits 2) |
| Dashboards | `data visualization/02_Analysis_Code/` |
| Alive-vs-dead dashboard / survivorship stats | `survivorship_analysis.py` (compute) + `build_v1_alive_dead_dashboard.py` (V1) or `build_v2_alive_dead_dashboard.py` (V2 professor CSV); rebuild V2 after the final production CSV lands |
| Eval dashboard (Classifier Eval Suite) | `evals/dashboard_metrics.py` (metrics + robustness checks) + `build_eval_dashboard.py` (three tabs: robustness / benchmarks / confidence; mock fixture until paid matrix runs; `--runs` for real data) |
| Kept dashboard builds / instance index | `evals/instances.py` (numbering, registry, index page) |
| Eval matrix / scoring | `evals/config.py` (`EVAL_MODELS` + `MATRIX_PASS_B_EFFORTS`); paid path `cost-preview` → `run-evals` → `open-dashboard` (`evals/orchestrate.py`); lower-level `run-classification` / `matrix` / `score --confidence-from-raw` |

## Maintaining this file

This file is the project's onboarding memory. Keep it self-healing: when your work
changes the repo, update the relevant section **in the same session/PR** — don't
wait to be asked.

Update triggers → what to edit:
- Strand/milestone started or finished → **Status/roadmap** table + the `Last updated` line.
- Top-level module/script added, removed, or renamed → **Repository layout**.
- Data flow, schema, or domain model changed → **Architecture & data flow** / **Domain model**.
- New dependency, command, or invariant → **Tech stack** / **Development commands** / **Conventions**.
- Active branch changed → the `Active branch` line.

Rules: surgical edits only, preserve structure and tone, keep entries one line,
no session chatter. Global policy: `~/.cursor/user-rules/agents-md-maintenance.md`.
