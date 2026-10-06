"""All filesystem paths for the Wayback Machine sub-project.

One place for every path so no module hard-codes a string. Paths are rooted
here, keeping generated artifacts inside ``wayback_machine/`` (so the folder
is self-contained and liftable).

Both ``data/`` and ``outputs/`` are git-ignored: ``data/`` holds frozen inputs,
``outputs/`` holds everything the scrape generates (raw API responses, resume
state, processed evidence, logs).
"""

from __future__ import annotations

from pathlib import Path

# wayback_machine/ -> project root is one level up.
WAYBACK_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = WAYBACK_ROOT.parent

# ---------------------------------------------------------------------------
# Frozen inputs (git-ignored)
# ---------------------------------------------------------------------------

DATA_DIR = WAYBACK_ROOT / "data"
COHORT_CSV = DATA_DIR / "wayback_cohort.csv"
COVERAGE_SAMPLE_CSV = DATA_DIR / "coverage_sample.csv"
COVERAGE_FULL_CSV = DATA_DIR / "coverage_full.csv"
SCRAPE_TARGETS_CSV = DATA_DIR / "scrape_targets.csv"

# Metadata base for the final CSV. This is the SAME static file the live crawl
# joined against, so the 2023 output carries byte-identical company metadata and
# the only thing that differs from the live input is the website evidence.
MASTER_CSV = PROJECT_ROOT / "data" / "master_csv.csv"

# ---------------------------------------------------------------------------
# Generated artifacts (git-ignored)
# ---------------------------------------------------------------------------

OUTPUTS_DIR = WAYBACK_ROOT / "outputs"
RAW_DIR = OUTPUTS_DIR / "raw"
PROCESSED_DIR = OUTPUTS_DIR / "processed"
LOGS_DIR = OUTPUTS_DIR / "logs"

SNAPSHOTS_JSONL = RAW_DIR / "snapshots.jsonl"
EXTRACT_STATE_JSON = RAW_DIR / "extract_state.json"
RUN_MANIFEST_CSV = RAW_DIR / "run_manifest.csv"
SPIKE_JSONL = RAW_DIR / "spike_extract.jsonl"

SCRAPE_PROCESSED_CSV = PROCESSED_DIR / "scrape_processed.csv"
CLASSIFIER_INPUT_2023_CSV = PROCESSED_DIR / "classifier_input_2023.csv"

EXTRACT_LOG = LOGS_DIR / "extract.log"

# ---------------------------------------------------------------------------
# Survivorship dead-cohort strand (GO: archive crawl). Suffixed so the dead run
# never collides with the historical 2023 extract artifacts above.
# ---------------------------------------------------------------------------

DEATH_COVERAGE_CSV = DATA_DIR / "death_coverage.csv"
SCRAPE_TARGETS_DEAD_CSV = DATA_DIR / "scrape_targets_dead.csv"

CRAWL_DEAD_JSONL = RAW_DIR / "crawl_dead.jsonl"
CRAWL_STATE_DEAD_JSON = RAW_DIR / "crawl_state_dead.json"
RUN_MANIFEST_DEAD_CSV = RAW_DIR / "run_manifest_dead.csv"

SCRAPE_PROCESSED_DEAD_CSV = PROCESSED_DIR / "scrape_processed_dead.csv"
CLASSIFIER_INPUT_DEAD_CSV = PROCESSED_DIR / "classifier_input_dead.csv"

CRAWL_DEAD_LOG = LOGS_DIR / "crawl_dead.log"
