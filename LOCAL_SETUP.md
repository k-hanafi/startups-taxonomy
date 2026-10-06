# Local Development Setup

This guide helps you set up this repository on your local machine to work alongside the Cloud Agent environment.

## Prerequisites

- Python ≥3.11
- Git
- A GitHub account with access to this repository

## Initial Setup

### 1. Clone the Repository

```bash
# If you haven't cloned yet:
git clone https://github.com/k-hanafi/startups-taxonomy.git
cd startups-taxonomy

# Start on main (default branch)
git checkout main
git pull origin main
```

### 2. Set Up Python Environment

```bash
# Create a virtual environment
python3 -m venv .venv

# Activate it
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"
```

### 3. Configure API Keys

Create the `keys/` directory (git-ignored) and add your API keys:

```bash
mkdir -p keys

# OpenAI API key
echo 'OPENAI_API_KEY=your_openai_key_here' > keys/openai.env

# Tavily API key
echo 'TAVILY_API_KEY=your_tavily_key_here' > keys/tavily.env
```

**Important:** These files are git-ignored and will never be committed. Each developer needs their own keys.

### 4. Verify Setup

```bash
# Run tests to verify everything works (no data files required)
OPENAI_API_KEY=placeholder pytest
OPENAI_API_KEY=placeholder PYTHONPATH=. pytest wayback_machine/tests

# Optional: cost estimate once classifier input exists locally
# OPENAI_API_KEY=placeholder python -m single_pass_classifier prepare --dry-run
# (requires outputs/tavilycrawl/processed/classifier_input.csv from a completed crawl)
```

## Working with the Repository

### Understanding the Data Flow

The repository has two classifiers and three evidence strands.

`python -m two_pass_classifier` is the production classifier. `python -m single_pass_classifier` is the legacy batch classifier. Historical and dead-company runs still use the legacy one.

1. **Live strand** (DONE): Classifies companies based on current websites
2. **Historical strand** (FROZEN): Re-classifies using March 2023 Internet Archive snapshots (infra built, paid extract not run)
3. **Survivorship strand** (FROZEN): Dead-cohort extract is complete; classify/merge not landed in this snapshot

**Current focus:** Local setup only. The research snapshot on `main` is frozen.

### Key Commands

```bash
# Tests. placeholder is a fake key so the legacy package can load.
OPENAI_API_KEY=placeholder pytest

# Production classifier. These commands make no API calls.
# Default inputs are local files and are not in git.
python -m two_pass_classifier build-manifest
python -m two_pass_classifier cost-preview

# Legacy batch classifier. Paid commands need a real key.
python -m single_pass_classifier prepare --dry-run
python -m single_pass_classifier status

# Live website enrichment
python -m tavily_crawler liveness
python -m tavily_crawler crawl

# Dead-company recovery (see wayback_machine/README.md)
python wayback_machine/scripts/probe_death_coverage.py
python wayback_machine/scripts/build_targets_dead.py
python wayback_machine/scripts/run_extract_dead.py
```

### Understanding Directory Structure

```
/
├── single_pass_classifier/ # Live V1 classifier application
├── two_pass_classifier/ # Production V2 classifier and contract owner
├── tavily_crawler/       # Live liveness and crawl application
├── evals/                # Golden-set research harness over production V2
├── wayback_machine/      # Historical + survivorship strands
├── scripts/              # Supporting utilities
├── data/                 # Input data (git-ignored, not indexed)
├── outputs/              # Generated results (git-ignored)
├── keys/                 # API keys (git-ignored)
└── AGENTS.md             # Agent briefing (read this!)
```

**Important:** `data/` and `outputs/` are not indexed by Cursor. Read them via terminal or Read tool.

## Syncing with Cloud Agent Work

### Pull Latest Changes

```bash
# Fetch all branches from remote
git fetch origin

# Pull your current branch
git pull origin "$(git branch --show-current)"

# See what branches the cloud agent has created
git branch -r | grep cursor/
```

### View Cloud Agent Commits

```bash
# See recent commits
git log --oneline -10

# See what changed in a specific commit
git show <commit-hash>

# Compare your local changes with remote
git diff "origin/$(git branch --show-current)"
```

### Working in Parallel

**Recommended workflow:**

1. Cloud agent works on a feature branch (often `cursor/*`)
2. You pull your branch regularly: `git pull origin "$(git branch --show-current)"`
3. Make your own changes in a separate branch if needed:
   ```bash
   git checkout -b feature/my-local-work
   ```
4. Push your branch and create a PR when ready

**Avoid conflicts:** Don't edit the same files the cloud agent is actively working on. Check recent commits first.

## Data Files

### Large Files (Not in Git)

These directories are git-ignored and must be downloaded separately or regenerated:

- `data/master_csv.csv` (44,387 companies)
- `outputs/production_csvs/production_classifications.csv` (live results)
- `wayback_machine/data/death_coverage.csv` (death probe results)
- `wayback_machine/data/scrape_targets_dead.csv` (survivorship targets)

**To get these files:**

1. Download from your cloud storage/backup
2. Or regenerate them by running the pipeline stages
3. Or ask the cloud agent to copy critical files to a shared location

### Checking Data Status

```bash
# See what data files exist
ls -lh data/
ls -lh outputs/production_csvs/
ls -lh wayback_machine/data/

# Check file sizes
du -sh data/ outputs/ wayback_machine/data/ wayback_machine/outputs/
```

## Running Network Operations Locally

**Important:** Scripts that make network calls or API requests should run outside the Cursor sandbox (i.e., in your local terminal).

### For Long-Running Operations

```bash
# Prevent your Mac from sleeping during long operations
caffeinate -ims python wayback_machine/scripts/run_extract_dead.py

# Or use tmux for session persistence
tmux new -s wayback-crawl
python wayback_machine/scripts/run_extract_dead.py
# Detach: Ctrl+B, then D
# Reattach: tmux attach -t wayback-crawl
```

## Troubleshooting

### Tests Fail with "OPENAI_API_KEY not set"

```bash
# For tests that don't need a real key:
OPENAI_API_KEY=placeholder pytest

# For integration tests:
source keys/openai.env  # Load real key
pytest
```

### "Module not found" errors

```bash
# Reinstall in development mode
pip install -e ".[dev]"
```

### Can't find data files

```bash
# Create expected directories
mkdir -p data outputs/production_csvs wayback_machine/data wayback_machine/outputs

# Check what's in .gitignore
cat .gitignore
```

### Git remote issues

```bash
# Verify remote is set
git remote -v

# If no remote, add it
git remote add origin https://github.com/k-hanafi/startups-taxonomy.git

# Fetch all branches
git fetch origin
```

## Best Practices

1. **Always activate your venv:** `source .venv/bin/activate`
2. **Never commit API keys:** They live in `keys/` (git-ignored)
3. **Read AGENTS.md first:** It's the authoritative project guide
4. **Check branch before making changes:** `git status` and `git branch`
5. **Pull before pushing:** `git pull origin <branch>` to avoid conflicts
6. **Use descriptive commits:** The cloud agent does too
7. **Run tests before committing:** `pytest` should pass

## Getting Help

- Read `AGENTS.md` for project architecture and data flow
- Read `wayback_machine/README.md` for survivorship pipeline details
- Run commands with `--help`: `python -m single_pass_classifier --help`
- Check package test folders for usage examples

## Next Steps

After setup:

1. Read `AGENTS.md` thoroughly
2. Run `pytest` to verify everything works
3. Read `two_pass_classifier/README.md` for the production command order
4. Check what the cloud agent has been working on: `git log --oneline -20`
5. Review current branch status: `git status`

---

**Remember:** Live, historical, and survivorship strands differ by `website_evidence`. The production classifier is `python -m two_pass_classifier`. The legacy batch classifier remains for historical runs and for the user message the production runner sends.
