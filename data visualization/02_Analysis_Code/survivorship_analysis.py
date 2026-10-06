#!/usr/bin/env python3
"""Compute the survivor-vs-dead metrics dict consumed by the alive/dead dashboard.

Pure compute for the alive and dead dashboards. summarize_death_coverage.py
writes the coverage JSON. This module owns the comparison metrics and returns
one JSON-able dict. build_v1_alive_dead_dashboard.py renders it.

Evidence-only universe (locked design decision): the alive cohort is companies
Tavily successfully scraped (non-empty live website_evidence) and the dead
cohort is companies with recovered archive evidence (evidence_source ==
"wayback_dead"). Metadata-only classifications appear nowhere except the
coverage funnel counts.

It assembles a single analysis frame from the merged survivor-vs-dead dataset
(survivorship_corrected.csv), joins the static metadata it needs (funding,
category, founded date, liveness) plus death timing, defines the survivor / dead
cohorts, and computes the descriptive sections and the logistic-regression
models. If survivorship_corrected.csv is absent it builds a structurally
identical PREVIEW frame from production_classifications.csv tagged by the frozen
dead work list (metadata-only stand-in dead cohort), so the dashboard renders
before the classify-dead run lands.

Run directly to print cohort sizes + the metrics JSON (optionally --out FILE).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Verdict fields (mirrors merge_survivorship.VERDICT_COLS); CompanyID is the join key.
VERDICT_COLS = ["ai_native", "subclass", "rad_score", "cohort",
                "conf_classification", "conf_rad"]

SUBCLASS_ORDER = ["1A", "1B", "1C", "1D", "1E", "1F", "1G", "0A", "0B", "0C"]
RAD_ORDER = ["RAD-H", "RAD-M", "RAD-L", "RAD-NA"]
AI_RAD_ORDER = ["RAD-H", "RAD-M", "RAD-L"]

# Defensibility grouping for the RAD-validation story. 1C (thin wrapper) and 1G
# (generative content) are the most commoditizable; 1A/1B/1E carry structural or
# domain moats. 1D/1F sit in between.
SUBCLASS_GROUP = {
    "1A": "Defensible AI", "1B": "Defensible AI", "1E": "Defensible AI",
    "1C": "Commoditizable AI", "1G": "Commoditizable AI",
    "1D": "Other AI", "1F": "Other AI",
    "0A": "Not AI", "0B": "Not AI", "0C": "Not AI",
}
SUBCLASS_GROUP_ORDER = ["Commoditizable AI", "Other AI", "Defensible AI", "Not AI"]

FUNDING_BUCKETS = [
    ("unknown", -np.inf, 0),
    ("<$1M", 1, 1_000_000),
    ("$1-10M", 1_000_001, 10_000_000),
    ("$10-100M", 10_000_001, 100_000_000),
    ("$100M+", 100_000_001, np.inf),
]
FUNDING_BUCKET_ORDER = [b[0] for b in FUNDING_BUCKETS]

# Confident, well-known release dates only. Used as reference lines on the
# deaths-over-time chart; no speculative dates.
MODEL_RELEASES = [
    ("2022-11", "ChatGPT"),
    ("2023-03", "GPT-4"),
    ("2023-11", "GPT-4 Turbo"),
    ("2024-05", "GPT-4o"),
]
GPT4_LAUNCH = "2023-03-14"
TEMPORAL_START = pd.Timestamp("2022-01-01")
TOP_CATEGORY_GROUPS = 8

# Snapshot-age robustness cut: days between the snapshot we classified and the
# company's death anchor (last Wayback capture).
SNAPSHOT_AGE_BUCKETS = [("0-30d", 0, 30), ("31-90d", 31, 90),
                        ("91-365d", 91, 365), ("365d+", 366, np.inf)]
Z_95 = 1.96
BH_ALPHA = 0.05


@dataclass
class Paths:
    """Overridable input paths so the module can run against a fixture."""
    corrected: Path = PROJECT_ROOT / "outputs" / "wayback_dead" / "survivorship_corrected.csv"
    production: Path = PROJECT_ROOT / "outputs" / "production_csvs" / "production_classifications.csv"
    master: Path = PROJECT_ROOT / "data" / "master_csv.csv"
    classifier_input: Path = PROJECT_ROOT / "outputs" / "tavilycrawl" / "processed" / "classifier_input.csv"
    targets_dead: Path = PROJECT_ROOT / "wayback_machine" / "data" / "scrape_targets_dead.csv"
    not_found: Path = PROJECT_ROOT / "wayback_machine" / "data" / "not_found_cohort.csv"


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, on_bad_lines="skip")


def _join_covariates(base: pd.DataFrame, paths: Paths) -> pd.DataFrame:
    """Attach master covariates and dead-target provenance to a verdict frame."""
    if not paths.master.exists():
        raise SystemExit(f"master_csv not found: {paths.master}")
    master = _read(paths.master)
    meta_cols = ["org_uuid", "founded_date", "employee_count", "total_funding_usd",
                 "category_list", "category_groups_list", "website_alive"]
    master_slim = master[[c for c in meta_cols if c in master.columns]].copy()
    df = base.merge(master_slim, left_on="CompanyID", right_on="org_uuid", how="left")

    targets = _read(paths.targets_dead) if paths.targets_dead.exists() else pd.DataFrame()
    if not targets.empty:
        tcols = ["org_uuid", "death_ts", "days_before_death", "closest_ts", "thin_history"]
        tgt = targets[[c for c in tcols if c in targets.columns]].copy()
        df = df.merge(tgt, left_on="CompanyID", right_on="org_uuid", how="left", suffixes=("", "_tgt"))
    for c in ["death_ts", "days_before_death", "closest_ts", "thin_history"]:
        if c not in df.columns:
            df[c] = ""
    return df


def load_frame(paths: Paths) -> tuple[pd.DataFrame, dict]:
    """Build the analysis frame and a meta dict (incl. the preview flag)."""
    targets = _read(paths.targets_dead) if paths.targets_dead.exists() else pd.DataFrame()
    dead_ids = set(targets["org_uuid"]) if "org_uuid" in targets else set()

    preview = not paths.corrected.exists()
    if preview:
        if not paths.production.exists():
            raise SystemExit(
                "Neither survivorship_corrected.csv nor production_classifications.csv "
                f"found ({paths.corrected} / {paths.production})."
            )
        base = _read(paths.production)
        base["evidence_source"] = np.where(base["CompanyID"].isin(dead_ids), "dead_metadata", "live")
    else:
        base = _read(paths.corrected)
        # Corrected CSV should carry evidence_source from merge_survivorship. If an
        # older export omitted it, target-list members are recovered archive dead
        # (wayback_dead), not metadata stand-ins.
        if "evidence_source" not in base.columns:
            base["evidence_source"] = np.where(
                base["CompanyID"].isin(dead_ids), "wayback_dead", "live"
            )

    df = _join_covariates(base, paths)

    # Live evidence presence: a true survivor has non-empty website_evidence.
    if paths.classifier_input.exists():
        ci = _read(paths.classifier_input)[["org_uuid", "website_evidence"]]
        df = df.merge(ci, left_on="CompanyID", right_on="org_uuid", how="left", suffixes=("", "_ci"))
        df["has_live_evidence"] = df["website_evidence"].fillna("").str.strip().ne("")
    else:
        df["has_live_evidence"] = df["evidence_source"].eq("live")

    df = _derive(df, preview)
    meta = {
        "preview": preview,
        "partial": False,
        "classifier_version": "v1",
        "gpt4_launch": GPT4_LAUNCH,
        "n_total": int(len(df)),
        "n_universe": int((df["is_survivor"] | df["is_dead"]).sum()),
        "n_survivor": int(df["is_survivor"].sum()),
        "n_dead": int(df["is_dead"].sum()),
        "n_dead_recovered": int(df["is_dead_recovered"].sum()),
        "n_dead_strict": int(df["is_dead_strict"].sum()),
        "n_excluded": int(df["is_excluded"].sum()),
        "n_expected": None,
        "n_completed": int(len(df)),
    }
    return df, meta


def load_v2_frame(
    paths: Paths,
    v2_csv: Path,
    *,
    expected_rows: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Load a V2 professor CSV into the survivorship analysis frame.

    Maps company_alive to evidence_source (yes=live, no=wayback_dead). Keeps the
    three V2 confidence fields and mirrors subclass/rad confidence into the V1
    conf_* columns so shared base-metric helpers keep working.

    expected_rows is optional. Leave it None for a final production build (no
    PARTIAL banner). Pass a manifest size only when you want a partial-run warning.
    """
    if not v2_csv.exists():
        raise SystemExit(f"V2 classifications CSV not found: {v2_csv}")
    raw = _read(v2_csv)
    required = {
        "company_id", "company_name", "cohort", "company_alive",
        "ai_native", "subclass", "rad_score",
        "ai_native_confidence", "subclass_confidence", "rad_confidence",
    }
    missing = sorted(required - set(raw.columns))
    if missing:
        raise SystemExit(f"V2 CSV missing columns: {missing}")

    base = pd.DataFrame({
        "CompanyID": raw["company_id"],
        "CompanyName": raw["company_name"],
        "cohort": raw["cohort"],
        "company_alive": raw["company_alive"].str.lower().str.strip(),
        "website_snapshot_date": raw.get("website_snapshot_date", ""),
        "ai_native": raw["ai_native"],
        "subclass": raw["subclass"],
        "rad_score": raw["rad_score"],
        "ai_native_confidence": raw["ai_native_confidence"],
        "subclass_confidence": raw["subclass_confidence"],
        "rad_confidence": raw["rad_confidence"],
    })
    alive = base["company_alive"].eq("yes")
    base["evidence_source"] = np.where(alive, "live", "wayback_dead")
    # Shared V1 helpers expect these names for discrete 1-5 charts / filters.
    base["conf_classification"] = base["subclass_confidence"]
    base["conf_rad"] = base["rad_confidence"]

    df = _join_covariates(base, paths)
    # V2 production rows are evidence-only by construction.
    df["has_live_evidence"] = df["evidence_source"].eq("live")
    preview = False
    df = _derive(df, preview)
    n_completed = int(len(df))
    partial = bool(expected_rows is not None and n_completed < expected_rows)
    meta = {
        "preview": preview,
        "partial": partial,
        "classifier_version": "v2",
        "gpt4_launch": GPT4_LAUNCH,
        "n_total": n_completed,
        "n_universe": int((df["is_survivor"] | df["is_dead"]).sum()),
        "n_survivor": int(df["is_survivor"].sum()),
        "n_dead": int(df["is_dead"].sum()),
        "n_dead_recovered": int(df["is_dead_recovered"].sum()),
        "n_dead_strict": int(df["is_dead_strict"].sum()),
        "n_excluded": int(df["is_excluded"].sum()),
        "n_expected": expected_rows,
        "n_completed": n_completed,
        "v2_csv": str(v2_csv),
    }
    return df, meta


def _funding_bucket(v: float) -> str:
    if pd.isna(v):
        return "unknown"
    for label, lo, hi in FUNDING_BUCKETS:
        if lo <= v <= hi:
            return label
    return "unknown"


def _derive(df: pd.DataFrame, preview: bool) -> pd.DataFrame:
    df["ai_native_int"] = pd.to_numeric(df["ai_native"], errors="coerce").fillna(0).astype(int)
    df["funding_usd"] = pd.to_numeric(df["total_funding_usd"], errors="coerce")
    df["log_funding"] = np.log10(df["funding_usd"].fillna(0).clip(lower=0) + 1.0)
    df["funding_bucket"] = df["funding_usd"].apply(_funding_bucket)
    df["founded_year"] = pd.to_numeric(df["founded_date"].str[:4], errors="coerce")
    df["founding_era"] = df["cohort"].where(df["cohort"].isin(["PRE-GENAI", "GENAI-ERA"]), "unknown")
    df["subclass_group"] = df["subclass"].map(SUBCLASS_GROUP).fillna("Not AI")
    df["category_group"] = (
        df["category_groups_list"].fillna("").str.split(",").str[0].str.strip().replace("", "Unknown")
    )
    thin = df["thin_history"].astype(str).str.lower().isin(["true", "1"])
    website_dead = df["website_alive"].astype(str).str.lower().eq("false")

    df["is_dead_recovered"] = df["evidence_source"].eq("wayback_dead")
    df["is_dead_full"] = df["evidence_source"].isin(["wayback_dead", "dead_metadata"])
    df["is_survivor"] = df["evidence_source"].eq("live") & df["has_live_evidence"]
    df["is_excluded"] = df["evidence_source"].eq("live") & ~df["has_live_evidence"]
    # Evidence-only universe: the dead cohort is the recovered-evidence set.
    # PREVIEW (pre-merge) has zero wayback_dead rows, so the full work list
    # stands in behind the banner until the corrected CSV lands.
    df["is_dead"] = df["is_dead_full"] if preview else df["is_dead_recovered"]
    df["is_dead_strict"] = df["is_dead"] & website_dead & ~thin
    return df


# --------------------------------------------------------------------------- #
# Section computations (every helper takes the analysis frame explicitly)
# --------------------------------------------------------------------------- #

def _dist(f: pd.DataFrame, mask: pd.Series, col: str, order: list[str]) -> dict:
    counts = f[mask][col].value_counts().to_dict()
    n = int(mask.sum())
    return {
        "n": n,
        "counts": {k: int(counts.get(k, 0)) for k in order},
        "share": {k: round(counts.get(k, 0) / n * 100, 1) if n else 0.0 for k in order},
    }


def _wilson(k: int, n: int) -> dict:
    """Wilson 95% score interval for a proportion, in percent."""
    if n == 0:
        return {"lo": 0.0, "hi": 0.0}
    p = k / n
    z2 = Z_95 * Z_95
    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = Z_95 * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    return {"lo": round((center - half) * 100, 2), "hi": round((center + half) * 100, 2)}


def _ai_rate(f: pd.DataFrame, mask: pd.Series) -> dict:
    sub = f[mask]
    n = int(len(sub))
    ai = int((sub["ai_native_int"] == 1).sum())
    return {"n": n, "ai_native": ai, "rate": round(ai / n * 100, 1) if n else 0.0,
            "ci": _wilson(ai, n)}


def _mortality_by(f: pd.DataFrame, col: str, order: list[str]) -> list[dict]:
    """Per-level mortality = dead / (dead + survivor) within the outcome frame."""
    out = []
    for level in order:
        sel = f[col] == level
        dead = int((sel & f["is_dead"]).sum())
        surv = int((sel & f["is_survivor"]).sum())
        tot = dead + surv
        if tot:
            out.append({"label": level, "dead": dead, "survivor": surv, "n": tot,
                        "mortality": round(dead / tot * 100, 1)})
    return out


def _correction(f: pd.DataFrame) -> dict:
    """Biased (evidence-based alive only) vs corrected (alive + evidence-based
    dead) global distributions, both on the evidence-only universe."""
    biased = f["is_survivor"]
    corrected = f["is_survivor"] | f["is_dead"]
    return {
        "biased": {
            "ai_native": _ai_rate(f, biased),
            "subclass": _dist(f, biased, "subclass", SUBCLASS_ORDER),
            "rad": _dist(f, biased, "rad_score", RAD_ORDER),
        },
        "corrected": {
            "ai_native": _ai_rate(f, corrected),
            "subclass": _dist(f, corrected, "subclass", SUBCLASS_ORDER),
            "rad": _dist(f, corrected, "rad_score", RAD_ORDER),
        },
    }


def _ai_vs_survival(f: pd.DataFrame) -> dict:
    surv = _dist(f, f["is_survivor"], "subclass", SUBCLASS_ORDER)
    dead = _dist(f, f["is_dead"], "subclass", SUBCLASS_ORDER)
    lift = []
    for s in SUBCLASS_ORDER:
        sv, dv = surv["share"][s], dead["share"][s]
        lift.append({"subclass": s, "survivor": sv, "dead": dv,
                     "lift": round(dv / sv, 2) if sv else None})
    return {
        "ai_rate": {
            "survivor": _ai_rate(f, f["is_survivor"]),
            "dead": _ai_rate(f, f["is_dead"]),
            "dead_strict": _ai_rate(f, f["is_dead_strict"]),
        },
        "subclass_survivor": surv,
        "subclass_dead": dead,
        "subclass_lift": lift,
    }


def _subclass_tests(f: pd.DataFrame) -> dict:
    """Per-subclass two-proportion z-tests (survivor vs dead share) with a
    Benjamini-Hochberg correction, plus one omnibus chi-square on the 10x2
    contingency table."""
    try:
        from scipy.stats import chi2_contingency
        from statsmodels.stats.proportion import proportions_ztest
    except Exception as exc:
        return {"available": False, "error": f"scipy/statsmodels unavailable: {exc}"}

    surv, dead = f[f["is_survivor"]], f[f["is_dead"]]
    n1, n2 = len(surv), len(dead)
    if not n1 or not n2:
        return {"available": False, "error": "empty cohort"}

    rows = []
    for s in SUBCLASS_ORDER:
        k1 = int((surv["subclass"] == s).sum())
        k2 = int((dead["subclass"] == s).sum())
        if k1 + k2 == 0:
            continue
        _, p = proportions_ztest([k1, k2], [n1, n2])
        rows.append({"subclass": s, "survivor_share": round(k1 / n1 * 100, 2),
                     "dead_share": round(k2 / n2 * 100, 2),
                     "delta_pp": round(k2 / n2 * 100 - k1 / n1 * 100, 2),
                     "pvalue": float(p), "k1": k1, "k2": k2})

    # Benjamini-Hochberg at BH_ALPHA across the family of subclass tests.
    m = len(rows)
    for rank, row in enumerate(sorted(rows, key=lambda r: r["pvalue"]), start=1):
        row["bh_threshold"] = rank / m * BH_ALPHA
    cutoff_ranks = [r for r in rows if r["pvalue"] <= r["bh_threshold"]]
    max_p = max((r["pvalue"] for r in cutoff_ranks), default=-1.0)
    for row in rows:
        row["significant"] = row["pvalue"] <= max_p
        del row["bh_threshold"]

    table = [[r["k1"] for r in rows], [r["k2"] for r in rows]]
    chi2, chi_p, dof, _ = chi2_contingency(table)
    return {"available": True, "alpha": BH_ALPHA, "method": "BH",
            "n_survivor": n1, "n_dead": n2, "tests": rows,
            "chi2": {"stat": round(float(chi2), 2), "pvalue": float(chi_p),
                     "dof": int(dof)}}


def _funding_by_survival(f: pd.DataFrame) -> dict:
    """Funding-bucket shares per cohort plus per-bucket mortality."""
    surv, dead = f[f["is_survivor"]], f[f["is_dead"]]
    n1, n2 = len(surv), len(dead)
    shares = []
    for fb in FUNDING_BUCKET_ORDER:
        k1 = int((surv["funding_bucket"] == fb).sum())
        k2 = int((dead["funding_bucket"] == fb).sum())
        shares.append({"bucket": fb,
                       "survivor_share": round(k1 / n1 * 100, 1) if n1 else 0.0,
                       "dead_share": round(k2 / n2 * 100, 1) if n2 else 0.0,
                       "survivor": k1, "dead": k2})
    return {"shares": shares,
            "mortality": _mortality_by(f, "funding_bucket", FUNDING_BUCKET_ORDER)}


def _sensitivity(f: pd.DataFrame) -> dict:
    """Act 4 robustness: does the dead-cohort AI-native rate hinge on thin
    archives or stale snapshots?"""
    dead = f[f["is_dead"]]
    thin_mask = dead["thin_history"].astype(str).str.lower().isin(["true", "1"])
    by_thin = [
        {"label": "thin archive", **_ai_rate(dead, thin_mask)},
        {"label": "regular archive", **_ai_rate(dead, ~thin_mask)},
    ]
    age = pd.to_numeric(dead["days_before_death"], errors="coerce")
    by_age = []
    for label, lo, hi in SNAPSHOT_AGE_BUCKETS:
        m = (age >= lo) & (age <= hi)
        if int(m.sum()):
            by_age.append({"label": label, **_ai_rate(dead, m.fillna(False))})
    return {"by_thin_history": by_thin, "by_snapshot_age": by_age}


def _funnel(f: pd.DataFrame, paths: Paths, preview: bool = False) -> dict:
    """Coverage funnel: not-found cohort -> archive targets -> classified with
    recovered evidence. The residuals are what survivorship correction still
    cannot see.

    In PREVIEW mode the final stage uses the metadata stand-in dead count (same
    `is_dead` definition the rest of the dashboard uses) so the funnel does not
    collapse to zero while hero metrics show thousands of dead firms.
    """
    n_not_found = None
    if paths.not_found.exists():
        n_not_found = int(len(_read(paths.not_found)))
    n_targets = None
    if paths.targets_dead.exists():
        n_targets = int(len(_read(paths.targets_dead)))
    if preview:
        n_classified = int(f["is_dead"].sum())
        classified_label = "Preview dead (metadata stand-in)"
    else:
        n_classified = int(f["is_dead_recovered"].sum())
        classified_label = "Classified on recovered evidence"
    stages = []
    if n_not_found is not None:
        stages.append({"label": "Not found by Tavily", "n": n_not_found})
    if n_targets is not None:
        stages.append({"label": "Usable pre-death snapshot", "n": n_targets})
    stages.append({"label": classified_label, "n": n_classified})
    out = {"stages": stages}
    if n_not_found is not None and n_targets is not None:
        out["no_archive"] = n_not_found - n_targets
        out["extract_shortfall"] = n_targets - n_classified
    return out


def _rad_survival(f: pd.DataFrame) -> dict:
    # RAD-H/M/L mortality is an AI-native story (chart + insight say so); keep
    # non-AI rows out so a stray RAD label cannot dilute the rates.
    ai = f[f["ai_native_int"] == 1]
    return {
        "by_rad": _mortality_by(ai, "rad_score", AI_RAD_ORDER),
        "by_subclass": _mortality_by(f, "subclass", SUBCLASS_ORDER),
        "by_subclass_group": _mortality_by(f, "subclass_group", SUBCLASS_GROUP_ORDER),
        "by_era": _mortality_by(f, "founding_era", ["PRE-GENAI", "GENAI-ERA"]),
    }


def _dependency_trap(f: pd.DataFrame) -> dict:
    """Mortality grid among AI-native firms: funding bucket (rows) x RAD (cols)."""
    ai = f[f["ai_native_int"] == 1]
    cells = []
    for fb in FUNDING_BUCKET_ORDER:
        row = []
        for rad in AI_RAD_ORDER:
            sel = (ai["funding_bucket"] == fb) & (ai["rad_score"] == rad)
            dead = int((sel & ai["is_dead"]).sum())
            surv = int((sel & ai["is_survivor"]).sum())
            tot = dead + surv
            row.append({"dead": dead, "survivor": surv, "n": tot,
                        "mortality": round(dead / tot * 100, 1) if tot else None})
        cells.append(row)
    return {"funding_order": FUNDING_BUCKET_ORDER, "rad_order": AI_RAD_ORDER, "cells": cells}


def _vertical(f: pd.DataFrame, top_n: int = TOP_CATEGORY_GROUPS) -> list[dict]:
    """AI-native mortality by primary category group, busiest groups first."""
    ai = f[f["ai_native_int"] == 1]
    rows = []
    for grp, sub in ai.groupby("category_group"):
        dead = int(sub["is_dead"].sum())
        surv = int(sub["is_survivor"].sum())
        tot = dead + surv
        if tot >= 1 and grp not in ("", "Unknown"):
            rows.append({"group": grp, "dead": dead, "survivor": surv, "n": tot,
                         "mortality": round(dead / tot * 100, 1)})
    rows.sort(key=lambda r: r["n"], reverse=True)
    return rows[:top_n]


def _temporal(f: pd.DataFrame) -> dict:
    dead = f[f["is_dead"]].copy()
    ts = pd.to_datetime(dead["death_ts"], format="%Y%m%d%H%M%S", errors="coerce")
    dead = dead.assign(_death=ts)
    dead = dead[dead["_death"] >= TEMPORAL_START]
    months_idx = dead["_death"].dt.to_period("M").astype(str)
    total = months_idx.value_counts().to_dict()
    commod = months_idx[dead["subclass_group"] == "Commoditizable AI"].value_counts().to_dict()
    defens = months_idx[dead["subclass_group"] == "Defensible AI"].value_counts().to_dict()
    months = sorted(total)
    return {
        "months": months,
        "total": [int(total.get(m, 0)) for m in months],
        "commoditizable": [int(commod.get(m, 0)) for m in months],
        "defensible": [int(defens.get(m, 0)) for m in months],
        "releases": [{"month": m, "label": lbl} for m, lbl in MODEL_RELEASES],
    }


def _confidence(f: pd.DataFrame) -> dict:
    def stats(mask: pd.Series) -> dict:
        v = pd.to_numeric(f[mask]["conf_classification"], errors="coerce").dropna()
        dist = {int(k): int(c) for k, c in v.value_counts().sort_index().items()}
        return {"n": int(len(v)), "mean": round(float(v.mean()), 2) if len(v) else None,
                "median": float(v.median()) if len(v) else None, "dist": dist}
    return {
        "survivor": stats(f["is_survivor"]),
        "dead": stats(f["is_dead"]),
        "dead_recovered": stats(f["is_dead_recovered"]),
    }


def _hist_bins(values: pd.Series, edges: list[float]) -> dict:
    """Return counts keyed by left-edge label for a fixed bin grid."""
    v = pd.to_numeric(values, errors="coerce").dropna()
    if v.empty:
        return {"n": 0, "mean": None, "median": None, "bins": [], "counts": []}
    labels = []
    counts = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if i == len(edges) - 2:
            mask = (v >= lo) & (v <= hi)
        else:
            mask = (v >= lo) & (v < hi)
        labels.append(f"{lo:.1f}-{hi:.1f}")
        counts.append(int(mask.sum()))
    return {
        "n": int(len(v)),
        "mean": round(float(v.mean()), 4),
        "median": round(float(v.median()), 4),
        "bins": labels,
        "counts": counts,
    }


def _v2_confidence(f: pd.DataFrame) -> dict:
    """Distributions for the three V2 professor confidence fields."""
    edges = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0000001]

    def discrete(mask: pd.Series, col: str) -> dict:
        v = pd.to_numeric(f[mask][col], errors="coerce").dropna()
        dist = {int(k): int(c) for k, c in v.value_counts().sort_index().items()}
        return {
            "n": int(len(v)),
            "mean": round(float(v.mean()), 2) if len(v) else None,
            "median": float(v.median()) if len(v) else None,
            "dist": dist,
        }

    universe = f["is_survivor"] | f["is_dead"]
    by_subclass = {}
    for s in SUBCLASS_ORDER:
        vals = pd.to_numeric(
            f[universe & f["subclass"].eq(s)]["subclass_confidence"],
            errors="coerce",
        ).dropna()
        by_subclass[s] = {
            "mean": round(float(vals.mean()), 2) if len(vals) else 0.0,
            "count": int(len(vals)),
        }

    return {
        "ai_native": {
            "all": _hist_bins(f[universe]["ai_native_confidence"], edges),
            "survivor": _hist_bins(f[f["is_survivor"]]["ai_native_confidence"], edges),
            "dead": _hist_bins(f[f["is_dead"]]["ai_native_confidence"], edges),
        },
        "subclass": {
            "all": discrete(universe, "subclass_confidence"),
            "by_subclass": by_subclass,
        },
        "rad": {
            "all": discrete(universe & f["rad_score"].ne("RAD-NA"), "rad_confidence"),
        },
    }


def _flips(f: pd.DataFrame, paths: Paths, preview: bool) -> dict:
    """Metadata-only (production) vs recovered-evidence (corrected) verdict flips
    on the dead cohort. Only meaningful once the real corrected CSV exists."""
    if preview or not paths.production.exists():
        return {"available": False, "reason": "preview" if preview else "no production csv"}
    before = _read(paths.production).set_index("CompanyID")
    dead = f[f["is_dead_recovered"]]
    # Skip IDs absent from production: reindex NaNs cast to "nan" and look like
    # flips even though there is no before-label to compare.
    dead = dead[dead["CompanyID"].isin(before.index)]
    n = int(len(dead))
    out = {"available": True, "n": n}
    for col in ("ai_native", "subclass", "rad_score"):
        if col not in before.columns:
            continue
        b = before.reindex(dead["CompanyID"])[col].astype(str).values
        a = dead[col].astype(str).values
        changed = int((b != a).sum())
        out[col] = {"changed": changed, "pct": round(changed / n * 100, 1) if n else 0.0}
    return out


def _regression_cis_stable(terms: list[dict], max_log_width: float = 5.0) -> bool:
    """True when every odds-ratio CI is finite, positive, and not explosively wide.

    max_log_width=5 means hi/lo may span at most e^5 ≈ 148x on the OR scale.
    Wider intervals usually mean separation or a sparse cell, not a usable estimate.
    """
    if not terms:
        return False
    for t in terms:
        lo, hi = t["ci_low"], t["ci_high"]
        if not (math.isfinite(lo) and math.isfinite(hi) and lo > 0 and hi > 0):
            return False
        if math.log(hi / lo) > max_log_width:
            return False
    return True


def _regression(f: pd.DataFrame) -> dict:
    """Logistic models. died=1 for the dead cohort, 0 for survivors.

    Model 1 (full outcome frame): does AI-nativeness predict death, net of
    funding, founding era, and vertical. Model 2 (AI-native only): does RAD
    dependency predict death among AI firms (the RAD-validation test).
    Model 3 (AI-native only): Model 2 plus a funding x RAD interaction; the
    builder renders it only if it converged with stable CIs.
    """
    try:
        import statsmodels.formula.api as smf
    except Exception as exc:  # statsmodels missing
        unavailable = {"available": False, "error": f"statsmodels unavailable: {exc}"}
        return {"model1": unavailable, "model2": unavailable, "model3": unavailable}

    base = f[f["is_dead"] | f["is_survivor"]].copy()
    base["died"] = base["is_dead"].astype(int)
    base = base[base["founding_era"] != "unknown"]
    top_cats = base["category_group"].value_counts().head(TOP_CATEGORY_GROUPS).index
    base["cat_grp"] = base["category_group"].where(base["category_group"].isin(top_cats), "Other")

    def fit(frame: pd.DataFrame, formula: str) -> dict:
        frame = frame.dropna(subset=["log_funding"])
        if frame["died"].nunique() < 2 or len(frame) < 30:
            return {"available": False, "error": "insufficient variation"}
        try:
            res = smf.logit(formula, data=frame).fit(disp=0, maxiter=100)
        except Exception as exc:
            return {"available": False, "error": str(exc)}
        ci = res.conf_int()
        terms = []
        for name in res.params.index:
            if name == "Intercept":
                continue
            terms.append({
                "term": name,
                "odds_ratio": round(float(np.exp(res.params[name])), 3),
                "ci_low": round(float(np.exp(ci.loc[name, 0])), 3),
                "ci_high": round(float(np.exp(ci.loc[name, 1])), 3),
                "pvalue": float(res.pvalues[name]),
            })
        return {"available": True, "n": int(res.nobs),
                "pseudo_r2": round(float(res.prsquared), 4), "terms": terms}

    model1 = fit(base, "died ~ ai_native_int + log_funding + C(founding_era) + C(cat_grp)")
    ai = base[base["ai_native_int"] == 1].copy()
    ai = ai[ai["rad_score"].isin(AI_RAD_ORDER)]
    model2 = fit(
        ai,
        "died ~ C(rad_score, Treatment('RAD-L')) + C(subclass_group) + log_funding + C(founding_era)",
    )
    # Attempt-then-fall-back: funding x RAD interaction. Ship only when the fit
    # converges with stable CIs; otherwise the descriptive heatmap stands alone.
    model3 = fit(
        ai,
        "died ~ C(rad_score, Treatment('RAD-L')) * log_funding + C(founding_era)",
    )
    if model3.get("available") and not _regression_cis_stable(model3["terms"]):
        model3 = {"available": False, "error": "unstable confidence intervals"}
    return {"model1": model1, "model2": model2, "model3": model3}


def compute_metrics(df: pd.DataFrame, meta: dict, paths: Paths) -> dict:
    out = {
        "meta": meta,
        "correction": _correction(df),
        "ai_vs_survival": _ai_vs_survival(df),
        "subclass_tests": _subclass_tests(df),
        "rad_survival": _rad_survival(df),
        "dependency_trap": _dependency_trap(df),
        "funding_survival": _funding_by_survival(df),
        "vertical": _vertical(df),
        "temporal": _temporal(df),
        "confidence": _confidence(df),
        "sensitivity": _sensitivity(df),
        "funnel": _funnel(df, paths, meta["preview"]),
        "flips": _flips(df, paths, meta["preview"]),
        "regression": _regression(df),
    }
    if meta.get("classifier_version") == "v2" and "ai_native_confidence" in df.columns:
        out["v2_confidence"] = _v2_confidence(df)
    return out


def analyze(paths: Paths | None = None) -> dict:
    paths = paths or Paths()
    df, meta = load_frame(paths)
    return compute_metrics(df, meta, paths)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corrected", type=Path, default=None)
    parser.add_argument("--production", type=Path, default=None)
    parser.add_argument("--master", type=Path, default=None)
    parser.add_argument("--classifier-input", type=Path, default=None)
    parser.add_argument("--targets-dead", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None, help="Also write the metrics JSON here.")
    args = parser.parse_args()

    p = Paths()
    for attr, val in [("corrected", args.corrected), ("production", args.production),
                      ("master", args.master), ("classifier_input", args.classifier_input),
                      ("targets_dead", args.targets_dead)]:
        if val is not None:
            setattr(p, attr, val)

    metrics = analyze(p)
    m = metrics["meta"]
    mode = "PREVIEW (metadata-only dead verdicts)" if m["preview"] else "evidence-based"
    print(f"[{mode}] total={m['n_total']:,}  universe={m['n_universe']:,}  "
          f"survivor={m['n_survivor']:,}  dead={m['n_dead']:,}  "
          f"dead_strict={m['n_dead_strict']:,}  excluded={m['n_excluded']:,}",
          file=sys.stderr)
    blob = json.dumps(metrics, indent=2, ensure_ascii=False, default=str)
    if args.out:
        args.out.write_text(blob + "\n", encoding="utf-8")
    print(blob)


if __name__ == "__main__":
    main()
