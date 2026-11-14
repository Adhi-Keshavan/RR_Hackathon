"""Phase 2: pre-delivery match state and a transparent resource-pressure score.

Run ``python -m src.pressure_model`` from the project root. The source lacks
exact extras runs for most deliveries, so the 0-100 score deliberately uses
verified pre-ball time, wicket, and knockout state. Exact score/target/RRR
features are populated only where the supplied rows prove them.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import DataValidationError
from src.phase1 import build_phase1_dataset


ROOT = Path(__file__).resolve().parents[1]
OVER_KEY = ["match_id", "innings", "overid"]
INNINGS_KEY = ["match_id", "innings"]


def _legal_ball_clock(deliveries: pd.DataFrame) -> pd.DataFrame:
    """Use only information observable before each ball for the legal clock.

An over transition establishes six completed legal balls, so the clock can
restart at the next over even if an earlier extra was ambiguous. Within the
current over, any prior ambiguous extra leaves the clock unknown. Completed
overs with fewer than six or more than six known/possible legal balls are
flagged from the following over, never retroactively.
    """
    d = deliveries.copy()
    original = d["is_legal_delivery"].astype("boolean")
    known = original.fillna(False).astype("int64")
    unknown = original.isna().astype("int64")
    over = d.assign(_known=known, _unknown=unknown).groupby(OVER_KEY).agg(
        known_legal=("_known", "sum"), unknown=("_unknown", "sum")
    )
    over = over.join(
        d.groupby(INNINGS_KEY).overid.max().rename("last_over"),
        on=INNINGS_KEY,
    )
    over_number = over.index.get_level_values("overid")
    completed = over_number < over.last_over
    over["clock_conflict"] = (
        over.known_legal.gt(6) |
        (completed & over.known_legal.add(over.unknown).lt(6))
    )
    bad_over = (over.loc[over.clock_conflict].reset_index()
                .groupby(INNINGS_KEY).overid.min().rename("first_bad_over"))
    d = d.join(bad_over, on=INNINGS_KEY)
    d["legal_delivery_phase2"] = original
    legal_in_over_before = (known.groupby(
        [d[column] for column in OVER_KEY]
    ).cumsum() - known)
    unknown_in_over_before = (unknown.groupby(
        [d[column] for column in OVER_KEY]
    ).cumsum() - unknown)
    d["legal_clock_conflict"] = (
        (d.first_bad_over.notna() & d.overid.gt(d.first_bad_over)) |
        legal_in_over_before.ge(6)
    )
    clock = (d.overid.sub(1).mul(6) + legal_in_over_before).astype("Int64")
    d["legal_balls_before_phase2"] = clock.mask(
        unknown_in_over_before.gt(0) | d.legal_clock_conflict
    )
    return d.drop(columns=["first_bad_over"])


def _score_and_target(deliveries: pd.DataFrame) -> pd.DataFrame:
    """Compute exact pre-ball score and target only when extras do not intervene."""
    d = deliveries.copy()
    runs = d.runs.astype("int64")
    extras = d.extras_category.ne("none").astype("int64")
    group = [d[column] for column in INNINGS_KEY]
    recorded_before = runs.groupby(group).cumsum() - runs
    extras_before = extras.groupby(group).cumsum() - extras
    d["recorded_batter_runs_before"] = recorded_before
    d["team_score_before"] = recorded_before.astype("Int64").mask(
        extras_before.gt(0)
    )
    d["team_score_exact"] = d.team_score_before.notna()
    minimum_total = runs.copy()
    guaranteed_extra = d.extras_category.isin(
        ["wide", "no_ball", "byes", "legbyes"]
    )
    minimum_total.loc[guaranteed_extra] = runs.loc[guaranteed_extra].clip(lower=1)
    d["minimum_team_score_before"] = (
        minimum_total.groupby(group).cumsum() - minimum_total
    )

    inning = d.groupby(INNINGS_KEY).agg(
        batter_runs=("runs", "sum"), extras_events=("extras_category",
        lambda s: int(s.ne("none").sum())),
    )
    first = inning.xs(1, level="innings")
    first_target = first.batter_runs.add(1).where(first.extras_events.eq(0))
    third_target = pd.Series(dtype="float64")
    if 3 in inning.index.get_level_values("innings"):
        third = inning.xs(3, level="innings")
        third_target = third.batter_runs.add(1).where(third.extras_events.eq(0))
    target = pd.Series(pd.NA, index=d.index, dtype="Int64")
    second = d.innings.eq(2) & ~d.is_revised_match
    fourth = d.innings.eq(4) & ~d.is_revised_match
    target.loc[second] = d.loc[second, "match_id"].map(first_target).astype("Int64")
    target.loc[fourth] = d.loc[fourth, "match_id"].map(third_target).astype("Int64")
    d["target_before"] = target
    required = (target - d.team_score_before).clip(lower=0)
    d["required_runs_before"] = required.astype("Int64")
    d["required_run_rate_before"] = (
        required.mul(6).div(d.balls_remaining_before.replace(0, pd.NA))
    ).astype("Float64")
    return d


def build_pressure_features(deliveries: pd.DataFrame) -> pd.DataFrame:
    """Add Phase 2 features without using the outcome of the current ball."""
    required = {
        "match_id", "innings", "overid", "innings_ball_num", "runs",
        "extras_category", "is_legal_delivery", "wickets_before", "knock_out",
        "first_inn_revised_overs", "second_inn_revised_overs",
        "revised_target",
    }
    missing = required - set(deliveries.columns)
    if missing:
        raise DataValidationError(f"Phase 2 missing columns: {sorted(missing)}")
    d = deliveries.sort_values(
        ["match_id", "innings", "innings_ball_num"], kind="stable"
    ).reset_index(drop=True)
    d["is_revised_match"] = d[[
        "first_inn_revised_overs", "second_inn_revised_overs", "revised_target"
    ]].notna().any(axis=1)
    d["is_super_over"] = d.innings.gt(2)
    d = _legal_ball_clock(d)
    limit = pd.Series(np.where(d.is_super_over, 6, 120), index=d.index,
                      dtype="Int64")
    limit = limit.mask(d.is_revised_match)
    d["innings_ball_limit"] = limit
    remaining = limit - d.legal_balls_before_phase2
    d["balls_remaining_before"] = remaining.mask(
        remaining.lt(0) | remaining.gt(limit)
    ).astype("Int64")
    d = _score_and_target(d)

    stage = d.legal_balls_before_phase2.div(limit).astype("Float64")
    wicket_limit = pd.Series(np.where(d.is_super_over, 2, 10), index=d.index)
    wicket_fraction = d.wickets_before.div(wicket_limit).clip(0, 1)
    importance = pd.to_numeric(d.knock_out, errors="coerce")
    if not importance.isin([0, 1]).all():
        raise DataValidationError("knock_out must be 0 or 1")
    base = (0.40 * stage + 0.35 * wicket_fraction +
            0.20 * stage * wicket_fraction + 0.05 * importance)
    ordinary_score = 100 * base
    super_score = 75 + 25 * base
    score = ordinary_score.where(~d.is_super_over, super_score)
    d["pressure_score"] = score.clip(0, 100).round(2).astype("Float64")
    d["pressure_model"] = "resource_only_v1"
    d["pressure_regime"] = np.where(d.is_super_over, "super_over", "ordinary")
    return d


def phase2_report(features: pd.DataFrame) -> dict:
    """Summarize supported features and unresolved source limitations."""
    score = features.pressure_score.dropna().to_numpy(dtype=float)
    quantiles = [float(x) for x in np.quantile(score, [0, .25, .5, .75, 1])]
    return {
        "status": "PASS_WITH_LIMITED_SCORE_DATA",
        "model": "resource_only_v1",
        "delivery_rows": len(features),
        "pressure_score_rows": int(features.pressure_score.notna().sum()),
        "pressure_score_missing_rows": int(features.pressure_score.isna().sum()),
        "pressure_score_quantiles_min_p25_p50_p75_max": quantiles,
        "revised_match_rows": int(features.is_revised_match.sum()),
        "super_over_rows": int(features.is_super_over.sum()),
        "legal_clock_recovered_rows": int(
            (features.legal_balls_before.isna() &
             features.legal_balls_before_phase2.notna()).sum()
        ),
        "legal_status_unresolved_rows": int(features.legal_delivery_phase2.isna().sum()),
        "clock_conflict_rows": int(features.legal_clock_conflict.sum()),
        "exact_score_before_rows": int(features.team_score_before.notna().sum()),
        "exact_target_before_rows": int(features.target_before.notna().sum()),
        "exact_required_run_rate_rows": int(features.required_run_rate_before.notna().sum()),
        "exact_target_matches": int(features.loc[
            features.target_before.notna(), "match_id"
        ].nunique()),
        "bye_legbye_rows_with_nonzero_runs_field": int((
            features.extras_category.isin(["byes", "legbyes"]) &
            features.runs.gt(0)
        ).sum()),
    }


def _markdown(report: dict) -> str:
    lines = [
        "# Phase 2 pressure feature validation", "",
        f"**{report['status']}**. Model: `{report['model']}`.", "",
        f"Feature file: `{report['feature_file']}`.",
        ("The usual `outputs/pressure_features.csv` was open or protected, so "
         "this run used a new filename." if not report["stable_file_updated"]
         else "The usual feature file was updated successfully."),
        "",
        "| Measure | Rows |", "|---|---:|",
        f"| Delivery rows preserved | {report['delivery_rows']:,} |",
        f"| 0-100 pressure score available | {report['pressure_score_rows']:,} |",
        f"| Pressure score unavailable | {report['pressure_score_missing_rows']:,} |",
        f"| Exact pre-ball team score available | {report['exact_score_before_rows']:,} |",
        f"| Exact target available | {report['exact_target_before_rows']:,} |",
        f"| Exact required run rate available | {report['exact_required_run_rate_rows']:,} |",
        f"| Revised-match rows (excluded from score) | {report['revised_match_rows']:,} |",
        f"| Super-over rows (separate score regime) | {report['super_over_rows']:,} |",
        f"| Pre-ball clocks recovered at later over boundaries | {report['legal_clock_recovered_rows']:,} |",
        f"| Legality still unresolved | {report['legal_status_unresolved_rows']:,} |",
        f"| Rows at/after a contradictory completed over | {report['clock_conflict_rows']:,} |",
        "", "## Definition", "",
        "All inputs are measured before the current delivery. For an ordinary",
        "innings, let `t = legal_balls_before / 120`, `w = wickets_before / 10`,",
        "and `k = knock_out` (0 or 1). The provisional 0-100 score is",
        "`100 * (0.40t + 0.35w + 0.20tw + 0.05k)`. A super over uses a six-ball",
        "and two-wicket clock, then `75 + 25 * base`; it is marked as a separate",
        "regime. An over transition resets the clock using only prior events;",
        "individual `other`/`Penalty` deliveries are never retroactively",
        "classified. Revised matches and contradictory/unknown legal clocks have a",
        "null score. Quantiles (min, 25th, median, 75th, max):",
        f"`{report['pressure_score_quantiles_min_p25_p50_p75_max']}`.",
        "", "## Score and target limitations", "",
        "`team_score_before` is exact only until the first extras event in an",
        "innings. `minimum_team_score_before` is a conservative lower bound",
        "using the larger of the `runs` field and one for a wide, no-ball,",
        "bye, or leg-bye. One leg-bye row has a nonzero `runs` field; this",
        "rule avoids double counting it. The lower bound is not treated",
        "as exact. The target is calculated only for a chase whose previous",
        "innings had no extras and whose match has no revised-innings fields.",
        "A revised target is never applied to balls before an unknown revision",
        "time. Required run rate is populated only if target, score, and",
        "balls remaining are all exact. It is **not** used in this provisional",
        "pressure score because its coverage is too sparse for fair comparison.",
        "", "This score measures resource and stage pressure, not match closeness.",
        "Do not use it for final player rankings until this limitation and the",
        "pressure methodology are reviewed.",
    ]
    return "\n".join(lines) + "\n"


def write_feature_csv(features: pd.DataFrame, columns: list[str],
                      output_dir: Path,
                      stable_name: str = "pressure_features.csv") -> Path:
    """Write once, then replace a stable CSV if Windows allows it.

    An open spreadsheet can prevent replacement of the stable filename. In
    that case the complete newly written CSV remains available under its
    timestamped name instead of losing the run or truncating the old file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stable = output_dir / stable_name
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    fresh = output_dir / f"{stable.stem}_{stamp}_{os.getpid()}{stable.suffix}"
    features[columns].to_csv(fresh, index=False)
    try:
        os.replace(fresh, stable)
    except PermissionError:
        return fresh
    return stable


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        phase1 = build_phase1_dataset(args.root)
        features = build_pressure_features(phase1.deliveries)
        if len(features) != len(phase1.deliveries):
            raise DataValidationError("Phase 2 changed delivery count")
        report = phase2_report(features)
    except (DataValidationError, FileNotFoundError) as exc:
        print(f"PHASE 2 FAIL: {exc}")
        return 1
    output_dir = args.root / "outputs"
    report_dir = args.root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "match_id", "innings", "overid", "over_ball_num", "innings_ball_num",
        "striker_id", "bowler_id", "wickets_before", "is_super_over",
        "is_revised_match", "legal_delivery_phase2",
        "legal_clock_conflict", "legal_balls_before_phase2", "innings_ball_limit",
        "balls_remaining_before", "recorded_batter_runs_before",
        "minimum_team_score_before", "team_score_before", "team_score_exact",
        "target_before", "required_runs_before", "required_run_rate_before",
        "pressure_score", "pressure_model", "pressure_regime",
    ]
    try:
        feature_path = write_feature_csv(features, columns, output_dir)
    except PermissionError as exc:
        print(f"PHASE 2 FAIL: Cannot write a new CSV in {output_dir}: {exc}")
        return 1
    report["feature_file"] = str(feature_path.relative_to(args.root)).replace("\\", "/")
    report["stable_file_updated"] = feature_path.name == "pressure_features.csv"
    (report_dir / "phase2_validation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (report_dir / "phase2_validation.md").write_text(
        _markdown(report), encoding="utf-8"
    )
    print(f"PHASE 2 {report['status']}: {report['pressure_score_rows']:,} "
          f"of {report['delivery_rows']:,} deliveries scored; "
          f"features in {report['feature_file']}; "
          "reports/phase2_validation.md has details")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
