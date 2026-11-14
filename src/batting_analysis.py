"""Phase 3: descriptive batting performance under provisional pressure.

Run ``python -m src.batting_analysis`` from the project root, or
``python batting_analysis.py`` from src/. This phase creates player and context
tables, not final rankings. Phase 5 will add reliability safeguards.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import DataValidationError
from src.analysis_common import select_pressure_windows
from src.phase1 import build_phase1_dataset
from src.pressure_model import build_pressure_features, write_feature_csv


ROOT = Path(__file__).resolve().parents[1]
CONTEXT = ["player_id", "competition_group", "innings"]
COUNT_FIELDS = ["batting_runs", "balls_faced", "legal_balls_faced",
                "striker_deliveries", "dismissals"]


@dataclass
class BattingAnalysis:
    by_context: pd.DataFrame
    comparison: pd.DataFrame
    player_summary: pd.DataFrame
    thresholds: pd.DataFrame
    report: dict


def _rate(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.mul(100).div(denominator.where(denominator.gt(0)))


def make_batting_events(analysis_rows: pd.DataFrame) -> pd.DataFrame:
    """Make one striker event plus a separate event for each known dismissal."""
    d = analysis_rows
    safe = {"none", "wide", "no_ball", "byes", "legbyes"}
    if not d.extras_category.isin(safe).all():
        raise DataValidationError("Ambiguous extras cannot enter batting events")
    if d.pressure_score.isna().any() or d.competition_group.isna().any():
        raise DataValidationError("Batting events need a pressure score and cohort")
    striker = d[["match_id", "competition_group", "innings", "high_pressure",
                 "striker_id", "runs", "extras_category",
                 "legal_delivery_phase2"]].rename(columns={"striker_id": "player_id"}).copy()
    striker["batting_runs"] = striker.runs.where(
        ~striker.extras_category.isin(["byes", "legbyes", "wide"]), 0
    ).astype("int64")
    striker["balls_faced"] = striker.extras_category.ne("wide").astype("int64")
    striker["legal_balls_faced"] = striker.legal_delivery_phase2.eq(True).fillna(False).astype("int64")
    striker["striker_deliveries"] = 1
    striker["dismissals"] = 0
    striker = striker.drop(columns=["runs", "extras_category", "legal_delivery_phase2"])

    dismissal = d.loc[
        d.is_wicket & d.dismissed_player_id.notna(),
        ["match_id", "competition_group", "innings", "high_pressure",
         "dismissed_player_id"],
    ].rename(columns={"dismissed_player_id": "player_id"}).copy()
    dismissal["player_id"] = dismissal.player_id.astype("int64")
    for column in COUNT_FIELDS:
        dismissal[column] = 1 if column == "dismissals" else 0
    events = pd.concat([striker, dismissal], ignore_index=True)
    events["player_id"] = events.player_id.astype("int64")
    return events


def _aggregate(events: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    return events.groupby(keys + ["high_pressure"], as_index=False).agg(
        **{column: (column, "sum") for column in COUNT_FIELDS},
        matches=("match_id", "nunique"),
    )


def _paired(aggregate: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    pressure = aggregate.loc[aggregate.high_pressure].drop(columns="high_pressure")
    ordinary = aggregate.loc[~aggregate.high_pressure].drop(columns="high_pressure")
    pressure = pressure.rename(columns={
        col: f"pressure_{col}" for col in pressure.columns if col not in keys
    })
    ordinary = ordinary.rename(columns={
        col: f"ordinary_{col}" for col in ordinary.columns if col not in keys
    })
    paired = pressure.merge(ordinary, on=keys, how="outer", validate="one_to_one")
    for prefix in ("pressure", "ordinary"):
        for column in COUNT_FIELDS + ["matches"]:
            paired[f"{prefix}_{column}"] = paired[
                f"{prefix}_{column}"
            ].fillna(0).astype("int64")
        paired[f"{prefix}_strike_rate"] = _rate(
            paired[f"{prefix}_batting_runs"], paired[f"{prefix}_balls_faced"]
        )
        paired[f"{prefix}_dismissals_per_100_balls"] = _rate(
            paired[f"{prefix}_dismissals"], paired[f"{prefix}_balls_faced"]
        )
    paired["strike_rate_delta"] = (
        paired.pressure_strike_rate - paired.ordinary_strike_rate
    )
    paired["dismissal_rate_delta_per_100_balls"] = (
        paired.pressure_dismissals_per_100_balls -
        paired.ordinary_dismissals_per_100_balls
    )
    paired["pressure_runs_above_own_ordinary_rate"] = (
        paired.pressure_batting_runs -
        paired.pressure_balls_faced *
        paired.ordinary_batting_runs.div(
            paired.ordinary_balls_faced.where(paired.ordinary_balls_faced.gt(0))
        )
    ).where(paired.pressure_balls_faced.gt(0))
    return paired


def analyze_batting(features: pd.DataFrame,
                    player_mapping: pd.DataFrame,
                    pressure_quantile: float = 0.80) -> BattingAnalysis:
    """Compare high-pressure batting with each player's ordinary windows."""
    required = {
        "extras_category", "legal_delivery_phase2",
        "striker_id", "runs", "is_wicket", "dismissed_player_id",
        "same_batter_at_both_ends",
    }
    missing = required - set(features.columns)
    if missing:
        raise DataValidationError(f"Phase 3 missing columns: {sorted(missing)}")
    candidate, thresholds, selection = select_pressure_windows(
        features, pressure_quantile
    )
    ambiguous_extra = candidate.extras_category.isin(["other", "penalty"])
    invalid_identity = candidate.same_batter_at_both_ends
    safe = candidate.loc[~ambiguous_extra & ~invalid_identity].copy()
    events = make_batting_events(safe)
    aggregate = _aggregate(events, CONTEXT)

    names = player_mapping[["player_id", "player_name"]].drop_duplicates()
    if names.duplicated("player_id").any():
        raise DataValidationError("A player ID has multiple canonical names")
    missing_names = set(events.player_id) - set(names.player_id)
    if missing_names:
        raise DataValidationError(f"Batting player IDs missing from mapping: {len(missing_names)}")
    by_context = aggregate.merge(names, on="player_id", validate="many_to_one")
    by_context["strike_rate"] = _rate(by_context.batting_runs, by_context.balls_faced)
    by_context["dismissals_per_100_balls"] = _rate(
        by_context.dismissals, by_context.balls_faced
    )
    by_context = by_context.sort_values(
        ["player_id", "competition_group", "innings", "high_pressure"]
    ).reset_index(drop=True)

    comparison = _paired(aggregate, CONTEXT)
    comparison = comparison.merge(names, on="player_id", validate="many_to_one")
    comparison = comparison.sort_values(CONTEXT).reset_index(drop=True)
    player_aggregate = _aggregate(events, ["player_id"])
    player_summary = _paired(player_aggregate, ["player_id"])
    matched = comparison.loc[
        comparison.pressure_runs_above_own_ordinary_rate.notna()
    ].groupby("player_id").agg(
        matched_pressure_balls=("pressure_balls_faced", "sum"),
        pressure_runs_above_matched_baseline=(
            "pressure_runs_above_own_ordinary_rate", "sum"
        ),
    )
    player_summary = player_summary.merge(
        matched, on="player_id", how="left", validate="one_to_one"
    ).merge(names, on="player_id", validate="many_to_one")
    player_summary["matched_pressure_balls"] = (
        player_summary.matched_pressure_balls.fillna(0).astype("int64")
    )
    player_summary = player_summary.sort_values("player_id").reset_index(drop=True)

    dismissal_rows = safe.is_wicket & safe.dismissed_player_id.notna()
    dismissed_id = pd.to_numeric(
        safe.dismissed_player_id, errors="coerce"
    ).astype("Int64")
    non_striker_dismissals = dismissal_rows & dismissed_id.eq(
        safe.non_striker_id.astype("Int64")
    ).fillna(False)
    report = {
        "status": "PASS_DESCRIPTIVE_ONLY",
        "pressure_model": "resource_only_v1",
        "pressure_quantile": pressure_quantile,
        "source_deliveries": len(features),
        **selection,
        "ambiguous_extra_rows_excluded": int(ambiguous_extra.sum()),
        "same_batter_identity_rows_excluded": int(invalid_identity.sum()),
        "analysis_delivery_rows": len(safe),
        "high_pressure_delivery_rows": int(safe.high_pressure.sum()),
        "ordinary_delivery_rows": int((~safe.high_pressure).sum()),
        "pressure_balls_faced": int(events.loc[events.high_pressure, "balls_faced"].sum()),
        "players_with_pressure_balls": int(player_summary.pressure_balls_faced.gt(0).sum()),
        "players_with_both_windows": int((
            player_summary.pressure_balls_faced.gt(0) &
            player_summary.ordinary_balls_faced.gt(0)
        ).sum()),
        "known_dismissals": int(dismissal_rows.sum()),
        "non_striker_dismissals_credited": int(non_striker_dismissals.sum()),
        "wickets_missing_dismissed_player_id": int((
            safe.is_wicket & safe.dismissed_player_id.isna()
        ).sum()),
        "thresholds": thresholds.to_dict("records"),
    }
    return BattingAnalysis(by_context, comparison, player_summary,
                           thresholds, report)


def _markdown(report: dict) -> str:
    lines = [
        "# Phase 3 batting analysis validation", "",
        f"**{report['status']}**. These are descriptive comparisons, not player rankings.",
        "", "| Measure | Count |", "|---|---:|",
        f"| Official delivery rows | {report['source_deliveries']:,} |",
        f"| Eligible ordinary IPL/T20I rows analysed | {report['analysis_delivery_rows']:,} |",
        f"| High-pressure delivery rows | {report['high_pressure_delivery_rows']:,} |",
        f"| Ordinary delivery rows | {report['ordinary_delivery_rows']:,} |",
        f"| High-pressure batting balls faced | {report['pressure_balls_faced']:,} |",
        f"| Players with pressure balls | {report['players_with_pressure_balls']:,} |",
        f"| Players with pressure and ordinary balls | {report['players_with_both_windows']:,} |",
        f"| Known player dismissals credited | {report['known_dismissals']:,} |",
        f"| Non-striker dismissals credited | {report['non_striker_dismissals_credited']:,} |",
        "", "## Pressure split", "",
        "The high-pressure group is the upper 20% of Phase 2 scores within each",
        "competition group (IPL or T20I) and innings (first or second). The",
        "cutoffs depend only on pre-ball match state, not batting outcomes.",
        "They are provisional until the pressure methodology is reviewed.",
        "", "| Group | Innings | 80th-percentile cutoff |",
        "|---|---:|---:|",
    ]
    for row in report["thresholds"]:
        lines.append(f"| {row['competition_group']} | {row['innings']} | {row['pressure_cutoff']:.2f} |")
    lines += [
        "", "## Metric definitions", "",
        "- Batting runs are the recorded `runs` field on ordinary and no-ball",
        "  deliveries. Wides, byes, and leg-byes add zero batter runs. Ambiguous",
        "  `other` and `Penalty` rows are excluded.",
        "- Conventional balls faced exclude wides but include no-balls, byes,",
        "  and leg-byes. `legal_balls_faced` is shown separately and excludes",
        "  both wides and no-balls. Strike rate = 100 x runs / balls faced.",
        "- Dismissals are credited to `dismissed_player_id`, including",
        "  non-striker run-outs. A wicket with no dismissed-player ID cannot",
        "  be assigned to a batter.",
        "- The ordinary group is each player's lower-pressure baseline.",
        "  Strike-rate and dismissal-rate deltas compare that player in the",
        "  same competition group and innings. Player summaries also show",
        "  a context-matched runs-above-baseline total where both windows exist.",
        "  These are descriptive and do not adjust for opponent quality or",
        "  sample-size uncertainty; those belong to Phase 5.",
        "", "## Exclusions and limits", "",
        f"- {report['non_ipl_t20_rows_excluded']:,} non-IPL T20 rows are outside the stated IPL/T20I cohort.",
        f"- {report['super_over_rows_excluded']:,} super-over rows use a separate Phase 2 regime and are excluded here.",
        f"- {report['unscored_core_ordinary_rows_excluded']:,} core ordinary rows have no trustworthy Phase 2 pressure score.",
        f"- {report['ambiguous_extra_rows_excluded']:,} ambiguous extras rows and {report['same_batter_identity_rows_excluded']} same-batter identity rows were excluded from batting attribution.",
        f"- {report['wickets_missing_dismissed_player_id']} analysed wickets lack a dismissed-player ID.",
        "", "No eligibility threshold, shrinkage, uncertainty interval,",
        "or final ranking is applied in this phase. Bowling is reported",
        "separately in `reports/phase4_validation.md`.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        phase1 = build_phase1_dataset(args.root)
        features = build_pressure_features(phase1.deliveries)
        analysis = analyze_batting(
            features, phase1.tables["competition_team_player_mapping"]
        )
    except (DataValidationError, FileNotFoundError, ValueError) as exc:
        print(f"PHASE 3 FAIL: {exc}")
        return 1
    output_dir = args.root / "outputs"
    report_dir = args.root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    produced = {}
    for label, frame in (
        ("batting_by_context", analysis.by_context),
        ("batting_pressure_vs_baseline", analysis.comparison),
        ("batting_player_summary", analysis.player_summary),
    ):
        try:
            path = write_feature_csv(
                frame, list(frame.columns), output_dir, f"{label}.csv"
            )
        except PermissionError as exc:
            print(f"PHASE 3 FAIL: Cannot write {label} in {output_dir}: {exc}")
            return 1
        produced[label] = str(path.relative_to(args.root)).replace("\\", "/")
    analysis.report["files"] = produced
    (report_dir / "phase3_validation.json").write_text(
        json.dumps(analysis.report, indent=2), encoding="utf-8"
    )
    (report_dir / "phase3_validation.md").write_text(
        _markdown(analysis.report), encoding="utf-8"
    )
    print(f"PHASE 3 {analysis.report['status']}: "
          f"{analysis.report['analysis_delivery_rows']:,} deliveries analysed; "
          "reports/phase3_validation.md has details")
    for path in produced.values():
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
