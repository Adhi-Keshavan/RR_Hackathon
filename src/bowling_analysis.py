"""Phase 4: descriptive bowling performance under provisional pressure.

Run ``python -m src.bowling_analysis`` from the project root, or
``python bowling_analysis.py`` from src/. Exact bowler economy is nullable
when the official extras fields do not reveal all runs conceded.
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

from src.analysis_common import select_pressure_windows
from src.data import DataValidationError
from src.phase1 import build_phase1_dataset
from src.pressure_model import build_pressure_features, write_feature_csv


ROOT = Path(__file__).resolve().parents[1]
CONTEXT = ["player_id", "competition_group", "innings"]
COUNT_FIELDS = [
    "bowler_deliveries", "legal_deliveries", "bowler_wickets",
    "batter_runs_allowed", "runs_conceded_floor",
    "known_exact_runs_conceded", "uncertain_concession_deliveries",
]


@dataclass
class BowlingAnalysis:
    by_context: pd.DataFrame
    comparison: pd.DataFrame
    player_summary: pd.DataFrame
    thresholds: pd.DataFrame
    report: dict


def _per_100(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.mul(100).div(denominator.where(denominator.gt(0)))


def _economy(runs: pd.Series, legal_deliveries: pd.Series) -> pd.Series:
    return runs.mul(6).div(legal_deliveries.where(legal_deliveries.gt(0)))


def make_bowling_events(analysis_rows: pd.DataFrame) -> pd.DataFrame:
    """Attribute legal balls, bowler wickets, and cautious run measures."""
    d = analysis_rows
    safe = {"none", "wide", "no_ball", "byes", "legbyes"}
    if not d.extras_category.isin(safe).all():
        raise DataValidationError("Ambiguous extras cannot enter bowling events")
    if d.pressure_score.isna().any() or d.competition_group.isna().any():
        raise DataValidationError("Bowling events need a pressure score and cohort")
    if d.legal_delivery_phase2.isna().any():
        raise DataValidationError("Bowling events need known delivery legality")
    events = d[["match_id", "competition_group", "innings", "high_pressure",
                "bowler_id", "runs", "extras_category", "bowler_wicket",
                "legal_delivery_phase2"]].rename(
                    columns={"bowler_id": "player_id"}
                ).copy()
    events["bowler_deliveries"] = 1
    events["legal_deliveries"] = events.legal_delivery_phase2.astype("int64")
    events["bowler_wickets"] = events.bowler_wicket.astype("int64")
    non_bowler_runs = events.extras_category.isin(["wide", "byes", "legbyes"])
    events["batter_runs_allowed"] = events.runs.where(~non_bowler_runs, 0).astype("int64")
    floor = events.batter_runs_allowed.copy()
    wide = events.extras_category.eq("wide")
    no_ball = events.extras_category.eq("no_ball")
    floor.loc[wide] = 1
    floor.loc[no_ball] = floor.loc[no_ball].clip(lower=1)
    events["runs_conceded_floor"] = floor
    uncertain = wide | no_ball
    events["uncertain_concession_deliveries"] = uncertain.astype("int64")
    events["known_exact_runs_conceded"] = events.batter_runs_allowed.where(
        ~uncertain, 0
    )
    return events.drop(columns=[
        "runs", "extras_category", "bowler_wicket", "legal_delivery_phase2",
    ])


def _aggregate(events: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    return events.groupby(keys + ["high_pressure"], as_index=False).agg(
        **{column: (column, "sum") for column in COUNT_FIELDS},
        matches=("match_id", "nunique"),
    )


def _add_rates(frame: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    legal = frame[f"{prefix}legal_deliveries"]
    wickets = frame[f"{prefix}bowler_wickets"]
    uncertain = frame[f"{prefix}uncertain_concession_deliveries"]
    frame[f"{prefix}exact_runs_conceded"] = frame[
        f"{prefix}known_exact_runs_conceded"
    ].where(uncertain.eq(0))
    frame[f"{prefix}economy_exact"] = _economy(
        frame[f"{prefix}exact_runs_conceded"], legal
    )
    frame[f"{prefix}economy_floor"] = _economy(
        frame[f"{prefix}runs_conceded_floor"], legal
    )
    frame[f"{prefix}wickets_per_100_legal"] = _per_100(wickets, legal)
    frame[f"{prefix}legal_balls_per_wicket"] = legal.div(wickets.where(wickets.gt(0)))
    return frame


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
    for prefix in ("pressure_", "ordinary_"):
        for column in COUNT_FIELDS + ["matches"]:
            paired[f"{prefix}{column}"] = paired[
                f"{prefix}{column}"
            ].fillna(0).astype("int64")
        paired = _add_rates(paired, prefix)
    paired["wicket_rate_delta_per_100_legal"] = (
        paired.pressure_wickets_per_100_legal -
        paired.ordinary_wickets_per_100_legal
    )
    paired["economy_floor_delta"] = (
        paired.pressure_economy_floor - paired.ordinary_economy_floor
    )
    paired["exact_economy_delta"] = (
        paired.pressure_economy_exact - paired.ordinary_economy_exact
    )
    paired["pressure_wickets_above_own_ordinary_rate"] = (
        paired.pressure_bowler_wickets -
        paired.pressure_legal_deliveries * paired.ordinary_bowler_wickets.div(
            paired.ordinary_legal_deliveries.where(
                paired.ordinary_legal_deliveries.gt(0)
            )
        )
    ).where(paired.pressure_legal_deliveries.gt(0))
    return paired


def analyze_bowling(features: pd.DataFrame,
                    player_mapping: pd.DataFrame,
                    pressure_quantile: float = 0.80) -> BowlingAnalysis:
    """Compare high-pressure bowling with each bowler's ordinary windows."""
    required = {"extras_category", "legal_delivery_phase2", "bowler_id",
                "bowler_wicket", "runs"}
    missing = required - set(features.columns)
    if missing:
        raise DataValidationError(f"Phase 4 missing columns: {sorted(missing)}")
    candidate, thresholds, selection = select_pressure_windows(
        features, pressure_quantile
    )
    ambiguous_extra = candidate.extras_category.isin(["other", "penalty"])
    safe = candidate.loc[~ambiguous_extra].copy()
    events = make_bowling_events(safe)
    aggregate = _aggregate(events, CONTEXT)

    names = player_mapping[["player_id", "player_name"]].drop_duplicates()
    if names.duplicated("player_id").any():
        raise DataValidationError("A player ID has multiple canonical names")
    missing_names = set(events.player_id) - set(names.player_id)
    if missing_names:
        raise DataValidationError(f"Bowling player IDs missing from mapping: {len(missing_names)}")
    by_context = aggregate.merge(names, on="player_id", validate="many_to_one")
    by_context = _add_rates(by_context).sort_values(
        ["player_id", "competition_group", "innings", "high_pressure"]
    ).reset_index(drop=True)

    comparison = _paired(aggregate, CONTEXT).merge(
        names, on="player_id", validate="many_to_one"
    ).sort_values(CONTEXT).reset_index(drop=True)
    player_summary = _paired(_aggregate(events, ["player_id"]), ["player_id"])
    matched = comparison.loc[
        comparison.pressure_wickets_above_own_ordinary_rate.notna()
    ].groupby("player_id").agg(
        matched_pressure_legal_deliveries=("pressure_legal_deliveries", "sum"),
        pressure_wickets_above_matched_baseline=(
            "pressure_wickets_above_own_ordinary_rate", "sum"
        ),
    )
    player_summary = player_summary.merge(
        matched, on="player_id", how="left", validate="one_to_one"
    ).merge(names, on="player_id", validate="many_to_one")
    player_summary["matched_pressure_legal_deliveries"] = (
        player_summary.matched_pressure_legal_deliveries.fillna(0).astype("int64")
    )
    player_summary = player_summary.sort_values("player_id").reset_index(drop=True)

    report = {
        "status": "PASS_DESCRIPTIVE_WITH_RUNS_LIMITATION",
        "pressure_model": "resource_only_v1",
        "pressure_quantile": pressure_quantile,
        "source_deliveries": len(features),
        **selection,
        "ambiguous_extra_rows_excluded": int(ambiguous_extra.sum()),
        "analysis_delivery_rows": len(safe),
        "high_pressure_delivery_rows": int(safe.high_pressure.sum()),
        "ordinary_delivery_rows": int((~safe.high_pressure).sum()),
        "pressure_legal_deliveries": int(events.loc[
            events.high_pressure, "legal_deliveries"
        ].sum()),
        "pressure_bowler_wickets": int(events.loc[
            events.high_pressure, "bowler_wickets"
        ].sum()),
        "uncertain_concession_delivery_rows": int(
            events.uncertain_concession_deliveries.sum()
        ),
        "players_with_pressure_legal_deliveries": int(
            player_summary.pressure_legal_deliveries.gt(0).sum()
        ),
        "players_with_both_windows": int((
            player_summary.pressure_legal_deliveries.gt(0) &
            player_summary.ordinary_legal_deliveries.gt(0)
        ).sum()),
        "players_with_exact_pressure_economy": int(
            player_summary.pressure_economy_exact.notna().sum()
        ),
        "players_with_exact_economy_in_both_windows": int(
            player_summary.exact_economy_delta.notna().sum()
        ),
        "thresholds": thresholds.to_dict("records"),
    }
    return BowlingAnalysis(by_context, comparison, player_summary,
                           thresholds, report)


def _markdown(report: dict) -> str:
    lines = [
        "# Phase 4 bowling analysis validation", "",
        f"**{report['status']}**. These are descriptive comparisons, not player rankings.",
        "", "| Measure | Count |", "|---|---:|",
        f"| Official delivery rows | {report['source_deliveries']:,} |",
        f"| Eligible ordinary IPL/T20I rows analysed | {report['analysis_delivery_rows']:,} |",
        f"| High-pressure delivery rows | {report['high_pressure_delivery_rows']:,} |",
        f"| High-pressure legal deliveries | {report['pressure_legal_deliveries']:,} |",
        f"| High-pressure bowler-creditable wickets | {report['pressure_bowler_wickets']:,} |",
        f"| Wide/no-ball rows with uncertain runs conceded | {report['uncertain_concession_delivery_rows']:,} |",
        f"| Players with pressure legal deliveries | {report['players_with_pressure_legal_deliveries']:,} |",
        f"| Players with both pressure and ordinary legal deliveries | {report['players_with_both_windows']:,} |",
        f"| Players with exact pressure economy | {report['players_with_exact_pressure_economy']:,} |",
        f"| Players with exact economy in both windows | {report['players_with_exact_economy_in_both_windows']:,} |",
        "", "## Pressure split", "",
        "Phase 4 uses the exact same upper-20% cutoffs by IPL/T20I and innings",
        "as Phase 3. Cutoffs depend on pre-ball match state, not bowling outcomes.",
        "", "| Group | Innings | 80th-percentile cutoff |",
        "|---|---:|---:|",
    ]
    for row in report["thresholds"]:
        lines.append(f"| {row['competition_group']} | {row['innings']} | {row['pressure_cutoff']:.2f} |")
    lines += [
        "", "## Bowling definitions", "",
        "- Legal deliveries exclude wides and no-balls. Stumpings on wides",
        "  still count as bowler wickets. Run-outs, retire-outs, and obstruction",
        "  do not count as bowler wickets.",
        "- Ordinary-ball `runs` are charged to the bowler. Byes and leg-byes",
        "  are not. A wide or no-ball contributes at least one bowler run, but",
        "  the exact extras amount is absent from the supplied fields; exact",
        "  runs conceded and economy are null for any player/window with one.",
        "- `runs_conceded_floor` is a conservative lower bound. `economy_floor`",
        "  is six times that floor divided by legal deliveries. `economy_exact`",
        "  is populated only when every delivery in that player/window has",
        "  known bowler-conceded runs. Neither floor nor batter-runs-allowed",
        "  should be presented as exact economy.",
        "- Wicket rate is bowler wickets per 100 legal deliveries. Player",
        "  baselines compare pressure with ordinary deliveries in the same",
        "  competition group and innings. Wickets above own ordinary rate are",
        "  summed only where both windows have legal-ball exposure. These",
        "  comparisons have no sample-size shrinkage or opponent adjustment.",
        "", "## Exclusions and limits", "",
        f"- {report['non_ipl_t20_rows_excluded']:,} non-IPL T20 rows are outside the stated IPL/T20I cohort.",
        f"- {report['super_over_rows_excluded']:,} super-over rows are excluded from ordinary-innings comparisons.",
        f"- {report['unscored_core_ordinary_rows_excluded']:,} core ordinary rows have no trustworthy Phase 2 pressure score.",
        f"- {report['ambiguous_extra_rows_excluded']:,} ambiguous `other`/`Penalty` rows are excluded from bowling attribution.",
        "", "No eligibility threshold, uncertainty interval, or final ranking",
        "is applied in this phase.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        phase1 = build_phase1_dataset(args.root)
        features = build_pressure_features(phase1.deliveries)
        analysis = analyze_bowling(
            features, phase1.tables["competition_team_player_mapping"]
        )
    except (DataValidationError, FileNotFoundError, ValueError) as exc:
        print(f"PHASE 4 FAIL: {exc}")
        return 1
    output_dir = args.root / "outputs"
    report_dir = args.root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    produced = {}
    for label, frame in (
        ("bowling_by_context", analysis.by_context),
        ("bowling_pressure_vs_baseline", analysis.comparison),
        ("bowling_player_summary", analysis.player_summary),
    ):
        try:
            path = write_feature_csv(
                frame, list(frame.columns), output_dir, f"{label}.csv"
            )
        except PermissionError as exc:
            print(f"PHASE 4 FAIL: Cannot write {label} in {output_dir}: {exc}")
            return 1
        produced[label] = str(path.relative_to(args.root)).replace("\\", "/")
    analysis.report["files"] = produced
    (report_dir / "phase4_validation.json").write_text(
        json.dumps(analysis.report, indent=2), encoding="utf-8"
    )
    (report_dir / "phase4_validation.md").write_text(
        _markdown(analysis.report), encoding="utf-8"
    )
    print(f"PHASE 4 {analysis.report['status']}: "
          f"{analysis.report['analysis_delivery_rows']:,} deliveries analysed; "
          "reports/phase4_validation.md has details")
    for path in produced.values():
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
