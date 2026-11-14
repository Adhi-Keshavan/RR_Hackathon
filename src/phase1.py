"""Complete, reproducible Phase 1 checks for the official data pack.

Run ``python -m src.phase1`` from the project root. A nonzero exit means a
structural validation failed. Reported source anomalies remain visible and are
never silently deleted from delivery history.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):
    # Allow `python phase1.py` from src/ as well as `python -m src.phase1`.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import DataValidationError, attach_match_context, load_official_data
from src.preprocessing import normalize_playing_eleven, prepare_deliveries


ROOT = Path(__file__).resolve().parents[1]
ROLES = {
    "striker": ("batting_team_id", "striker_id", "striker_name"),
    "non_striker": ("batting_team_id", "non_striker_id", "non_striker_name"),
    "bowler": ("bowling_team_id", "bowler_id", "bowler_name"),
}


@dataclass
class Phase1Data:
    """Validated tables, with one prepared row per original delivery."""

    deliveries: pd.DataFrame
    playing_eleven: pd.DataFrame
    tables: dict[str, pd.DataFrame]
    report: dict


def validate_innings(deliveries: pd.DataFrame) -> dict:
    """Verify structural innings sequence, teams, overs, and wicket limits."""
    d = deliveries.sort_values(
        ["match_id", "innings", "innings_ball_num"], kind="stable"
    )
    if not d["innings"].isin([1, 2, 3, 4]).all():
        raise DataValidationError("Unexpected innings number")
    groups = d.groupby(["match_id", "innings"], sort=False)
    summary = groups.agg(
        rows=("innings_ball_num", "size"),
        first_ball=("innings_ball_num", "min"),
        last_ball=("innings_ball_num", "max"),
        batting_teams=("batting_team_id", "nunique"),
        bowling_teams=("bowling_team_id", "nunique"),
        wickets=("is_wicket", "sum"),
        first_over=("overid", "min"),
        last_over=("overid", "max"),
        distinct_overs=("overid", "nunique"),
    )
    if (summary.first_ball.ne(1) |
            summary.last_ball.sub(summary.first_ball).add(1).ne(summary.rows)).any():
        raise DataValidationError("Innings delivery numbering has gaps or does not start at 1")
    if (summary.batting_teams.ne(1) | summary.bowling_teams.ne(1)).any():
        raise DataValidationError("Batting or bowling team changes within an innings")
    if (summary.first_over.ne(1) |
            summary.last_over.sub(summary.first_over).add(1).ne(summary.distinct_overs)).any():
        raise DataValidationError("Over numbering has gaps or does not start at 1")
    if d.groupby(["match_id", "innings"]).overid.diff().lt(0).any():
        raise DataValidationError("Over numbers go backwards within an innings")
    over = d.groupby(["match_id", "innings", "overid"]).over_ball_num.agg(
        rows="size", first="min", last="max"
    )
    if (over["first"].ne(1) | over["last"].ne(over.rows)).any():
        raise DataValidationError("Delivery numbering within an over is inconsistent")
    patterns = groups.size().reset_index().groupby("match_id").innings.apply(tuple)
    allowed = {(1,), (1, 2), (1, 2, 3, 4)}
    if not patterns.isin(allowed).all():
        raise DataValidationError("Unexpected ordinary/super-over innings pattern")
    ordinary = summary.index.get_level_values("innings") <= 2
    if summary.loc[ordinary, "last_over"].gt(20).any():
        raise DataValidationError("Ordinary innings exceeds 20 overs")
    if summary.loc[~ordinary, "last_over"].gt(1).any():
        raise DataValidationError("Super-over innings spans more than one over")
    if summary.loc[ordinary, "wickets"].gt(10).any():
        raise DataValidationError("More than 10 wickets in an ordinary innings")
    if summary.loc[~ordinary, "wickets"].gt(2).any():
        raise DataValidationError("More than 2 wickets in a super over")
    return {
        "innings_groups": int(len(summary)),
        "one_innings_matches": int(patterns.map(lambda p: p == (1,)).sum()),
        "two_innings_matches": int(patterns.map(lambda p: p == (1, 2)).sum()),
        "super_over_matches": int(patterns.map(lambda p: p == (1, 2, 3, 4)).sum()),
        "maximum_over_ball_num": int(d.over_ball_num.max()),
    }


def _checked_merge(left: pd.DataFrame, right: pd.DataFrame, on: list[str],
                   label: str) -> pd.DataFrame:
    merged = left.merge(right, on=on, how="left", validate="many_to_one",
                        indicator=True)
    if not merged._merge.eq("both").all():
        raise DataValidationError(f"Missing {label} foreign-key match")
    return merged.drop(columns="_merge")


def validate_relationships(tables: dict[str, pd.DataFrame],
                           deliveries: pd.DataFrame,
                           roster: pd.DataFrame) -> dict:
    """Check all dimension keys, player IDs/names, and optional XI coverage."""
    matches = tables["matches"]
    competitions = tables["competitions"]
    teams = tables["teams"]
    mapping = tables["competition_team_player_mapping"]

    _checked_merge(matches[["comp_id"]], competitions[["comp_id"]],
                   ["comp_id"], "competition")
    _checked_merge(teams[["comp_id"]], competitions[["comp_id"]],
                   ["comp_id"], "team competition")
    for side in ("home", "away"):
        matched = _checked_merge(
            matches[["comp_id", f"{side}_team_id", f"{side}_team"]].rename(
                columns={f"{side}_team_id": "team_id"}),
            teams[["comp_id", "team_id", "team_name"]],
            ["comp_id", "team_id"], f"{side} team",
        )
        if not matched[f"{side}_team"].eq(matched.team_name).all():
            raise DataValidationError(f"{side} team names disagree with teams.csv")

    named_mapping = _checked_merge(
        mapping, competitions[["comp_id", "comp_name"]].rename(
            columns={"comp_name": "competition_name"}),
        ["comp_id"], "player competition",
    )
    if not named_mapping.comp_name.eq(named_mapping.competition_name).all():
        raise DataValidationError("Competition names disagree in player mapping")
    named_mapping = _checked_merge(
        named_mapping, teams[["comp_id", "team_id", "team_name"]].rename(
            columns={"team_name": "canonical_team_name"}),
        ["comp_id", "team_id"], "player team",
    )
    if not named_mapping.team_name.eq(named_mapping.canonical_team_name).all():
        raise DataValidationError("Team names disagree in player mapping")

    roster_match = _checked_merge(
        roster, matches[["match_id", "comp_id", "home_team_id", "away_team_id"]],
        ["match_id"], "playing-XI match",
    )
    valid_roster_team = (roster_match.team_id.eq(roster_match.home_team_id) |
                         roster_match.team_id.eq(roster_match.away_team_id))
    if not valid_roster_team.all():
        raise DataValidationError("Playing-XI team is not in its match")
    roster_map = roster_match.merge(
        mapping[["comp_id", "team_id", "player_id"]],
        on=["comp_id", "team_id", "player_id"], how="left",
        validate="many_to_one", indicator=True,
    )
    roster_missing_mapping = int(roster_map._merge.ne("both").sum())
    if roster_map.loc[roster_map._merge.ne("both"), "is_played"].any():
        raise DataValidationError("An active playing-XI player is absent from mapping")

    role_coverage = {}
    for role, (team_col, player_col, name_col) in ROLES.items():
        role_frame = deliveries[["match_id", "comp_id", team_col, player_col,
                                 name_col]].rename(columns={
            team_col: "team_id", player_col: "player_id", name_col: "delivery_name",
        })
        role_map = _checked_merge(
            role_frame, mapping[["comp_id", "team_id", "player_id", "player_name"]],
            ["comp_id", "team_id", "player_id"], f"{role} player",
        )
        if not role_map.delivery_name.eq(role_map.player_name).all():
            raise DataValidationError(f"{role} name disagrees with player ID mapping")
        xi = role_frame.merge(
            roster[["match_id", "team_id", "player_id", "is_played"]],
            on=["match_id", "team_id", "player_id"], how="left",
            validate="many_to_one", indicator=True,
        )
        if xi.loc[xi._merge.eq("both"), "is_played"].eq(False).any():
            raise DataValidationError(f"A {role} is marked as not played")
        role_coverage[role] = {
            "mapping_missing_rows": 0,
            "playing_eleven_missing_rows": int(xi._merge.ne("both").sum()),
        }
    wicket = deliveries["is_wicket"]
    dismissed = deliveries["dismissed_player_id"]
    invalid_dismissed = (wicket & dismissed.notna() &
                         ~dismissed.eq(deliveries.striker_id) &
                         ~dismissed.eq(deliveries.non_striker_id))
    if invalid_dismissed.any():
        raise DataValidationError("Dismissed player ID is neither batter at the crease")
    return {
        "role_coverage": role_coverage,
        "roster_rows_missing_player_mapping": roster_missing_mapping,
        "same_batter_at_both_ends": int(deliveries.same_batter_at_both_ends.sum()),
        "wickets_missing_dismissed_id": int(deliveries.wicket_missing_dismissed_id.sum()),
    }


def build_phase1_dataset(root: Path | str | None = None) -> Phase1Data:
    """Build the Phase 1 analytical view, with validated joins and audit flags."""
    tables = load_official_data(root)
    exact_duplicates = {
        name: int(frame.duplicated().sum()) for name, frame in tables.items()
    }
    # Current official files have no exact duplicate rows. Keep this explicit:
    # dropping any future duplicate before reviewing it could erase a delivery.
    if any(exact_duplicates.values()):
        raise DataValidationError(f"Exact duplicate source rows need review: {exact_duplicates}")
    roster = normalize_playing_eleven(tables["playing_eleven"])
    deliveries = attach_match_context(
        prepare_deliveries(tables["ball_by_ball"]),
        tables["matches"], tables["competitions"],
    )
    innings = validate_innings(deliveries)
    relationships = validate_relationships(tables, deliveries, roster)
    if len(deliveries) != len(tables["ball_by_ball"]):
        raise DataValidationError("Phase 1 changed the number of deliveries")
    report = {
        "status": "PASS",
        "source_rows": {name: len(frame) for name, frame in tables.items()},
        "exact_duplicate_rows": exact_duplicates,
        "deliveries_after_join": len(deliveries),
        "playing_eleven_rows_after_key_normalization": len(roster),
        "playing_eleven_repeated_key_rows_removed":
            len(tables["playing_eleven"]) - len(roster),
        "batting_order_conflict_keys": int(roster.batting_order_conflict.sum()),
        "innings": innings,
        "relationships": relationships,
        "matches_without_deliveries": int(
            (~tables["matches"].match_id.isin(deliveries.match_id)).sum()
        ),
        "legal_delivery_classification_unknown_rows":
            int(deliveries.is_legal_delivery.isna().sum()),
        "legal_balls_before_unknown_rows":
            int(deliveries.legal_balls_before.isna().sum()),
    }
    warnings_present = (
        report["playing_eleven_repeated_key_rows_removed"]
        + report["matches_without_deliveries"]
        + report["legal_delivery_classification_unknown_rows"]
        + relationships["roster_rows_missing_player_mapping"]
        + relationships["same_batter_at_both_ends"]
        + relationships["wickets_missing_dismissed_id"]
        + sum(x["playing_eleven_missing_rows"] for x in relationships["role_coverage"].values())
    )
    if warnings_present:
        report["status"] = "PASS_WITH_SOURCE_WARNINGS"
    return Phase1Data(deliveries, roster, tables, report)


def _markdown(report: dict) -> str:
    rows = [
        "# Phase 1 validation", "",
        f"**{report['status']}**. Generated by `python -m src.phase1`.", "",
        "All six CSVs loaded, validated, and joined with one row per delivery.",
        "The checked match/competition/team/player foreign keys and innings",
        "sequences passed. No exact source rows were duplicated, so none were",
        "removed. Repeated playing-XI keys were normalized with conflicts flagged.",
        "", "| Check | Result |", "|---|---:|",
        f"| Deliveries after match and competition joins | {report['deliveries_after_join']:,} |",
        f"| Exact duplicate rows across all CSVs | {sum(report['exact_duplicate_rows'].values())} |",
        f"| Playing-XI repeated-key rows collapsed | {report['playing_eleven_repeated_key_rows_removed']} |",
        f"| Conflicting batting-order keys | {report['batting_order_conflict_keys']} |",
        f"| Validated innings groups | {report['innings']['innings_groups']:,} |",
        f"| Matches with super-over innings | {report['innings']['super_over_matches']} |",
        "", "## Source warnings retained for Phase 2", "",
        f"- {report['matches_without_deliveries']} match metadata rows have no delivery records.",
        f"- {report['relationships']['same_batter_at_both_ends']} delivery lists the same batter at both ends; the row is flagged, not deleted.",
        f"- {report['relationships']['wickets_missing_dismissed_id']} wicket lacks a dismissed-player ID; its dismissal is a retirement and is not bowler-credited.",
        f"- {report['relationships']['roster_rows_missing_player_mapping']} normalized, unplayed roster rows lack a competition-player mapping.",
        f"- {report['legal_delivery_classification_unknown_rows']} `other`/`Penalty` deliveries have unresolved legal-ball status, leaving {report['legal_balls_before_unknown_rows']:,} later pre-ball legal counts unknown.",
    ]
    for role, coverage in report["relationships"]["role_coverage"].items():
        rows.append(f"- {role}: {coverage['playing_eleven_missing_rows']} delivery rows are absent from the playing XI, but all map to valid player IDs and names.")
    rows += [
        "", "The extras fields do not yet support exact innings totals. This is a",
        "Phase 2 methodology issue; no required run rate or pressure score is",
        "calculated in this phase. See `reports/data_audit.md` for every column,",
        "missing value, and competition.",
    ]
    return "\n".join(rows) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        report = build_phase1_dataset(args.root).report
    except (DataValidationError, FileNotFoundError) as exc:
        print(f"PHASE 1 FAIL: {exc}")
        return 1
    report_dir = args.root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "phase1_validation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (report_dir / "phase1_validation.md").write_text(
        _markdown(report), encoding="utf-8"
    )
    print(f"PHASE 1 {report['status']}: {report['deliveries_after_join']:,} "
          "deliveries preserved; reports/phase1_validation.md has details")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
