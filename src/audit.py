"""Read-only inventory of the supplied cricket CSV files.

Run with ``python -m src.audit`` from the project root. The JSON output is a
local, reproducible companion to the human-readable audit in reports/.
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path

import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "ball_by_ball": ROOT / "data" / "ball_by_ball.csv",
    "matches": ROOT / "data" / "matches.csv",
    "playing_eleven": ROOT / "6af3166e7b922_data_pack" / "playing_eleven.csv",
    "competitions": ROOT / "6af3166e7b922_data_pack" / "competitions.csv",
    "teams": ROOT / "6af3166e7b922_data_pack" / "teams.csv",
    "competition_team_player_mapping": ROOT
    / "6af3166e7b922_data_pack"
    / "competition_team_player_mapping.csv",
}


def inventory(path: Path) -> dict:
    frame = pd.read_csv(path, low_memory=False)
    counts = frame.nunique(dropna=True)
    return {
        "path": str(path.relative_to(ROOT)),
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "dtypes": {k: str(v) for k, v in frame.dtypes.items()},
        "missing": {k: int(v) for k, v in frame.isna().sum().items()},
        "distinct": {k: int(v) for k, v in counts.items()},
        "duplicate_rows": int(frame.duplicated().sum()),
    }


def diagnostics() -> dict:
    b = pd.read_csv(FILES["ball_by_ball"], low_memory=False)
    m = pd.read_csv(FILES["matches"], low_memory=False)
    c = pd.read_csv(FILES["competitions"], low_memory=False)
    t = pd.read_csv(FILES["teams"], low_memory=False)
    p = pd.read_csv(FILES["playing_eleven"], low_memory=False)
    mapping = pd.read_csv(FILES["competition_team_player_mapping"], low_memory=False)
    bm = b.merge(m[["match_id", "comp_id", "match_date", "revised_target"]],
                 on="match_id", how="left", validate="many_to_one", indicator=True)
    mc = m.merge(c, on="comp_id", how="left", validate="many_to_one", indicator=True)
    cids = bm["comp_id"].dropna().astype(int)
    covered = m[m.match_id.isin(b.match_id)]
    deliveries_per_match = b.groupby("match_id").size().to_numpy()
    by_comp = m.groupby("comp_id").agg(
        matches=("match_id", "size"), first_match=("match_date", "min"),
        last_match=("match_date", "max")
    )
    by_comp["delivery_rows"] = bm.groupby("comp_id").size()
    comp_coverage = c.merge(by_comp.reset_index(), on="comp_id", how="left")
    comp_coverage[["matches", "delivery_rows"]] = comp_coverage[
        ["matches", "delivery_rows"]
    ].fillna(0).astype(int)
    scoped = bm.merge(c[["comp_id", "comp_name", "match_format"]],
                      on="comp_id", how="left", validate="many_to_one")
    non_ipl_t20 = scoped.match_format.eq("T20") & ~scoped.comp_name.str.startswith("IPL ")
    # Keep the relationship checks explicit; ball context above only brings
    # competition/date fields, while this join brings the two match teams.
    bt = b[["match_id", "batting_team_id", "bowling_team_id"]].merge(
        m[["match_id", "home_team_id", "away_team_id"]],
        on="match_id", validate="many_to_one"
    )
    team_keys = t[["comp_id", "team_id"]].drop_duplicates()
    pt = p[["match_id", "team_id", "player_id"]].drop_duplicates()
    mp = mapping[["comp_id", "team_id", "player_id"]]
    role_checks = {}
    for role, team_col, player_col in (
        ("striker", "batting_team_id", "striker_id"),
        ("non_striker", "batting_team_id", "non_striker_id"),
        ("bowler", "bowling_team_id", "bowler_id"),
    ):
        roster = b[["match_id", team_col, player_col]].rename(
            columns={team_col: "team_id", player_col: "player_id"}
        )
        on_xi = roster.merge(pt, on=["match_id", "team_id", "player_id"],
                             how="left", validate="many_to_one", indicator=True)
        roster["comp_id"] = bm["comp_id"].to_numpy()
        on_map = roster.merge(mp, on=["comp_id", "team_id", "player_id"],
                              how="left", validate="many_to_one", indicator=True)
        role_checks[role] = {
            "rows_absent_from_playing_eleven": int((on_xi._merge != "both").sum()),
            "rows_absent_from_competition_mapping": int((on_map._merge != "both").sum()),
        }
    batting_teams = bm[["comp_id", "batting_team_id"]].rename(columns={"batting_team_id": "team_id"})
    bowling_teams = bm[["comp_id", "bowling_team_id"]].rename(columns={"bowling_team_id": "team_id"})
    batting_team_check = batting_teams.merge(team_keys, on=["comp_id", "team_id"], how="left", validate="many_to_one", indicator=True)
    bowling_team_check = bowling_teams.merge(team_keys, on=["comp_id", "team_id"], how="left", validate="many_to_one", indicator=True)
    return {
        "keys": {
            "matches_match_id_duplicates": int(m.duplicated("match_id").sum()),
            "competitions_comp_id_duplicates": int(c.duplicated("comp_id").sum()),
            "teams_comp_team_duplicates": int(t.duplicated(["comp_id", "team_id"]).sum()),
            "playing_match_team_player_duplicates": int(p.duplicated(["match_id", "team_id", "player_id"]).sum()),
            "mapping_comp_team_player_duplicates": int(mapping.duplicated(["comp_id", "team_id", "player_id"]).sum()),
            "ball_match_innings_innings_ball_duplicates": int(b.duplicated(["match_id", "innings", "innings_ball_num"]).sum()),
            "ball_match_innings_over_ball_duplicates": int(b.duplicated(["match_id", "innings", "overid", "over_ball_num"]).sum()),
        },
        "coverage": {
            "delivery_matches": int(b.match_id.nunique()),
            "deliveries_per_match_p10_p50_p90": [float(x) for x in np.quantile(deliveries_per_match, [0.1, 0.5, 0.9])],
            "matches_without_deliveries": sorted(set(m.match_id) - set(b.match_id)),
            "delivery_matches_without_metadata": sorted(set(b.match_id) - set(m.match_id)),
            "matches_without_competition": int((mc._merge != "both").sum()),
            "delivery_rows_without_match": int((bm._merge != "both").sum()),
            "ball_team_not_in_match": int((~(bt.batting_team_id.eq(bt.home_team_id) | bt.batting_team_id.eq(bt.away_team_id)) | ~(bt.bowling_team_id.eq(bt.home_team_id) | bt.bowling_team_id.eq(bt.away_team_id)) | bt.batting_team_id.eq(bt.bowling_team_id)).sum()),
            "batting_team_rows_absent_from_teams": int((batting_team_check._merge != "both").sum()),
            "bowling_team_rows_absent_from_teams": int((bowling_team_check._merge != "both").sum()),
            "role_checks": role_checks,
            "match_year_counts": m.match_date.str.slice(0, 4).value_counts().sort_index().to_dict(),
            "delivery_match_year_counts": covered.match_date.str.slice(0, 4).value_counts().sort_index().to_dict(),
            "match_date_min": str(m.match_date.min()),
            "match_date_max": str(m.match_date.max()),
            "delivery_time_min": str(b.delivery_ist_time.min()),
            "delivery_time_max": str(b.delivery_ist_time.max()),
            "match_formats": mc.match_format.value_counts(dropna=False).to_dict(),
            "delivery_formats": bm.merge(c[["comp_id", "match_format"]], on="comp_id", how="left", validate="many_to_one").match_format.value_counts(dropna=False).to_dict(),
            "non_ipl_t20_delivery_rows": int(non_ipl_t20.sum()),
            "non_ipl_t20_matches": int(scoped.loc[non_ipl_t20, "match_id"].nunique()),
            "non_ipl_t20_competitions": scoped.loc[non_ipl_t20, "comp_name"].drop_duplicates().tolist(),
            "comp_seasons": c.season.value_counts().sort_index().to_dict(),
            "match_stages": m.stage.value_counts(dropna=False).to_dict(),
            "innings": b.innings.value_counts().sort_index().to_dict(),
            "matches_by_competition": mc.groupby(["comp_id", "comp_name", "match_format"], dropna=False).size().sort_values(ascending=False).head(12).to_dict().__str__(),
            "competition_coverage": comp_coverage.sort_values(
                ["match_format", "season", "comp_name"]
            ).astype(object).where(lambda x: pd.notna(x), None).to_dict("records"),
        },
        "delivery_fields": {
            "extras_type": b.extras_type.value_counts(dropna=False).to_dict(),
            "extras1_runs": b.extras1_runs.value_counts(dropna=False).sort_index().to_dict(),
            "extras_2_runs": b.extras_2_runs.value_counts(dropna=False).sort_index().to_dict(),
            "runs": b.runs.value_counts(dropna=False).sort_index().to_dict(),
            "dismissal_type": b.dismissal_type.value_counts(dropna=False).to_dict(),
            "wicket_true": int(b.is_wicket.sum()),
            "wicket_without_dismissal": int((b.is_wicket & b.dismissal_type.isna()).sum()),
            "dismissal_without_wicket": int((~b.is_wicket & b.dismissal_type.notna()).sum()),
            "is_four_runs": pd.crosstab(b.is_four, b.runs).to_dict().__str__(),
            "is_six_runs": pd.crosstab(b.is_six, b.runs).to_dict().__str__(),
            "extras_by_type": b.groupby("extras_type", dropna=False)[["runs", "extras1_runs", "extras_2_runs"]].agg(["min", "max", "mean"]).round(3).to_dict().__str__(),
            "over_ball_max": int(b.over_ball_num.max()),
            "innings_ball_max": int(b.innings_ball_num.max()),
            "revised_matches": int(m.revised_target.notna().sum()),
            "super_over_matches": int(b.loc[b.innings > 2, "match_id"].nunique()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    args = parser.parse_args()
    report = {name: inventory(path) for name, path in FILES.items()}
    report["diagnostics"] = diagnostics()
    if args.format == "json":
        print(json.dumps(report, indent=2))
    else:
        print(render_markdown(report))


def render_markdown(report: dict) -> str:
    diag = report["diagnostics"]
    coverage = diag["coverage"]
    keys = diag["keys"]
    lines = [
        "# Official data audit - initial phase", "",
        "Generated by `python -m src.audit --format markdown` from the six supplied CSVs.",
        "Pandas-inferred dtypes are shown; object/string fields are not assumed to be semantic types.",
        "", "## Inventory", "",
        "| File | Rows | Columns | Exact duplicate rows |",
        "|---|---:|---:|---:|",
    ]
    for name, item in report.items():
        if name != "diagnostics":
            lines.append(f"| `{item['path']}` | {item['rows']:,} | {len(item['columns'])} | {item['duplicate_rows']} |")
    lines += ["", "## Columns, inferred types, and missing cells", ""]
    for name, item in report.items():
        if name == "diagnostics":
            continue
        lines += [f"### {name}", "", "| Column | Pandas dtype | Missing | Distinct non-null |",
                  "|---|---|---:|---:|"]
        for column in item["columns"]:
            lines.append(f"| `{column}` | `{item['dtypes'][column]}` | "
                         f"{item['missing'][column]:,} | {item['distinct'][column]:,} |")
        lines.append("")
    lines += [
        "## Keys and relationships", "",
        "| Check | Result |", "|---|---:|",
    ]
    for key, value in keys.items():
        lines.append(f"| {key.replace('_', ' ')} | {value:,} |")
    for key in ("delivery_rows_without_match", "matches_without_competition",
                "ball_team_not_in_match", "batting_team_rows_absent_from_teams",
                "bowling_team_rows_absent_from_teams"):
        lines.append(f"| {key.replace('_', ' ')} | {coverage[key]:,} |")
    for role, checks in coverage["role_checks"].items():
        for key, value in checks.items():
            lines.append(f"| {role} {key.replace('_', ' ')} | {value:,} |")
    lines += [
        "", "`matches.match_id` is the parent of `ball_by_ball.match_id` and",
        "`playing_eleven.match_id`. `matches.comp_id` joins to `competitions.comp_id`;",
        "`(comp_id, team_id)` joins to `teams`, and `(comp_id, team_id, player_id)`",
        "joins to the competition-player mapping. All three delivery player roles",
        "match that mapping. Playing-XI keys require normalization before a join.",
        "", "## Coverage", "",
        f"- Match dates: {coverage['match_date_min']} through {coverage['match_date_max']}.",
        f"- Delivery timestamps: {coverage['delivery_time_min']} through {coverage['delivery_time_max']}.",
        f"- Matches: {report['matches']['rows']:,} metadata rows; {coverage['delivery_matches']:,} with deliveries; "
        f"{len(coverage['matches_without_deliveries'])} without deliveries.",
        f"- Deliveries per covered match (10th/50th/90th percentiles): "
        f"{coverage['deliveries_per_match_p10_p50_p90']}.",
        f"- Match formats: {coverage['match_formats']}; delivery rows by format: {coverage['delivery_formats']}.",
        f"- Non-IPL T20: {coverage['non_ipl_t20_matches']} matches, "
        f"{coverage['non_ipl_t20_delivery_rows']:,} deliveries in "
        f"{coverage['non_ipl_t20_competitions']}.",
        f"- Matches by year: {coverage['match_year_counts']}.",
        f"- Competitions by season label: {coverage['comp_seasons']}.",
        f"- Match stages: {coverage['match_stages']}.",
        f"- Delivery rows by innings: {coverage['innings']}.",
        f"- All {report['competitions']['distinct']['comp_id']} competition IDs are unique; "
        f"matches reference {report['matches']['distinct']['comp_id']} of them.",
        f"- Matches with revised targets: {diag['delivery_fields']['revised_matches']}; "
        f"matches with innings 3/4: {diag['delivery_fields']['super_over_matches']}.",
        "", "## Data quality and interpretation", "",
        "- `extras1_runs` is constant within each extras category (1=Overstep, 2=wide, "
        "3=legbyes, 4=byes, 5=other, 6=no extras, 7=Penalty). It is a code, "
        "not an extras run count. `extras_2_runs` is zero on almost every row, "
        "including every wide, bye, and leg-bye. Exact team totals and required "
        "run rates cannot be reconstructed safely from these fields yet.",
        "- `over_ball_num` reaches 13; it is a delivery index within an over, "
        "not necessarily a legal-ball number. Wides and `Overstep` are treated "
        "as illegal. The legality of `other` and `Penalty` is unresolved.",
        "- Revised-overs fields contain values such as 114 and 42, apparently "
        "ball counts despite their names. This interpretation needs validation "
        "before pressure scoring uses an innings limit.",
        f"- {keys['playing_match_team_player_duplicates']} playing-XI rows repeat "
        "`(match_id, team_id, player_id)`; 35 groups have conflicting batting orders. "
        "The normalized roster leaves those orders missing and preserves a conflict flag.",
        "- The playing XI does not cover a small number of delivery roles "
        f"({coverage['role_checks']}). The competition-player mapping does cover them.",
        "- Six matches contain innings 3/4 and require separate super-over handling. "
        "Sixteen match metadata rows have no delivery records and cannot contribute "
        "to delivery-based ratings.",
        "- The statement describes IPL and T20I data, but the pack also contains "
        "two non-IPL T20 competitions (30 matches). The core ranking cohort "
        "should use IPL and T20I unless the organizers clarify their inclusion.",
        "- The statement mentions supporting player data, but the archive has "
        "no separate `players.csv`; player IDs and names are available in the "
        "competition-team-player mapping and delivery records.",
        "- `delivery_ist_time` contains the literal `UTC` in its values. Use "
        "`match_date` for season chronology until timezone semantics are confirmed.",
        "", "Full machine-readable counts and category distributions are in `reports/audit.json`.",
    ]
    lines += [
        "", "## Competition coverage (all supplied competition IDs)", "",
        "| ID | Competition | Format | Season | Matches | Delivery rows | First match | Last match |",
        "|---:|---|---|---|---:|---:|---|---|",
    ]
    for item in coverage["competition_coverage"]:
        lines.append(
            f"| {item['comp_id']} | {item['comp_name']} | {item['match_format']} | "
            f"{item['season']} | {item['matches']} | {item['delivery_rows']} | "
            f"{item['first_match'] or ''} | {item['last_match'] or ''} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
