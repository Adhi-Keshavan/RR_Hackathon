"""Conservative delivery and roster cleaning; no pressure rating is computed here."""

from __future__ import annotations

import pandas as pd

from .data import DataValidationError, validate_schema


EXTRA_TYPES = {
    "none": "none", "wide": "wide", "overstep": "no_ball",
    "byes": "byes", "legbyes": "legbyes", "other": "other",
    "penalty": "penalty",
}
EXTRA_CODES = {
    "none": 6, "wide": 2, "no_ball": 1, "byes": 4,
    "legbyes": 3, "other": 5, "penalty": 7,
}
BOWLER_DISMISSALS = {"caught", "bowled", "lbw", "stump", "hit wicket"}


def _booleans(series: pd.Series, label: str) -> pd.Series:
    normalized = series.astype("string").str.strip().str.lower()
    values = normalized.map({"true": True, "false": False, "1": True, "0": False})
    unexpected = normalized.notna() & values.isna()
    if unexpected.any():
        raise DataValidationError(f"Invalid boolean values in {label}")
    return values.astype("boolean")


def prepare_deliveries(frame: pd.DataFrame) -> pd.DataFrame:
    """Create safe pre-ball state from the official row order within an innings.

    ``extras1_runs`` is an extras category code in this pack, so this function
    never treats it as a run count. ``other`` and ``penalty`` have unresolved
    legal-ball semantics; legal-ball counts after either event remain null.
    """
    validate_schema("ball_by_ball", frame)
    d = frame.copy().sort_values(
        ["match_id", "innings", "innings_ball_num"], kind="stable"
    ).reset_index(drop=True)
    d["extras_category"] = (
        d["extras_type"].astype("string").str.strip().str.lower()
        .fillna("none").map(EXTRA_TYPES)
    )
    if d["extras_category"].isna().any():
        unknown = d.loc[d["extras_category"].isna(), "extras_type"].unique()
        raise DataValidationError(f"Unknown extras types: {unknown}")
    codes = pd.to_numeric(d["extras1_runs"], errors="coerce")
    if not codes.eq(d["extras_category"].map(EXTRA_CODES)).all():
        raise DataValidationError("extras1_runs no longer matches the observed category code")
    d["is_wicket"] = _booleans(d["is_wicket"], "is_wicket")
    if d["is_wicket"].isna().any():
        raise DataValidationError("Null is_wicket in deliveries")
    dismissal = d["dismissal_type"].astype("string").str.strip().str.lower()
    if ((d["is_wicket"] & dismissal.isna()) |
            (~d["is_wicket"] & dismissal.notna())).any():
        raise DataValidationError("Wicket flag and dismissal type disagree")
    d["bowler_wicket"] = d["is_wicket"] & dismissal.isin(BOWLER_DISMISSALS)
    d["recorded_batter_runs"] = d["runs"].astype("int64")
    d["same_batter_at_both_ends"] = d["striker_id"].eq(d["non_striker_id"])
    d["wicket_missing_dismissed_id"] = d["is_wicket"] & d["dismissed_player_id"].isna()

    legal = pd.Series(pd.NA, index=d.index, dtype="boolean")
    legal.loc[d.extras_category.isin(["none", "byes", "legbyes"])] = True
    legal.loc[d.extras_category.isin(["wide", "no_ball"])] = False
    d["is_legal_delivery"] = legal
    group = [d["match_id"], d["innings"]]
    legal_number = legal.fillna(False).astype("int64")
    unresolved = legal.isna().astype("int64")
    legal_before = legal_number.groupby(group).cumsum() - legal_number
    unresolved_before = unresolved.groupby(group).cumsum() - unresolved
    d["legal_balls_before"] = legal_before.astype("Int64").mask(unresolved_before.gt(0))
    wickets = d["is_wicket"].astype("int64")
    d["wickets_before"] = wickets.groupby(group).cumsum() - wickets
    return d


def normalize_playing_eleven(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse repeated roster keys without inventing conflicting batting order."""
    validate_schema("playing_eleven", frame)
    p = frame.copy()
    key = ["match_id", "team_id", "player_id"]
    for flag in ("is_played", "is_captain", "is_wk", "is_impact", "is_impact_off"):
        p[flag] = _booleans(p[flag], flag)
    grouped = p.groupby(key, sort=False, dropna=False)
    result = grouped.size().rename("source_rows").to_frame()
    orders = grouped["batting_order"].apply(lambda s: tuple(pd.unique(s.dropna())))
    result["batting_order_conflict"] = orders.map(lambda values: len(values) > 1)
    result["batting_order"] = orders.map(
        lambda values: values[0] if len(values) == 1 else pd.NA
    ).astype("Float64")
    for flag in ("is_played", "is_captain", "is_wk", "is_impact", "is_impact_off"):
        result[flag] = grouped[flag].agg(lambda s: bool(s.fillna(False).any()))
    return result.reset_index()


def attach_roster_role(
    deliveries: pd.DataFrame, roster: pd.DataFrame, role: str
) -> pd.DataFrame:
    """Optional roster enrichment with a cardinality check for one player role."""
    roles = {"striker": ("batting_team_id", "striker_id"),
             "non_striker": ("batting_team_id", "non_striker_id"),
             "bowler": ("bowling_team_id", "bowler_id")}
    if role not in roles:
        raise ValueError(f"Unsupported role: {role}")
    team_col, player_col = roles[role]
    normalized = normalize_playing_eleven(roster)
    lookup = normalized.rename(columns={
        "team_id": team_col, "player_id": player_col,
        **{col: f"{role}_{col}" for col in normalized.columns
           if col not in {"match_id", "team_id", "player_id"}},
    })
    result = deliveries.merge(
        lookup, on=["match_id", team_col, player_col], how="left",
        validate="many_to_one",
    )
    if len(result) != len(deliveries):
        raise DataValidationError("Roster join changed delivery count")
    return result
