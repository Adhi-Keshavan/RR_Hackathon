"""Locate, load, and validate the official CSV tables."""

from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import pandas as pd


CSV_FILES = {
    "ball_by_ball": "ball_by_ball.csv",
    "matches": "matches.csv",
    "playing_eleven": "playing_eleven.csv",
    "competitions": "competitions.csv",
    "teams": "teams.csv",
    "competition_team_player_mapping": "competition_team_player_mapping.csv",
}

REQUIRED_COLUMNS = {
    "ball_by_ball": {
        "match_id", "innings", "overid", "over_ball_num", "innings_ball_num",
        "batting_team_id", "bowling_team_id", "striker_id", "non_striker_id",
        "bowler_id", "runs", "is_wicket", "dismissal_type", "extras_type",
        "extras1_runs", "extras_2_runs", "striker_name", "non_striker_name",
        "bowler_name", "dismissed_player_id",
    },
    "matches": {
        "match_id", "comp_id", "match_date", "stage", "home_team_id",
        "away_team_id", "home_team", "away_team",
        "first_inn_revised_overs", "second_inn_revised_overs",
        "revised_target", "knock_out",
    },
    "playing_eleven": {
        "match_id", "team_id", "player_id", "batting_order", "is_played",
        "is_captain", "is_wk", "is_impact", "is_impact_off",
    },
    "competitions": {"comp_id", "comp_name", "match_format", "season"},
    "teams": {"comp_id", "team_id", "team_name"},
    "competition_team_player_mapping": {
        "comp_id", "comp_name", "team_id", "team_name", "player_id", "player_name",
    },
}

KEYS = {
    "ball_by_ball": ["match_id", "innings", "innings_ball_num"],
    "matches": ["match_id"],
    "competitions": ["comp_id"],
    "teams": ["comp_id", "team_id"],
    "competition_team_player_mapping": ["comp_id", "team_id", "player_id"],
    # playing_eleven is deliberately absent: its official file repeats 40 keys.
}

POSITIVE_ID_COLUMNS = {
    "ball_by_ball": ["match_id", "batting_team_id", "bowling_team_id",
                    "striker_id", "non_striker_id", "bowler_id"],
    "matches": ["match_id", "comp_id", "home_team_id", "away_team_id"],
    "playing_eleven": ["match_id", "team_id", "player_id"],
    "competitions": ["comp_id"],
    "teams": ["comp_id", "team_id"],
    "competition_team_player_mapping": ["comp_id", "team_id", "player_id"],
}


class DataValidationError(ValueError):
    """An official table cannot be used safely as supplied."""


def _read_from_root(root: Path, filename: str) -> pd.DataFrame:
    candidates = [root / "data" / filename]
    candidates.extend(sorted(root.glob(f"*_data_pack/{filename}")))
    for path in candidates:
        if path.is_file():
            return pd.read_csv(path, low_memory=False)

    for archive in sorted(root.glob("*data_pack.zip")):
        with ZipFile(archive) as bundle:
            matches = [name for name in bundle.namelist() if Path(name).name == filename]
            if len(matches) == 1:
                with bundle.open(matches[0]) as source:
                    return pd.read_csv(source, low_memory=False)
            if len(matches) > 1:
                raise DataValidationError(f"Ambiguous {filename} in {archive}")
    raise FileNotFoundError(f"Official file {filename} not found below {root}")


def validate_schema(name: str, frame: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS[name] - set(frame.columns)
    if missing:
        raise DataValidationError(f"{name}: missing columns {sorted(missing)}")
    for column in POSITIVE_ID_COLUMNS[name]:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or (values % 1 != 0).any() or (values <= 0).any():
            raise DataValidationError(f"{name}: invalid positive ID in {column}")
    key = KEYS.get(name)
    if key:
        if frame[key].isna().any().any():
            raise DataValidationError(f"{name}: null primary key in {key}")
        if frame.duplicated(key).any():
            raise DataValidationError(f"{name}: duplicate primary key in {key}")
    if name == "ball_by_ball":
        numeric = ["innings", "overid", "over_ball_num", "innings_ball_num", "runs"]
        for column in numeric:
            values = pd.to_numeric(frame[column], errors="coerce")
            if values.isna().any() or (values % 1 != 0).any():
                raise DataValidationError(f"ball_by_ball: invalid integer {column}")
            if column != "runs" and (values <= 0).any():
                raise DataValidationError(f"ball_by_ball: nonpositive {column}")
        if (pd.to_numeric(frame["runs"]) < 0).any():
            raise DataValidationError("ball_by_ball: negative recorded runs")
    if name == "matches" and pd.to_datetime(frame["match_date"], errors="coerce").isna().any():
        raise DataValidationError("matches: invalid match_date")


def load_official_data(root: Path | str | None = None) -> dict[str, pd.DataFrame]:
    """Load all six tables without modifying source data."""
    base = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    tables = {name: _read_from_root(base, filename) for name, filename in CSV_FILES.items()}
    for name, frame in tables.items():
        validate_schema(name, frame)
    return tables


def attach_match_context(
    deliveries: pd.DataFrame,
    matches: pd.DataFrame,
    competitions: pd.DataFrame,
) -> pd.DataFrame:
    """Join only unique match and competition dimensions, preserving every ball."""
    validate_schema("ball_by_ball", deliveries)
    validate_schema("matches", matches)
    validate_schema("competitions", competitions)
    match_fields = [
        "match_id", "comp_id", "match_date", "stage", "knock_out",
        "home_team_id", "away_team_id", "first_inn_revised_overs",
        "second_inn_revised_overs", "revised_target",
    ]
    result = deliveries.merge(
        matches[match_fields], on="match_id", how="left", validate="many_to_one",
        indicator="_match_join",
    )
    if (result["_match_join"] != "both").any():
        raise DataValidationError("Some deliveries have no match metadata")
    result = result.drop(columns="_match_join")
    result = result.merge(
        competitions[["comp_id", "comp_name", "match_format", "season"]],
        on="comp_id", how="left", validate="many_to_one", indicator="_comp_join",
    )
    if (result["_comp_join"] != "both").any():
        raise DataValidationError("Some deliveries have no competition metadata")
    result = result.drop(columns="_comp_join")
    if len(result) != len(deliveries):
        raise DataValidationError("Context join changed delivery count")
    home, away = result["home_team_id"], result["away_team_id"]
    batting, bowling = result["batting_team_id"], result["bowling_team_id"]
    valid_teams = ((batting.eq(home) & bowling.eq(away)) |
                   (batting.eq(away) & bowling.eq(home)))
    if not valid_teams.all():
        raise DataValidationError("Delivery teams disagree with match teams")
    return result
