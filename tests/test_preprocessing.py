"""Behavioural checks for ball state and safe enrichment."""

import pandas as pd
import pytest

from src.data import DataValidationError, attach_match_context, validate_schema
from src.preprocessing import (
    attach_roster_role,
    normalize_playing_eleven,
    prepare_deliveries,
)


def deliveries() -> pd.DataFrame:
    rows = [
        # Deliberately unordered: innings_ball_num defines delivery order.
        (4, None, 1, False, None),
        (1, None, 0, False, None),
        (3, "Overstep", 2, True, "RunOut"),
        (2, "wide", 0, False, None),
        (5, None, 0, True, "Caught"),
    ]
    frame = pd.DataFrame(rows, columns=[
        "innings_ball_num", "extras_type", "runs", "is_wicket", "dismissal_type",
    ])
    frame["match_id"] = 1
    frame["innings"] = 1
    frame["overid"] = 1
    frame["over_ball_num"] = frame["innings_ball_num"]
    frame["batting_team_id"] = 10
    frame["bowling_team_id"] = 20
    frame["striker_id"] = 100
    frame["striker_name"] = "Striker"
    frame["non_striker_id"] = 101
    frame["non_striker_name"] = "Partner"
    frame["bowler_id"] = 200
    frame["bowler_name"] = "Bowler"
    frame["dismissed_player_id"] = frame.dismissal_type.map({
        "RunOut": 101, "Caught": 100,
    })
    frame["extras1_runs"] = frame.extras_type.map({"wide": 2, "Overstep": 1}).fillna(6)
    frame["extras_2_runs"] = 0
    return frame


def match_table() -> pd.DataFrame:
    return pd.DataFrame([{
        "match_id": 1, "comp_id": 7, "match_date": "2024-01-01",
        "stage": "League", "knock_out": 0, "home_team_id": 10,
        "away_team_id": 20, "first_inn_revised_overs": None,
        "home_team": "Home", "away_team": "Away",
        "second_inn_revised_overs": None, "revised_target": None,
    }])


def roster_table() -> pd.DataFrame:
    return pd.DataFrame([
        {"match_id": 1, "team_id": 10, "player_id": 100,
         "batting_order": 1, "is_played": True, "is_captain": False,
         "is_wk": False, "is_impact": 0, "is_impact_off": 0},
        {"match_id": 1, "team_id": 10, "player_id": 100,
         "batting_order": 2, "is_played": True, "is_captain": False,
         "is_wk": False, "is_impact": 0, "is_impact_off": 0},
    ])


def test_preball_counts_handle_illegal_balls_and_dismissals() -> None:
    result = prepare_deliveries(deliveries())
    assert result.innings_ball_num.tolist() == [1, 2, 3, 4, 5]
    assert result.is_legal_delivery.tolist() == [True, False, False, True, True]
    assert result.legal_balls_before.tolist() == [0, 1, 1, 1, 2]
    assert result.wickets_before.tolist() == [0, 0, 0, 1, 1]
    assert result.bowler_wicket.tolist() == [False, False, False, False, True]
    assert result.recorded_batter_runs.tolist() == [0, 0, 2, 1, 0]


def test_unknown_extra_does_not_create_false_legal_ball_count() -> None:
    frame = deliveries()
    frame.loc[frame.innings_ball_num.eq(2), "extras_type"] = "other"
    frame.loc[frame.innings_ball_num.eq(2), "extras1_runs"] = 5
    result = prepare_deliveries(frame)
    assert pd.isna(result.loc[1, "is_legal_delivery"])
    assert result.loc[1, "legal_balls_before"] == 1
    assert pd.isna(result.loc[2, "legal_balls_before"])


def test_roster_conflict_is_visible_and_join_preserves_rows() -> None:
    roster = roster_table()
    normalized = normalize_playing_eleven(roster)
    assert len(normalized) == 1
    assert normalized.loc[0, "source_rows"] == 2
    assert bool(normalized.loc[0, "batting_order_conflict"])
    assert pd.isna(normalized.loc[0, "batting_order"])
    joined = attach_roster_role(deliveries(), roster, "striker")
    assert len(joined) == 5


def test_match_and_competition_joins_preserve_rows() -> None:
    competitions = pd.DataFrame([{
        "comp_id": 7, "comp_name": "Sample", "match_format": "T20",
        "season": "2024",
    }])
    joined = attach_match_context(deliveries(), match_table(), competitions)
    assert len(joined) == 5
    assert joined.comp_id.eq(7).all()


def test_duplicate_dimension_key_and_missing_match_fail() -> None:
    competitions = pd.DataFrame([{
        "comp_id": 7, "comp_name": "Sample", "match_format": "T20",
        "season": "2024",
    }])
    with pytest.raises(DataValidationError, match="duplicate primary key"):
        attach_match_context(deliveries(), pd.concat([match_table()] * 2), competitions)
    with pytest.raises(DataValidationError, match="no match metadata"):
        attach_match_context(deliveries(), match_table().assign(match_id=2), competitions)


def test_schema_rejects_duplicate_delivery_key() -> None:
    with pytest.raises(DataValidationError, match="duplicate primary key"):
        validate_schema("ball_by_ball", pd.concat([deliveries()] * 2))


def test_extras_code_drift_is_rejected() -> None:
    frame = deliveries()
    frame.loc[frame.innings_ball_num.eq(2), "extras1_runs"] = 3
    with pytest.raises(DataValidationError, match="category code"):
        prepare_deliveries(frame)
