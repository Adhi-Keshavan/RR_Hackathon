"""Batting attribution and descriptive baseline checks."""

import subprocess
import sys
from pathlib import Path

import pandas as pd

from src.batting_analysis import analyze_batting, make_batting_events


def event_rows() -> pd.DataFrame:
    return pd.DataFrame([
        (1, 100, 101, 4, "none", True, False, pd.NA),
        (1, 100, 101, 2, "no_ball", False, False, pd.NA),
        (1, 100, 101, 0, "wide", False, True, 101),
        (1, 100, 101, 1, "legbyes", True, False, pd.NA),
    ], columns=[
        "match_id", "striker_id", "non_striker_id", "runs", "extras_category",
        "legal_delivery_phase2", "is_wicket", "dismissed_player_id",
    ]).assign(competition_group="IPL", innings=1,
             high_pressure=True, pressure_score=80.0)


def test_no_ball_wide_byes_and_non_striker_dismissal() -> None:
    events = make_batting_events(event_rows())
    striker = events.loc[events.player_id.eq(100)]
    partner = events.loc[events.player_id.eq(101)]
    assert striker.batting_runs.sum() == 6
    assert striker.balls_faced.sum() == 3
    assert striker.legal_balls_faced.sum() == 2
    assert striker.dismissals.sum() == 0
    assert partner.dismissals.sum() == 1
    assert partner.balls_faced.sum() == 0


def comparison_rows() -> pd.DataFrame:
    rows = pd.DataFrame([
        (1, 1, 0, 10.0),
        (1, 2, 2, 10.0),
        (2, 1, 4, 80.0),
        (2, 2, 6, 80.0),
    ], columns=["match_id", "innings_ball_num", "runs", "pressure_score"])
    rows["innings"] = 1
    rows["match_format"] = "T20"
    rows["comp_name"] = "IPL 2024"
    rows["is_super_over"] = False
    rows["extras_category"] = "none"
    rows["legal_delivery_phase2"] = True
    rows["striker_id"] = 100
    rows["non_striker_id"] = 101
    rows["is_wicket"] = False
    rows["dismissed_player_id"] = pd.NA
    rows["same_batter_at_both_ends"] = False
    return rows


def test_pressure_performance_compared_with_own_baseline() -> None:
    mapping = pd.DataFrame({"player_id": [100], "player_name": ["Batter"]})
    analysis = analyze_batting(comparison_rows(), mapping)
    player = analysis.player_summary.iloc[0]
    assert player.pressure_batting_runs == 10
    assert player.pressure_balls_faced == 2
    assert player.ordinary_batting_runs == 2
    assert player.ordinary_balls_faced == 2
    assert player.pressure_strike_rate == 500
    assert player.ordinary_strike_rate == 100
    assert player.strike_rate_delta == 400
    assert player.pressure_runs_above_matched_baseline == 8
    assert "rank" not in analysis.player_summary.columns


def test_scope_and_ambiguous_extras_are_reported() -> None:
    rows = comparison_rows()
    extra = rows.iloc[[0]].copy()
    extra["match_id"] = 3
    extra["extras_category"] = "other"
    extra["pressure_score"] = 90.0
    outside = rows.iloc[[0]].copy()
    outside["match_id"] = 4
    outside["comp_name"] = "Other T20 Cup"
    rows = pd.concat([rows, extra, outside], ignore_index=True)
    mapping = pd.DataFrame({"player_id": [100], "player_name": ["Batter"]})
    analysis = analyze_batting(rows, mapping)
    assert analysis.report["non_ipl_t20_rows_excluded"] == 1
    assert analysis.report["ambiguous_extra_rows_excluded"] == 1


def test_phase3_starts_from_src_directory() -> None:
    script = Path(__file__).resolve().parents[1] / "src" / "batting_analysis.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"], cwd=script.parent,
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
