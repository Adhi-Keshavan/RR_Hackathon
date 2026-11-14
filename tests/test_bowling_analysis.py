"""Bowling attribution, economy uncertainty, and baseline checks."""

import subprocess
import sys
from pathlib import Path

import pandas as pd

from src.bowling_analysis import analyze_bowling, make_bowling_events


def event_rows() -> pd.DataFrame:
    return pd.DataFrame([
        (1, 4, "none", True, True),
        (1, 0, "wide", False, True),
        (1, 2, "no_ball", False, False),
        (1, 1, "legbyes", True, False),
    ], columns=["match_id", "runs", "extras_category",
                "legal_delivery_phase2", "bowler_wicket"]).assign(
        competition_group="IPL", innings=1, high_pressure=True,
        pressure_score=80.0, bowler_id=200,
    )


def test_wickets_and_conceded_run_quality_by_extra_type() -> None:
    events = make_bowling_events(event_rows())
    assert events.legal_deliveries.sum() == 2
    assert events.bowler_wickets.sum() == 2  # caught and stumped on wide
    assert events.batter_runs_allowed.sum() == 6
    assert events.runs_conceded_floor.sum() == 7
    assert events.uncertain_concession_deliveries.sum() == 2
    assert events.known_exact_runs_conceded.sum() == 4


def comparison_rows() -> pd.DataFrame:
    rows = pd.DataFrame([
        (1, 1, 4, 10.0, False, "none"),
        (1, 2, 0, 10.0, False, "none"),
        (2, 1, 1, 80.0, False, "none"),
        (2, 2, 0, 80.0, True, "none"),
    ], columns=["match_id", "innings_ball_num", "runs", "pressure_score",
                "bowler_wicket", "extras_category"])
    rows["innings"] = 1
    rows["match_format"] = "T20"
    rows["comp_name"] = "IPL 2024"
    rows["is_super_over"] = False
    rows["legal_delivery_phase2"] = True
    rows["bowler_id"] = 200
    return rows


def test_exact_economy_wicket_rate_and_baseline_delta() -> None:
    mapping = pd.DataFrame({"player_id": [200], "player_name": ["Bowler"]})
    analysis = analyze_bowling(comparison_rows(), mapping)
    player = analysis.player_summary.iloc[0]
    assert player.pressure_legal_deliveries == 2
    assert player.pressure_bowler_wickets == 1
    assert player.pressure_economy_exact == 3
    assert player.ordinary_economy_exact == 12
    assert player.exact_economy_delta == -9
    assert player.pressure_wickets_per_100_legal == 50
    assert player.pressure_wickets_above_matched_baseline == 1
    assert "rank" not in analysis.player_summary.columns


def test_wide_makes_exact_economy_unavailable() -> None:
    rows = comparison_rows()
    rows.loc[rows.match_id.eq(2) & rows.innings_ball_num.eq(2),
             "extras_category"] = "wide"
    rows.loc[rows.match_id.eq(2) & rows.innings_ball_num.eq(2),
             "legal_delivery_phase2"] = False
    mapping = pd.DataFrame({"player_id": [200], "player_name": ["Bowler"]})
    player = analyze_bowling(rows, mapping).player_summary.iloc[0]
    assert pd.isna(player.pressure_economy_exact)
    assert player.pressure_economy_floor == 12
    assert pd.isna(player.exact_economy_delta)


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
    mapping = pd.DataFrame({"player_id": [200], "player_name": ["Bowler"]})
    analysis = analyze_bowling(rows, mapping)
    assert analysis.report["non_ipl_t20_rows_excluded"] == 1
    assert analysis.report["ambiguous_extra_rows_excluded"] == 1


def test_phase4_starts_from_src_directory() -> None:
    script = Path(__file__).resolve().parents[1] / "src" / "bowling_analysis.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"], cwd=script.parent,
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
