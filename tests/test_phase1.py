"""Structural failure cases that Phase 1 must catch."""

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.data import DataValidationError
from src.phase1 import validate_innings


def test_phase1_can_start_as_a_script_from_src() -> None:
    script = Path(__file__).resolve().parents[1] / "src" / "phase1.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"], cwd=script.parent,
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr


def innings_sample() -> pd.DataFrame:
    return pd.DataFrame([
        {"match_id": 1, "innings": 1, "innings_ball_num": 1,
         "overid": 1, "over_ball_num": 1, "batting_team_id": 10,
         "bowling_team_id": 20, "is_wicket": False},
        {"match_id": 1, "innings": 1, "innings_ball_num": 2,
         "overid": 1, "over_ball_num": 2, "batting_team_id": 10,
         "bowling_team_id": 20, "is_wicket": True},
        {"match_id": 1, "innings": 2, "innings_ball_num": 1,
         "overid": 1, "over_ball_num": 1, "batting_team_id": 20,
         "bowling_team_id": 10, "is_wicket": False},
    ])


def test_valid_innings_sequence() -> None:
    summary = validate_innings(innings_sample())
    assert summary["innings_groups"] == 2
    assert summary["two_innings_matches"] == 1


def test_delivery_gap_is_rejected() -> None:
    broken = innings_sample()
    broken.loc[1, "innings_ball_num"] = 3
    with pytest.raises(DataValidationError, match="delivery numbering"):
        validate_innings(broken)


def test_mid_innings_team_switch_is_rejected() -> None:
    broken = innings_sample()
    broken.loc[1, "batting_team_id"] = 20
    with pytest.raises(DataValidationError, match="team changes"):
        validate_innings(broken)


def test_unpaired_super_over_is_rejected() -> None:
    broken = innings_sample()
    broken.loc[len(broken)] = [1, 3, 1, 1, 1, 10, 20, False]
    with pytest.raises(DataValidationError, match="innings pattern"):
        validate_innings(broken)
