"""Phase 2 checks for pre-ball causality, extras, targets, and exclusions."""

import pandas as pd

from src.pressure_model import build_pressure_features, write_feature_csv


def context(rows: list[tuple], revised: bool = False) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=[
        "match_id", "innings", "overid", "innings_ball_num", "runs",
        "extras_category", "is_legal_delivery", "wickets_before",
    ])
    frame["knock_out"] = 0
    frame["first_inn_revised_overs"] = 60 if revised else pd.NA
    frame["second_inn_revised_overs"] = 60 if revised else pd.NA
    frame["revised_target"] = 50 if revised else pd.NA
    return frame


def test_score_target_and_rrr_use_only_prior_deliveries() -> None:
    rows = [
        (1, 1, 1, 1, 4, "none", True, 0),
        (1, 1, 1, 2, 6, "none", True, 0),
        (1, 2, 1, 1, 6, "none", True, 0),
        (1, 2, 1, 2, 0, "wide", False, 0),
        (1, 2, 1, 3, 2, "no_ball", False, 0),
        (1, 2, 1, 4, 0, "none", True, 0),
    ]
    features = build_pressure_features(context(rows))
    chase = features[features.innings.eq(2)].reset_index(drop=True)
    assert chase.team_score_before.tolist()[:2] == [0, 6]
    assert pd.isna(chase.loc[2, "team_score_before"])
    assert chase.minimum_team_score_before.tolist() == [0, 6, 7, 9]
    assert chase.target_before.tolist() == [11, 11, 11, 11]
    assert chase.balls_remaining_before.tolist() == [120, 119, 119, 119]
    assert abs(chase.loc[0, "required_run_rate_before"] - 0.55) < 1e-12
    assert pd.isna(chase.loc[2, "required_run_rate_before"])
    changed = context(rows)
    changed.loc[changed.innings.eq(2) & changed.innings_ball_num.eq(1), "runs"] = 0
    changed_features = build_pressure_features(changed)
    changed_first = changed_features[changed_features.innings.eq(2)].iloc[0]
    assert changed_first.team_score_before == chase.loc[0, "team_score_before"]
    assert changed_first.pressure_score == chase.loc[0, "pressure_score"]


def test_ambiguous_extra_only_blocks_clock_until_next_over() -> None:
    rows = [
        (2, 1, 1, 1, 0, "none", True, 0),
        (2, 1, 1, 2, 0, "other", pd.NA, 0),
        (2, 1, 1, 3, 0, "none", True, 0),
        (2, 1, 1, 4, 0, "none", True, 0),
        (2, 1, 1, 5, 0, "none", True, 0),
        (2, 1, 1, 6, 0, "none", True, 0),
        (2, 1, 1, 7, 0, "none", True, 0),
        (2, 1, 2, 8, 0, "none", True, 0),
    ]
    features = build_pressure_features(context(rows))
    assert features.loc[1, "legal_balls_before_phase2"] == 1
    assert pd.isna(features.loc[2, "legal_balls_before_phase2"])
    assert features.loc[7, "legal_balls_before_phase2"] == 6
    assert features.loc[7, "balls_remaining_before"] == 114
    assert not bool(features.loc[7, "legal_clock_conflict"])
    prefix = build_pressure_features(context(rows[:4]))
    assert prefix.pressure_score.tolist() == features.pressure_score.iloc[:4].tolist()


def test_revised_match_does_not_apply_final_target_early() -> None:
    rows = [
        (3, 1, 1, 1, 4, "none", True, 0),
        (3, 2, 1, 1, 1, "none", True, 0),
    ]
    features = build_pressure_features(context(rows, revised=True))
    assert features.pressure_score.isna().all()
    assert features.target_before.isna().all()
    assert features.required_run_rate_before.isna().all()


def test_super_over_uses_separate_regime() -> None:
    rows = [
        (4, 1, 1, 1, 0, "none", True, 0),
        (4, 2, 1, 1, 0, "none", True, 0),
        (4, 3, 1, 1, 0, "none", True, 0),
        (4, 4, 1, 1, 0, "none", True, 0),
    ]
    features = build_pressure_features(context(rows))
    assert features.loc[2, "innings_ball_limit"] == 6
    assert features.loc[2, "pressure_regime"] == "super_over"
    assert features.loc[2, "pressure_score"] == 75
    assert features.loc[3, "target_before"] == 1


def test_legbye_nonzero_runs_does_not_double_count_lower_bound() -> None:
    rows = [
        (5, 1, 1, 1, 1, "legbyes", True, 0),
        (5, 1, 1, 2, 0, "none", True, 0),
    ]
    features = build_pressure_features(context(rows))
    assert features.loc[1, "minimum_team_score_before"] == 1
    assert pd.isna(features.loc[1, "team_score_before"])


def test_open_stable_csv_keeps_new_output_under_unique_name(tmp_path, monkeypatch) -> None:
    stable = tmp_path / "pressure_features.csv"
    stable.write_text("old output\n", encoding="utf-8")

    def locked(*_args) -> None:
        raise PermissionError("file is open")

    monkeypatch.setattr("src.pressure_model.os.replace", locked)
    result = write_feature_csv(pd.DataFrame({"match_id": [1]}),
                               ["match_id"], tmp_path)
    assert result != stable
    assert result.read_text(encoding="utf-8") == "match_id\n1\n"
    assert stable.read_text(encoding="utf-8") == "old output\n"
