"""Full-pack regression check for Phase 1 data integrity."""

from src.phase1 import build_phase1_dataset
from src.pressure_model import build_pressure_features, phase2_report
from src.batting_analysis import analyze_batting
from src.bowling_analysis import analyze_bowling


def test_official_phase1_pipeline() -> None:
    prepared = build_phase1_dataset()
    report = prepared.report
    assert report["status"] == "PASS_WITH_SOURCE_WARNINGS"
    assert report["deliveries_after_join"] == 200999
    assert len(prepared.deliveries) == len(prepared.tables["ball_by_ball"])
    assert not prepared.deliveries.duplicated(
        ["match_id", "innings", "innings_ball_num"]
    ).any()
    assert report["playing_eleven_repeated_key_rows_removed"] == 40
    assert report["batting_order_conflict_keys"] == 35
    assert report["innings"]["super_over_matches"] == 6
    assert all(
        item["mapping_missing_rows"] == 0
        for item in report["relationships"]["role_coverage"].values()
    )
    features = build_pressure_features(prepared.deliveries)
    phase2 = phase2_report(features)
    assert phase2["delivery_rows"] == 200999
    assert phase2["pressure_score_rows"] == 195504
    assert phase2["exact_required_run_rate_rows"] == 20
    assert phase2["bye_legbye_rows_with_nonzero_runs_field"] == 1
    assert features.pressure_score.dropna().between(0, 100).all()
    batting = analyze_batting(
        features, prepared.tables["competition_team_player_mapping"]
    )
    assert batting.report["analysis_delivery_rows"] == 188271
    assert batting.report["non_striker_dismissals_credited"] == 252
    assert batting.report["players_with_pressure_balls"] == 678
    assert not batting.player_summary.player_id.duplicated().any()
    bowling = analyze_bowling(
        features, prepared.tables["competition_team_player_mapping"]
    )
    assert bowling.report["analysis_delivery_rows"] == 188271
    assert bowling.report["pressure_bowler_wickets"] == 2832
    assert bowling.report["players_with_exact_economy_in_both_windows"] == 46
    assert bowling.thresholds.equals(batting.thresholds)
    assert not bowling.player_summary.player_id.duplicated().any()
