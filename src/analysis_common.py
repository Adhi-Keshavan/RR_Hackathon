"""Shared, outcome-free pressure cohort for batting and bowling."""

from __future__ import annotations

import pandas as pd

from .data import DataValidationError


def select_pressure_windows(
    features: pd.DataFrame, pressure_quantile: float = 0.80
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Label the upper pressure quantile within IPL/T20I and innings.

    Only pre-ball pressure scores determine the cutoffs. This is a descriptive
    split; Phase 5 must freeze a training-only cutoff for holdout validation.
    """
    if not 0 < pressure_quantile < 1:
        raise ValueError("pressure_quantile must be between 0 and 1")
    required = {"match_id", "innings", "innings_ball_num", "match_format",
                "comp_name", "is_super_over", "pressure_score"}
    missing = required - set(features.columns)
    if missing:
        raise DataValidationError(f"Pressure cohort missing columns: {sorted(missing)}")
    if features.duplicated(["match_id", "innings", "innings_ball_num"]).any():
        raise DataValidationError("Pressure cohort received duplicate delivery keys")
    d = features.copy()
    ipl = d.comp_name.astype("string").str.fullmatch(r"IPL 20\d{2}").fillna(False)
    t20i = d.match_format.eq("T20I")
    d["competition_group"] = pd.Series(pd.NA, index=d.index, dtype="string")
    d.loc[ipl, "competition_group"] = "IPL"
    d.loc[t20i, "competition_group"] = "T20I"
    in_cohort = d.competition_group.notna()
    ordinary_innings = ~d.is_super_over & d.innings.isin([1, 2])
    score_available = d.pressure_score.notna()
    candidate = d.loc[in_cohort & ordinary_innings & score_available].copy()
    if candidate.empty:
        raise DataValidationError("No eligible pressure-scored ordinary innings")
    thresholds = (candidate.groupby(["competition_group", "innings"])
                  .pressure_score.quantile(pressure_quantile)
                  .rename("pressure_cutoff").reset_index())
    candidate = candidate.merge(
        thresholds, on=["competition_group", "innings"],
        how="left", validate="many_to_one",
    )
    candidate["high_pressure"] = candidate.pressure_score.ge(
        candidate.pressure_cutoff
    )
    coverage = {
        "non_ipl_t20_rows_excluded": int((~in_cohort).sum()),
        "super_over_rows_excluded": int((in_cohort & ~ordinary_innings).sum()),
        "unscored_core_ordinary_rows_excluded": int((
            in_cohort & ordinary_innings & ~score_available
        ).sum()),
    }
    return candidate, thresholds, coverage
