# Phase 2 pressure feature validation

**PASS_WITH_LIMITED_SCORE_DATA**. Model: `resource_only_v1`.

Feature file: `outputs/pressure_features_20261113T144313963102Z_15132.csv`.
The usual `outputs/pressure_features.csv` was open or protected, so this run used a new filename.

| Measure | Rows |
|---|---:|
| Delivery rows preserved | 200,999 |
| 0-100 pressure score available | 195,504 |
| Pressure score unavailable | 5,495 |
| Exact pre-ball team score available | 31,933 |
| Exact target available | 231 |
| Exact required run rate available | 20 |
| Revised-match rows (excluded from score) | 4,433 |
| Super-over rows (separate score regime) | 38 |
| Pre-ball clocks recovered at later over boundaries | 8,498 |
| Legality still unresolved | 265 |
| Rows at/after a contradictory completed over | 164 |

## Definition

All inputs are measured before the current delivery. For an ordinary
innings, let `t = legal_balls_before / 120`, `w = wickets_before / 10`,
and `k = knock_out` (0 or 1). The provisional 0-100 score is
`100 * (0.40t + 0.35w + 0.20tw + 0.05k)`. A super over uses a six-ball
and two-wicket clock, then `75 + 25 * base`; it is marked as a separate
regime. An over transition resets the clock using only prior events;
individual `other`/`Penalty` deliveries are never retroactively
classified. Revised matches and contradictory/unknown legal clocks have a
null score. Quantiles (min, 25th, median, 75th, max):
`[0.0, 13.97, 29.67, 48.33, 94.02]`.

## Score and target limitations

`team_score_before` is exact only until the first extras event in an
innings. `minimum_team_score_before` is a conservative lower bound
using the larger of the `runs` field and one for a wide, no-ball,
bye, or leg-bye. One leg-bye row has a nonzero `runs` field; this
rule avoids double counting it. The lower bound is not treated
as exact. The target is calculated only for a chase whose previous
innings had no extras and whose match has no revised-innings fields.
A revised target is never applied to balls before an unknown revision
time. Required run rate is populated only if target, score, and
balls remaining are all exact. It is **not** used in this provisional
pressure score because its coverage is too sparse for fair comparison.

This score measures resource and stage pressure, not match closeness.
Do not use it for final player rankings until this limitation and the
pressure methodology are reviewed.
