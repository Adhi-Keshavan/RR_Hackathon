# Phase 3 batting analysis validation

**PASS_DESCRIPTIVE_ONLY**. These are descriptive comparisons, not player rankings.

| Measure | Count |
|---|---:|
| Official delivery rows | 200,999 |
| Eligible ordinary IPL/T20I rows analysed | 188,271 |
| High-pressure delivery rows | 37,765 |
| Ordinary delivery rows | 150,506 |
| High-pressure batting balls faced | 35,807 |
| Players with pressure balls | 678 |
| Players with pressure and ordinary balls | 426 |
| Known player dismissals credited | 10,116 |
| Non-striker dismissals credited | 252 |

## Pressure split

The high-pressure group is the upper 20% of Phase 2 scores within each
competition group (IPL or T20I) and innings (first or second). The
cutoffs depend only on pre-ball match state, not batting outcomes.
They are provisional until the pressure methodology is reviewed.

| Group | Innings | 80th-percentile cutoff |
|---|---:|---:|
| IPL | 1 | 52.40 |
| IPL | 2 | 50.80 |
| T20I | 1 | 54.17 |
| T20I | 2 | 52.00 |

## Metric definitions

- Batting runs are the recorded `runs` field on ordinary and no-ball
  deliveries. Wides, byes, and leg-byes add zero batter runs. Ambiguous
  `other` and `Penalty` rows are excluded.
- Conventional balls faced exclude wides but include no-balls, byes,
  and leg-byes. `legal_balls_faced` is shown separately and excludes
  both wides and no-balls. Strike rate = 100 x runs / balls faced.
- Dismissals are credited to `dismissed_player_id`, including
  non-striker run-outs. A wicket with no dismissed-player ID cannot
  be assigned to a batter.
- The ordinary group is each player's lower-pressure baseline.
  Strike-rate and dismissal-rate deltas compare that player in the
  same competition group and innings. Player summaries also show
  a context-matched runs-above-baseline total where both windows exist.
  These are descriptive and do not adjust for opponent quality or
  sample-size uncertainty; those belong to Phase 5.

## Exclusions and limits

- 6,985 non-IPL T20 rows are outside the stated IPL/T20I cohort.
- 26 super-over rows use a separate Phase 2 regime and are excluded here.
- 5,473 core ordinary rows have no trustworthy Phase 2 pressure score.
- 244 ambiguous extras rows and 0 same-batter identity rows were excluded from batting attribution.
- 1 analysed wickets lack a dismissed-player ID.

No eligibility threshold, shrinkage, uncertainty interval,
or final ranking is applied in this phase. Bowling is reported
separately in `reports/phase4_validation.md`.
