# Phase 4 bowling analysis validation

**PASS_DESCRIPTIVE_WITH_RUNS_LIMITATION**. These are descriptive comparisons, not player rankings.

| Measure | Count |
|---|---:|
| Official delivery rows | 200,999 |
| Eligible ordinary IPL/T20I rows analysed | 188,271 |
| High-pressure delivery rows | 37,765 |
| High-pressure legal deliveries | 35,659 |
| High-pressure bowler-creditable wickets | 2,832 |
| Wide/no-ball rows with uncertain runs conceded | 7,778 |
| Players with pressure legal deliveries | 489 |
| Players with both pressure and ordinary legal deliveries | 479 |
| Players with exact pressure economy | 172 |
| Players with exact economy in both windows | 46 |

## Pressure split

Phase 4 uses the exact same upper-20% cutoffs by IPL/T20I and innings
as Phase 3. Cutoffs depend on pre-ball match state, not bowling outcomes.

| Group | Innings | 80th-percentile cutoff |
|---|---:|---:|
| IPL | 1 | 52.40 |
| IPL | 2 | 50.80 |
| T20I | 1 | 54.17 |
| T20I | 2 | 52.00 |

## Bowling definitions

- Legal deliveries exclude wides and no-balls. Stumpings on wides
  still count as bowler wickets. Run-outs, retire-outs, and obstruction
  do not count as bowler wickets.
- Ordinary-ball `runs` are charged to the bowler. Byes and leg-byes
  are not. A wide or no-ball contributes at least one bowler run, but
  the exact extras amount is absent from the supplied fields; exact
  runs conceded and economy are null for any player/window with one.
- `runs_conceded_floor` is a conservative lower bound. `economy_floor`
  is six times that floor divided by legal deliveries. `economy_exact`
  is populated only when every delivery in that player/window has
  known bowler-conceded runs. Neither floor nor batter-runs-allowed
  should be presented as exact economy.
- Wicket rate is bowler wickets per 100 legal deliveries. Player
  baselines compare pressure with ordinary deliveries in the same
  competition group and innings. Wickets above own ordinary rate are
  summed only where both windows have legal-ball exposure. These
  comparisons have no sample-size shrinkage or opponent adjustment.

## Exclusions and limits

- 6,985 non-IPL T20 rows are outside the stated IPL/T20I cohort.
- 26 super-over rows are excluded from ordinary-innings comparisons.
- 5,473 core ordinary rows have no trustworthy Phase 2 pressure score.
- 244 ambiguous `other`/`Penalty` rows are excluded from bowling attribution.

No eligibility threshold, uncertainty interval, or final ranking
is applied in this phase.
