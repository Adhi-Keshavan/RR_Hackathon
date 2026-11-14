# Pressure methodology proposal - review before rankings

Phase 1 data preparation is complete. A provisional Phase 2 resource-pressure
score is implemented, with feature coverage and limitations in
`reports/phase2_validation.md`. The extras and revised-target limitations
remain unresolved; no final player rankings should use this score yet.

Phase 3 descriptive batting tables are implemented in `src/batting_analysis.py`.
The provisional high-pressure split, conventional balls-faced rules, player
dismissal attribution, and baseline comparisons are documented in
`reports/phase3_validation.md`. Phase 5 must still address sample size,
opponent/context adjustment, and uncertainty before any ranking.

Phase 4 descriptive bowling tables are implemented in `src/bowling_analysis.py`.
They share Phase 3's pressure cutoffs. Bowler-creditable wickets, legal balls,
economy floors, and exact economy only where supported are documented in
`reports/phase4_validation.md`. Phase 5 must resolve comparison reliability;
the source extras limitation remains material for run-based bowling ratings.

## Official objective and scope

The Round 1 task is to compare players' high-pressure performance with their usual performance, then select ranked top-10 batter and bowler lists. The report must explain pressure, scoring, fair comparisons, evidence, reliability, and limitations. The statement leaves pressure and eligibility definitions to the entrant and supplies no numeric judging weights or minimum sample sizes.

The statement calls the data historical IPL and T20I. The actual pack has 30 matches from two other T20 competitions. Proposed core cohort: competitions whose name is `IPL YYYY` plus matches labelled `T20I`; keep non-IPL T20 out of the core ratings, but show a sensitivity check if appropriate. This is a methodology choice to review, not a stated eligibility rule.

## Pre-ball match state

Order deliveries by `(match_id, innings, innings_ball_num)`. For each ball, calculate all features from **prior** rows in the same innings. Wides and `Overstep` are illegal deliveries; byes and leg-byes are legal. The legality of `other` and `Penalty` is unresolved, so later legal-ball counts in that over are null until the next over boundary restores the clock. Count all dismissals toward wickets lost, while bowler wicket credit excludes run-outs, retire-outs, and obstruction.

Use the innings limit (normally 120 legal balls, 6 in a super over) only after validating the revised-overs fields and interrupted matches. Treat innings 3/4 as a separate super-over regime. Do not compute a chase target from final innings scores, final result, or any post-ball field. `revised_target` may be used only for balls after the revision took effect; the pack does not give a revision timestamp, so revised chases may need exclusion from target-based features.

### Proposed pressure definition

Use an interpretable, pre-registered index based on:

1. **Resource scarcity:** wickets already lost, with rising weight when few remain.
2. **Time scarcity:** legal balls remaining and innings stage; later balls receive more weight only when the outcome remains consequential.
3. **Chase urgency and closeness:** required runs per remaining legal ball relative to a format/phase expectation, if exact pre-ball team runs and the operative target can be verified. In the first innings, compare resource state with an out-of-fold par estimate rather than an unavailable target.
4. **Match importance:** the supplied `knock_out`/stage fields can make a small, declared multiplier. `Others` and `Super 4` are not silently assumed to be knockout matches.

Scale components within format and innings using training data, and define pressure as a fixed upper quantile after checking that selected events actually concentrate late/close/resource-constrained states. The same pre-ball match-state definition should apply to batters and bowlers. Batting and bowling have separate response models. A wicket-heavy collapse or a late but already decided chase should not receive a high score merely because one raw variable is extreme.

**Current blocker:** `extras1_runs` is an extras-type code (1 through 7), and `extras_2_runs` is zero on almost all byes, leg-byes, and wides. Accurate team totals, required run rate, and run-based bowling economy are therefore unsupported for most deliveries. Phase 2 calculates exact values only where the supplied rows prove them; the provisional 0-100 score uses legal-ball stage, wickets, and match importance. First reconcile extras with a data dictionary or another official supplied source before adding run-rate pressure.

## Separate performance models

### Batting

For each eligible striker delivery, estimate expected batter runs and dismissal probability from pre-ball context (format, innings, stage, wickets, opposition and season where support is adequate). Use cross-fitted or chronological out-of-fold predictions. Convert observed minus expected contributions to runs above expectation, with a dismissal cost estimated only from the supplied data. Compare each player's high-pressure rate with their own ordinary-situation rate and with a peer context baseline. Report both pressure effectiveness and pressure uplift; neither one alone should determine rank.

### Bowling

For each bowler delivery, estimate context-expected runs conceded and bowler-creditable wicket probability. Use accurate extras only after their encoding is resolved; until then, any `runs` measure is a **batter-runs-allowed proxy**, not bowler economy. Exclude run-outs, retire-outs, and obstruction from bowler wickets. Compare context-adjusted high-pressure performance with each bowler's ordinary baseline. Normalize by legal deliveries and matches rather than raw wicket totals.

### Reliability and ranking

- Freeze eligibility after examining **exposure distributions**, before looking at ranked names. Require multiple distinct matches and a minimum effective sample size under pressure; report both raw and weighted exposures. Choose and document numeric cutoffs at that point.
- Shrink sparse player effects toward a format/role peer mean using empirical Bayes or a hierarchical model. Use match-cluster bootstrap intervals so multiple balls in one match do not masquerade as independent evidence.
- Rank by a documented combination of adjusted pressure level, uplift from the player's baseline, and conservative uncertainty bound. Give the component values and intervals for each selected player.
- Check rank stability under alternate pressure thresholds, exclusion of revised matches/super overs, and IPL-only versus T20I-only fits. Use a chronological holdout to check that the model generalizes to later seasons. Do not tune weights to favor known players.
- Make uncertainty visible when players are statistically indistinguishable; use a deterministic tie rule declared in the report.

## Exact next steps after review

1. Resolve extras-run encoding and validate team score reconstruction against official match data; otherwise lock the reduced score-free pressure model. Verify whether revised-overs values are legal-ball limits and identify any revision timing information.
2. Freeze the IPL/T20I cohort, super-over policy, revised-match policy, pressure components, and eligibility thresholds. Record the reason for every exclusion.
3. Implement pre-ball innings-state and pressure quantiles, then test wides, no-balls, wickets, over changes, super overs, and revised targets with synthetic cases and real-match spot checks.
4. Build separate cross-fitted batting and bowling expectation models; calculate player baselines, adjusted pressure effects, shrinkage, and match-bootstrap uncertainty.
5. Validate ranking stability by season, competition, pressure threshold, and match exclusions. Inspect data anomalies and compare with a simple transparent baseline.
6. Produce static Matplotlib diagnostics and an auditable candidate table; only then create the two ordered top-10 lists and final methodology document for the Unstop dropdowns/report.
