# SupeRR Selector Season 2 - pressure performance

**Phases 1 through 4 are implemented through descriptive batting and bowling comparisons.** This local Python project audits the official Round 1 pack and prepares delivery records for reproducible pressure analysis. **No player rankings have been produced.** The Phase 2 score omits run-rate pressure because the supplied extras fields do not support exact innings scores for most deliveries. Review [Phase 2](reports/phase2_validation.md), [Phase 3](reports/phase3_validation.md), and [Phase 4](reports/phase4_validation.md) limitations before interpreting player results.

## Official task

Rank 10 batters and 10 bowlers in order of performance under pressure, compared with their usual levels. The final Unstop submission uses 20 dropdown selections and a methodology document. The [official statement](6af3166e7b922_data_pack/Round%201%20Problem%20Statement%20_%20SupeRR%20Selector%20Season%202.docx.pdf) gives no fixed eligibility threshold or numeric judging rubric.

## Project layout

| Path | Purpose |
|---|---|
| `data/` | Supplied ball-by-ball and match CSVs; ignored by Git. |
| `6af3166e7b922_data_pack/` | Remaining official CSVs and statement; ignored by Git. |
| `src/data.py` | Discover six CSVs (including zip fallback), validate schemas and keys, safely join match/competition data. |
| `src/preprocessing.py` | Normalize extras and dismissals, compute safe pre-ball counters, normalize duplicate playing-XI records. |
| `src/phase1.py` | Build the validated Phase 1 view and run structural innings, team, and player-ID checks. |
| `src/pressure_model.py` | Build causal pre-ball state, exact score/target/RRR where supported, and a provisional 0-100 pressure score. |
| `src/batting_analysis.py` | Split IPL/T20I ordinary innings by provisional pressure quantiles; calculate batting metrics and player baselines without ranking. |
| `src/analysis_common.py` | Use one outcome-free IPL/T20I pressure cohort and set of cutoffs for batting and bowling. |
| `src/bowling_analysis.py` | Calculate bowler wickets, legal deliveries, cautious runs conceded/economy, and ordinary baselines without ranking. |
| `src/audit.py` | Reproducible inventory and relationship audit. |
| `notebooks/exploration.ipynb` | First-pass exploration without player rankings. |
| `tests/` | Unit and full-data cardinality tests. |
| `reports/data_audit.md` / `reports/audit.json` | Human and machine-readable audits. |
| `reports/phase1_validation.md` / `.json` | PASS/FAIL evidence and retained source warnings for Phase 1. |
| `reports/phase2_validation.md` / `.json` | Pressure formula, feature coverage, and limitations. |
| `reports/phase3_validation.md` / `.json` | Batting definitions, pressure thresholds, exclusions, and coverage. |
| `reports/phase4_validation.md` / `.json` | Bowling definitions, exact-economy coverage, exclusions, and limitations. |
| `outputs/pressure_features.csv` | Derived per-delivery Phase 2 features; ignored by Git. |
| `outputs/batting_*.csv` | Descriptive player/context tables from Phase 3; ignored by Git. |
| `outputs/bowling_*.csv` | Descriptive player/context tables from Phase 4; ignored by Git. |
| `reports/methodology_plan.md` | Proposed pressure models and implementation roadmap. |
| `outputs/` | Reserved for reviewed final results; current CSV files are empty pre-existing placeholders. |

The pre-existing zero-byte `src/ranking.py` and `reports/methodology.pdf` remain placeholders for later phases.

## Set up and reproduce

From this folder, create a Python 3.11+ environment and install the project stack:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m src.audit > reports\audit.json
.venv\Scripts\python -m src.audit --format markdown > reports\data_audit.md
.venv\Scripts\python -m src.phase1
.venv\Scripts\python -m src.pressure_model
.venv\Scripts\python -m src.batting_analysis
.venv\Scripts\python -m src.bowling_analysis
.venv\Scripts\python -m pytest -q
```

The loader finds individual CSVs in `data/`, in the extracted `*_data_pack/` folder, or in a `*data_pack.zip` archive. It reads but never rewrites the official files.

`python -m src.phase1` is the Phase 1 check from the project root. If your terminal is already in `src/`, `python phase1.py` works too. It exits with code 1 on a structural failure and writes [phase1_validation.md](reports/phase1_validation.md) when successful. The current pack reports `PASS_WITH_SOURCE_WARNINGS`: 200,999 delivery rows preserved, all match/competition/team/player foreign keys valid, all innings numbering valid, and no exact duplicate rows.

`python -m src.pressure_model` runs Phase 2 from the project root; `python pressure_model.py` works from `src/`. It writes [phase2_validation.md](reports/phase2_validation.md) and a local [pressure_features.csv](outputs/pressure_features.csv). If that CSV is open in another Windows application, the new run writes a timestamped CSV in `outputs/` and names it in the report instead of failing. The current pack reports `PASS_WITH_LIMITED_SCORE_DATA`: 195,504 deliveries scored out of 200,999.

`python -m src.batting_analysis` runs Phase 3 from the project root; `python batting_analysis.py` works from `src/`. It regenerates Phase 1 and 2 features in memory, then writes [phase3_validation.md](reports/phase3_validation.md) and three descriptive batting CSVs under `outputs/`. A locked CSV is handled with a timestamped alternative filename. The report names the files produced by that run. No final top-10 list is produced.

`python -m src.bowling_analysis` runs Phase 4 from the project root; `python bowling_analysis.py` works from `src/`. It uses the same pressure cutoffs and writes [phase4_validation.md](reports/phase4_validation.md) plus three descriptive bowling CSVs. Exact bowling economy is null in windows containing a wide or no-ball because the supplied extras fields do not provide exact runs conceded. The output also includes a conservative economy floor, clearly labelled as a lower bound. No final top-10 list is produced.

## Current findings

- Six CSVs contain 200,999 deliveries across 867 matches with deliveries. The full [data audit](reports/data_audit.md) gives every column, inferred dtype, null count, key check, competition coverage, and date range.
- `extras1_runs` is a category code, despite its name. Exact extras and team totals are not recoverable from the current columns without further verification. Required run rate is therefore **not yet computed**.
- The playing-XI table repeats 40 `(match_id, team_id, player_id)` keys. Joining it directly would multiply some delivery rows; preprocessing collapses those keys and flags conflicting batting orders.
- One delivery has the same batter ID at both ends, one retirement wicket lacks a dismissed-player ID, and five unplayed roster entries lack competition-player mapping. These are flagged, not silently dropped. All delivery player roles map to valid IDs and names.
- The pack contains two non-IPL T20 competitions (30 matches, 6,985 deliveries). The official statement describes IPL and T20I; the proposed core cohort excludes those 30 matches pending review.
- The statement mentions player data, but no standalone `players.csv` is present; names and IDs come from the supplied mapping and delivery files.
- Wides and oversteps do not consume a legal ball. `other` and `Penalty` remain legally ambiguous, so later balls in the same over may have an unknown pre-ball clock; the next over boundary restores it. Super overs use a separate scoring regime, and revised matches are left unscored.
- Phase 2 uses only information available before each delivery. Over changes recover the clock after an ambiguous extra, but the ambiguous ball itself is not reclassified using future rows. Revised matches are left unscored because the revision timing is unavailable. Exact required run rate is available for only 20 delivery rows, so the provisional pressure formula uses time, wickets, and match importance only.

No raw pack, data CSV, ranking CSV, or PDF is committed by the supplied `.gitignore`. No commit or push was made.
