# Training status and research scope

> **Current release:** V6 is a real player-**season** scoring-rate exploration pipeline spanning five European leagues. Its trained Ridge model and retrospective evaluation CSVs are included in `artifacts/five_leagues_v6/`. This is separate from the earlier **player-match** experiments documented below.

## V6 season-level results

- Cohort: 1,316 players with at least 450 minutes in **both** the 2024/25 and 2025/26 seasons.
- Held-out player split: 329 test players; Ridge MAE **0.06797** goals/90 versus previous-season baseline **0.07902**.
- RMSE: Ridge **0.09735**, baseline **0.12302**.
- Caveats: retrospective rather than prospective validation; eligibility uses future-season minutes; dashboard rankings include in-sample predictions. Not validated for upcoming matches.

## Historical player-match work

The notes below describe an **earlier, separate match-level workflow**. Statements about absent five-league *player-match* training do not mean the V6 *player-season* model is missing.

---


## What is implemented
- Match-level goal count regression (Poisson gradient boosting)
- Lagged player form; chronological train/test split
- Holdout MAE/RMSE versus constant-mean baseline
- Final model refit on all completed history **after** holdout evaluation
- Next-match goal expectation and fixture-sum season projections
- Data audit, CLI training, Streamlit UI, and automated tests

## What is NOT established
- **No verified five-league real player-match dataset has been downloaded or trained in this build.**
- The bundled `data/demo_history.csv` is synthetic. Any metric produced from it is a demo metric, not evidence of real football forecasting performance.
- Season totals in third-party player-season datasets cannot substitute for player-match labels.
- No live fixture feed or real next-match prediction has been validated.

## How to finish real training
1. Obtain licensed **player-match** data with one row per completed player match, dates, goals, minutes, club, league, opponent and home/away for all five leagues. Keep original source license and attribution with the data; don't publish third-party datasets unless allowed.
2. Normalize into `date,player,team,league,opponent,home,minutes,goals` and check consistent player IDs across transfers. Current adapter uses names, so name collisions must be resolved before production.
3. `python -m scripts.audit_data --history data/real_player_matches.csv` and fix validation issues.
4. Choose a cutoff date that leaves sufficient *future* holdout matches, then `python -m scripts.train --history data/real_player_matches.csv --cutoff YYYY-MM-DD --data-kind real --out artifacts/real`.
5. Check MAE/RMSE against baseline and preferably stronger player-rate baselines. Repeat across multiple rolling time cutoffs; inspect by league and minutes-played groups.
6. Validate the model against unseen future dates, then supply a verified upcoming fixture list and inspect predictions. Forecasts assume players will appear; absence/injury/minutes uncertainty is not yet modeled.

## Reproducible synthetic smoke test
`python -m scripts.train --history data/demo_history.csv --cutoff 2025-05-01 --data-kind synthetic --out artifacts/demo`

Do not describe this project as a trained and validated real-world predictor until steps 1–6 are completed.

### 2026-11-23 update

Added a verified Understat-style player-match converter and a Barcelona-only
real-data download/training entry point. The download/training entry point has
**not** been run in this environment because external network access is
unavailable. There are no verified real-data model metrics yet. The repository
now has 11 passing local tests, including converter tests.
