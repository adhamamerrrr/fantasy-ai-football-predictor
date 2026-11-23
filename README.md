# Fantasy AI — Top Five Leagues Goal Predictor

A Python machine-learning application for estimating **goals in a player's next match** and **total goals by the end of a season** across the Premier League, La Liga, Serie A, Bundesliga and Ligue 1.

> **Status:** Working prototype with a training pipeline, evaluation, tests and a Streamlit interface. Real-world accuracy has not been established; sample model artifacts are trained on synthetic data only.

## What it does

- Learns from completed **player-match** records with a time-based train/test split.
- Builds pre-match lagged form features: previous matches, cumulative goals per 90, rolling five-match goals per 90, and previous minutes played.
- Predicts **expected goals for a specified upcoming fixture**, with team, opponent, league and home/away information.
- Projects **end-of-season goal totals** by adding goals already scored to expected goals across a **user-provided remaining fixture schedule**.
- Reports MAE and RMSE on a later-date holdout, compared against a simple mean-prediction baseline.
- Exports match and season predictions from Streamlit.

## Important limitations

- **No bundled real football data.** The optional generated dataset is entirely synthetic, for testing the pipeline only.
- This project **does not fetch live fixtures or player data**; you must supply properly licensed historical records and future fixtures. Predictions are not betting advice.
- Player availability, lineup selection, transfers, injuries and match difficulty are not directly modeled. Fixture rows implicitly assume the named player participates.
- For robust research evaluation, fit category encoders on the training period only, use rolling-origin validation, compare against Poisson and xG baselines, and monitor calibration. The current code uses fixed categorical encodings from the provided history; no future target labels are included in features.
- Expected goals are continuous nonnegative estimates, **not exact goal counts or goal-scoring probabilities**.

## Quick start

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m scripts.make_demo_data
python -m scripts.train --history data/demo_history.csv --cutoff 2025-05-01
python -m pytest -q
streamlit run app.py
```

Upload `data/demo_history.csv` to explore the demo dashboard. For real predictions, upload a real historical CSV and future player fixtures CSV.

## Data contract

Historical player-match CSV columns:

| Column | Example | Meaning |
| --- | --- | --- |
| date | 2025-08-16 | Completed match date |
| player | Player A | Stable player identifier/name |
| team | Club A | Club at that match |
| league | Premier League | One of five supported leagues |
| opponent | Club B | Opposing club |
| home | 1 | 1 home, 0 away |
| minutes | 90 | Minutes played |
| goals | 2 | Actual goals scored |

Upcoming fixtures CSV needs `date,player,team,league,opponent,home`. **All fixture dates must be later than the final historical date.** Each player must have one row for each remaining fixture they are expected to play. This is an input assumption, not a lineup forecast.

## Source layout

- `fantasy_ai/core.py`: validation, lag features, training, evaluation, match/season forecasts
- `scripts/make_demo_data.py`: clearly labeled synthetic demo records
- `scripts/train.py`: CLI training, metrics, holdout predictions and serialized model
- `app.py`: interactive Streamlit dashboard
- `tests/test_core.py`: leakage, evaluation, forecast and schema tests

## Model

The baseline implementation uses `HistGradientBoostingRegressor(loss="poisson")` from scikit-learn, appropriate for nonnegative count-like targets. It predicts expected goals, not a calibrated count distribution. The trained artifact is created locally in `artifacts/` and should not be presented as a real-data-trained model when generated from synthetic demo records.

## License

MIT. Any real-world football data remains subject to its original provider's terms.

## Optional external season data

The optional season-data adapter accepts compatible player-season CSV files. **External datasets are not bundled**; obtain data from providers with appropriate rights, and follow their terms. Player-season summaries cannot substitute for match-level training records.

```bash
# Put the downloaded upstream CSV at data/player_seasons.csv
python -m scripts.evaluate_top5_seasons --csv data/player_seasons.csv
```

This generates `artifacts/top5_season_baseline/{metrics.json,predictions.csv}`. The adapter checks for common column names and fails clearly if the upstream schema differs.

**Scientific limitation:** With only two seasons of aggregate data, this evaluates a *retrospective previous-season goals-per-minute baseline*, scaled by the next season's **observed minutes**. Because those minutes are known only after the season, **it is not a valid prospective end-of-season forecast**. It is useful for data-integration checks and exploratory comparisons only. Genuine forward season prediction requires a minutes/availability forecast and additional historical seasons. Next-match goal prediction requires match-level player data and fixtures; do not fabricate them from season totals.

## Current implementation and training readiness (November 2026)

See [TRAINING_STATUS.md](TRAINING_STATUS.md) for the verified scope and remaining real-data steps. The project now includes `scripts/audit_data.py`, a chronological evaluation followed by full-history model refitting, and tests of the training-to-forecast path. The `artifacts/demo/` outputs are generated **only from synthetic example data** and are not real-world accuracy claims.

```bash
pip install -r requirements.txt
python -m pytest -q
python -m scripts.audit_data --history data/demo_history.csv
python -m scripts.train --history data/demo_history.csv --cutoff 2025-05-01 --data-kind synthetic --out artifacts/demo
streamlit run app.py
```

For real training replace the history with legitimately obtained player-match records and pass `--data-kind real`. The program does not automatically download or license third-party football datasets.

## Real player-match training (Barcelona proof of concept)

The project does **not** include or redistribute third-party datasets. With an
internet connection, run the following from the repository root:

```bash
python -m scripts.download_and_train_barcelona --cutoff 2025-01-01
```

This downloads a source CSV locally, converts its Understat-style player-match
rows, evaluates a chronological holdout, and saves a final refitted model and
metrics under `artifacts/barcelona_real/`. **This is only Barcelona coverage, not
an evaluation of all five leagues.** Review the data provider's terms before
using or redistributing the downloaded data. To train across five leagues,
obtain properly licensed player-match histories covering all five leagues and
run `python -m scripts.train --history data/real_history.csv --data-kind real
--cutoff YYYY-MM-DD --out artifacts/real`.
