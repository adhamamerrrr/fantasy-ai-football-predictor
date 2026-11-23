# Fantasy AI Football Predictor

A Streamlit machine-learning project exploring football player scoring rates across the **Premier League, La Liga, Bundesliga, Serie A and Ligue 1**.

## Current dashboard

The current `app.py` is a **retrospective player-season goals-per-90 explorer**, not a live fixture predictor. It provides player rankings, league and position filters, player comparisons and model evaluation. The V6 pipeline uses prior-season features to estimate the following season's goals per 90.

**Important:** The dashboard requires locally generated files in `artifacts/five_leagues_v6/`. Those files are **not currently committed to this repository**. Follow the setup below before running the premium dashboard.

## Setup

Requires Python with compatible versions of the packages in `requirements.txt`.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

### Prepare the real season dataset and train V6

Obtain the upstream `player_seasons.csv` from [m-mahadi/top5-football-dataset](https://github.com/m-mahadi/top5-football-dataset) and place it at `data/master/player_seasons.csv`. Review the upstream dataset's usage terms.

```bash
python -m scripts.prepare_v6
python -m scripts.train_v6
python -m scripts.evaluate_v6_leagues
python -m streamlit run app.py
```

If the preparation script requires a different input path, check its CLI options/source before running. The scripts generate the model, matched-player records, predictions and league evaluations used by the dashboard.

## Model and evaluation

V6 is a scikit-learn **Ridge regression** pipeline using previous-season scoring and playing-time features, with preprocessing for numeric and categorical fields. In the recorded retrospective held-out-player evaluation (329 players):

| Metric | Baseline | V6 Ridge |
| --- | ---: | ---: |
| MAE (goals per 90) | 0.07902 | 0.06797 |
| RMSE (goals per 90) | 0.12302 | 0.09735 |

This corresponds to approximately **13.99% lower MAE** than the baseline in that split. The project also includes leave-one-league-out evaluation code.

**Limitations:** This is retrospective player-season evaluation, **not verified future-fixture prediction accuracy**. The matched-player sample requires sufficient playing time in both seasons; this introduces selection bias. Exploratory dashboard predictions can mix in-sample and held-out records. Observed next-season team and participation information must not be treated as known in advance.

## Repository structure

- `app.py` — Streamlit dashboard
- `fantasy_ai/` — core data and prediction utilities
- `scripts/prepare_v6.py` — prepare player-season features
- `scripts/train_v6.py` — train and evaluate the V6 Ridge model
- `scripts/evaluate_v6_leagues.py` — league-level evaluation
- `scripts/train_v3.py`, `train_v4.py`, `train_v5.py` — earlier experiments
- `tests/` — automated checks
- `data/` — demo data and instructions; upstream real data not redistributed
- `TRAINING_STATUS.md` — earlier experimental workflow and history

Earlier match-level models and the synthetic demo workflow are separate experiments and should not be confused with the current V6 dashboard.

## Tests

```bash
python -m pytest -q
```

## License

MIT for this repository's code. Third-party football datasets remain subject to their respective terms.
