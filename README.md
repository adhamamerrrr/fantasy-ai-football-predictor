# Fantasy AI Football Predictor

A Streamlit machine-learning project exploring football player scoring rates across the **Premier League, La Liga, Bundesliga, Serie A and Ligue 1**.

## Current dashboard

The current `app.py` is a **retrospective player-season goals-per-90 explorer**, not a live fixture predictor. It provides player rankings, league and position filters, player comparisons and model evaluation. The V6 pipeline uses prior-season features to estimate the following season's goals per 90.

**Ready to explore:** The repository includes the V6 model (`v6_ridge_model.joblib`), matched-player table, held-out predictions and league evaluation CSVs under `artifacts/five_leagues_v6/`. The dashboard can use these files immediately after installing dependencies. Raw upstream football data is not included.

## Setup

Requires Python with compatible versions of the packages in `requirements.txt`.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

### Launch the dashboard

```bash
python -m streamlit run app.py
```

### Reproduce the V6 training outputs

Download the upstream `player_seasons.csv` from [m-mahadi/top5-football-dataset](https://github.com/m-mahadi/top5-football-dataset), review its terms, and save it as `artifacts/five_leagues_v6/player_seasons.csv` (the exact path currently expected by `scripts/prepare_v6.py`). From the repository root, run:

```bash
python -m scripts.prepare_v6
python -m scripts.train_v6
python -m scripts.evaluate_v6_leagues
```

These commands regenerate the matched-player data, model and evaluation outputs. The upstream dataset is not redistributed in this repository.

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

## Research integrity

The saved model was fitted on the training portion of the matched cohort for held-out evaluation. Dashboard-wide rankings also score players who were in the training split; therefore **dashboard-wide rankings are not an out-of-sample performance evaluation**. The reported MAE and RMSE refer to the held-out subset only. This project is an analytical prototype, not a live fantasy points optimizer, betting model, or production forecast service.

## Tests

```bash
python -m pytest -q
```

## License

MIT for this repository's code. Third-party football datasets remain subject to their respective terms.
