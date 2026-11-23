"""Feature engineering and match/remaining-season goal forecasts.

Input: one row per completed player-match, sorted chronologically.
No future-match outcomes or post-match stats are used as predictors.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

LEAGUES = ["Premier League", "La Liga", "Serie A", "Bundesliga", "Ligue 1"]
REQUIRED = ["date", "player", "team", "league", "goals", "minutes", "home", "opponent"]
FEATURES = ["prior_matches", "prior_goals_per90", "last5_goals_per90", "last5_minutes", "prior_minutes", "home", "league_code", "opponent_code", "team_code"]


def validate(df: pd.DataFrame) -> pd.DataFrame:
    missing = set(REQUIRED) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="raise")
    if out[list(["player", "team", "league", "opponent"])].isna().any().any():
        raise ValueError("Missing player/team/league/opponent")
    if out[["goals", "minutes", "home"]].isna().any().any():
        raise ValueError("Missing goals/minutes/home")
    if (out["goals"] < 0).any() or ((out["minutes"] < 0) | (out["minutes"] > 130)).any():
        raise ValueError("Invalid goals or minutes")
    if not out["home"].isin([0, 1]).all():
        raise ValueError("home must be 0 or 1")
    if not out["league"].isin(LEAGUES).all():
        raise ValueError("Only the five supported leagues are allowed")
    if out.duplicated(["date", "player", "team"]).any():
        raise ValueError("Duplicate player/team/date records")
    return out.sort_values(["date", "player", "team"]).reset_index(drop=True)


def _codes(series: pd.Series, vocab: list[str]) -> pd.Series:
    mapping = {s: i + 1 for i, s in enumerate(vocab)}
    return series.map(mapping).fillna(0).astype(int)


def engineer(df: pd.DataFrame, vocab: dict | None = None):
    """Lag features, never incorporating a row's target into its features."""
    df = validate(df)
    if vocab is None:
        vocab = {col: sorted(df[col].unique().tolist()) for col in ["league", "team", "opponent"]}
    group = df.groupby("player", sort=False)
    df["prior_matches"] = group.cumcount()
    prior_goals = group["goals"].transform(lambda s: s.shift().expanding().sum())
    prior_mins = group["minutes"].transform(lambda s: s.shift().expanding().sum())
    df["prior_goals_per90"] = (90 * prior_goals / prior_mins.replace(0, np.nan)).fillna(0)
    df["prior_minutes"] = group["minutes"].transform(lambda s: s.shift().expanding().mean()).fillna(0)
    last5goals = group["goals"].transform(lambda s: s.shift().rolling(5, min_periods=1).sum())
    last5mins = group["minutes"].transform(lambda s: s.shift().rolling(5, min_periods=1).sum())
    df["last5_goals_per90"] = (90 * last5goals / last5mins.replace(0, np.nan)).fillna(0)
    df["last5_minutes"] = group["minutes"].transform(lambda s: s.shift().rolling(5, min_periods=1).mean()).fillna(0)
    for col in ["league", "team", "opponent"]:
        df[f"{col}_code"] = _codes(df[col], vocab[col])
    return df, vocab


def fit_and_evaluate(df: pd.DataFrame, cutoff: str):
    """Train before cutoff, evaluate on later dates; no random split."""
    frame, vocab = engineer(df)
    cutoff = pd.Timestamp(cutoff)
    train = frame[frame.date < cutoff]
    test = frame[frame.date >= cutoff]
    if len(train) < 20 or len(test) < 5:
        raise ValueError("Need >=20 training and >=5 later test rows; adjust cutoff")
    model = HistGradientBoostingRegressor(loss="poisson", max_iter=130, max_leaf_nodes=12, l2_regularization=2, random_state=42)
    model.fit(train[FEATURES], train.goals)
    baseline = DummyRegressor(strategy="mean").fit(train[FEATURES], train.goals)
    predictions = np.maximum(0, model.predict(test[FEATURES]))
    base_preds = np.maximum(0, baseline.predict(test[FEATURES]))
    metrics = {"mae": float(mean_absolute_error(test.goals, predictions)), "rmse": float(np.sqrt(mean_squared_error(test.goals, predictions))), "baseline_mae": float(mean_absolute_error(test.goals, base_preds)), "baseline_rmse": float(np.sqrt(mean_squared_error(test.goals, base_preds))), "train_rows": len(train), "test_rows": len(test), "cutoff": str(cutoff.date())}
    return model, vocab, metrics, test.assign(predicted_goals=predictions)


def upcoming_features(history: pd.DataFrame, fixtures: pd.DataFrame, vocab: dict) -> pd.DataFrame:
    """One prediction row per upcoming player fixture using historical form only."""
    hist = validate(history)
    required = {"date", "player", "team", "league", "opponent", "home"}
    if not required.issubset(fixtures.columns):
        raise ValueError(f"Fixtures require {sorted(required)}")
    future = fixtures.copy()
    future["date"] = pd.to_datetime(future["date"])
    if not future["home"].isin([0, 1]).all() or not future["league"].isin(LEAGUES).all():
        raise ValueError("Invalid fixture home or league")
    if len(future) and future.date.min() <= hist.date.max():
        raise ValueError("Fixture dates must be after all historical results")
    future["goals"] = 0
    future["minutes"] = 0
    # One future row at a time prevents earlier future fixtures from contaminating later feature histories.
    rows = []
    for _, fixture in future.iterrows():
        combined = pd.concat([hist, pd.DataFrame([fixture])], ignore_index=True)
        engineered, _ = engineer(combined, vocab)
        match = engineered[(engineered.date == fixture.date) & (engineered.player == fixture.player) & (engineered.team == fixture.team)]
        if len(match) != 1:
            raise ValueError("Ambiguous fixture row")
        rows.append(match.iloc[0])
    return pd.DataFrame(rows).reset_index(drop=True) if rows else pd.DataFrame(columns=FEATURES)


def forecast(model, history: pd.DataFrame, fixtures: pd.DataFrame, vocab: dict):
    features = upcoming_features(history, fixtures, vocab)
    result = fixtures.copy().reset_index(drop=True)
    result["expected_goals"] = np.maximum(0, model.predict(features[FEATURES])) if len(features) else []
    return result


def season_forecast(model, history: pd.DataFrame, remaining_fixtures: pd.DataFrame, vocab: dict, season_start: str):
    """Sum expected goals over supplied remaining fixtures; no schedule is invented."""
    predicted = forecast(model, history, remaining_fixtures, vocab)
    actual = validate(history)
    actual = actual[actual.date >= pd.Timestamp(season_start)].groupby("player").goals.sum().rename("goals_so_far")
    future = predicted.groupby("player").expected_goals.sum().rename("expected_remaining_goals")
    result = pd.concat([actual, future], axis=1).fillna(0)
    result["projected_season_goals"] = result.goals_so_far + result.expected_remaining_goals
    return result.sort_values("projected_season_goals", ascending=False).reset_index()


def fit_final_model(df: pd.DataFrame):
    """Fit the deployment model on all observed completed matches.

    Call only after evaluating a separate chronological holdout. Historical
    feature rows contain lagged values, never their own match outcome.
    """
    frame, vocab = engineer(df)
    if len(frame) < 20:
        raise ValueError('Need at least 20 completed player-match rows')
    model = HistGradientBoostingRegressor(loss='poisson', max_iter=130,
        max_leaf_nodes=12, l2_regularization=2, random_state=42)
    model.fit(frame[FEATURES], frame.goals)
    return model, vocab
