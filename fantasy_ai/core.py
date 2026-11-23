"""Leakage-aware player-match goal forecasting (v2).

Optional historical columns: xg, shots, position. Missing optional fields are
handled, but xG/shots improvements require the importer to preserve them.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

LEAGUES = ["Premier League", "La Liga", "Serie A", "Bundesliga", "Ligue 1"]
REQUIRED = ["date", "player", "team", "league", "goals", "minutes", "home", "opponent"]
CATEGORICAL = ["league_code", "team_code", "opponent_code", "position_code"]
FEATURES = ["prior_matches", "prior_goals_per90", "last5_goals_per90", "last5_minutes",
            "prior_minutes", "prior_xg_per90", "last5_xg_per90", "last5_shots_per90",
            "home", *CATEGORICAL]


def validate(df: pd.DataFrame) -> pd.DataFrame:
    missing = set(REQUIRED) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="raise").dt.normalize()
    if out[["player", "team", "league", "opponent"]].isna().any().any():
        raise ValueError("Missing player/team/league/opponent")
    for c in ["goals", "minutes", "home"]:
        out[c] = pd.to_numeric(out[c], errors="raise")
    if out[["goals", "minutes", "home"]].isna().any().any():
        raise ValueError("Missing goals/minutes/home")
    if (out.goals < 0).any() or ((out.minutes < 0) | (out.minutes > 130)).any():
        raise ValueError("Invalid goals or minutes")
    if not out.home.isin([0, 1]).all() or not out.league.isin(LEAGUES).all():
        raise ValueError("Invalid home or league")
    if out.duplicated(["date", "player", "team"]).any():
        raise ValueError("Duplicate player/team/date records")
    for c in ["xg", "shots"]:
        if c not in out:
            out[c] = 0.0
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).clip(lower=0)
    if "position" not in out:
        out["position"] = "Unknown"
    out["position"] = out["position"].fillna("Unknown").astype(str)
    return out.sort_values(["date", "player", "team"]).reset_index(drop=True)


def _codes(series: pd.Series, vocab: list[str]) -> pd.Series:
    mapping = {s: i + 1 for i, s in enumerate(vocab)}
    return series.map(mapping).fillna(0).astype(int)


def _history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute features using strictly earlier calendar dates, never same-day rows."""
    # Aggregate a player's rows per day, so simultaneous fixtures cannot leak.
    daily = (df.groupby(["player", "date"], sort=False, as_index=False)
             .agg(goals=("goals", "sum"), minutes=("minutes", "sum"),
                  xg=("xg", "sum"), shots=("shots", "sum"), matches=("goals", "size")))
    daily = daily.sort_values(["player", "date"]).reset_index(drop=True)
    group = daily.groupby("player", sort=False)
    for c in ["goals", "minutes", "xg", "shots", "matches"]:
        daily[f"prior_{c}_sum"] = group[c].transform(lambda s: s.shift().fillna(0).cumsum())
        daily[f"last5_{c}_sum"] = group[c].transform(lambda s: s.shift().rolling(5, min_periods=1).sum()).fillna(0)
    daily["prior_matches"] = daily["prior_matches_sum"]
    daily["prior_minutes"] = daily["prior_minutes_sum"] / daily["prior_matches"].replace(0, np.nan)
    daily["last5_minutes"] = daily["last5_minutes_sum"] / daily["last5_matches_sum"].replace(0, np.nan)
    for name, num, den in [
        ("prior_goals_per90", "prior_goals_sum", "prior_minutes_sum"),
        ("last5_goals_per90", "last5_goals_sum", "last5_minutes_sum"),
        ("prior_xg_per90", "prior_xg_sum", "prior_minutes_sum"),
        ("last5_xg_per90", "last5_xg_sum", "last5_minutes_sum"),
        ("last5_shots_per90", "last5_shots_sum", "last5_minutes_sum"),
    ]:
        daily[name] = 90 * daily[num] / daily[den].replace(0, np.nan)
    derived = [c for c in FEATURES if c not in ["home", *CATEGORICAL]]
    return df.merge(daily[["player", "date", *derived]], on=["player", "date"], how="left").fillna({c: 0 for c in derived})


def engineer(df: pd.DataFrame, vocab: dict | None = None):
    frame = _history_features(validate(df))
    if vocab is None:
        vocab = {c: sorted(frame[c].unique().tolist()) for c in ["league", "team", "opponent", "position"]}
    for c in ["league", "team", "opponent", "position"]:
        frame[f"{c}_code"] = _codes(frame[c], vocab[c])
    return frame, vocab


def _model():
    return HistGradientBoostingRegressor(
        loss="poisson", max_iter=160, max_leaf_nodes=12,
        l2_regularization=2, categorical_features=[FEATURES.index(c) for c in CATEGORICAL],
        random_state=42)


def fit_and_evaluate(df: pd.DataFrame, cutoff: str):
    frame = _history_features(validate(df))
    cutoff = pd.Timestamp(cutoff)
    train = frame[frame.date < cutoff].copy()
    test = frame[frame.date >= cutoff].copy()
    if len(train) < 20 or len(test) < 5:
        raise ValueError("Need >=20 training and >=5 later test rows; adjust cutoff")
    # Fit category vocabulary on training rows only.
    vocab = {c: sorted(train[c].unique().tolist()) for c in ["league", "team", "opponent", "position"]}
    if any(len(v) >= 255 for v in vocab.values()):
        raise ValueError("Too many categorical values for HistGradientBoosting; use an encoding pipeline")
    for data in [train, test]:
        for c in vocab:
            data[f"{c}_code"] = _codes(data[c], vocab[c])
    model = _model().fit(train[FEATURES], train.goals)
    baseline = DummyRegressor(strategy="mean").fit(train[FEATURES], train.goals)
    predictions = np.maximum(0, model.predict(test[FEATURES]))
    base_preds = np.maximum(0, baseline.predict(test[FEATURES]))
    metrics = {
        "mae": float(mean_absolute_error(test.goals, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(test.goals, predictions))),
        "baseline_mae": float(mean_absolute_error(test.goals, base_preds)),
        "baseline_rmse": float(np.sqrt(mean_squared_error(test.goals, base_preds))),
        "train_rows": int(len(train)), "test_rows": int(len(test)),
        "cutoff": str(cutoff.date()),
        "actual_goals": float(test.goals.sum()),
        "predicted_goals": float(predictions.sum()),
        "historical_xg_available": bool(df.get("xg", pd.Series(dtype=float)).notna().any()),
        "historical_shots_available": bool(df.get("shots", pd.Series(dtype=float)).notna().any()),
    }
    return model, vocab, metrics, test.assign(predicted_goals=predictions)


def upcoming_features(history: pd.DataFrame, fixtures: pd.DataFrame, vocab: dict) -> pd.DataFrame:
    hist = validate(history)
    required = {"date", "player", "team", "league", "opponent", "home"}
    if not required.issubset(fixtures.columns):
        raise ValueError(f"Fixtures require {sorted(required)}")
    future = fixtures.copy().reset_index(drop=True)
    future["date"] = pd.to_datetime(future.date).dt.normalize()
    if not future.home.isin([0, 1]).all() or not future.league.isin(LEAGUES).all():
        raise ValueError("Invalid fixture home or league")
    if len(future) and (future.date <= hist.date.max()).any():
        raise ValueError("Fixture dates must be after all historical results")
    rows = []
    for _, fixture in future.iterrows():
        dummy = fixture.to_dict()
        dummy.update(goals=0, minutes=0, xg=0, shots=0)
        if "position" not in dummy:
            previous = hist[hist.player == fixture.player]
            dummy["position"] = previous.iloc[-1].position if len(previous) else "Unknown"
        combined = pd.concat([hist, pd.DataFrame([dummy])], ignore_index=True)
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
    predicted = forecast(model, history, remaining_fixtures, vocab)
    actual = validate(history)
    actual = actual[actual.date >= pd.Timestamp(season_start)].groupby("player").goals.sum().rename("goals_so_far")
    future = predicted.groupby("player").expected_goals.sum().rename("expected_remaining_goals")
    result = pd.concat([actual, future], axis=1).fillna(0)
    result["projected_season_goals"] = result.goals_so_far + result.expected_remaining_goals
    return result.sort_values("projected_season_goals", ascending=False).reset_index()


def fit_final_model(df: pd.DataFrame):
    frame, vocab = engineer(df)
    if len(frame) < 20:
        raise ValueError("Need at least 20 completed player-match rows")
    if any(len(v) >= 255 for v in vocab.values()):
        raise ValueError("Too many categorical values for HistGradientBoosting")
    return _model().fit(frame[FEATURES], frame.goals), vocab
