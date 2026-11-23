
"""Fantasy AI V5: player-ID-safe, pre-match goal prediction."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_poisson_deviance,
)

from fantasy_ai.core import FEATURES, CATEGORICAL, validate
from scripts.import_understat_matches import convert


SOURCE = Path(
    "artifacts/barcelona_v2_real_features/"
    "source_understat_player_matches.csv"
)
OUT = Path("artifacts/barcelona_v5")

TRAIN_END = pd.Timestamp("2025-06-01")
VALID_START = pd.Timestamp("2025-08-01")
TEST_START = pd.Timestamp("2026-01-01")

CATS = ["league", "team", "opponent", "position"]

HISTORY_COLS = [
    "prior_matches",
    "prior_goals_per90",
    "last5_goals_per90",
    "last5_minutes",
    "prior_minutes",
    "prior_xg_per90",
    "last5_xg_per90",
    "last5_shots_per90",
]


def evaluate(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.maximum(
        np.asarray(predicted, dtype=float), 1e-9
    )

    return {
        "mae": float(
            mean_absolute_error(actual, predicted)
        ),
        "rmse": float(
            np.sqrt(mean_squared_error(actual, predicted))
        ),
        "poisson_deviance": float(
            mean_poisson_deviance(actual, predicted)
        ),
        "actual_goals": float(actual.sum()),
        "predicted_goals": float(predicted.sum()),
    }


def ranking_metrics(actual, predicted):
    data = pd.DataFrame({
        "actual": np.asarray(actual, dtype=float),
        "predicted": np.asarray(predicted, dtype=float),
    })

    correlation = data["actual"].corr(
        data["predicted"], method="spearman"
    )

    top20 = data.nlargest(20, "predicted")

    return {
        "spearman": (
            float(correlation)
            if pd.notna(correlation)
            else None
        ),
        "top20_actual_goals": int(
            top20["actual"].sum()
        ),
    }


def prepare_data():
    raw = pd.read_csv(SOURCE)

    if "player_id" not in raw.columns:
        raise ValueError("Source is missing player_id")

    if raw["player_id"].isna().any():
        raise ValueError("Missing player IDs")

    if raw.duplicated(["game", "player_id"]).any():
        raise ValueError("Duplicate game/player IDs")

    converted = convert(raw)

    # The importer does not retain player_id.
    # Restore it through the original row identity.
    raw_ids = raw[[
        "game", "player", "team", "player_id"
    ]].copy()

    raw_ids["date"] = pd.to_datetime(
        raw_ids["game"].astype(str).str[:10]
    ).dt.normalize()

    if raw_ids.duplicated(
        ["date", "player", "team"]
    ).any():
        raise ValueError(
            "Ambiguous name/team/date mapping"
        )

    data = converted.merge(
        raw_ids[[
            "date", "player", "team", "player_id"
        ]],
        on=["date", "player", "team"],
        how="left",
        validate="one_to_one",
    )

    if data.player_id.isna().any():
        raise ValueError("Player ID merge failed")

    data = validate(data)

    data = data.sort_values(
        ["player_id", "date", "team"]
    ).reset_index(drop=True)

    return data


def make_features(data):
    daily = (
        data.groupby(
            ["player_id", "date"],
            as_index=False
        )
        .agg(
            goals=("goals", "sum"),
            minutes=("minutes", "sum"),
            xg=("xg", "sum"),
            shots=("shots", "sum"),
            matches=("goals", "size"),
        )
        .sort_values(["player_id", "date"])
        .reset_index(drop=True)
    )

    group = daily.groupby(
        "player_id", sort=False
    )

    for col in [
        "goals", "minutes", "xg",
        "shots", "matches"
    ]:
        daily[f"prior_{col}_sum"] = (
            group[col].transform(
                lambda s: s.shift().fillna(0).cumsum()
            )
        )

        daily[f"last5_{col}_sum"] = (
            group[col].transform(
                lambda s: (
                    s.shift()
                    .rolling(5, min_periods=1)
                    .sum()
                )
            ).fillna(0)
        )

    daily["prior_matches"] = (
        daily["prior_matches_sum"]
    )

    daily["prior_minutes"] = (
        daily["prior_minutes_sum"]
        / daily["prior_matches"].replace(
            0, np.nan
        )
    )

    daily["last5_minutes"] = (
        daily["last5_minutes_sum"]
        / daily["last5_matches_sum"].replace(
            0, np.nan
        )
    )

    ratios = [
        (
            "prior_goals_per90",
            "prior_goals_sum",
            "prior_minutes_sum",
        ),
        (
            "last5_goals_per90",
            "last5_goals_sum",
            "last5_minutes_sum",
        ),
        (
            "prior_xg_per90",
            "prior_xg_sum",
            "prior_minutes_sum",
        ),
        (
            "last5_xg_per90",
            "last5_xg_sum",
            "last5_minutes_sum",
        ),
        (
            "last5_shots_per90",
            "last5_shots_sum",
            "last5_minutes_sum",
        ),
    ]

    for name, numerator, denominator in ratios:
        daily[name] = (
            90 * daily[numerator]
            / daily[denominator].replace(
                0, np.nan
            )
        )

    frame = data.merge(
        daily[[
            "player_id", "date",
            *HISTORY_COLS
        ]],
        on=["player_id", "date"],
        how="left",
        validate="many_to_one",
    )

    frame[HISTORY_COLS] = (
        frame[HISTORY_COLS].fillna(0)
    )

    # Use only a previously observed position.
    # Sub is not treated as a playing position.
    daily_position = (
        data.assign(
            known_position=data["position"].where(
                ~data["position"].isin(
                    ["Sub", "Unknown", ""]
                )
            )
        )
        .groupby(
            ["player_id", "date"],
            as_index=False
        )
        .agg(
            known_position=(
                "known_position", "first"
            )
        )
        .sort_values(["player_id", "date"])
    )

    daily_position["historical_position"] = (
        daily_position.groupby(
            "player_id"
        )["known_position"].transform(
            lambda s: s.ffill().shift(1)
        )
    )

    frame = frame.merge(
        daily_position[[
            "player_id",
            "date",
            "historical_position",
        ]],
        on=["player_id", "date"],
        how="left",
        validate="many_to_one",
    )

    frame["position"] = (
        frame["historical_position"]
        .fillna("Unknown")
    )

    return frame.drop(
        columns="historical_position"
    )


def encode_categories(train, valid, test):
    vocab = {
        col: sorted(
            train[col].astype(str).unique().tolist()
        )
        for col in CATS
    }

    if any(
        len(values) >= 255
        for values in vocab.values()
    ):
        raise ValueError(
            "Too many categories for this model"
        )

    for frame in (train, valid, test):
        for col in CATS:
            mapping = {
                value: i + 1
                for i, value in enumerate(
                    vocab[col]
                )
            }

            frame[f"{col}_code"] = (
                frame[col].astype(str)
                .map(mapping)
                .fillna(0)
                .astype(int)
            )

    return vocab


def create_model(config):
    return HistGradientBoostingRegressor(
        loss="poisson",
        max_iter=config["max_iter"],
        max_leaf_nodes=config["max_leaf_nodes"],
        learning_rate=config["learning_rate"],
        l2_regularization=config["l2"],
        categorical_features=[
            FEATURES.index(col)
            for col in CATEGORICAL
        ],
        random_state=42,
    )


def main():
    OUT.mkdir(
        parents=True, exist_ok=True
    )

    data = prepare_data()
    frame = make_features(data)

    train = frame[
        frame.date < TRAIN_END
    ].copy()

    valid = frame[
        (frame.date >= VALID_START)
        & (frame.date < TEST_START)
    ].copy()

    test = frame[
        frame.date >= TEST_START
    ].copy()

    print("Training:", len(train))
    print("Validation:", len(valid))
    print("Testing:", len(test))

    if (len(train), len(valid), len(test)) != (
        1196, 569, 632
    ):
        raise ValueError(
            "Unexpected split sizes"
        )

    vocab = encode_categories(
        train, valid, test
    )

    configs = [
        {
            "name": "conservative",
            "max_iter": 100,
            "max_leaf_nodes": 8,
            "learning_rate": 0.05,
            "l2": 5,
        },
        {
            "name": "balanced",
            "max_iter": 150,
            "max_leaf_nodes": 12,
            "learning_rate": 0.05,
            "l2": 2,
        },
        {
            "name": "flexible",
            "max_iter": 200,
            "max_leaf_nodes": 16,
            "learning_rate": 0.05,
            "l2": 2,
        },
        {
            "name": "fast_learning",
            "max_iter": 130,
            "max_leaf_nodes": 12,
            "learning_rate": 0.1,
            "l2": 2,
        },
    ]

    trials = []
    best = None

    for config in configs:
        model = create_model(config)

        model.fit(
            train[FEATURES],
            train.goals,
        )

        pred = np.maximum(
            model.predict(
                valid[FEATURES]
            ),
            1e-9,
        )

        metrics = evaluate(
            valid.goals, pred
        )

        ranking = ranking_metrics(
            valid.goals, pred
        )

        trial = {
            "config": config,
            "validation": metrics,
            "ranking": ranking,
        }

        trials.append(trial)

        print(
            config["name"],
            "validation Poisson:",
            round(
                metrics["poisson_deviance"],
                5,
            ),
            "MAE:",
            round(metrics["mae"], 5),
            "Spearman:",
            ranking["spearman"],
        )

        if (
            best is None
            or metrics["poisson_deviance"]
            < best["validation"][
                "poisson_deviance"
            ]
        ):
            best = trial

    print(
        "Selected model:",
        best["config"]["name"]
    )

    # Evaluate the selected model on the test
    # period, without choosing based on test results.
    model = create_model(
        best["config"]
    )

    model.fit(
        train[FEATURES],
        train.goals,
    )

    predictions = np.maximum(
        model.predict(test[FEATURES]),
        1e-9,
    )

    baseline = DummyRegressor(
        strategy="mean"
    ).fit(
        train[FEATURES],
        train.goals,
    )

    baseline_pred = np.maximum(
        baseline.predict(
            test[FEATURES]
        ),
        1e-9,
    )

    results = {
        "train_rows": len(train),
        "validation_rows": len(valid),
        "test_rows": len(test),
        "selected_config": best["config"],
        "validation_trials": trials,
        "test_v5": evaluate(
            test.goals, predictions
        ),
        "test_ranking": ranking_metrics(
            test.goals, predictions
        ),
        "test_baseline": evaluate(
            test.goals, baseline_pred
        ),
        "note": (
            "Test period was examined in earlier "
            "versions; not an untouched holdout."
        ),
    }

    holdout = test[[
        "date", "player_id", "player",
        "team", "opponent",
        "position", "goals"
    ]].copy()

    holdout["predicted_goals"] = (
        predictions
    )

    holdout.to_csv(
        OUT / "holdout_predictions.csv",
        index=False,
    )

    joblib.dump(
        {
            "model": model,
            "vocab": vocab,
            "features": FEATURES,
            "config": best["config"],
            "version": "v5",
        },
        OUT / "model.joblib",
    )

    (OUT / "metrics.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print(
        json.dumps(results, indent=2)
    )


if __name__ == "__main__":
    main()
