
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

from fantasy_ai.core import (
    validate,
    _history_features,
    FEATURES,
    CATEGORICAL,
    _codes,
)
from scripts.import_understat_matches import convert

SOURCE = Path(
    "artifacts/barcelona_v2_real_features/source_understat_player_matches.csv"
)
OUT = Path("artifacts/barcelona_v4")
CATS = ["league", "team", "opponent", "position"]


def evaluate(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.maximum(np.asarray(predicted, dtype=float), 1e-9)

    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "poisson_deviance": float(
            mean_poisson_deviance(actual, predicted)
        ),
        "actual_goals": float(actual.sum()),
        "predicted_goals": float(predicted.sum()),
    }


def historical_positions(frame):
    frame = frame.sort_values(
        ["player", "date", "team"]
    ).copy()

    known = frame["position"].where(
        ~frame["position"].isin(["Sub", "Unknown", ""])
    )

    per_day = (
        frame.assign(known_position=known)
        .groupby(["player", "date"], as_index=False)[
            "known_position"
        ]
        .first()
    )

    per_day = per_day.sort_values(["player", "date"])

    per_day["historical_position"] = (
        per_day.groupby("player")["known_position"]
        .transform(lambda s: s.ffill().shift(1))
    )

    frame = frame.merge(
        per_day[
            ["player", "date", "historical_position"]
        ],
        on=["player", "date"],
        how="left",
    )

    frame["position"] = (
        frame["historical_position"].fillna("Unknown")
    )

    return frame.drop(columns="historical_position")


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(SOURCE)

    data = (
        convert(raw)
        if "game" in raw.columns
        else validate(raw)
    )

    frame = _history_features(validate(data))
    frame = historical_positions(frame)

    train = frame[
        frame.date < "2025-06-01"
    ].copy()

    valid = frame[
        (frame.date >= "2025-08-01")
        & (frame.date < "2026-01-01")
    ].copy()

    test = frame[
        frame.date >= "2026-01-01"
    ].copy()

    print(
        "Training:", len(train),
        "Validation:", len(valid),
        "Testing:", len(test),
    )

    if (len(train), len(valid), len(test)) != (
        1196, 569, 632
    ):
        raise ValueError(
            "Unexpected split sizes; inspect data before training"
        )

    vocab = {
        c: sorted(train[c].unique().tolist())
        for c in CATS
    }

    if any(
        len(values) >= 255
        for values in vocab.values()
    ):
        raise ValueError(
            "Too many categories; use a different encoder"
        )

    for part in (train, valid, test):
        for c in CATS:
            part[f"{c}_code"] = _codes(
                part[c], vocab[c]
            )

    model = HistGradientBoostingRegressor(
        loss="poisson",
        max_iter=130,
        max_leaf_nodes=12,
        l2_regularization=2,
        categorical_features=[
            FEATURES.index(c)
            for c in CATEGORICAL
        ],
        random_state=42,
    )

    model.fit(
        train[FEATURES],
        train.goals,
    )

    baseline = DummyRegressor(
        strategy="mean"
    ).fit(
        train[FEATURES],
        train.goals,
    )

    validation_pred = np.maximum(
        model.predict(valid[FEATURES]),
        1e-9,
    )

    test_pred = np.maximum(
        model.predict(test[FEATURES]),
        1e-9,
    )

    baseline_pred = np.maximum(
        baseline.predict(test[FEATURES]),
        1e-9,
    )

    results = {
        "train_rows": len(train),
        "validation_rows": len(valid),
        "test_rows": len(test),
        "validation": evaluate(
            valid.goals,
            validation_pred,
        ),
        "test_v4": evaluate(
            test.goals,
            test_pred,
        ),
        "test_baseline": evaluate(
            test.goals,
            baseline_pred,
        ),
        "note": (
            "2026 test period previously inspected in v3; "
            "not a pristine holdout"
        ),
    }

    predictions = test[
        [
            "date",
            "player",
            "team",
            "opponent",
            "position",
            "goals",
        ]
    ].copy()

    predictions["predicted_goals"] = test_pred

    predictions.to_csv(
        OUT / "holdout_predictions.csv",
        index=False,
    )

    joblib.dump(
        {
            "model": model,
            "vocab": vocab,
            "features": FEATURES,
        },
        OUT / "model.joblib",
    )

    (OUT / "metrics.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
