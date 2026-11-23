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

from fantasy_ai.core import engineer, FEATURES
from scripts.import_understat_matches import convert


SOURCE = Path(
    "artifacts/barcelona_v2_real_features/"
    "source_understat_player_matches.csv"
)
OUT = Path("artifacts/barcelona_v3")


def evaluate(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.maximum(
        np.asarray(predicted, dtype=float), 1e-9
    )
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "poisson_deviance": float(
            mean_poisson_deviance(actual, predicted)
        ),
        "actual_goals": float(actual.sum()),
        "predicted_goals": float(predicted.sum()),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(SOURCE)
    data = convert(raw) if "game" in raw.columns else raw

    frame, vocab = engineer(data)
    frame["date"] = pd.to_datetime(frame["date"])

    train = frame[frame.date < "2025-06-01"].copy()
    valid = frame[
        (frame.date >= "2025-08-01")
        & (frame.date < "2026-01-01")
    ].copy()
    test = frame[frame.date >= "2026-01-01"].copy()

    print("Training:", len(train))
    print("Validation:", len(valid))
    print("Testing:", len(test))

    assert len(train) == 1196
    assert len(valid) == 569
    assert len(test) == 632

    model = HistGradientBoostingRegressor(
        loss="poisson",
        max_iter=130,
        max_leaf_nodes=12,
        l2_regularization=2,
        random_state=42,
    )

    model.fit(train[FEATURES], train.goals)

    baseline = DummyRegressor(strategy="mean")
    baseline.fit(train[FEATURES], train.goals)

    valid_pred = np.maximum(
        model.predict(valid[FEATURES]), 1e-9
    )

    calibration = float(
        np.clip(
            valid.goals.sum() / valid_pred.sum(),
            0.25,
            4.0,
        )
    )

    raw_pred = np.maximum(
        model.predict(test[FEATURES]), 1e-9
    )
    calibrated_pred = raw_pred * calibration
    baseline_pred = np.maximum(
        baseline.predict(test[FEATURES]), 1e-9
    )

    results = {
        "train_rows": len(train),
        "validation_rows": len(valid),
        "test_rows": len(test),
        "calibration_factor": calibration,
        "validation": evaluate(valid.goals, valid_pred),
        "test_raw": evaluate(test.goals, raw_pred),
        "test_calibrated": evaluate(
            test.goals, calibrated_pred
        ),
        "test_baseline": evaluate(
            test.goals, baseline_pred
        ),
    }

    predictions = test[
        ["date", "player", "team", "opponent", "goals"]
    ].copy()
    predictions["predicted_raw"] = raw_pred
    predictions["predicted_calibrated"] = calibrated_pred

    predictions.to_csv(
        OUT / "holdout_predictions.csv", index=False
    )

    joblib.dump(
        {
            "model": model,
            "vocab": vocab,
            "features": FEATURES,
            "calibration_factor": calibration,
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

