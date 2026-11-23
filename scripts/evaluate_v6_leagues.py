
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error

import joblib

ROOT = Path("artifacts/five_leagues_v6")

df = pd.read_csv(ROOT / "matched_players.csv")

saved = joblib.load(ROOT / "v6_ridge_model.joblib")
template = saved["model"]
features = saved["features"]

print("\nFANTASY AI V6 - LEAVE-ONE-LEAGUE-OUT")
print("-" * 75)

results = []

for league in sorted(df["league_25"].unique()):
    test_mask = df["league_25"] == league

    train = df.loc[~test_mask]
    test = df.loc[test_mask]

    model = clone(template)

    model.fit(
        train[features],
        train["goals_p90_25"],
    )

    y = test["goals_p90_25"].to_numpy()

    baseline = np.maximum(
        test["goals_p90_24"].to_numpy(),
        0,
    )

    predictions = np.maximum(
        model.predict(test[features]),
        0,
    )

    baseline_mae = mean_absolute_error(y, baseline)
    model_mae = mean_absolute_error(y, predictions)

    improvement = (
        (baseline_mae - model_mae) / baseline_mae * 100
        if baseline_mae > 0 else np.nan
    )

    results.append({
        "league": league,
        "players": len(test),
        "baseline_mae": baseline_mae,
        "model_mae": model_mae,
        "improvement_pct": improvement,
        "model_rmse": np.sqrt(
            mean_squared_error(y, predictions)
        ),
    })

summary = pd.DataFrame(results)

print(summary.round(4).to_string(index=False))

summary.to_csv(
    ROOT / "v6_league_evaluation.csv",
    index=False,
)

print("\nLeagues improved:",
      int((summary["improvement_pct"] > 0).sum()),
      "out of",
      len(summary))

print("\nSaved league evaluation.")
