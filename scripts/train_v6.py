
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path("artifacts/five_leagues_v6")

df = pd.read_csv(ROOT / "matched_players.csv")

numeric = [
    "goals_p90_24",
    "xg_p90_24",
    "shots_p90_24",
    "playing_time_min_24",
    "playing_time_starts_24",
    "age_24",
]

categorical = [
    "pos_24",
    "league_24",
]

features = numeric + categorical

X = df[features].copy()
y = df["goals_p90_25"].astype(float)

train_idx, test_idx = train_test_split(
    np.arange(len(df)),
    test_size=0.25,
    random_state=42,
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]
y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

preprocess = ColumnTransformer([
    (
        "numeric",
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]),
        numeric,
    ),
    (
        "categorical",
        Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="Unknown",
                ),
            ),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]),
        categorical,
    ),
])

model = Pipeline([
    ("preprocess", preprocess),
    ("regressor", Ridge(alpha=20.0)),
])

model.fit(X_train, y_train)

predictions = np.maximum(model.predict(X_test), 0)

baseline = X_test["goals_p90_24"].to_numpy()
baseline = np.maximum(baseline, 0)

def evaluate(name, actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    print(
        f"{name}: "
        f"MAE={mae:.5f}, "
        f"RMSE={rmse:.5f}"
    )

    return mae

print("\nFANTASY AI V6 RESULTS")
print("Training players:", len(train_idx))
print("Testing players:", len(test_idx))

baseline_mae = evaluate(
    "Previous-season baseline",
    y_test,
    baseline,
)

model_mae = evaluate(
    "Ridge ML model",
    y_test,
    predictions,
)

print(
    "MAE improvement:",
    round(
        (baseline_mae - model_mae)
        / baseline_mae * 100,
        2,
    ),
    "%",
)

results = df.iloc[test_idx][[
    "player_25",
    "league_25",
    "team_25",
]].copy()

results["actual_goals_p90"] = y_test.to_numpy()
results["baseline_goals_p90"] = baseline
results["predicted_goals_p90"] = predictions

results.to_csv(
    ROOT / "v6_test_predictions.csv",
    index=False,
)

joblib.dump(
    {
        "model": model,
        "features": features,
        "numeric": numeric,
        "categorical": categorical,
        "train_players": len(train_idx),
        "test_players": len(test_idx),
    },
    ROOT / "v6_ridge_model.joblib",
)

print("\nSaved model and test predictions.")
