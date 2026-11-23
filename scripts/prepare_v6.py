
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("artifacts/five_leagues_v6")
df = pd.read_csv(ROOT / "player_seasons.csv")

# Only use verified SofaScore IDs.
df = df.dropna(subset=["ss_player_id", "born"]).copy()
df["ss_player_id"] = df["ss_player_id"].astype("int64")
df["born"] = df["born"].astype("int64")

# Reject IDs assigned to multiple birth years or names.
conflicts = df.groupby("ss_player_id").agg(
    names=("player", "nunique"),
    birth_years=("born", "nunique"),
)
bad_ids = conflicts.index[
    (conflicts["names"] > 1) |
    (conflicts["birth_years"] > 1)
]
df = df[~df["ss_player_id"].isin(bad_ids)].copy()

# Aggregate additive statistics across clubs.
additive = [
    "playing_time_min",
    "playing_time_mp",
    "playing_time_starts",
    "fb_gls",
    "fb_ast",
    "fb_sh",
    "fb_sot",
    "us_xg",
    "us_shots",
]

# Missing xG must remain missing, not become zero.
for col in additive:
    df[col] = pd.to_numeric(df[col], errors="coerce")

group_cols = ["season", "ss_player_id"]

totals = df.groupby(group_cols, as_index=False)[additive].sum(
    min_count=1
)

# Representative club is the one with the most minutes.
representative = (
    df.sort_values("playing_time_min", ascending=False)
    .drop_duplicates(group_cols)
    [group_cols + ["player", "born", "league", "team", "pos", "age"]]
)

clean = representative.merge(
    totals, on=group_cols, validate="one_to_one"
)

minutes = clean["playing_time_min"].replace(0, np.nan)

clean["goals_p90"] = clean["fb_gls"] * 90 / minutes
clean["shots_p90"] = clean["fb_sh"] * 90 / minutes
clean["xg_p90"] = clean["us_xg"] * 90 / minutes

clean.to_csv(ROOT / "clean_player_seasons.csv", index=False)

previous = clean[clean["season"] == 2425].copy()
following = clean[clean["season"] == 2526].copy()

paired = previous.merge(
    following,
    on="ss_player_id",
    suffixes=("_24", "_25"),
    validate="one_to_one",
)

# Require meaningful playing time in both seasons.
paired = paired[
    (paired["playing_time_min_24"] >= 450) &
    (paired["playing_time_min_25"] >= 450)
].copy()

paired.to_csv(ROOT / "matched_players.csv", index=False)

print("Excluded conflicting IDs:", len(bad_ids))
print("Clean player-season records:", len(clean))
print("Matched players:", len(paired))
print("2024/25 leagues:", previous["league"].nunique())
print("2025/26 leagues:", following["league"].nunique())
print("Saved cleaned data and matched players.")
