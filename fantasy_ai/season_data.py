"""Adapter and honest prior-season baseline for Top-5 player-season data.

This is not a match-level model: season totals cannot be used to reconstruct
match-by-match targets or predict upcoming fixtures without extra data.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def _pick(df, names, label):
    for name in names:
        if name in df.columns:
            return name
    raise ValueError(f"Missing {label}; tried {', '.join(names)}. Inspect dataset column names.")


def load_seasons(path):
    raw = pd.read_csv(path)
    cols = {
        'player': _pick(raw, ['player', 'player_name', 'name'], 'player'),
        'season': _pick(raw, ['season_label', 'season'], 'season'),
        'team': _pick(raw, ['team', 'squad', 'club'], 'team'),
        'goals': _pick(raw, ['fb_gls', 'fb_goals', 'goals', 'gls', 'us_goals'], 'goals'),
        'minutes': _pick(raw, ['fb_min', 'fb_minutes', 'minutes', 'min'], 'minutes'),
    }
    out = raw[[cols[k] for k in cols]].rename(columns={v:k for k,v in cols.items()}).copy()
    for field in ['goals', 'minutes']:
        out[field] = pd.to_numeric(out[field].astype(str).str.replace(',', '', regex=False), errors='coerce')
    out['season'] = out['season'].astype(str).str.replace('/', '-', regex=False)
    out = out.dropna(subset=['player', 'season', 'goals', 'minutes'])
    out = out[(out.goals >= 0) & (out.minutes > 0)]
    # Multiple club rows in one season: aggregate rather than silently duplicate.
    out = out.groupby(['player','season'],as_index=False).agg(goals=('goals','sum'),minutes=('minutes','sum'),team=('team','first'))
    return out


def evaluate_prior_season_baseline(df, earlier='2024-25', later='2025-26'):
    """Evaluate predicting the next season from the preceding season only.

    Same-player names can collide; use stable IDs for production-grade joins.
    Evaluation excludes players absent in either season and is not a prospective
    match-level or partial-season evaluation.
    """
    a = df[df.season == earlier][['player','goals','minutes']].rename(columns={'goals':'prior_goals','minutes':'prior_minutes'})
    b = df[df.season == later][['player','goals','minutes']].rename(columns={'goals':'actual_goals','minutes':'actual_minutes'})
    paired = a.merge(b,on='player',validate='one_to_one')
    if paired.empty:
        raise ValueError('No paired players across seasons; check season labels and source data')
    # Naive rate projection using actual next-season minutes is *retrospective only*.
    # For a true prospective forecast, minutes must be estimated separately.
    paired['predicted_goals'] = (paired.prior_goals / paired.prior_minutes * paired.actual_minutes).clip(lower=0)
    paired['absolute_error'] = (paired.actual_goals - paired.predicted_goals).abs()
    metrics = {'n_players':int(len(paired)), 'mae':float(paired.absolute_error.mean()),
               'rmse':float(np.sqrt(((paired.actual_goals-paired.predicted_goals)**2).mean())),
               'method':'prior season goals/90 scaled by observed next-season minutes (retrospective baseline)',
               'uses_future_minutes':True}
    return paired, metrics
