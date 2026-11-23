"""Audit a match-level dataset before model training (no data is modified)."""
import argparse
import json
import pandas as pd
from fantasy_ai.core import validate, LEAGUES


def audit(path):
    raw = pd.read_csv(path)
    report = {'file': str(path), 'rows': len(raw), 'columns': list(raw.columns),
              'missing_by_column': {k: int(v) for k, v in raw.isna().sum().items()},
              'duplicates_player_team_date': None, 'validation': 'not run'}
    try:
        valid = validate(raw)
        report.update(validation='passed', date_min=str(valid.date.min().date()),
                      date_max=str(valid.date.max().date()),
                      leagues={k: int(v) for k,v in valid.league.value_counts().items()},
                      players=int(valid.player.nunique()),
                      goal_rate=float(valid.goals.mean()))
        report['missing_leagues'] = sorted(set(LEAGUES)-set(valid.league))
    except (ValueError, KeyError, TypeError) as e:
        report['validation'] = f'failed: {e}'
    return report

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--history',required=True)
    args=parser.parse_args()
    print(json.dumps(audit(args.history),indent=2))
