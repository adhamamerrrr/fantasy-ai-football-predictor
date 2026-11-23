"""Convert Understat-style player-match CSV while retaining historical attacking stats."""
import argparse
from pathlib import Path
import pandas as pd

LEAGUE_MAP = {
    'ESP-La Liga': 'La Liga', 'ENG-Premier League': 'Premier League',
    'ITA-Serie A': 'Serie A', 'GER-Bundesliga': 'Bundesliga',
    'FRA-Ligue 1': 'Ligue 1',
}

def convert(source: pd.DataFrame) -> pd.DataFrame:
    needed = {'game', 'player', 'team', 'league', 'goals', 'minutes'}
    missing = needed - set(source.columns)
    if missing:
        raise ValueError(f'Missing source columns: {sorted(missing)}')
    df = source.copy()
    df['date'] = pd.to_datetime(df['game'].astype(str).str.slice(0, 10), errors='raise')
    fixture = df['game'].astype(str).str.slice(11)
    teams = fixture.str.split('-', n=1, expand=True)
    if teams.shape[1] != 2 or teams.isna().any().any():
        raise ValueError('Expected game format YYYY-MM-DD Home-Away')
    home_team, away_team = teams[0].str.strip(), teams[1].str.strip()
    is_home = df['team'].eq(home_team)
    is_away = df['team'].eq(away_team)
    if (~(is_home | is_away)).any():
        examples = df.loc[~(is_home | is_away), ['game', 'team']].head(3)
        raise ValueError(f'Team does not match fixture: {examples.to_dict("records")}')
    df['home'] = is_home.astype(int)
    df['opponent'] = away_team.where(is_home, home_team)
    df['league'] = df['league'].replace(LEAGUE_MAP)
    cols = ['date', 'player', 'team', 'league', 'goals', 'minutes', 'home', 'opponent']
    cols += [c for c in ('xg', 'shots', 'position') if c in df.columns]
    result = df[cols].copy()
    from fantasy_ai.core import validate
    return validate(result)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', default='data/real_history.csv')
    args = parser.parse_args()
    converted = convert(pd.read_csv(args.input))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    converted.to_csv(args.output, index=False)
    print(f'Converted {len(converted)} player-match rows to {args.output}')
    print('Leagues:', converted.league.value_counts().to_dict())
    print('Dates:', str(converted.date.min().date()), 'to', str(converted.date.max().date()))
    print('Preserved columns:', converted.columns.tolist())

if __name__ == '__main__':
    main()
