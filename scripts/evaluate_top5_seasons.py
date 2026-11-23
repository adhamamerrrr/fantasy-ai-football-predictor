"""Run: python -m scripts.evaluate_top5_seasons --csv data/player_seasons.csv"""
import argparse
import json
from pathlib import Path
from fantasy_ai.season_data import load_seasons, evaluate_prior_season_baseline


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--csv',required=True,help='Path to upstream player_seasons.csv')
    p.add_argument('--output',default='artifacts/top5_season_baseline')
    args=p.parse_args()
    df=load_seasons(args.csv)
    predictions, metrics=evaluate_prior_season_baseline(df)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    predictions.to_csv(out/'predictions.csv',index=False)
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    print(json.dumps(metrics,indent=2))

if __name__=='__main__':main()
