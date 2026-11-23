"""Optional real-data Barcelona proof-of-concept; not a five-league evaluation.

Run locally with internet access. The source data is downloaded only to the
user's machine; it is not included in the distributed project archive.
"""
import argparse
import json
from pathlib import Path
from urllib.request import urlretrieve
import pandas as pd
import joblib
from fantasy_ai.core import fit_and_evaluate, fit_final_model
from scripts.import_understat_matches import convert

URL = 'https://raw.githubusercontent.com/m-mahadi/top5-football-dataset/master/data/barcelona/understat_player_matches.csv'

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',default=URL,help='Source CSV URL or local CSV path')
    p.add_argument('--cutoff',default='2025-01-01')
    p.add_argument('--out',default='artifacts/barcelona_real')
    a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    raw=out/'source_understat_player_matches.csv'
    if a.source.startswith(('https://','http://')):
        urlretrieve(a.source,raw)
    else:
        raw=Path(a.source)
    df=convert(pd.read_csv(raw))
    print(f'Real source: {len(df)} rows, {df.league.nunique()} leagues, {df.player.nunique()} players')
    model,vocab,metrics,pred=fit_and_evaluate(df,a.cutoff)
    final_model,final_vocab=fit_final_model(df)
    metrics.update(data_kind='real',scope='Barcelona player-matches only; NOT five-league coverage',source=a.source)
    joblib.dump({'model':final_model,'vocab':final_vocab,'data_kind':'real','scope':metrics['scope']},out/'model.joblib')
    pred[['date','player','league','goals','predicted_goals']].to_csv(out/'holdout_predictions.csv',index=False)
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2))
    print(json.dumps(metrics,indent=2))

if __name__=='__main__':main()
