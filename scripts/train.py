"""Train/evaluate on a historical CSV and save model bundle and metrics."""
import argparse
import json
from pathlib import Path
import joblib
import pandas as pd
from fantasy_ai.core import fit_and_evaluate, fit_final_model

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--history',default='data/demo_history.csv')
    p.add_argument('--cutoff',default='2025-05-01')
    p.add_argument('--out',default='artifacts')
    p.add_argument('--data-kind', choices=['real','synthetic'], required=True, help='Explicit provenance declaration')
    args=p.parse_args()
    df=pd.read_csv(args.history)
    model,vocab,metrics,pred=fit_and_evaluate(df,args.cutoff)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    final_model, final_vocab = fit_final_model(df)
    joblib.dump({'model':final_model,'vocab':final_vocab,'cutoff':args.cutoff,'data_kind':args.data_kind},out/'model.joblib')
    metrics['data_kind'] = args.data_kind
    metrics['evaluation_model'] = 'chronological holdout'
    metrics['saved_model'] = 'refitted on all completed matches after evaluation'
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2))
    pred[['date','player','league','goals','predicted_goals']].to_csv(out/'holdout_predictions.csv',index=False)
    print(json.dumps(metrics,indent=2))

if __name__=='__main__':main()
