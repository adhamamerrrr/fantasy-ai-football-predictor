import pandas as pd
import pytest
from fantasy_ai.core import engineer,fit_and_evaluate,forecast,season_forecast
from scripts.make_demo_data import generate

def test_lags_exclude_current_goals():
    df=generate()
    a,_=engineer(df)
    altered=df.copy()
    altered.loc[altered.index[0],'goals']=50
    b,_=engineer(altered)
    key=a.loc[0,['date','player','team']]
    idx=b.index[(b.date==key.date)&(b.player==key.player)&(b.team==key.team)][0]
    assert a.loc[0,'prior_goals_per90']==b.loc[idx,'prior_goals_per90']==0

def test_training_and_forecast():
    history=generate()
    model,vocab,metrics,_=fit_and_evaluate(history,'2025-05-01')
    assert metrics['train_rows']>20 and metrics['test_rows']>5
    row=history.iloc[-1].copy()
    row['date']='2026-01-01'
    fixture=pd.DataFrame([row.drop(['goals','minutes'])])
    pred=forecast(model,history,fixture,vocab)
    assert pred.expected_goals.iloc[0]>=0
    season=season_forecast(model,history,fixture,vocab,'2025-08-01')
    assert season.projected_season_goals.iloc[0]>=0

def test_future_only():
    history=generate()
    model,vocab,_,_=fit_and_evaluate(history,'2025-05-01')
    fixture=history.iloc[0:1].drop(columns=['goals','minutes'])
    with pytest.raises(ValueError,match='after'):
        forecast(model,history,fixture,vocab)

def test_missing_columns():
    with pytest.raises(ValueError,match='Missing columns'):
        engineer(pd.DataFrame({'player':['A']}))
