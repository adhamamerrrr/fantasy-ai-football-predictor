import pandas as pd
import pytest
from fantasy_ai.core import fit_final_model, forecast, fit_and_evaluate
from scripts.audit_data import audit


def test_final_model_forecast():
    df=pd.read_csv('data/demo_history.csv')
    model,vocab=fit_final_model(df)
    latest=pd.to_datetime(df.date).max()
    player=df.iloc[0]
    fixture=pd.DataFrame([{'date':str((latest+pd.Timedelta(days=7)).date()),
        'player':player.player,'team':player.team,'league':player.league,
        'opponent':player.opponent,'home':1}])
    result=forecast(model,df,fixture,vocab)
    assert result.expected_goals.iloc[0] >= 0


def test_audit_demo():
    report=audit('data/demo_history.csv')
    assert report['validation']=='passed'
    assert report['rows'] > 20


def test_time_split_rejects_insufficient_holdout():
    df=pd.read_csv('data/demo_history.csv')
    with pytest.raises(ValueError):
        fit_and_evaluate(df,'2099-01-01')
