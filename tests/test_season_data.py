import pandas as pd
import pytest
from fantasy_ai.season_data import load_seasons,evaluate_prior_season_baseline


def test_top5_adapter_and_retrospective_baseline(tmp_path):
    p=tmp_path/'seasons.csv'
    pd.DataFrame({'player':['A','A','B','B'], 'season_label':['2024/25','2025/26']*2,
                  'team':['T']*4,'fb_gls':[10,8,2,4],'fb_min':['1,800','1,440','900','1,800']}).to_csv(p,index=False)
    frame=load_seasons(p)
    result,metrics=evaluate_prior_season_baseline(frame)
    assert len(result)==2
    assert metrics['uses_future_minutes'] is True
    assert metrics['mae']>=0


def test_missing_goals_rejected(tmp_path):
    p=tmp_path/'bad.csv'
    pd.DataFrame({'player':['A'],'season_label':['2024/25'],'team':['T'],'minutes':[90]}).to_csv(p,index=False)
    with pytest.raises(ValueError,match='goals'):
        load_seasons(p)
