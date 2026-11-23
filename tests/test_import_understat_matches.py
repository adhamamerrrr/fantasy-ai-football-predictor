import pandas as pd
import pytest
from scripts.import_understat_matches import convert

def test_convert_home_and_away():
    df=pd.DataFrame([
        dict(game='2024-08-17 Valencia-Barcelona',player='A',team='Barcelona',league='ESP-La Liga',goals=1,minutes=90),
        dict(game='2024-08-17 Valencia-Barcelona',player='B',team='Valencia',league='ESP-La Liga',goals=0,minutes=70),
    ])
    out=convert(df)
    assert out.home.tolist()==[0,1]
    assert out.opponent.tolist()==['Valencia','Barcelona']
    assert out.league.tolist()==['La Liga','La Liga']

def test_reject_unmatched_team():
    df=pd.DataFrame([dict(game='2024-08-17 Valencia-Barcelona',player='A',team='Arsenal',league='ESP-La Liga',goals=1,minutes=90)])
    with pytest.raises(ValueError,match='Team does not match'): convert(df)
