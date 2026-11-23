"""Generate clearly synthetic records for local smoke tests, not real football statistics."""
from pathlib import Path
import numpy as np
import pandas as pd
from fantasy_ai.core import LEAGUES

def generate(seed=42):
    rng=np.random.default_rng(seed)
    records=[]
    teams={league:[f'{league} Demo Club {i}' for i in range(1,5)] for league in LEAGUES}
    dates=pd.date_range('2024-08-01',periods=70,freq='7D')
    for league in LEAGUES:
        for t,team in enumerate(teams[league]):
            for n in range(3):
                player=f'{league} Demo Player {t*3+n+1}'
                skill=float(rng.uniform(0.05,0.9))
                for ix,date in enumerate(dates):
                    opp=teams[league][(t+1+ix%3)%4]
                    minutes=int(rng.choice([0,35,65,90],p=[.08,.08,.14,.70]))
                    lam=skill*minutes/90*(1.12 if ix%2 else .95)
                    records.append(dict(date=date.strftime('%Y-%m-%d'),player=player,team=team,league=league,opponent=opp,home=ix%2,minutes=minutes,goals=int(rng.poisson(lam))))
    return pd.DataFrame(records)

if __name__=='__main__':
    dest=Path('data/demo_history.csv');dest.parent.mkdir(parents=True,exist_ok=True)
    generate().to_csv(dest,index=False)
    print(f'Wrote {dest}: SYNTHETIC data, not actual football results')
