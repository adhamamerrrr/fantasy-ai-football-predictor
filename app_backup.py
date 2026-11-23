from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / 'artifacts' / 'five_leagues_v6'
st.set_page_config(page_title='Fantasy AI | Player Predictor', page_icon='⚽', layout='wide')

@st.cache_resource
def load_model():
    return joblib.load(ARTIFACTS / 'v6_ridge_model.joblib')

@st.cache_data
def load_data():
    return pd.read_csv(ARTIFACTS / 'matched_players.csv')

st.title('Fantasy AI')
st.caption('Five-league player scoring forecast | 2024/25 → 2025/26 retrospective evaluation')
st.info('This dashboard predicts season goals per 90 minutes, not individual match outcomes. The evaluation cohort requires at least 450 minutes in both seasons, which is not known at preseason prediction time.')

try:
    bundle = load_model()
    data = load_data()
except (OSError, KeyError, ValueError) as exc:
    st.error(f'Unable to load V6 artifacts: {exc}')
    st.code('python -m scripts.prepare_v6\npython -m scripts.train_v6')
    st.stop()

features = bundle['features']
missing = [name for name in features if name not in data.columns]
if missing:
    st.error(f'Missing model features: {missing}')
    st.stop()

@st.cache_data
def make_predictions(frame):
    pred = np.maximum(bundle['model'].predict(frame[features]), 0)
    result = frame.copy()
    result['predicted_goals_p90'] = pred
    result['previous_goals_p90'] = result['goals_p90_24']
    result['actual_goals_p90'] = result['goals_p90_25']
    result['predicted_goals_at_90x20'] = pred * 20
    return result

full = make_predictions(data)
league_names = {
    'ENG-Premier League': 'Premier League',
    'ESP-La Liga': 'La Liga',
    'ITA-Serie A': 'Serie A',
    'GER-Bundesliga': 'Bundesliga',
    'FRA-Ligue 1': 'Ligue 1',
}
full['league_display'] = full['league_25'].map(league_names).fillna(full['league_25'])

with st.sidebar:
    st.header('Filters')
    leagues = sorted(full['league_display'].dropna().unique())
    selected_leagues = st.multiselect('League (2025/26)', leagues, default=leagues)
    subset = full[full['league_display'].isin(selected_leagues)]
    teams = sorted(subset['team_25'].dropna().unique())
    selected_teams = st.multiselect('Team (2025/26)', teams)
    if selected_teams:
        subset = subset[subset['team_25'].isin(selected_teams)]
    search = st.text_input('Search player', placeholder='e.g., Salah')
    if search.strip():
        subset = subset[subset['player_25'].str.contains(search.strip(), case=False, na=False, regex=False)]
    min_minutes = st.slider('Minimum previous-season minutes', 450, 3000, 450, 50)
    subset = subset[subset['playing_time_min_24'] >= min_minutes]
    st.caption('Filters use 2025/26 club membership for retrospective browsing; future club assignments are not model features.')

if subset.empty:
    st.warning('No players match your filters.')
    st.stop()

col1, col2, col3 = st.columns(3)
col1.metric('Players', f'{len(subset):,}')
col2.metric('Average predicted goals/90', f'{subset.predicted_goals_p90.mean():.3f}')
col3.metric('Leagues', str(subset.league_display.nunique()))

st.subheader('AI player rankings')
view = subset.sort_values('predicted_goals_p90', ascending=False).copy()
view = view.rename(columns={
    'player_25': 'Player', 'league_display': 'League', 'team_25': 'Team',
    'pos_24': 'Position (2024/25)', 'predicted_goals_p90': 'AI goals/90',
    'previous_goals_p90': '2024/25 goals/90', 'actual_goals_p90': '2025/26 actual goals/90',
    'playing_time_min_24': '2024/25 minutes',
})
columns = ['Player', 'League', 'Team', 'Position (2024/25)', 'AI goals/90',
           '2024/25 goals/90', '2025/26 actual goals/90', '2024/25 minutes']
st.dataframe(view[columns].head(200), hide_index=True, use_container_width=True,
             column_config={
                 'AI goals/90': st.column_config.NumberColumn(format='%.3f'),
                 '2024/25 goals/90': st.column_config.NumberColumn(format='%.3f'),
                 '2025/26 actual goals/90': st.column_config.NumberColumn(format='%.3f'),
             })

st.subheader('Compare players')
choices = view['Player'].dropna().drop_duplicates().tolist()
selected = st.multiselect('Choose up to five players', choices, default=choices[:3], max_selections=5)
if selected:
    chart = view[view['Player'].isin(selected)].drop_duplicates('Player').set_index('Player')
    st.bar_chart(chart[['AI goals/90', '2024/25 goals/90', '2025/26 actual goals/90']])

st.subheader('Prediction details')
player_options = (view['Player'] + ' | ' + view['Team']).tolist()
choice = st.selectbox('Select a player record', player_options)
selected_row = view.iloc[player_options.index(choice)]
a, b, c = st.columns(3)
a.metric('Predicted goals per 90', f"{selected_row['AI goals/90']:.3f}")
b.metric('Previous goals per 90', f"{selected_row['2024/25 goals/90']:.3f}")
c.metric('Actual 2025/26 goals per 90', f"{selected_row['2025/26 actual goals/90']:.3f}")
st.caption('These are scoring rates, not probabilities. Actual outcomes are displayed only for retrospective comparison and are never passed to the model as input features.')

st.download_button('Export filtered predictions as CSV',
                   data=view[columns].to_csv(index=False).encode('utf-8'),
                   file_name='fantasy_ai_v6_predictions.csv', mime='text/csv')
