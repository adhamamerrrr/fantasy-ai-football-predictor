"""Fantasy AI | Five-league retrospective scoring-rate explorer.

Launch from the repository root: python -m streamlit run app.py
Requires artifacts/five_leagues_v6 created by prepare_v6/train_v6.
"""
from pathlib import Path
import html

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

st.set_page_config(page_title="Fantasy AI | Football Intelligence", page_icon="⚽", layout="wide", initial_sidebar_state="expanded")
ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts" / "five_leagues_v6"
LEAGUES = {
    "ENG-Premier League": "Premier League", "ESP-La Liga": "La Liga",
    "ITA-Serie A": "Serie A", "GER-Bundesliga": "Bundesliga",
    "FRA-Ligue 1": "Ligue 1",
}
GOLD = "#D9B76E"
BLUE = "#69B9F6"
MUTED = "#94A3B8"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], [data-testid="stApp"] {font-family: 'DM Sans', sans-serif;}
[data-testid="stApp"] {background: radial-gradient(ellipse at 75% 0%, #16232D 0%, #0C1119 40%, #090D13 100%); color:#EAF0F5;}
[data-testid="stSidebar"] {background:#111923; border-right:1px solid #273443;}
.block-container {max-width: 1450px; padding-top:2.2rem;}
h1,h2,h3 {letter-spacing:-.035em;}
.hero {padding:26px 30px; border:1px solid #354454; border-radius:20px; background:linear-gradient(115deg,#1B2B36,#101923 70%); margin-bottom:16px;}
.eyebrow {color:#D9B76E; font-size:12px; letter-spacing:2px; font-weight:800;}
.hero h1 {margin:6px 0 4px; font-size:42px; line-height:1.15; color:#F7FAFC;}
.hero p {color:#B2C1D0; margin:0; font-size:14px;}
.metric-card {border:1px solid #2B3947; background:#14202A; padding:19px 21px; border-radius:16px; min-height:115px;}
.metric-label {color:#9EAFBF; font-size:12px; font-weight:600; letter-spacing:.02em;}
.metric-value {color:#F4F7F9; font-size:29px; font-weight:800; margin-top:9px; line-height:1.1;}
.metric-foot {color:#C7A967; font-size:11px; margin-top:6px;}
.player-card {border:1px solid #344250; border-radius:16px; background:linear-gradient(145deg,#192934,#111A23); padding:20px; min-height:195px;}
.player-rank {color:#D9B76E; font-size:12px; font-weight:800; letter-spacing:1px;}
.player-name {color:#F2F6FA; font-size:21px; font-weight:800; margin:10px 0 3px;}
.player-meta {color:#A9B9C7; font-size:12px; min-height:34px;}
.player-score {color:#D9B76E; font-size:33px; font-weight:800; margin-top:10px;}
.player-unit {color:#A6B5C3; font-size:11px;}
.section-note {color:#9AAABA; font-size:13px; margin-bottom:10px;}
div[data-testid="stMetric"] {background:#14202A; border:1px solid #2B3947; border-radius:13px; padding:15px;}
[data-testid="stTabs"] button {font-weight:700;}
[data-testid="stDataFrame"] {border-radius:12px; overflow:hidden;}
[data-testid="stAlert"] {border-radius:12px;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_bundle():
    return joblib.load(ART / "v6_ridge_model.joblib")

@st.cache_data
def load_csv(name):
    return pd.read_csv(ART / name)

try:
    bundle = load_bundle()
    players = load_csv("matched_players.csv")
except Exception as exc:
    st.error(f"V6 files could not be loaded: {exc}")
    st.code("python -m scripts.prepare_v6\npython -m scripts.train_v6\npython -m streamlit run app.py", language="bat")
    st.stop()

features = bundle.get("features", [])
missing = [f for f in features if f not in players.columns]
if missing:
    st.error(f"Missing model features: {missing}")
    st.stop()

@st.cache_data(show_spinner=False)
def score_players(frame, _model, feature_names):
    scored = frame.copy()
    scored["prediction"] = np.maximum(_model.predict(frame[list(feature_names)]), 0)
    scored["prior"] = pd.to_numeric(scored["goals_p90_24"], errors="coerce")
    scored["actual"] = pd.to_numeric(scored["goals_p90_25"], errors="coerce")
    scored["league_label"] = scored["league_25"].map(LEAGUES).fillna(scored["league_25"])
    scored["player_label"] = scored["player_25"].astype(str) + " · " + scored["team_25"].astype(str) + " · " + scored["ss_player_id"].astype(str)
    return scored

full = score_players(players, bundle["model"], tuple(features))

with st.sidebar:
    st.markdown("### ⚽ FANTASY AI")
    st.caption("PLAYER INTELLIGENCE · V6")
    st.divider()
    st.markdown("#### Explore the data")
    league_options = sorted(full["league_label"].dropna().unique().tolist())
    selected_leagues = st.multiselect("Leagues (2025/26)", league_options, default=league_options)
    league_subset = full[full["league_label"].isin(selected_leagues)]
    team_options = sorted(league_subset["team_25"].dropna().unique().tolist())
    selected_teams = st.multiselect("Teams (2025/26)", team_options)
    search = st.text_input("Search footballers", placeholder="Name, e.g. Salah")
    minimum_minutes = st.slider("2024/25 minimum minutes", 450, 3000, 450, step=50)
    position_options = sorted(full["pos_24"].dropna().astype(str).unique().tolist())
    positions = st.multiselect("2024/25 positions", position_options)
    st.divider()
    st.caption("Club and league filters refer to the observed 2025/26 team. They are for retrospective browsing, not forecast inputs.")
    st.caption("All rates shown are goals per 90 minutes, not scoring probabilities.")

filtered = league_subset.copy()
if selected_teams:
    filtered = filtered[filtered["team_25"].isin(selected_teams)]
if search.strip():
    filtered = filtered[filtered["player_25"].str.contains(search.strip(), case=False, na=False, regex=False)]
if positions:
    filtered = filtered[filtered["pos_24"].isin(positions)]
filtered = filtered[filtered["playing_time_min_24"] >= minimum_minutes]
ranked = filtered.sort_values("prediction", ascending=False).reset_index(drop=True)

st.markdown("""<div class="hero"><div class="eyebrow">FIVE LEAGUES · ONE INTELLIGENCE PLATFORM</div>
<h1>FANTASY AI <span style="color:#D9B76E">/</span> PLAYER LAB</h1>
<p>Scouting-inspired player forecasts · 2024/25 inputs → 2025/26 retrospective outcomes</p></div>""", unsafe_allow_html=True)
st.info("Research dashboard: season scoring-rate forecasts, not upcoming-match probabilities. The 1,316-player cohort requires ≥450 minutes in both seasons; this future-season eligibility is not available in real preseason forecasting.")

def metric_card(label, value, foot=""):
    return (f'<div class="metric-card"><div class="metric-label">{html.escape(str(label))}</div>'
            f'<div class="metric-value">{html.escape(str(value))}</div>'
            f'<div class="metric-foot">{html.escape(str(foot))}</div></div>')

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(metric_card("PLAYERS IN VIEW", f"{len(ranked):,}", "Matched evaluation cohort"), unsafe_allow_html=True)
with m2:
    st.markdown(metric_card("LEAGUES IN VIEW", str(ranked["league_label"].nunique()), "Top five European leagues"), unsafe_allow_html=True)
with m3:
    st.markdown(metric_card("AVG AI GOALS / 90", f"{ranked['prediction'].mean():.3f}" if len(ranked) else "—", "Exploratory model output"), unsafe_allow_html=True)
with m4:
    st.markdown(metric_card("HELD-OUT TEST MAE", "0.06797", "329 test players · V6 Ridge"), unsafe_allow_html=True)

st.write("")
tab_overview, tab_explore, tab_compare, tab_performance = st.tabs(["Overview", "Player explorer", "Compare players", "Model performance"])

with tab_overview:
    st.subheader("Featured scoring profiles")
    st.markdown('<div class="section-note">Highest model-estimated scoring rates within your filters. These full-cohort predictions include training players.</div>', unsafe_allow_html=True)
    if ranked.empty:
        st.warning("No players match the selected filters.")
    else:
        top_cols = st.columns(3)
        for i, (_, p) in enumerate(ranked.head(3).iterrows()):
            with top_cols[i]:
                name = html.escape(str(p["player_25"]))
                team = html.escape(str(p["team_25"]))
                league = html.escape(str(p["league_label"]))
                st.markdown(f'<div class="player-card"><div class="player-rank">#{i+1:02d} · AI SCORING RANK</div>'
                            f'<div class="player-name">{name}</div><div class="player-meta">{team} · {league}</div>'
                            f'<div class="player-score">{p["prediction"]:.3f}</div><div class="player-unit">PREDICTED GOALS / 90</div></div>', unsafe_allow_html=True)
        st.write("")
        c1, c2 = st.columns([1.4, 1])
        with c1:
            st.markdown("#### Top 12 AI scoring rates")
            bars = ranked.head(12).copy()
            bars["display"] = bars["player_25"].astype(str) + " · " + bars["team_25"].astype(str)
            chart = alt.Chart(bars).mark_bar(cornerRadiusEnd=5, color=GOLD).encode(
                x=alt.X("prediction:Q", title="Predicted goals per 90"),
                y=alt.Y("display:N", sort="-x", title=None),
                tooltip=[alt.Tooltip("player_25:N", title="Player"), alt.Tooltip("team_25:N", title="Team"), alt.Tooltip("prediction:Q", title="AI goals/90", format=".3f")],
            ).properties(height=390)
            st.altair_chart(chart, use_container_width=True)
        with c2:
            st.markdown("#### Players by league")
            counts = ranked.groupby("league_label", as_index=False).size().rename(columns={"size": "players"})
            chart = alt.Chart(counts).mark_bar(cornerRadiusEnd=5, color=BLUE).encode(
                x=alt.X("players:Q", title="Players"),
                y=alt.Y("league_label:N", sort="-x", title=None),
                tooltip=["league_label:N", "players:Q"],
            ).properties(height=270)
            st.altair_chart(chart, use_container_width=True)
            st.caption("Counts reflect the filtered, retrospective cohort—not every player in each league.")

with tab_explore:
    st.subheader("Player explorer")
    st.caption("AI predictions use 2024/25 features. Actual 2025/26 results are shown separately below for retrospective analysis.")
    if ranked.empty:
        st.warning("No players match the selected filters.")
    else:
        options = ranked["player_label"].tolist()
        chosen = st.selectbox("Select a player", options, key="player_profile")
        p = ranked.loc[ranked["player_label"] == chosen].iloc[0]
        st.markdown(f"### {p['player_25']}  ·  {p['team_25']}")
        st.caption(f"{p['league_label']} · 2024/25 position: {p['pos_24']} · 2024/25 minutes: {int(p['playing_time_min_24']):,}")
        a, b, c = st.columns(3)
        a.metric("AI forecast · goals/90", f"{p['prediction']:.3f}")
        b.metric("2024/25 · goals/90", f"{p['prior']:.3f}")
        c.metric("2025/26 actual · goals/90", f"{p['actual']:.3f}")
        profile = pd.DataFrame({"Metric": ["AI forecast", "2024/25", "2025/26 actual"],
                                "Goals / 90": [p["prediction"], p["prior"], p["actual"]]})
        chart = alt.Chart(profile).mark_bar(cornerRadiusEnd=6).encode(
            x=alt.X("Metric:N", sort=["AI forecast", "2024/25", "2025/26 actual"], title=None),
            y=alt.Y("Goals / 90:Q", title="Goals per 90", scale=alt.Scale(zero=True)),
            color=alt.Color("Metric:N", scale=alt.Scale(domain=["AI forecast", "2024/25", "2025/26 actual"], range=[GOLD, BLUE, MUTED]), legend=None),
            tooltip=["Metric:N", alt.Tooltip("Goals / 90:Q", format=".3f")],
        ).properties(height=270)
        st.altair_chart(chart, use_container_width=True)
        st.markdown("#### AI player rankings")
        table = ranked[["player_25", "league_label", "team_25", "pos_24", "prediction", "prior", "playing_time_min_24"]].rename(columns={
            "player_25": "Player", "league_label": "League", "team_25": "Team", "pos_24": "Position",
            "prediction": "AI goals/90", "prior": "2024/25 goals/90", "playing_time_min_24": "2024/25 minutes"})
        st.dataframe(table, hide_index=True, use_container_width=True, height=400,
                     column_config={"AI goals/90": st.column_config.NumberColumn(format="%.3f"),
                                    "2024/25 goals/90": st.column_config.NumberColumn(format="%.3f")})
        st.download_button("Export filtered AI rankings (CSV)", table.to_csv(index=False).encode("utf-8-sig"),
                           "fantasy_ai_v6_rankings.csv", "text/csv", key="export_rankings")

with tab_compare:
    st.subheader("Head-to-head player comparison")
    st.caption("Grouped bars show independent scoring rates; they are not stacked or added together.")
    if ranked.empty:
        st.warning("No players match the selected filters.")
    else:
        options = ranked["player_label"].tolist()
        picked = st.multiselect("Choose up to five player records", options, default=options[:min(3, len(options))], max_selections=5)
        if picked:
            comp = ranked[ranked["player_label"].isin(picked)].copy()
            comp["display"] = comp["player_25"].astype(str) + " · " + comp["team_25"].astype(str)
            long = comp.melt(id_vars=["display"], value_vars=["prediction", "prior", "actual"], var_name="Metric", value_name="Goals / 90")
            long["Metric"] = long["Metric"].map({"prediction": "AI forecast", "prior": "2024/25", "actual": "2025/26 actual"})
            chart = alt.Chart(long).mark_bar(cornerRadiusEnd=4).encode(
                x=alt.X("display:N", title=None, sort=comp["display"].tolist(), axis=alt.Axis(labelAngle=-25, labelLimit=190)),
                xOffset=alt.XOffset("Metric:N", sort=["AI forecast", "2024/25", "2025/26 actual"]),
                y=alt.Y("Goals / 90:Q", title="Goals per 90", scale=alt.Scale(zero=True)),
                color=alt.Color("Metric:N", scale=alt.Scale(domain=["AI forecast", "2024/25", "2025/26 actual"], range=[GOLD, BLUE, MUTED]),
                                legend=alt.Legend(orient="bottom")),
                tooltip=["display:N", "Metric:N", alt.Tooltip("Goals / 90:Q", format=".3f")],
            ).properties(height=370)
            st.altair_chart(chart, use_container_width=True)
            st.dataframe(comp[["player_25", "team_25", "prediction", "prior", "actual"]].rename(columns={
                "player_25": "Player", "team_25": "Team", "prediction": "AI goals/90", "prior": "2024/25 goals/90", "actual": "2025/26 actual goals/90"}).round(3), hide_index=True, use_container_width=True)
        else:
            st.info("Choose at least one player to compare.")

with tab_performance:
    st.subheader("Model performance & methodology")
    st.caption("Separate evaluation outputs from the full-cohort exploratory predictions shown in the other tabs.")
    a, b, c = st.columns(3)
    a.metric("Held-out players", "329")
    b.metric("Held-out Ridge MAE", "0.06797")
    c.metric("Improvement vs baseline", "13.99%")
    overall = pd.DataFrame({"Metric": ["MAE", "RMSE"], "Previous-season baseline": [0.07902, 0.12302], "Ridge model": [0.06797, 0.09735]})
    melted = overall.melt("Metric", var_name="Method", value_name="Error")
    chart = alt.Chart(melted).mark_bar(cornerRadiusEnd=5).encode(
        x=alt.X("Metric:N", title=None), xOffset="Method:N", y=alt.Y("Error:Q", scale=alt.Scale(zero=True)),
        color=alt.Color("Method:N", scale=alt.Scale(domain=["Previous-season baseline", "Ridge model"], range=[MUTED, GOLD])),
        tooltip=["Metric:N", "Method:N", alt.Tooltip("Error:Q", format=".5f")],
    ).properties(height=275)
    st.altair_chart(chart, use_container_width=True)
    st.markdown("#### Leave-one-league-out evaluation")
    try:
        league_results = load_csv("v6_league_evaluation.csv")
        league_results["league"] = league_results["league"].map(LEAGUES).fillna(league_results["league"])
        st.dataframe(league_results.rename(columns={"league": "League", "players": "Players", "baseline_mae": "Baseline MAE", "model_mae": "Model MAE", "improvement_pct": "MAE improvement (%)", "model_rmse": "Model RMSE"}).round(4), hide_index=True, use_container_width=True)
        league_long = league_results.melt(id_vars="league", value_vars=["baseline_mae", "model_mae"], var_name="Method", value_name="MAE")
        league_long["Method"] = league_long["Method"].map({"baseline_mae": "Baseline", "model_mae": "Ridge"})
        chart = alt.Chart(league_long).mark_bar(cornerRadiusEnd=4).encode(
            x=alt.X("league:N", title=None, axis=alt.Axis(labelAngle=-20)), xOffset="Method:N", y=alt.Y("MAE:Q", scale=alt.Scale(zero=True)),
            color=alt.Color("Method:N", scale=alt.Scale(domain=["Baseline", "Ridge"], range=[MUTED, GOLD])),
            tooltip=["league:N", "Method:N", alt.Tooltip("MAE:Q", format=".4f")],
        ).properties(height=310)
        st.altair_chart(chart, use_container_width=True)
    except FileNotFoundError:
        st.warning("League evaluation CSV not found. Run: python -m scripts.evaluate_v6_leagues")
    st.markdown("#### Held-out predictions")
    try:
        heldout = load_csv("v6_test_predictions.csv")
        st.dataframe(heldout.head(100).round(3), hide_index=True, use_container_width=True, height=270)
        st.download_button("Download held-out predictions", heldout.to_csv(index=False).encode("utf-8-sig"),
                           "fantasy_ai_v6_heldout.csv", "text/csv", key="heldout_export")
    except FileNotFoundError:
        st.warning("Held-out predictions CSV not found. Run: python -m scripts.train_v6")
    st.markdown("#### Read before interpreting results")
    st.markdown("""
- **Target:** 2025/26 goals per 90 minutes; **inputs:** selected 2024/25 player features.
- **Held-out evaluation:** 987 players trained / 329 tested, randomly separated from one season transition. It is not an unseen-season test.
- **Full-cohort rankings:** include training players and therefore must not be treated as held-out predictions.
- **League robustness:** each fold excludes the target 2025/26 league, but players can change leagues between seasons; this is not a strict independent-player, future-season test.
- **Selection bias:** requiring 450+ minutes in 2025/26 uses information unavailable at preseason time.
- **Data scope:** five leagues, two seasons. These rates are not per-match scoring probabilities and should not be interpreted as live 2026/27 forecasts.
""")

st.divider()
st.caption("FANTASY AI · V6 research prototype · Built with Streamlit, scikit-learn, pandas and Altair")
