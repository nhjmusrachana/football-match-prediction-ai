import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import requests
import random

# PAGE CONFIG
st.set_page_config(
    page_title="Football Match Prediction AI",
    page_icon="⚽",
    layout="wide"
)

# LOAD MODEL
model = joblib.load("models/football_model.pkl")

# API CONFIG
API_KEY = "e0377f4de9bf43c6a763ca4d8e97ae42"

headers = {
    "X-Auth-Token": API_KEY
}

url = "https://api.football-data.org/v4/competitions/PL/standings"

# API REQUEST
response = requests.get(url, headers=headers)

standings_data = response.json()

# ERROR CHECK
if "standings" not in standings_data:
    st.error("API Error")
    st.write(standings_data)
    st.stop()

# CUSTOM CSS
st.markdown("""
<style>
.main {
    background-color: #0b0f1a;
}

.stButton>button {
    background-color: #00ff88;
    color: black;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-weight: bold;
}

h1, h2, h3 {
    color: white;
}

.metric-box {
    background-color: #111827;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# TITLE
st.title("Football Match Prediction AI ⚽")
st.markdown("## AI-Powered Football Analytics Dashboard")

# GET TEAMS FROM API
teams = {}

for team in standings_data["standings"][0]["table"]:

    team_name = team["team"]["shortName"]

    teams[team_name] = {
        "logo": "⚽",
        "form": random.randint(60, 100),
        "goals": round(random.uniform(1.0, 3.5), 1),
        "points": team["points"],
        "position": team["position"]
    }

# TEAM LIST
team_names = list(teams.keys())

# TEAM SELECTORS
col1, col2 = st.columns(2)

with col1:
    home_team = st.selectbox(
        "Select Home Team",
        team_names
    )

with col2:
    away_team = st.selectbox(
        "Select Away Team",
        team_names
    )

# TEAM STATS
home_form = teams[home_team]["form"]
away_form = teams[away_team]["form"]

home_goals = teams[home_team]["goals"]
away_goals = teams[away_team]["goals"]

# TEAM DISPLAY
st.markdown("---")

col3, col4 = st.columns(2)

with col3:
    st.markdown(f"""
    <div class="metric-box">
        <h2>{teams[home_team]['logo']} {home_team}</h2>
        <h3>League Position: {teams[home_team]['position']}</h3>
        <h3>Points: {teams[home_team]['points']}</h3>
        <h3>Form: {home_form}</h3>
        <h3>Goals/Game: {home_goals}</h3>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-box">
        <h2>{teams[away_team]['logo']} {away_team}</h2>
        <h3>League Position: {teams[away_team]['position']}</h3>
        <h3>Points: {teams[away_team]['points']}</h3>
        <h3>Form: {away_form}</h3>
        <h3>Goals/Game: {away_goals}</h3>
    </div>
    """, unsafe_allow_html=True)

# PREDICTION
if st.button("Predict Match"):

    features = [[
        home_form,
        away_form,
        home_goals,
        away_goals
    ]]

    prediction = model.predict(features)[0]

    probabilities = model.predict_proba(features)[0]

    results = {
        0: f"{away_team} Win",
        1: "Draw",
        2: f"{home_team} Win"
    }

    st.markdown("## Match Prediction")

    st.success(f"🏆 Prediction: {results[prediction]}")

    # PROBABILITY CHART
    df = pd.DataFrame({
        "Result": [
            f"{home_team} Win",
            "Draw",
            f"{away_team} Win"
        ],
        "Probability": probabilities * 100
    })

    fig = px.bar(
        df,
        x="Result",
        y="Probability",
        text="Probability",
        title="Win Probability"
    )

    st.plotly_chart(fig, use_container_width=True)

    # TEAM COMPARISON
    comparison = pd.DataFrame({
        "Category": ["Form", "Goals/Game", "Points"],
        home_team: [
            home_form,
            home_goals,
            teams[home_team]["points"]
        ],
        away_team: [
            away_form,
            away_goals,
            teams[away_team]["points"]
        ]
    })

    st.markdown("## Team Comparison")
    st.dataframe(comparison)

# LIVE STANDINGS
st.markdown("---")
st.markdown("## Live Premier League Standings")

table_data = []

for team in standings_data["standings"][0]["table"]:
    table_data.append({
        "Position": team["position"],
        "Team": team["team"]["name"],
        "Points": team["points"]
    })

table_df = pd.DataFrame(table_data)

st.dataframe(table_df, use_container_width=True)

# FOOTER
st.markdown("---")
st.caption("Built with Streamlit + Machine Learning + Football API ⚽🤖")