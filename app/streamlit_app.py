import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

# PAGE CONFIG
st.set_page_config(
    page_title="Football Match Prediction AI",
    page_icon="⚽",
    layout="wide"
)

# LOAD AI MODEL
model = joblib.load("models/football_model.pkl")

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

h1 {
    color: white;
    text-align: center;
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

# TEAM DATA
teams = {
    "Arsenal": {
        "logo": "🔴",
        "form": 85,
        "goals": 2.3
    },
    "Chelsea": {
        "logo": "🔵",
        "form": 72,
        "goals": 1.8
    },
    "Liverpool": {
        "logo": "🔴",
        "form": 90,
        "goals": 2.7
    },
    "Manchester City": {
        "logo": "🔵",
        "form": 95,
        "goals": 3.1
    },
    "Barcelona": {
        "logo": "🔴🔵",
        "form": 88,
        "goals": 2.5
    },
    "Real Madrid": {
        "logo": "⚪",
        "form": 92,
        "goals": 2.9
    }
}

# TEAM SELECTORS
col1, col2 = st.columns(2)

with col1:
    home_team = st.selectbox(
        "Select Home Team",
        list(teams.keys())
    )

with col2:
    away_team = st.selectbox(
        "Select Away Team",
        list(teams.keys())
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
        <h3>Form: {home_form}</h3>
        <h3>Goals/Game: {home_goals}</h3>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-box">
        <h2>{teams[away_team]['logo']} {away_team}</h2>
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
        "Category": ["Form", "Goals/Game"],
        home_team: [home_form, home_goals],
        away_team: [away_form, away_goals]
    })

    st.markdown("## Team Comparison")
    st.dataframe(comparison)

# FOOTER
st.markdown("---")
st.caption("Built with Streamlit + Machine Learning ⚽🤖")