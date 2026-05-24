import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import numpy as np

# =========================================
# LOAD DATASET
# =========================================

df = pd.read_csv("../data/matches.csv")

# =========================================
# LOAD MODEL FILES
# =========================================

model = pickle.load(
    open("../model/xgb_model.pkl", "rb")
)

encoders = pickle.load(
    open("../model/encoders.pkl", "rb")
)

class_mapping = pickle.load(
    open("../model/class_mapping.pkl", "rb")
)

# =========================================
# REVERSE CLASS MAPPING
# =========================================

reverse_mapping = {
    v: k for k, v in class_mapping.items()
}

# =========================================
# PAGE CONFIG
# =========================================

st.set_page_config(
    page_title="IPL AI Match Predictor",
    page_icon="🏏",
    layout="wide"
)

# =========================================
# CUSTOM CSS
# =========================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(to right, #141e30, #243b55);
    color: white;
}

.main-title {
    text-align: center;
    font-size: 55px;
    font-weight: bold;
    color: #FFD700;
    margin-bottom: 20px;
}

.prediction-box {
    background-color: rgba(0,255,0,0.2);
    border: 2px solid #00ff99;
    padding: 25px;
    border-radius: 20px;
    text-align: center;
    font-size: 28px;
    font-weight: bold;
    color: white;
    animation: pulse 2s infinite;
    line-height: 2;
}

.prediction-box hr {
    border: 1px solid rgba(255,255,255,0.3);
}

@keyframes pulse {
    0% {transform: scale(1);}
    50% {transform: scale(1.02);}
    100% {transform: scale(1);}
}

.stButton>button {
    background-color: #ff4b4b;
    color: white;
    border-radius: 12px;
    height: 3em;
    width: 100%;
    font-size: 20px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# =========================================
# TITLE
# =========================================

st.markdown(
    '<div class="main-title">🏏 IPL Match Winner Prediction</div>',
    unsafe_allow_html=True
)

st.write("")

# =========================================
# TEAM OPTIONS
# =========================================

teams = list(encoders['team1'].classes_)

# =========================================
# USER INPUTS
# =========================================

col1, col2 = st.columns(2)

with col1:

    team1 = st.selectbox(
        "Select Team 1",
        teams
    )

    toss_winner = st.selectbox(
        "Toss Winner",
        teams
    )

with col2:

    team2 = st.selectbox(
        "Select Team 2",
        teams
    )

    toss_decision = st.selectbox(
        "Toss Decision",
        encoders['toss_decision'].classes_
    )

# =========================================
# VENUE
# =========================================

venue = st.selectbox(
    "Select Venue",
    encoders['venue'].classes_
)

# =========================================
# PREDICT BUTTON
# =========================================

if st.button("Predict Winner"):

    # Prevent Same Team Selection
    if team1 == team2:

        st.error(
            "Please select two different teams."
        )

    else:

        # =====================================
        # CREATE INPUT DATA
        # =====================================

        input_df = pd.DataFrame({

            'team1': [
                encoders['team1'].transform(
                    [team1]
                )[0]
            ],

            'team2': [
                encoders['team2'].transform(
                    [team2]
                )[0]
            ],

            'toss_winner': [
                encoders['toss_winner'].transform(
                    [toss_winner]
                )[0]
            ],

            'toss_decision': [
                encoders['toss_decision'].transform(
                    [toss_decision]
                )[0]
            ],

            'venue': [
                encoders['venue'].transform(
                    [venue]
                )[0]
            ]

        })

        # =====================================
        # PREDICT PROBABILITIES
        # =====================================

        probabilities = model.predict_proba(
            input_df
        )[0]

        class_labels = model.classes_

        # =====================================
        # TEAM PROBABILITY DICTIONARY
        # =====================================

        team_probabilities = {}

        for idx, prob in enumerate(probabilities):

            original_class = reverse_mapping[
                class_labels[idx]
            ]

            team_name = encoders[
                'match_winner'
            ].inverse_transform(
                [original_class]
            )[0]

            team_probabilities[
                team_name
            ] = round(
                prob * 100,
                2
            )

        # =====================================
        # GET SELECTED TEAMS PROBABILITY
        # =====================================

        team1_prob = team_probabilities.get(
            team1,
            0
        )

        team2_prob = team_probabilities.get(
            team2,
            0
        )

        # =====================================
        # PREDICT WINNER
        # =====================================

        if team1_prob > team2_prob:

            winner = team1
            confidence = team1_prob

        else:

            winner = team2
            confidence = team2_prob

        # =====================================
        # DISPLAY RESULT
        # =====================================

        st.markdown(
            f"""
            <div class="prediction-box">

                🏆 Predicted Winner: {winner}

                🎯 {team1}: {round(team1_prob, 2)}%

                🎯 {team2}: {round(team2_prob, 2)}%


                🔥 Confidence: {round(confidence, 2)}%

            </div>
            """,
            unsafe_allow_html=True
        )

# =========================================
# TEAM ANALYSIS
# =========================================

st.write("")

st.subheader("📊 IPL Team Wins Analysis")

team_wins = df[
    'match_winner'
].value_counts().reset_index()

team_wins.columns = [
    'Team',
    'Wins'
]

fig = px.bar(
    team_wins,
    x='Team',
    y='Wins',
    title='IPL Team Wins'
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =========================================
# VENUE ANALYSIS
# =========================================

st.subheader("🏟️ Venue Analysis")

venue_stats = df[
    'venue'
].value_counts().reset_index()

venue_stats.columns = [
    'Venue',
    'Matches'
]

fig2 = px.pie(
    venue_stats.head(10),
    names='Venue',
    values='Matches',
    title='Top IPL Venues'
)

st.plotly_chart(
    fig2,
    use_container_width=True
)

# =========================================
# HEAD TO HEAD ANALYSIS
# =========================================

st.subheader("⚔️ Head-to-Head Analysis")

filtered = df[
    (
        (df['team1'] == team1) &
        (df['team2'] == team2)
    )
    |
    (
        (df['team1'] == team2) &
        (df['team2'] == team1)
    )
]

h2h = filtered[
    'match_winner'
].value_counts()

st.write(h2h)

# =========================================
# DATASET PREVIEW
# =========================================

st.subheader("📄 Dataset Preview")

st.dataframe(df.head())