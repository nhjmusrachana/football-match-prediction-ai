import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# LOAD DATA
df = pd.read_csv("data/matches.csv")

# FEATURES
X = df[[
    "home_form",
    "away_form",
    "home_goals_avg",
    "away_goals_avg"
]]

# TARGET
y = df["result"]

# TRAIN MODEL
model = RandomForestClassifier()

model.fit(X, y)

# SAVE MODEL
joblib.dump(model, "models/football_model.pkl")

print("✅ AI Model Trained Successfully")