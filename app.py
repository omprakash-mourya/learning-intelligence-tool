import streamlit as st
import pandas as pd
import joblib

from feature_engineering import build_features

# App title

st.title("📊 Learning Intelligence Tool")
st.write("Predict course completion, detect at-risk students, and identify difficult chapters.")

# Load trained mode

@st.cache_resource
def load_model():
    return joblib.load("models/completion_model.pkl")

model = load_model()

# File upload

uploaded_file = st.file_uploader(
    "Upload learner activity CSV",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Please upload a CSV file to run predictions.")
    st.stop()

# Load data

df = pd.read_csv(uploaded_file)

required_cols = [
    "student_id",
    "course_id",
    "chapter_id",
    "chapter_order",
    "time_spent_minutes",
    "score",
    "completed"
]

missing_cols = set(required_cols) - set(df.columns)
if missing_cols:
    st.error(f"Missing required columns: {missing_cols}")
    st.stop()

st.subheader("Raw Input Data")
st.dataframe(df.head())

# Feature engineering

features_df = build_features(df)

X = features_df.drop(columns=["student_id", "course_id", "completed"])

# Predictions

features_df["completion_probability"] = model.predict_proba(X)[:, 1]
features_df["completion_prediction"] = model.predict(X)

def risk_flag(p):
    if p < 0.4:
        return "HIGH RISK"
    elif p < 0.7:
        return "MEDIUM RISK"
    else:
        return "LOW RISK"

features_df["risk_level"] = features_df["completion_probability"].apply(risk_flag)

# Results display

st.subheader("📈 Student Completion Predictions")
st.dataframe(features_df[
    ["student_id", "course_id", "completion_probability", "risk_level"]
])

# High-risk students

st.subheader("🚨 High-Risk Students")
high_risk = features_df[features_df["risk_level"] == "HIGH RISK"]

if high_risk.empty:
    st.success("No high-risk students detected.")
else:
    st.dataframe(high_risk[
        ["student_id", "course_id", "completion_probability"]
    ])


# Chapter difficulty analysis

st.subheader("📉 Chapter Difficulty Analysis")

chapter_stats = (
    df.groupby("chapter_order")
    .agg(
        dropout_rate=("completed", lambda x: 1 - x.mean()),
        avg_score=("score", "mean"),
        avg_time_spent=("time_spent_minutes", "mean")
    )
    .reset_index()
)

chapter_stats["difficulty_score"] = (
    chapter_stats["dropout_rate"] * (1 / chapter_stats["avg_score"])
)

st.dataframe(
    chapter_stats.sort_values("difficulty_score", ascending=False)
)

# Insights

st.subheader("🧠 Key Insights")

num_high_risk = len(high_risk)
st.write(f"- **{num_high_risk} students** are currently at high risk of dropping out.")

hardest_chapter = chapter_stats.sort_values(
    "difficulty_score", ascending=False
).iloc[0]

st.write(
    f"- **Chapter {int(hardest_chapter.chapter_order)}** appears to be the most difficult "
    f"with a dropout rate of {hardest_chapter.dropout_rate:.2f}."
)

st.write(
    "- Time spent and assessment scores are strong indicators of course completion."
)

st.success("Inference complete.")
