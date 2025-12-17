import pandas as pd
from feature_engineering import build_features
import joblib

def test_model_prediction_range():
    df = pd.read_csv("data/synthetic_learning_data.csv")
    features = build_features(df)
    X = features.drop(columns=["student_id", "course_id", "completed"])

    model = joblib.load("models/completion_model.pkl")
    probs = model.predict_proba(X)[:, 1]

    assert (probs >= 0).all()
    assert (probs <= 1).all()
