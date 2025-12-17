import pandas as pd
import joblib
from feature_engineering import build_features
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

df = pd.read_csv("data/synthetic_learning_data.csv")

feature_df = build_features(df)

X = feature_df.drop(columns=["student_id", "course_id", "completed"])
y = feature_df["completed"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))

coefficients = pipeline.named_steps["model"].coef_[0]

importance = pd.Series(
    coefficients,
    index=X.columns
).sort_values(ascending=False)

importance

joblib.dump(pipeline, "models/completion_model.pkl")



