import pandas as pd

def build_features(df):
    grouped = df.groupby(["student_id", "course_id"])

    features = grouped.agg(
        total_time_spent=("time_spent_minutes", "sum"),
        avg_time_spent=("time_spent_minutes", "mean"),
        avg_score=("score", "mean"),
        min_score=("score", "min"),
        std_score=("score", "std"),
        chapters_attempted=("chapter_id", "count"),
        last_chapter_score=("score", "last"),
        completed=("completed", "max")
    ).reset_index()

    max_chapters = df["chapter_order"].max()
    features["chapters_completed_ratio"] = (
        features["chapters_attempted"] / max_chapters
    )

    features["std_score"] = features["std_score"].fillna(0)

    return features

