import pandas as pd
import numpy as np

np.random.seed(42)

rows = []

NUM_STUDENTS = 800
NUM_COURSES = 5
CHAPTERS_PER_COURSE = 12


for student_id in range(1, NUM_STUDENTS+1):
    course_id = np.random.randint(1, NUM_COURSES+1)
    persistence = np.random.uniform(0.3, 1.0)  # student seriousness

    completed_course = 1

    for chapter in range(1, CHAPTERS_PER_COURSE+1):
        difficulty = chapter / CHAPTERS_PER_COURSE
        time_spent = np.random.normal(40, 10) * persistence * (1 - difficulty)
        time_spent = max(5, time_spent)

        score = np.random.normal(70, 15) * persistence * (1 - difficulty)
        score = np.clip(score, 0, 100)

        dropout_prob = (difficulty * 0.6) + (score < 40) * 0.4

        if np.random.rand() < dropout_prob:
            completed_course = 0
            break

        rows.append([
            student_id,
            course_id,
            chapter,
            chapter,
            round(time_spent, 1),
            round(score, 1),
            1
        ])

    if completed_course == 0:
        rows.append([
            student_id,
            course_id,
            chapter,
            chapter,
            round(time_spent, 1),
            round(score, 1),
            0
        ])

df = pd.DataFrame(rows, columns=[
    "student_id",
    "course_id",
    "chapter_id",
    "chapter_order",
    "time_spent_minutes",
    "score",
    "completed"
])

df.to_csv("data/synthetic_learning_data.csv", index=False)
