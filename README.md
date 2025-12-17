# 📊 AI-Powered Learning Intelligence Tool

## Overview
This project is an executable AI-powered tool designed to analyze learner activity data for an internship or training platform.  
It predicts course completion, identifies at-risk students early, detects difficult chapters, and generates human-readable insights for mentors and administrators.

The focus of this project is on AI system design and deployment, not notebook-based experimentation.

---

## Key AI Capabilities
- Course Completion Prediction  
  Binary classification to predict whether a student will complete a course.

- Early Risk Detection  
  Students are flagged as high, medium, or low risk based on predicted completion probability.

- Chapter Difficulty Detection  
  Chapters are ranked using dropout rate, average score, and time spent.

- Insight Generation  
  The tool produces actionable insights such as:
  - High-risk student lists
  - Key behavioral indicators affecting completion
  - Chapters needing improvement

---

## System Architecture
CSV Input  
↓  
Data Validation  
↓  
Feature Engineering  
↓  
Trained ML Model (Logistic Regression)  
↓  
Predictions + Risk Flags  
↓  
Insights & Reports  
↓  
Streamlit Interface  

Training and inference are separated to ensure reproducibility and clean system design.

---

## Technology Stack
- Python 3
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib
- Pytest

---

## Input Data Format
The tool accepts a CSV file containing learner activity data with the following columns:

- student_id  
- course_id  
- chapter_id  
- chapter_order  
- time_spent_minutes  
- score  
- completed (0 = dropped out, 1 = completed)

A sample dataset is provided in the data directory.

---

## Model Details
- Model Type: Logistic Regression  
- Target Variable: Course completion (completed)  
- Training: Performed offline using aggregated student–course features  
- Inference: Model is loaded from a saved file for reproducible predictions  

Features used:
- Total and average time spent
- Average, minimum, and standard deviation of scores
- Number of chapters attempted
- Chapter completion ratio
- Last chapter score

Logistic Regression was chosen due to its interpretability, stability, and suitability for structured tabular data.

---

## Synthetic Data Usage
Due to privacy constraints and the absence of publicly available learner behavior datasets, synthetic data was generated to simulate realistic learning patterns.

The synthetic data generation logic models:
- Increasing difficulty across chapters
- Correlation between effort, scores, and completion
- Natural dropout behavior

This project does not claim real-world accuracy and is intended to demonstrate AI system design and deployment.

---

## How to Run the Tool

Install dependencies:
pip install -r requirements.txt

Run the application:
streamlit run app.py

Upload the provided sample CSV or your own dataset in the specified format to view predictions and insights.

---

## Testing
Basic unit tests are included to validate:
- Feature generation
- Model prediction sanity (probabilities remain within valid bounds)

Run tests using:
pytest

---

## AI Usage Disclosure
AI tools (including ChatGPT) were used for guidance on project structuring and code review.  
All machine learning logic, feature design, synthetic data generation logic, and system integration were implemented and verified independently.

---

## Limitations
- The model is trained on synthetic data and does not represent real-world learner performance.
- Results reflect controlled assumptions used during data generation.
- The system is designed as a prototype for learning intelligence workflows.

---

## Conclusion
This project demonstrates the ability to:
- Build a usable AI-powered tool
- Integrate machine learning into a production-style pipeline
- Expose AI functionality via an executable interface
- Apply responsible and transparent AI practices
