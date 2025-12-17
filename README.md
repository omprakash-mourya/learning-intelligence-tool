# 📊 AI-Powered Learning Intelligence Tool

## Overview
This project is an **executable AI-powered learning intelligence tool** built for an internship or training platform use case.  
It analyzes learner activity data to predict course completion, identify at-risk students early, detect difficult chapters, and generate actionable insights for mentors and administrators.

The project is intentionally designed as a **production-style AI system**, not a notebook-based experiment.

---

## 🔗 Live Deployment (Streamlit)
The tool is deployed as a live, publicly accessible Streamlit application:

**Live App URL:**  
https://learning-intelligence-tool.streamlit.app/

This deployment allows evaluators to directly interact with the AI system without local setup.

---

## Key AI Capabilities
- **Course Completion Prediction**  
  Binary classification to predict whether a student will complete a course.

- **Early Risk Detection**  
  Students are categorized into high, medium, or low risk groups based on predicted completion probability.

- **Chapter Difficulty Detection**  
  Chapters are analyzed and ranked using dropout rate, average score, and time spent.

- **Insight Generation**  
  Human-readable insights such as:
  - High-risk student lists  
  - Key behavioral factors affecting completion  
  - Chapters requiring instructional improvement  

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
Executable Streamlit Interface  

Training and inference are cleanly separated to ensure reproducibility and maintainable system design.

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

A sample dataset is provided in the `data/` directory.

---

## Model Details
- **Model Type:** Logistic Regression  
- **Target Variable:** Course completion (completed)  
- **Training:** Performed offline using aggregated student–course level features  
- **Inference:** Model is loaded from a saved file for reproducible predictions  

### Features Used
- Total and average time spent  
- Average, minimum, and standard deviation of scores  
- Number of chapters attempted  
- Chapter completion ratio  
- Last chapter score  

Logistic Regression was selected due to its interpretability, stability, and suitability for structured tabular data.

---

## Synthetic Data Usage
Due to privacy constraints and the absence of publicly available learner behavior datasets, **synthetic data** was generated to simulate realistic learning patterns.

The synthetic data generation logic models:
- Increasing difficulty across chapters  
- Correlation between learner effort, assessment scores, and completion  
- Natural dropout behavior  

This project does **not** claim real-world accuracy and is intended to demonstrate AI system design and deployment capability.

---

## How to Run the Tool Locally (Executable Deployment)

### 1. Install dependencies
pip install -r requirements.txt

### 2. Run the application
streamlit run app.py


### 3. Use the tool
Upload the provided sample CSV or your own dataset in the specified format to view predictions, risk flags, chapter difficulty analysis, and insights.

---

## Testing
Basic unit tests are included to validate:
- Feature engineering logic  
- Model prediction sanity (probabilities remain within valid bounds)  

Run tests using:
pytest


---

## AI Usage Disclosure
AI tools (including ChatGPT) were used for **guidance on project structuring, debugging, and code review**.  
All machine learning logic, feature engineering, synthetic data generation logic, model training, and system integration were implemented, validated, and understood independently.

---

## Limitations
- The model is trained on synthetic data and does not represent real-world learner performance.  
- Results reflect controlled assumptions used during data generation.  
- The system is intended as a prototype to demonstrate learning intelligence workflows.

---

## Conclusion
This project demonstrates the ability to:
- Build a usable and executable AI-powered tool  
- Integrate machine learning into a production-style pipeline  
- Expose AI functionality via a runnable interface  
- Apply responsible, transparent, and explainable AI practices  
