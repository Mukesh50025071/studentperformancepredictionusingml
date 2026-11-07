from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "student_performance_model.joblib"
FEATURES_PATH = ROOT / "models" / "feature_names.json"

st.set_page_config(page_title="Student Performance Prediction", page_icon="🎓", layout="centered")
st.title("🎓 Student Performance Prediction System")
st.write("An educational machine-learning project that estimates a student's performance category from study and learning indicators.")
st.info("This is a classroom demonstration using synthetic data. It is not validated for real student assessment and must not be used for grading, admissions, disciplinary action, or other high-impact decisions.")

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists() or not FEATURES_PATH.exists():
        return None, None
    return joblib.load(MODEL_PATH), json.loads(FEATURES_PATH.read_text())

model, features = load_model()
if model is None:
    st.error("Model files are missing. Run `python generate_dataset.py` followed by `python train_model.py`.")
    st.stop()

st.subheader("Enter student learning details")
with st.form("student_inputs"):
    study = st.slider("Average study hours per day", 0.0, 12.0, 3.0, 0.5)
    attendance = st.slider("Attendance (%)", 0.0, 100.0, 80.0, 1.0)
    previous = st.slider("Previous exam score (%)", 0.0, 100.0, 65.0, 1.0)
    assignments = st.slider("Assignment completion (%)", 0.0, 100.0, 75.0, 1.0)
    sleep = st.slider("Average sleep hours per day", 0.0, 12.0, 7.0, 0.5)
    extracurricular = st.selectbox("Extracurricular participation", ["No", "Yes"])
    absences = st.number_input("Number of absences", min_value=0, max_value=100, value=3, step=1)
    submitted = st.form_submit_button("Predict performance", type="primary")

if submitted:
    row = pd.DataFrame([{
        "study_hours_per_day": study,
        "attendance_percentage": attendance,
        "previous_exam_score": previous,
        "assignment_completion_percentage": assignments,
        "sleep_hours_per_day": sleep,
        "extracurricular_participation": 1 if extracurricular == "Yes" else 0,
        "absences": absences
    }], columns=features)
    prediction = model.predict(row)[0]
    probabilities = model.predict_proba(row)[0]
    probability_map = dict(zip(model.classes_, probabilities))
    st.subheader("Prediction result")
    if prediction == "High":
        st.success(f"Predicted category: {prediction}")
    elif prediction == "Medium":
        st.warning(f"Predicted category: {prediction}")
    else:
        st.error(f"Predicted category: {prediction}")
    st.metric("Model confidence for predicted category", f"{probability_map[prediction] * 100:.1f}%")
    st.caption("This confidence is a model output, not a guarantee of a student's future result.")

st.divider()
st.subheader("Project information")
st.markdown("""
- **Algorithm:** Random Forest Classifier
- **Input features:** Study hours, attendance, previous score, assignment completion, sleep, extracurricular participation, and absences.
- **Output:** Low, Medium, or High performance category.
- **Dataset:** Reproducible synthetic educational data generated for this project.
""")
