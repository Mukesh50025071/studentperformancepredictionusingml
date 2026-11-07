"""Train and evaluate a student-performance classification model."""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "student_performance.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
FEATURES = [
    "study_hours_per_day", "attendance_percentage", "previous_exam_score",
    "assignment_completion_percentage", "sleep_hours_per_day",
    "extracurricular_participation", "absences"
]

def main():
    if not DATA.exists():
        raise FileNotFoundError("Dataset not found. Run: python generate_dataset.py")
    df = pd.read_csv(DATA)
    X, y = df[FEATURES], df["performance_category"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=250, max_depth=10, min_samples_leaf=2,
        random_state=42, class_weight="balanced"
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision_weighted": float(precision_score(y_test, pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_test, pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_test, pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, pred, labels=["High", "Low", "Medium"]).tolist(),
        "labels_order": ["High", "Low", "Medium"],
        "test_samples": int(len(y_test)),
        "dataset_type": "synthetic educational data"
    }
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR / "student_performance_model.joblib")
    (MODEL_DIR / "feature_names.json").write_text(json.dumps(FEATURES, indent=2))
    (REPORT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (REPORT_DIR / "classification_report.txt").write_text(
        classification_report(y_test, pred, zero_division=0)
    )
    print(json.dumps(metrics, indent=2))
    print(classification_report(y_test, pred, zero_division=0))

if __name__ == "__main__":
    main()
