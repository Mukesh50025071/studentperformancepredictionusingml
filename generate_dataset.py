"""Generate a reproducible synthetic student-performance dataset for learning."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
OUT = ROOT / "data" / "student_performance.csv"

def main():
    rng = np.random.default_rng(42)
    n = 1200
    study_hours = rng.uniform(0, 8, n)
    attendance = rng.uniform(45, 100, n)
    previous_score = rng.uniform(25, 100, n)
    assignment_completion = rng.uniform(0, 100, n)
    sleep_hours = rng.uniform(3.5, 10, n)
    extracurricular = rng.integers(0, 2, n)
    absences = rng.integers(0, 25, n)

    noise = rng.normal(0, 8, n)
    score = (
        8 + 4.2 * study_hours + 0.20 * attendance
        + 0.22 * previous_score + 0.12 * assignment_completion
        + 1.1 * sleep_hours + 2.0 * extracurricular
        - 0.75 * absences + noise
    )
    score = np.clip(score, 0, 100)
    category = np.where(score >= 75, "High",
               np.where(score >= 50, "Medium", "Low"))

    df = pd.DataFrame({
        "study_hours_per_day": study_hours.round(1),
        "attendance_percentage": attendance.round(1),
        "previous_exam_score": previous_score.round(1),
        "assignment_completion_percentage": assignment_completion.round(1),
        "sleep_hours_per_day": sleep_hours.round(1),
        "extracurricular_participation": extracurricular,
        "absences": absences,
        "final_score_simulated": score.round(1),
        "performance_category": category
    })
    OUT.parent.mkdir(exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Created {len(df)} synthetic records at {OUT}")

if __name__ == "__main__":
    main()
