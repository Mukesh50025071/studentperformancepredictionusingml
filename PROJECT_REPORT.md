# PROJECT REPORT
## Student Performance Prediction System Using Machine Learning

### 1. Abstract
The Student Performance Prediction System is an educational machine-learning application designed to demonstrate how learning-related indicators can be used to classify simulated student performance. The project uses Python, Pandas, NumPy, and Scikit-learn to generate a reproducible synthetic dataset, train a Random Forest classifier, and evaluate the model using a held-out test set. A Streamlit interface allows users to enter study hours, attendance, previous exam score, assignment completion, sleep hours, extracurricular participation, and absences. The model returns one of three categories: Low, Medium, or High. The project demonstrates the stages of an AI implementation workflow, including data preparation, model development, evaluation, deployment, and documentation. The dataset is synthetic, so the results are intended only for academic demonstration and must not be used to make decisions about real students.

### 2. Introduction
Student achievement can be associated with several factors, including study habits, attendance, prior academic performance, assignment completion, and learning environment. Machine learning provides methods for learning patterns from data and producing predictions. This project builds a three-class classification prototype using simulated data to demonstrate a complete machine-learning workflow.

### 3. Problem statement
It can be useful in an academic exercise to understand how multiple learning indicators can be combined by a classification algorithm. The project demonstrates this process by training a model to map input indicators to simulated performance categories. It does not claim to measure a student's ability or predict actual academic outcomes.

### 4. Objectives
- Generate a reproducible educational dataset for experimentation.
- Apply a supervised machine-learning classification algorithm.
- Divide data into training and testing subsets.
- Evaluate the model with standard classification metrics.
- Build a simple interactive web application.
- Document project planning, implementation, testing, and limitations.

### 5. Scope
The project is limited to a synthetic dataset with 1,200 generated records. The model predicts Low, Medium, or High categories according to a simulated score formula. It is a learning prototype and is not intended for real student evaluation or high-impact decisions.

### 6. Dataset description
The dataset is generated locally by `generate_dataset.py` using a fixed random seed. It contains seven input variables: study hours per day, attendance percentage, previous exam score, assignment completion percentage, sleep hours per day, extracurricular participation, and absences. A simulated final score is calculated with a formula and random noise. Scores below 50 are labelled Low, scores from 50 to below 75 are Medium, and scores of 75 or above are High. The generated CSV is saved as `data/student_performance.csv`.

### 7. Technologies used
- **Python:** Main programming language.
- **NumPy:** Reproducible numerical data generation.
- **Pandas:** Dataset handling.
- **Scikit-learn:** Data splitting, Random Forest model, and evaluation.
- **Streamlit:** User interface.
- **Joblib:** Model persistence.

### 8. Methodology
1. Generate 1,200 synthetic student records.
2. Separate seven input features (X) and the performance category (y).
3. Split records into 80% training and 20% testing using stratification.
4. Train a Random Forest classifier with 250 trees.
5. Predict categories for the test subset.
6. Calculate accuracy, weighted precision, weighted recall, weighted F1-score, and confusion matrix.
7. Save the trained model and feature list.
8. Load the saved model in Streamlit and accept sample input values.

### 9. System modules
- **Data generation module:** Creates simulated student records.
- **Preprocessing module:** Selects features and splits the dataset.
- **Training module:** Trains the Random Forest classifier.
- **Evaluation module:** Calculates classification metrics.
- **Model storage module:** Saves the model and feature names.
- **Prediction interface:** Accepts user inputs and displays the predicted category.

### 10. System requirements
**Hardware:** A computer with at least 4 GB RAM and a modern dual-core processor is sufficient for this small demonstration.
**Software:** Python 3.10 or newer, pip, and a web browser.

### 11. Testing
Run `python generate_dataset.py` and confirm that `data/student_performance.csv` is created with 1,200 rows. Run `python train_model.py` and confirm that model files and evaluation reports are created. Run `streamlit run app.py`, enter values, and press **Predict performance**. Confirm that a category and model confidence are displayed.

### 12. Results
After running the training script, copy the values from `reports/metrics.json` into this table.

| Metric | Result |
|---|---|
| Accuracy | Fill from metrics.json |
| Weighted precision | Fill from metrics.json |
| Weighted recall | Fill from metrics.json |
| Weighted F1-score | Fill from metrics.json |
| Confusion matrix | Fill from metrics.json |

Because the dataset is synthetic and generated from a score formula, these metrics only demonstrate learning of the simulated pattern. They do not measure real-world educational effectiveness.

### 13. Advantages
- Easy to understand and reproduce.
- Does not require downloading private student records.
- Demonstrates a multi-class classification problem.
- Includes an interactive interface.
- Separates data generation, training, and prediction.

### 14. Limitations and ethical considerations
- The dataset is synthetic, not collected from real students.
- The target categories are generated from an artificial formula.
- Real educational outcomes depend on many contextual factors not represented here.
- Predictions may reinforce misleading assumptions if used outside this demonstration.
- The application must not be used for grading, admissions, scholarships, discipline, or other decisions affecting real students.

### 15. Future enhancements
- Evaluate a properly governed, consented, representative real-world dataset for research purposes.
- Compare Random Forest with Logistic Regression, Decision Tree, and Gradient Boosting.
- Add cross-validation and model calibration.
- Study fairness across relevant groups with appropriate safeguards.
- Add explainability and a model card describing intended use and limitations.

### 16. Conclusion
The Student Performance Prediction System demonstrates an end-to-end supervised machine-learning workflow. It generates data, trains and evaluates a Random Forest classifier, saves the model, and provides a Streamlit interface. The project offers practical experience in AI implementation and technical project documentation while clearly distinguishing a synthetic classroom exercise from a validated educational assessment system.

### References
1. Scikit-learn documentation: https://scikit-learn.org/stable/
2. Random Forest documentation: https://scikit-learn.org/stable/modules/ensemble.html#forest
3. Streamlit documentation: https://docs.streamlit.io/
