[README (1).md](https://github.com/user-attachments/files/32383331/README.1.md)
# Stroke Prediction using Machine Learning

This project predicts whether a patient is likely to suffer a **stroke** based on demographic and health-related attributes such as age, average glucose level, BMI, hypertension, heart disease, smoking status, and lifestyle information.

Stroke is one of the leading causes of death and long-term disability worldwide. Early identification of high-risk individuals from routine health data can help prioritize preventive care and medical attention. This project explores how machine learning models can flag patients who are more likely to experience a stroke, using easily-collected clinical and demographic features.

## Dataset

- **File:** `12.csv`
- **Rows:** 5,110 patients
- **Target variable:** `stroke` (1 = had a stroke, 0 = no stroke) — a binary classification problem
- **Features:** `gender`, `age`, `hypertension`, `heart_disease`, `ever_married`, `work_type`, `Residence_type`, `avg_glucose_level`, `bmi`, `smoking_status`

## Project Workflow

1. **Data Description & EDA** — Explored feature distributions, class balance, and relationships between features and stroke outcome using histograms, count plots, box plots, and a correlation heatmap.
2. **Preprocessing** — Handled missing `bmi` values (filled with median), removed duplicates, and label-encoded categorical features.
3. **Handling Class Imbalance** — The dataset is highly imbalanced (~4.9% stroke cases). Addressed using `class_weight='balanced'` for Logistic Regression and Decision Tree, and a manually implemented **SMOTE** oversampling technique for the Neural Network.
4. **Supervised Models** — Trained and compared:
   - Logistic Regression
   - Decision Tree Classifier
   - Neural Network (MLPClassifier)
5. **Evaluation** — Compared models using Accuracy, Precision, Recall, F1-score, ROC-AUC, and confusion matrices.
6. **Unsupervised Learning** — Applied K-Means clustering (with PCA visualization) to see how well unsupervised clusters align with actual stroke labels.

## Key Results

| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|---|---|---|---|---|---|
| Logistic Regression | ~0.75 | ~0.14 | ~0.80 | ~0.24 | **0.839** |
| Decision Tree | ~0.79 | — | ~0.62 | — | 0.718 |
| Neural Network | **0.823** | — | 0.38 | — | 0.758 |

**Best model:** Logistic Regression — highest AUC and recall, meaning it catches the most actual stroke cases, which matters most for a medical screening task.

> Note: Before handling class imbalance, all models scored ~95% accuracy but 0.0000 precision/recall/F1 on the stroke class — they were simply predicting "no stroke" for everyone. This is a classic example of the **accuracy paradox** on imbalanced data.

## Tech Stack

- Python
- pandas, numpy
- matplotlib, seaborn
- scikit-learn

## How to Run

1. Clone this repository
   ```
   git clone <your-repo-url>
   cd <your-repo-name>
   ```
2. Install dependencies
   ```
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
3. Run the script
   ```
   python group_12_24101221_24101589.py
   ```
   (Make sure `12.csv` is in the same folder.)

## Files

- `group_12_24101221_24101589.py` — Main analysis and modeling script
- `12.csv` — Stroke prediction dataset

## Authors

- Group 12 (ID: 24101221, 24101589)
