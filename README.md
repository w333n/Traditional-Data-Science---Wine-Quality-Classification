# Red Wine Quality Classification

## Project Overview

This notebook performs a complete machine learning pipeline on the **UCI Red Wine Quality dataset**, covering:

1. **Exploratory Data Analysis (EDA)** — data overview, class distribution, univariate & bivariate analysis
2. **Feature Selection** — Filter methods (Pearson correlation, ANOVA F-test) and Embedded methods (Lasso, Random Forest importance)
3. **Modelling** — Logistic Regression, Decision Tree, and Random Forest classifiers
4. **Model Comparison** — Accuracy, Precision, Recall, F1, Confusion Matrix

**Target variable:** Binary classification
- `lower` (0): quality score 3–5
- `better` (1): quality score 6–8

## How to Run the Streamlit App

**Prerequisite:** Python 3.10+ installed. Both `app.py` and `winequality-red.csv` must be in the same directory.

**Steps:**

1. Place `app.py` and `winequality-red.csv` in the same folder.
2. Install dependencies:
   ```bash
   pip install streamlit pandas numpy scikit-learn matplotlib seaborn
3. Launch the app: streamlit run app.py
4. A browser window opens automatically at:http://localhost:8501
