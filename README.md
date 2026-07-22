# Heart Disease Risk Prediction

## The Problem

Heart disease is one of the leading causes of death globally. Early identification of high-risk patients allows healthcare providers to intervene proactively, potentially saving lives. However, manually reviewing clinical data across large patient populations is time-consuming and prone to inconsistency.

This project builds a machine learning pipeline that predicts heart disease risk from clinical indicators, enabling faster, data-driven risk assessment to support medical decision-making.

---

## Results

| Model | Baseline AUC | After Tuning AUC |
|---|---|---|
| Logistic Regression | 0.935 | 0.931 |
| Support Vector Machine | 0.944 | 0.946 |
| Decision Tree | 0.770 | 0.952 |
| **Random Forest** | 0.912 | **0.957** |

**Best model: Random Forest with RandomizedSearchCV - AUC = 0.957**

The Random Forest with randomized hyperparameter search achieved the highest AUC across all models and tuning strategies, correctly identifying the majority of high-risk patients while maintaining low false-positive rates.

---

## Live App

An interactive Streamlit app is deployed publicly, allowing users to input clinical values and receive an instant heart disease risk prediction.

**App:** [Heart Disease Predictor on Streamlit Cloud](https://heart-disease-predictor-udacnzdexe5owihe5lkxd4.streamlit.app/)
---

## Dataset

- **Source:** UCI Heart Disease Dataset
- **Size:** 303 patients, 13 clinical features
- **Target:** Binary classification (heart disease present / absent)
- **Class distribution:** Slightly imbalanced - handled via `class_weight='balanced'`

**Features include:** age, sex, chest pain type, resting blood pressure, cholesterol, fasting blood sugar, resting ECG results, maximum heart rate, exercise-induced angina, ST depression, slope, number of major vessels, thalassemia type

---

## Pipeline

The project is structured as 6 sequential notebooks:

### 01 - Data Preprocessing
- Loaded and inspected UCI dataset
- Identified and imputed missing values (mode imputation for `ca` and `thal`)
- Train/test split with stratification to preserve class balance
- Saved cleaned data for downstream notebooks

### 02 - PCA Analysis
- Applied StandardScaler before PCA
- Determined that 11 principal components are needed to retain 95% of variance
- Visualized 2D PCA projection to assess class separability

### 03 - Feature Selection
- Applied three feature selection methods in parallel:
  - Random Forest feature importance
  - Recursive Feature Elimination (RFE)
  - SelectKBest (chi-squared)
- Top features identified: `oldpeak`, `ca`, `cp` (chest pain type)

### 04 - Supervised Learning (Baseline Models)
- Trained 4 baseline classifiers: Logistic Regression, SVM, Decision Tree, Random Forest
- Evaluated using AUC-ROC as the primary metric (appropriate for class imbalance)
- Applied `class_weight='balanced'` across all models

### 05 - Unsupervised Learning (Exploratory)
- Applied K-Means and Hierarchical clustering on the same dataset
- Elbow method used to select optimal k
- Cross-tabulated cluster assignments against actual disease labels
- K-Means Cluster 1 captured 74 of 89 disease-positive patients - confirming natural groupings in the data

### 06 - Hyperparameter Tuning
- Tuned all 4 models using GridSearchCV and RandomizedSearchCV
- Best result: Random Forest with RandomizedSearchCV - AUC = 0.957
- Final model saved with joblib for deployment

---

## Project Structure

```
├── 01_data_preprocessing.ipynb
├── 02_pca_analysis.ipynb
├── 03_feature_selection.ipynb
├── 04_supervised_learning.ipynb
├── 5_unsupervised_learning.ipynb
├── 06_hyperparameter_tuning.ipynb
├── app.py                          # Streamlit app
├── requirements.txt
└── model/                          # Saved model artifacts (joblib)
```

---

## Tools & Libraries

Python, Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn, Streamlit, Joblib

---

