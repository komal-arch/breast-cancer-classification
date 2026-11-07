# ML Assignment 2

| | |
|---|---|
| Name | Komal Kumari |
| BITS ID | 2026ak05718 |
| Programme | M.Tech (AIML), BITS Pilani WILP |

## a. Problem Statement

For this assignment I had to train a few different classification models on
the same dataset, see how each one performs, and then wrap them in a
Streamlit app so someone can upload test data, pick a model, and see the
results without needing to retrain anything.

## b. Dataset Description

**Dataset used:** Breast Cancer Wisconsin (Diagnostic) Dataset. I loaded it
straight from `sklearn.datasets.load_breast_cancer` instead of downloading a
CSV - it's the same dataset, originally from the UCI Machine Learning
Repository, just bundled with scikit-learn.

- **Task:** Binary classification - predict if a tumor is **malignant** or
  **benign**.
- **Rows:** 569 (assignment needs at least 500)
- **Features:** 30 numeric features (assignment needs at least 12), such as
  radius, texture, perimeter, area, smoothness, compactness, concavity,
  symmetry and fractal dimension, each measured as a mean, standard error
  and "worst" value.
- **Class split:** 212 malignant, 357 benign.
- **Train/test split:** 80/20, stratified on the target, random_state=42.
- **Scaling:** Features were standardized using `StandardScaler` for the
  models that care about scale - Logistic Regression, KNN and SVM. Decision
  Tree and Naive Bayes were trained on the raw features.
- **Test data for the app:** The 20% test split (114 rows) is saved as
  [`test_data.csv`](test_data.csv) and can be uploaded directly into the app.

## c. GitHub Repository Link

https://github.com/komal-arch/breast-cancer-classification

## d. Models Used

All five models were trained on the same train/test split, and the metrics
below are calculated on the held-out test set.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9123 | 0.9157 | 0.9559 | 0.9028 | 0.9286 | 0.8174 |
| KNN | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| Naive Bayes (Gaussian) | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| Support Vector Machine | 0.9825 | 0.9950 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |

(Numbers come from running [`train.py`](train.py), also saved in
[`model_comparison.csv`](model_comparison.csv).)

## e. Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Performed the best. Once the features are scaled, the two classes are almost linearly separable, so a simple linear model does a great job. Highest accuracy and MCC, tied with SVM. |
| Decision Tree | Weakest of the five. A single tree with no depth limit tends to overfit the training data and gets confused by borderline cases, so it ends up with the lowest accuracy, AUC and MCC. |
| KNN | Did quite well once the data was scaled. Since distance between points actually means something after scaling, the nearest neighbours line up well with the real classes. Recall was especially good here. |
| Naive Bayes (Gaussian) | Second-best AUC even though it assumes the features are independent, which isn't really true (radius, perimeter and area are obviously related). That assumption costs it a bit on precision and MCC compared to Logistic Regression and SVM. |
| Support Vector Machine | Basically tied with Logistic Regression for the best model. The RBF kernel finds a similarly good boundary once the features are scaled - same accuracy, precision, recall, F1 and MCC. |
| **Overall, Winner for your dataset?** | **Logistic Regression and SVM are tied** at the top (accuracy 0.9825, F1 0.9861, MCC 0.9623). Between the two, Logistic Regression is the better pick to actually use since it's simpler, easier to explain, and has a slightly higher AUC (0.9954 vs 0.9950). |

## Streamlit App

**Live app:** https://2026ak05718-breast-cancer-classification.streamlit.app/

What it lets you do:
1. Upload a test CSV (the 30 features plus a `target` column).
2. Pick which of the 5 trained models to use from a dropdown.
3. See all 6 metrics for that model on your uploaded data.
4. See a confusion matrix and the full classification report.

It only loads the already-trained models from `model/` and the saved
`StandardScaler` - nothing gets retrained on whatever you upload.

## What's in This Repo

- `app.py` - the Streamlit app
- `train.py` - trains all 5 models and saves them, plus `test_data.csv`
- `requirements.txt` - packages needed to run everything
- `test_data.csv` - held-out test rows, ready to upload into the app
- `model_comparison.csv` - the raw metrics table
- `model/` - the saved models (`.joblib` files), the scaler, and `meta.json`

## Running It Locally

```bash
pip install -r requirements.txt
python train.py        # optional, only needed to regenerate the models
streamlit run app.py
```

## BITS Virtual Lab Execution

Screenshot of this assignment running on the BITS Virtual Lab:
`<ADD_SCREENSHOT_REFERENCE_HERE>`
