# ML Assignment 2

M.Tech (AIML), BITS Pilani WILP

| Field | Value |
|---|---|
| Name | Komal Kumari |
| BITS ID | 2026ak05718 |
| Assignment | ML Assignment 2 |

## 1. Problem Statement

For this assignment I had to train a few different classification models on
the same dataset, see how each one does, and then wrap them in a small
Streamlit app so someone can upload test data, pick a model, and see the
results without needing to retrain anything.

## 2. Dataset Description

**Dataset used:** Breast Cancer Wisconsin (Diagnostic) Dataset. I loaded it
straight from `sklearn.datasets.load_breast_cancer` instead of downloading a
CSV - it's the same dataset, originally from the UCI Machine Learning
Repository, just bundled with scikit-learn so there's one less thing to go
wrong.

- **Task:** binary classification - is a tumor **malignant** or **benign**?
- **Rows:** 569 (assignment asks for at least 500)
- **Features:** 30 numeric features (assignment asks for at least 12) - things
  like radius, texture, perimeter, area, smoothness, compactness, concavity,
  symmetry and fractal dimension, each given as a mean, a standard error, and
  a "worst" value.
- **Class split:** 212 malignant, 357 benign.
- **Train/test split:** 80/20, stratified on the target, random_state=42.
- **Scaling:** I standardized the features with `StandardScaler` for the
  models that actually care about scale - Logistic Regression, KNN and SVM.
  Decision Tree and Naive Bayes were trained on the raw features since they
  don't need it.
- **Test data for the app:** the 20% test split (114 rows) is saved as
  [`test_data.csv`](test_data.csv), ready to upload into the app as is.

## 3. GitHub Repository Link

https://github.com/komal-arch/breast-cancer-classification

## 4. Models Used

All five models were trained on the same train/test split, and these numbers
are from the held-out test set.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9123 | 0.9157 | 0.9559 | 0.9028 | 0.9286 | 0.8174 |
| KNN | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| Naive Bayes (Gaussian) | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| Support Vector Machine | 0.9825 | 0.9950 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |

(These come straight from running [`train.py`](train.py); also saved in
[`model_comparison.csv`](model_comparison.csv) if you want the raw numbers.)

## 5. Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Came out on top. Once the features are scaled, the two classes are almost linearly separable, so even a simple linear model does really well. Highest accuracy and MCC, tied with SVM. |
| Decision Tree | The weakest of the five. A single tree with no depth limit overfits the training data a bit and gets tripped up on borderline cases, so it ends up lowest on accuracy, AUC and MCC. |
| KNN | Did quite well once scaled. Distance between points actually means something after scaling, so the nearest neighbours line up well with the real classes. Recall was especially good. |
| Naive Bayes (Gaussian) | Second-best AUC, even though it assumes the features are independent, which isn't really true here (radius, perimeter and area are obviously related). That assumption costs it a little on precision and MCC. |
| Support Vector Machine | Pretty much tied with Logistic Regression. The RBF kernel finds a similarly good boundary once the data is scaled - same accuracy, precision, recall, F1 and MCC. |
| **Overall, Winner for your dataset?** | **Logistic Regression and SVM, tied** at the top (accuracy 0.9825, F1 0.9861, MCC 0.9623). I'd go with Logistic Regression between the two since it's simpler, easier to explain, and has a slightly higher AUC (0.9954 vs 0.9950). |

## Streamlit App

**Live app:** https://2026ak05718-breast-cancer-classification.streamlit.app/

What it lets you do:
1. Upload a test CSV (the 30 features plus a `target` column).
2. Pick which of the 5 trained models to use.
3. See all 6 metrics for that model on your data.
4. See a confusion matrix and the full classification report.

It only loads the models that were already trained and saved in `model/`,
along with the `StandardScaler` - nothing gets retrained on whatever you
upload.

## What's in This Repo

- `app.py` - the Streamlit app
- `train.py` - trains all 5 models and saves them, plus `test_data.csv`
- `run-local.sh` - installs dependencies, trains the models, and starts the app, all in one go
- `requirements.txt` - packages needed to run everything
- `test_data.csv` - held-out test rows, ready to upload into the app
- `model_comparison.csv` - the raw metrics table
- `model/` - the saved models (`.joblib` files), the scaler, and `meta.json`

## Running It Locally

```bash
chmod +x run-local.sh
./run-local.sh
```

Or step by step:

```bash
pip install -r requirements.txt
python train.py        # optional, only needed to regenerate the models
streamlit run app.py
```

## BITS Virtual Lab Execution

Screenshot of this assignment running on the BITS Virtual Lab:
`<ADD_SCREENSHOT_REFERENCE_HERE>`
