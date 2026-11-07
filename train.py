"""
Trains 5 classification models on the Breast Cancer Wisconsin dataset and
saves the models, the test data, and a metrics table for the Streamlit app.

Dataset: sklearn.datasets.load_breast_cancer
569 rows, 30 numeric features, binary target (malignant=0, benign=1)
"""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report,
)

RANDOM_STATE = 42

data = load_breast_cancer(as_frame=True)
X = data.data
y = data.target  # 0 = malignant, 1 = benign
target_names = list(data.target_names)

X.to_csv("data_full.csv", index=False)
full_df = X.copy()
full_df["target"] = y
full_df.to_csv("data_full_with_target.csv", index=False)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, "model/scaler.joblib")

# this is what goes into the Streamlit app: raw features + target, just like
# a normal exported CSV of held-out rows
test_data = X_test.copy()
test_data["target"] = y_test.values
test_data.to_csv("test_data.csv", index=False)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": SVC(probability=True, random_state=RANDOM_STATE),
}

# these models need scaled input, the rest work fine on raw features
SCALED_MODELS = {"Logistic Regression", "KNN", "Support Vector Machine"}

results = []
for name, model in models.items():
    if name in SCALED_MODELS:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_proba),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }
    results.append(metrics)

    print(f"\n=== {name} ===")
    print(f"Accuracy : {metrics['Accuracy']:.4f}")
    print(f"AUC      : {metrics['AUC']:.4f}")
    print(f"Precision: {metrics['Precision']:.4f}")
    print(f"Recall   : {metrics['Recall']:.4f}")
    print(f"F1       : {metrics['F1']:.4f}")
    print(f"MCC      : {metrics['MCC']:.4f}")
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=target_names))

    filename = "model/" + name.lower().replace(" ", "_") + ".joblib"
    joblib.dump(model, filename)

results_df = pd.DataFrame(results).set_index("Model").round(4)
results_df.to_csv("model_comparison.csv")

meta = {
    "scaled_models": sorted(SCALED_MODELS),
    "feature_names": list(X.columns),
    "target_names": target_names,
    "model_files": {
        name: "model/" + name.lower().replace(" ", "_") + ".joblib" for name in models
    },
}
with open("model/meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print("\n\n=== Comparison Table ===")
print(results_df.to_string())
print("\nSaved models to model/, test data to test_data.csv, comparison to model_comparison.csv")
