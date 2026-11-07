# Streamlit app for ML Assignment 2 - upload test data, pick a model, see how it does.
import json

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)


@st.cache_resource
def load_meta():
    with open("model/meta.json") as f:
        return json.load(f)


@st.cache_resource
def load_model(path):
    return joblib.load(path)


meta = load_meta()
scaler = load_model("model/scaler.joblib")
feature_names = meta["feature_names"]
target_names = meta["target_names"]
scaled_models = set(meta["scaled_models"])
model_files = meta["model_files"]

st.title("Breast Cancer Classifier")

uploaded_file = st.file_uploader("Upload test data CSV", type=["csv"])
model_name = st.selectbox("Pick a model", list(model_files.keys()))

if uploaded_file is None:
    st.info("Upload a CSV with the features and a `target` column "
             "(or use `test_data.csv` from the repo).")
else:
    df = pd.read_csv(uploaded_file)

    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        st.error(f"Your file is missing some columns it needs: {missing}")
    elif "target" not in df.columns:
        st.error("Your file needs a `target` column with the actual labels in it.")
    else:
        X = df[feature_names]
        y_true = df["target"]

        model = load_model(model_files[model_name])
        X_input = scaler.transform(X) if model_name in scaled_models else X

        y_pred = model.predict(X_input)
        y_proba = model.predict_proba(X_input)[:, 1]

        accuracy = accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_proba)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)

        st.subheader(f"{model_name} results")
        st.write(f"Accuracy: {accuracy:.4f}")
        st.write(f"AUC: {auc:.4f}")
        st.write(f"Precision: {precision:.4f}")
        st.write(f"Recall: {recall:.4f}")
        st.write(f"F1 Score: {f1:.4f}")
        st.write(f"MCC: {mcc:.4f}")

        st.write("Confusion matrix:")
        cm = confusion_matrix(y_true, y_pred)
        st.dataframe(pd.DataFrame(cm, index=target_names, columns=target_names))

        st.write("Classification report:")
        report = classification_report(
            y_true, y_pred, target_names=target_names, output_dict=True
        )
        st.dataframe(pd.DataFrame(report).transpose().round(4))

        st.write("Uploaded data (first 5 rows):")
        st.dataframe(df.head())

st.caption("Komal Kumari (2026ak05718) - ML Assignment 2")
