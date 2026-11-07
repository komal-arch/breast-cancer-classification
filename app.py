"""
Streamlit app for ML Assignment 2.
Upload a test CSV, pick a trained model, and see how it scores -
no retraining happens here, the models are just loaded and used.
"""
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)

st.set_page_config(page_title="Breast Cancer Classifier Comparison", layout="wide")


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

st.title("Breast Cancer Classification - Model Comparison")
st.write(
    "Upload a test CSV (the features plus a `target` column), pick one of "
    "the five models on the left, and see how it does on your data. The "
    "models are already trained, nothing gets retrained here."
)

st.sidebar.header("1. Upload test data")
uploaded_file = st.sidebar.file_uploader("Test data CSV", type=["csv"])

st.sidebar.header("2. Choose a model")
model_name = st.sidebar.selectbox("Model", list(model_files.keys()))

if uploaded_file is None:
    st.info("Upload a CSV with the 30 breast-cancer features plus a `target` "
             "column (0 = malignant, 1 = benign) to get started. There's a "
             "ready-made `test_data.csv` in this repo you can use.")
    st.subheader("Expected columns")
    st.code(", ".join(feature_names + ["target"]))
else:
    df = pd.read_csv(uploaded_file)

    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        st.error(f"Uploaded CSV is missing required feature columns: {missing}")
    elif "target" not in df.columns:
        st.error("Uploaded CSV must contain a `target` column with the true labels.")
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

        st.subheader(f"Results - {model_name}")
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Accuracy", f"{accuracy:.4f}")
        m2.metric("AUC", f"{auc:.4f}")
        m3.metric("Precision", f"{precision:.4f}")
        m4.metric("Recall", f"{recall:.4f}")
        m5.metric("F1 Score", f"{f1:.4f}")
        m6.metric("MCC", f"{mcc:.4f}")

        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("**Confusion Matrix**")
            cm = confusion_matrix(y_true, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=target_names, yticklabels=target_names, ax=ax,
            )
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

        with col_right:
            st.markdown("**Classification Report**")
            report = classification_report(
                y_true, y_pred, target_names=target_names, output_dict=True
            )
            st.dataframe(pd.DataFrame(report).transpose().round(4))

        with st.expander("Preview uploaded data"):
            st.dataframe(df.head())

st.sidebar.markdown("---")
st.sidebar.caption("ML Assignment 2 - Breast Cancer Wisconsin (Diagnostic) dataset")
