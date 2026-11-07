# Streamlit app for ML Assignment 2 - upload test data, pick a model, see how it does.
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

st.set_page_config(page_title="Breast Cancer Classifier", layout="wide")


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
st.write(
    "This is a small app I built for my ML assignment. I trained five "
    "different classification models on the same breast cancer dataset, "
    "and here you can try each one out on test data and compare how they do. "
    "Nothing is being retrained on this page, the models were trained "
    "beforehand and are just loaded here."
)

st.sidebar.header("1. Upload test data")
uploaded_file = st.sidebar.file_uploader("Test data CSV", type=["csv"])

st.sidebar.header("2. Pick a model")
model_name = st.sidebar.selectbox("Model", list(model_files.keys()))

if uploaded_file is None:
    st.info("Upload a CSV with the 30 breast-cancer features and a `target` "
             "column (0 for malignant, 1 for benign) to see it in action. "
             "If you don't have one handy, there's a `test_data.csv` file "
             "in the repo you can use.")
    st.subheader("Columns it expects")
    st.code(", ".join(feature_names + ["target"]))
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
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Accuracy", f"{accuracy:.4f}")
        m2.metric("AUC", f"{auc:.4f}")
        m3.metric("Precision", f"{precision:.4f}")
        m4.metric("Recall", f"{recall:.4f}")
        m5.metric("F1 Score", f"{f1:.4f}")
        m6.metric("MCC", f"{mcc:.4f}")

        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("**Confusion matrix**")
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
            st.markdown("**Classification report**")
            report = classification_report(
                y_true, y_pred, target_names=target_names, output_dict=True
            )
            st.dataframe(pd.DataFrame(report).transpose().round(4))

        with st.expander("See the data I uploaded"):
            st.dataframe(df.head())

st.sidebar.markdown("---")
st.sidebar.caption("Komal Kumari (2026ak05718) - ML Assignment 2")
