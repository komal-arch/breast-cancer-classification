# Streamlit app for ML Assignment 2 - upload test data, pick a model, see how it does.
import json

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)

METRIC_ORDER = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
METRIC_INFO = {
    "Accuracy": "How many predictions it got right, out of all of them",
    "AUC": "How good the model is at telling malignant and benign apart",
    "Precision": "When it says malignant, how often that's actually true",
    "Recall": "Out of all the real malignant cases, how many it found",
    "F1": "One number that balances precision and recall",
    "MCC": "One number for how good the predictions are overall, even if one class has way more cases than the other",
}
CMAP = "Blues"

st.set_page_config(page_title="Breast Cancer Classifier", layout="wide")

st.markdown(
    "<style>.block-container { padding-left: 4rem; padding-right: 4rem; }</style>",
    unsafe_allow_html=True,
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

st.caption("M.Tech (AIML), BITS Pilani WILP")
info_rows = [
    ("Name", "Komal Kumari"),
    ("BITS ID", "2026ak05718"),
    ("Assignment", "ML Assignment 2"),
]
rows_html = "".join(
    f"<tr style='border-bottom:1px solid #e6e6e6;'>"
    f"<td style='padding:6px 12px; color:#666;'>{field}</td>"
    f"<td style='padding:6px 12px;'>{value}</td></tr>"
    for field, value in info_rows
)
st.markdown(
    f"<table style='width:420px; max-width:100%; margin:8px 0; "
    f"border-collapse:collapse; font-size:14px;'>{rows_html}</table>",
    unsafe_allow_html=True,
)

with st.container(border=True):
    st.subheader("Input")
    uploaded_file = st.file_uploader("Upload test data CSV", type=["csv"])
    model_name = st.selectbox("Pick a model", list(model_files.keys()))

    if uploaded_file is None:
        st.info("Upload a CSV with the features and a `target` column "
                 "(or use `test_data.csv` from the repo).")

if uploaded_file is not None:
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

        metrics_row = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "AUC": roc_auc_score(y_true, y_proba),
            "Precision": precision_score(y_true, y_pred),
            "Recall": recall_score(y_true, y_pred),
            "F1": f1_score(y_true, y_pred),
            "MCC": matthews_corrcoef(y_true, y_pred),
        }

        with st.container(border=True):
            st.subheader(f"{model_name} Results")

            metrics_df = pd.DataFrame({
                "Metric": METRIC_ORDER,
                "Value": [metrics_row[m] for m in METRIC_ORDER],
            }).set_index("Metric")
            st.dataframe(
                metrics_df.style.format("{:.4f}").background_gradient(cmap=CMAP),
                use_container_width=True,
            )
            with st.expander("What do these metrics mean?"):
                for m in METRIC_ORDER:
                    st.write(f"**{m}** - {METRIC_INFO[m]}")

            st.divider()

            col_left, col_right = st.columns(2)

            with col_left:
                st.write("Confusion matrix")
                cm = confusion_matrix(y_true, y_pred)
                cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
                st.dataframe(
                    cm_df.style.background_gradient(cmap=CMAP, axis=None),
                    use_container_width=True,
                )

            with col_right:
                st.write("Classification report")
                report = classification_report(
                    y_true, y_pred, target_names=target_names, output_dict=True
                )
                report_df = pd.DataFrame(report).transpose().round(4)
                st.dataframe(
                    report_df.style.background_gradient(
                        cmap=CMAP, subset=["precision", "recall", "f1-score"]
                    ),
                    use_container_width=True,
                )
