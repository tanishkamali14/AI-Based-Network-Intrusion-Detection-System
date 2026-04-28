import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px

# Config
DATA_DIR        = "data/processed"
PLOTS_DIR       = "outputs/plots"
REPORTS_DIR     = "outputs/reports"
MODEL_DIR       = "models"
ANOMALY_MODELS  = ["isolation_forest", "lof", "oneclass_svm", "ensemble"]

# ---------------------
# Helper functions
# ---------------------
def load_metrics():
    unified_path = os.path.join(REPORTS_DIR, "unified_metrics.csv")
    combined_path = os.path.join(REPORTS_DIR, "combined_metrics.csv")
    unified = pd.read_csv(unified_path)
    combined = pd.read_csv(combined_path)
    return unified, combined

def display_bar_charts(metrics_df):
    # Bar charts for precision, recall, and F1
    for metric in ["precision","recall","f1"]:
        fig = px.bar(metrics_df, x="model", y=metric,
                     title=f"{metric.capitalize()} by Model",
                     color="model")
        st.plotly_chart(fig, use_container_width=True)

    for metric in ["roc_auc", "pr_auc"]:
        fig = px.bar(metrics_df, x="model", y=metric,
                     title=f"{metric.upper()} by Model",
                     color="model")
        st.plotly_chart(fig, use_container_width=True)

def load_plot_image(filename):
    path = os.path.join(PLOTS_DIR, filename)
    if os.path.exists(path):
        return path
    return None

def show_plot(image_file, title=""):
    if image_file:
        st.image(image_file, caption=title, use_column_width=True)

def load_models():
    models = {}
    # Load supervised models
    for name in ["best_rf.pkl", "best_gb.pkl"]:
        path = os.path.join(MODEL_DIR, name)
        if os.path.exists(path):
            models[name[:-4]] = joblib.load(path)
    # Load unsupervised models
    for name in ANOMALY_MODELS:
        pkl = f"{name}.pkl"
        path = os.path.join(MODEL_DIR, pkl)
        if os.path.exists(path):
            models[name] = joblib.load(path)
    return models

def run_supervised(model, df):
    X = df.drop(columns=["Label_Encoded"], errors="ignore")
    y_pred = model.predict(X)
    return y_pred

def run_unsupervised(model, df, threshold):
    # Use decision function / score samples
    X = df.drop(columns=["Label_Encoded"], errors="ignore")
    if hasattr(model, "predict"):
        # isolation_forest returns -1 for anomaly
        pred_raw = model.predict(X)
        # Convert to 1 (anomaly) / 0 (benign)
        return np.where(pred_raw == -1, 1, 0)
    scores = model.decision_function(X)
    return (scores > threshold).astype(int)

# ---------------------
# Streamlit UI
# ---------------------
st.set_page_config(page_title="NIDS Performance Dashboard",
                   layout="wide")

st.title("Network Intrusion Detection System – Performance Dashboard")

# Tabs for organization
tabs = st.tabs(["Overview", "Visualizations", "Anomaly Scoring"])

# ---------------------
# Overview Tab
# ---------------------
with tabs[0]:
    st.header("Overview")
    unified, combined = load_metrics()
    st.subheader("Supervised Model Metrics")
    st.dataframe(unified)
    st.markdown("*Precision, recall, F1 and AUC scores are summarized here.*")

    st.subheader("Combined (Supervised + Unsupervised) Metrics")
    st.dataframe(combined)
    st.markdown("*Includes thresholds and modes for unsupervised models*")

    st.header("Summary Plots")
    display_bar_charts(unified)

# ---------------------
# Visualization Tab
# ---------------------
with tabs[1]:
    st.header("Precision‑Recall & ROC Curves")
    model_choice = st.selectbox("Choose a model for PR/ROC curves",
                                ["random_forest", "gradient_boosting",
                                 "balanced_random_forest",
                                 "isolation_forest", "lof", "oneclass_svm", "ensemble"])

    pr_file = f"{model_choice}_pr.png"
    roc_file = f"{model_choice}_roc.png"

    show_plot(load_plot_image(pr_file),
              title=f"{model_choice} – Precision‑Recall Curve")
    show_plot(load_plot_image(roc_file),
              title=f"{model_choice} – ROC Curve")

# ---------------------
# Anomaly Scoring Tab
# ---------------------
with tabs[2]:
    st.header("Run Models on Uploaded Data")
    st.markdown("""
    Upload a CSV with the same structure as your processed dataset (scaled numeric columns plus optional `Label_Encoded`).  
    You can score it using supervised (RF, GB) or unsupervised models (IF, LOF, OCSVM, Ensemble).
    """)

    uploaded_file = st.file_uploader("Upload CSV for Scoring", type=["csv"])
    if uploaded_file:
        df_upload = pd.read_csv(uploaded_file)
        models = load_models()

        model_select = st.selectbox("Select Model to Score",
                                    list(models.keys()))
        threshold = None
        # For unsupervised models requiring threshold, ask
        if model_select in ["ensemble"]:
            # The threshold used in combined_metrics CSV for the ensemble
            threshold = float(combined.loc[combined['model'] == "ensemble",
                                           'threshold'].iloc[0])
        if st.button("Run Scoring"):
            try:
                model = models[model_select]
                if model_select.startswith("best_"):  # supervised
                    preds = run_supervised(model, df_upload)
                else:
                    preds = run_unsupervised(model, df_upload, threshold)
                df_upload["Predictions"] = preds
                st.write("Preview of Predictions:")
                st.dataframe(df_upload.head())

                # Optionally allow the user to download results
                csv_download = df_upload.to_csv(index=False)
                st.download_button("Download Predictions CSV",
                                   csv_download,
                                   file_name="predictions.csv",
                                   mime="text/csv")
            except Exception as e:
                st.error(f"Error scoring: {e}")