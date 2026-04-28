import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, classification_report

# -----------------------------
# CONFIGURATION
# -----------------------------
DATA_DIR = "data/processed"
MODEL_DIR = "models"
PLOTS_DIR = "outputs/plots"
REPORTS_DIR = "outputs/reports"

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
@st.cache_data
def load_metrics():
    """Load training metrics CSVs for overview."""
    unified = pd.DataFrame()
    combined = pd.DataFrame()
    if os.path.exists(os.path.join(REPORTS_DIR, "unified_metrics.csv")):
        unified = pd.read_csv(os.path.join(REPORTS_DIR, "unified_metrics.csv"))
    if os.path.exists(os.path.join(REPORTS_DIR, "combined_metrics.csv")):
        combined = pd.read_csv(os.path.join(REPORTS_DIR, "combined_metrics.csv"))
    return unified, combined

def display_bar_charts(metrics_df):
    """Display bar charts for key metrics (precision, recall, F1, ROC-AUC, PR-AUC)."""
    if metrics_df.empty:
        st.write("_No metrics to display._")
        return
    for metric in ["precision", "recall", "f1"]:
        if metric in metrics_df.columns:
            fig = px.bar(
                metrics_df, x="model", y=metric, title=f"{metric.capitalize()} by Model",
                color="model", template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)
    for metric in ["roc_auc", "pr_auc"]:
        if metric in metrics_df.columns:
            fig = px.bar(
                metrics_df, x="model", y=metric, title=f"{metric.upper()} by Model",
                color="model", template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)

def load_plot_image(filename):
    """Return path to a plot image if it exists."""
    path = os.path.join(PLOTS_DIR, filename)
    return path if os.path.exists(path) else None

def show_plot(image_path, title=""):
    """Display an image with caption."""
    if image_path:
        st.image(image_path, caption=title, use_column_width=True)

@st.cache_resource
def load_models():
    """Dynamically load all models (.pkl) from MODEL_DIR, skipping any scaler files."""
    models = {}
    if not os.path.isdir(MODEL_DIR):
        return models
    for fname in os.listdir(MODEL_DIR):
        if fname.lower().endswith(".pkl") and "scaler" not in fname.lower():
            key = fname[:-4]  # model name without .pkl
            try:
                models[key] = joblib.load(os.path.join(MODEL_DIR, fname))
            except Exception:
                continue
    return models

def pretty_name(key: str) -> str:
    """Return a human-friendly name for a model key."""
    if key.startswith("best_"):
        name = key[len("best_"):].replace("_", " ").title()
        return f"{name} (Supervised)"
    unsupervised_prefixes = ["isolation_forest", "lof", "lof_novelty", "oneclass_svm", "ensemble"]
    for prefix in unsupervised_prefixes:
        if key.startswith(prefix):
            name = key.replace("_", " ").title()
            if "Novelty" in name:
                return f"{name} (Unsupervised - Novelty)"
            else:
                return f"{name} (Unsupervised)"
    # Fallback
    return key.replace("_", " ").title()

# -----------------------------
# MAIN APPLICATION
# -----------------------------
def main():
    st.set_page_config(page_title="NIDS Performance Dashboard", layout="wide")
    st.title("Network-IDS Anomaly Detection Dashboard")

    # Load training metrics for Overview
    unified_metrics, combined_metrics = load_metrics()

    # Sidebar: file upload & model selection
    st.sidebar.header("Anomaly Scoring")
    uploaded_file = st.sidebar.file_uploader("Upload CSV (numeric features)", type=["csv"])
    models = load_models()
    if not models:
        st.sidebar.error("No models found in 'models' directory.")
        return
    # Build a list of pretty names for the selectbox
    model_names = sorted(models.keys())
    pretty_to_key = {}
    model_labels = []
    for key in model_names:
        label = pretty_name(key)
        pretty_to_key[label] = key
        model_labels.append(label)
    selected_label = st.sidebar.selectbox("Select Model", sorted(model_labels))
    selected_key = pretty_to_key[selected_label]
    model = models[selected_key]

    # Tabs
    tabs = st.tabs(["Overview", "PR/ROC Curves", "Run Scoring"])

    # -----------------------------
    # Overview Tab
    # -----------------------------
    with tabs[0]:
        st.header("Model Performance (Training)")
        if not unified_metrics.empty:
            st.subheader("Supervised Model Metrics")
            st.dataframe(unified_metrics, use_container_width=True)
        else:
            st.write("_No supervised metrics available._")
        if not combined_metrics.empty:
            st.subheader("Combined (Supervised + Unsupervised) Metrics")
            st.dataframe(combined_metrics, use_container_width=True)
        else:
            st.write("_No combined metrics available._")
        st.header("Metric Bar Charts")
        display_bar_charts(unified_metrics)

    # -----------------------------
    # PR/ROC Curves Tab
    # -----------------------------
    with tabs[1]:
        st.header("Precision-Recall & ROC Curves")
        # List available models by looking at plot filenames
        plots = []
        if os.path.isdir(PLOTS_DIR):
            for f in os.listdir(PLOTS_DIR):
                if f.endswith("_pr.png"):
                    plots.append(f.split("_")[0])
        if plots:
            curve_model = st.selectbox("Choose model for curves", sorted(set(plots)))
            pr_path = load_plot_image(f"{curve_model}_pr.png")
            roc_path = load_plot_image(f"{curve_model}_roc.png")
            show_plot(pr_path, title=f"{curve_model} – Precision-Recall Curve")
            show_plot(roc_path, title=f"{curve_model} – ROC Curve")
        else:
            st.write("_No precomputed PR/ROC plots available._")

    # -----------------------------
    # Run Scoring Tab
    # -----------------------------
    with tabs[2]:
        st.header("Score New Data")
        st.write("Upload a CSV with the same structure as your training data (numeric features, optional `Label_Encoded`). Select a model and run scoring. The app will display predictions and evaluation metrics if true labels are present.")
        if uploaded_file is not None:
            st.success("File uploaded successfully.")
            if st.button("Run Model"):
                try:
                    # Read the entire CSV
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file)
                    st.write("### Data Preview")
                    st.dataframe(df.head())

                    # Validation
                    if df.shape[1] == 0:
                        st.error("Uploaded CSV is empty or invalid.")
                        return
                    if "Label_Encoded" in df.columns:
                        y_true = df["Label_Encoded"].copy()
                        X = df.drop(columns=["Label_Encoded"])
                    else:
                        y_true = None
                        X = df.copy()
                    # Keep numeric columns only
                    X = X.select_dtypes(include=[np.number])
                    if X.shape[1] == 0:
                        st.error("No numeric features found in the CSV.")
                        return

                    # Run the selected model
                    with st.spinner("Running model..."):
                        if selected_key.startswith("best_"):
                            preds = model.predict(X.values)
                        else:
                            if hasattr(model, "predict"):
                                raw = model.predict(X.values)
                                preds = (raw == -1).astype(int)
                            else:
                                scores = model.decision_function(X.values)
                                thresh = float(np.median(scores))
                                preds = (scores > thresh).astype(int)
                    df["Predictions"] = preds
                    st.success("Model scoring complete.")

                    st.write("### Predictions Preview")
                    st.dataframe(df.head())

                    # Evaluation metrics
                    if y_true is not None:
                        st.subheader("Evaluation Metrics")
                        accuracy = accuracy_score(y_true, preds)
                        precision = precision_score(y_true, preds, zero_division=0)
                        recall = recall_score(y_true, preds, zero_division=0)
                        f1 = f1_score(y_true, preds, zero_division=0)
                        try:
                            roc = roc_auc_score(y_true, preds)
                        except:
                            roc = None
                        pr_auc = average_precision_score(y_true, preds)
                        metrics = {
                            "Accuracy": accuracy,
                            "Precision": precision,
                            "Recall": recall,
                            "F1 Score": f1,
                            "ROC-AUC": roc,
                            "PR-AUC": pr_auc
                        }
                        st.table(metrics)

                        st.subheader("Classification Report")
                        report_dict = classification_report(y_true, preds, output_dict=True, zero_division=0)
                        report_df = pd.DataFrame(report_dict).transpose()
                        st.dataframe(report_df, use_container_width=True)

                    # Download predictions
                    csv_data = df.to_csv(index=False).encode('utf-8')
                    st.download_button("⬇ Download Predictions", csv_data, "predictions.csv", "text/csv")

                except Exception as e:
                    st.error(f"Error during scoring: {e}")
        else:
            st.info("Please upload a CSV file to begin.")

if __name__ == "__main__":
    main()
