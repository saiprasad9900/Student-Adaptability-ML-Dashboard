import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from student_adaptability_ml import (
    FeatureEngineeringConfig,
    FrequencyEncoder,
    predict_from_bundle,
)

# Compatibility shim: previously saved joblib bundles were created when the
# training script ran as "__main__", so custom classes were pickled under that
# module path. Expose matching names in this module for unpickling.
FeatureEngineeringConfig = FeatureEngineeringConfig
FrequencyEncoder = FrequencyEncoder


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = PROJECT_DIR / "students_adaptability_data.csv"
DEFAULT_MODEL_PATH = PROJECT_DIR / "best_model.joblib"
DEFAULT_OUTPUTS_DIR = PROJECT_DIR / "outputs"
DEFAULT_PLOTS_DIR = DEFAULT_OUTPUTS_DIR / "plots"


st.set_page_config(
    page_title="Student Adaptability ML Dashboard",
    page_icon="🎓",
    layout="wide",
)

st.markdown(
    """
    <style>
        .main-title {font-size: 2.0rem; font-weight: 700; margin-bottom: 0.2rem;}
        .subtle {color: #808080;}
        .card {
            background-color: #111827;
            border: 1px solid #2f3746;
            border-radius: 10px;
            padding: 0.8rem 1rem;
            margin-bottom: 0.8rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_dataset(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_model_comparison(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_resource(show_spinner=False)
def load_bundle(path: Path):
    if path.exists():
        return joblib.load(path)
    return None


def load_uploaded_dataset(uploaded_file) -> pd.DataFrame:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(uploaded_file)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(uploaded_file)
    raise ValueError("Unsupported format. Please upload CSV or Excel file.")


def metric_explanation_block() -> None:
    st.markdown("### Metric Meaning (Detailed)")
    st.markdown(
        """
        - **Accuracy**: overall fraction of correct predictions across all students.
        - **Precision (Macro)**: average of class-wise precision; tells how reliable positive predictions are for each class.
        - **Recall (Macro)**: average of class-wise recall; tells how well each class is captured.
        - **F1 (Macro)**: balance of precision and recall across classes; a strong choice for imbalanced multiclass data.
        - **ROC-AUC OVR (Macro)**: ranking quality across classes using one-vs-rest strategy; closer to 1.0 is better.
        """
    )


def run_training(data_path: Path, target_col: str) -> str:
    command = [
        sys.executable,
        str(PROJECT_DIR / "student_adaptability_ml.py"),
        "--data",
        str(data_path),
        "--target-col",
        target_col,
        "--save-model",
        str(DEFAULT_MODEL_PATH),
    ]
    process = subprocess.run(
        command,
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True,
        check=False,
    )
    output = process.stdout + ("\n" + process.stderr if process.stderr else "")
    return output


def describe_univariate_plot(stem: str) -> str:
    feature = stem.replace("univariate_countplot__", "").replace("_", " ")
    return (
        f"This chart shows how values of '{feature}' are distributed in the dataset. "
        "Taller bars mean that category appears more often."
    )


def describe_bivariate_plot(stem: str) -> str:
    feature = stem.replace("bivariate_stacked__", "").replace("_", " ")
    return (
        f"This stacked chart compares '{feature}' with adaptivity levels. "
        "Each bar is one feature category, and colors show the split of Low/Moderate/High."
    )


def describe_confusion_matrix(stem: str) -> str:
    model = stem.replace("confusion_matrix__", "").replace("_", " ")
    return (
        f"Confusion matrix for {model}. "
        "Diagonal cells are correct predictions; off-diagonal cells are mistakes."
    )


def show_plot_grid(image_paths: List[Path], columns: int = 2, description_fn=None):
    cols = st.columns(columns)
    for i, img_path in enumerate(image_paths):
        with cols[i % columns]:
            st.image(str(img_path), caption=img_path.stem.replace("_", " "), width="stretch")
            if description_fn is not None:
                st.caption(description_fn(img_path.stem))


st.markdown('<div class="main-title">🎓 Student Adaptability ML Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtle">End-to-end interface for data insights, model evaluation, and prediction.</div>',
    unsafe_allow_html=True,
)

default_dataset_df = load_dataset(DEFAULT_DATA_PATH)
comparison_df = load_model_comparison(DEFAULT_OUTPUTS_DIR / "model_comparison_results.csv")
bundle = load_bundle(DEFAULT_MODEL_PATH)

if "active_dataset_df" not in st.session_state:
    st.session_state["active_dataset_df"] = default_dataset_df
    st.session_state["active_dataset_name"] = DEFAULT_DATA_PATH.name

st.markdown("### Upload Dataset for Analysis")
uploaded_analysis = st.file_uploader(
    "Upload CSV/Excel dataset",
    type=["csv", "xlsx", "xls"],
    key="analysis_dataset_uploader",
)
u1, u2 = st.columns([1, 1])
with u1:
    if st.button("Use Uploaded Dataset", disabled=uploaded_analysis is None):
        try:
            loaded_uploaded_df = load_uploaded_dataset(uploaded_analysis)
            st.session_state["active_dataset_df"] = loaded_uploaded_df
            st.session_state["active_dataset_name"] = uploaded_analysis.name
            st.success(f"Using uploaded dataset: {uploaded_analysis.name}")
        except Exception as e:
            st.error(f"Could not load uploaded dataset: {e}")
with u2:
    if st.button("Reset to Default Dataset"):
        st.session_state["active_dataset_df"] = default_dataset_df
        st.session_state["active_dataset_name"] = DEFAULT_DATA_PATH.name
        st.info("Reset to default dataset.")

dataset_df = st.session_state.get("active_dataset_df", default_dataset_df)
active_dataset_name = st.session_state.get("active_dataset_name", DEFAULT_DATA_PATH.name)

st.markdown("### Project Status")
st.markdown(
    f"""
    <div class="card">
    <b>Dataset:</b> <code>{active_dataset_name}</code><br/>
    <b>Model bundle:</b> <code>{DEFAULT_MODEL_PATH.name}</code><br/>
    <b>Outputs directory:</b> <code>{DEFAULT_OUTPUTS_DIR.name}</code>
    </div>
    """,
    unsafe_allow_html=True,
)

tabs = st.tabs(
    [
        "📊 Dashboard",
        "🧪 Model Performance",
        "🖼️ EDA & Visual Outputs",
        "🔮 Prediction Lab",
        "⚙️ Train / Re-Train",
    ]
)

with tabs[0]:
    st.markdown("## Dashboard Overview")
    if dataset_df.empty:
        st.error("Dataset not found. Please keep `students_adaptability_data.csv` in the project root.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{len(dataset_df):,}")
        c2.metric("Columns", f"{dataset_df.shape[1]}")
        target_col = "Adaptivity Level" if "Adaptivity Level" in dataset_df.columns else dataset_df.columns[-1]
        c3.metric("Target Column", target_col)
        c4.metric("Target Classes", dataset_df[target_col].nunique())

        st.markdown("### Dataset Preview")
        st.dataframe(dataset_df.head(15), width="stretch")

        st.markdown("### Target Class Distribution")
        dist_df = dataset_df[target_col].value_counts().rename_axis("Class").reset_index(name="Count")
        st.bar_chart(dist_df.set_index("Class"))
        st.caption(
            "This chart shows how many students are in each adaptivity class. "
            "If one class has much higher bars, the data is imbalanced."
        )
        st.dataframe(dist_df, width="stretch")

        st.markdown("### Feature Snapshot")
        profile_df = pd.DataFrame(
            {
                "Column": dataset_df.columns,
                "Data Type": dataset_df.dtypes.astype(str).values,
                "Unique Values": [dataset_df[c].nunique() for c in dataset_df.columns],
                "Missing Values": [dataset_df[c].isna().sum() for c in dataset_df.columns],
            }
        )
        st.dataframe(profile_df, width="stretch")

with tabs[1]:
    st.markdown("## Model Performance")
    if comparison_df.empty:
        st.warning("No model comparison results found. Run training from the Train tab.")
    else:
        comparison_show = comparison_df.copy()
        metric_cols = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc_ovr_macro"]
        for col in metric_cols:
            if col in comparison_show.columns:
                comparison_show[col] = (comparison_show[col] * 100).round(2).astype(str) + "%"
        st.dataframe(comparison_show, width="stretch")

        best_row = comparison_df.sort_values("f1_macro", ascending=False).iloc[0]
        st.success(
            f"Best model by F1 Macro: {best_row['model']} | "
            f"F1 Macro={best_row['f1_macro']:.4f}, Accuracy={best_row['accuracy']:.4f}"
        )

        st.markdown("### Performance Explanation")
        metric_explanation_block()

        st.markdown("### Classification Reports")
        report_files = sorted(DEFAULT_PLOTS_DIR.glob("classification_report__*.txt"))
        if not report_files:
            st.info("No classification reports found yet.")
        else:
            for rep_path in report_files:
                with st.expander(rep_path.stem.replace("classification_report__", "").replace("_", " "), expanded=False):
                    st.code(rep_path.read_text(encoding="utf-8"), language="text")

        st.markdown("### Confusion Matrix Gallery")
        cm_files = sorted(DEFAULT_PLOTS_DIR.glob("confusion_matrix__*.png"))
        if not cm_files:
            st.info("No confusion matrices found yet.")
        else:
            show_plot_grid(cm_files, columns=3, description_fn=describe_confusion_matrix)

with tabs[2]:
    st.markdown("## EDA & Visual Outputs")
    if not dataset_df.empty:
        inferred_target = "Adaptivity Level" if "Adaptivity Level" in dataset_df.columns else dataset_df.columns[-1]
        st.markdown("### Live Analysis on Current Dataset")
        st.caption("These charts are generated directly from the dataset currently selected above.")

        live_dist_df = dataset_df[inferred_target].astype(str).value_counts().rename_axis("Class").reset_index(name="Count")
        st.bar_chart(live_dist_df.set_index("Class"))
        st.caption("Live target distribution for the selected dataset.")

        categorical_live_cols = [c for c in dataset_df.columns if c != inferred_target and dataset_df[c].dtype == "object"]
        if categorical_live_cols:
            c_a, c_b = st.columns(2)
            with c_a:
                uni_col = st.selectbox("Choose feature for univariate analysis", categorical_live_cols, key="live_uni_col")
                uni_df = dataset_df[uni_col].astype(str).value_counts().head(20).rename_axis(uni_col).reset_index(name="Count")
                st.bar_chart(uni_df.set_index(uni_col))
                st.caption(f"Live univariate chart for '{uni_col}'.")
            with c_b:
                bi_col = st.selectbox("Choose feature for bivariate analysis", categorical_live_cols, key="live_bi_col")
                bi_df = (
                    dataset_df[[bi_col, inferred_target]]
                    .astype(str)
                    .groupby([bi_col, inferred_target])
                    .size()
                    .reset_index(name="Count")
                )
                bi_pivot = bi_df.pivot(index=bi_col, columns=inferred_target, values="Count").fillna(0)
                st.bar_chart(bi_pivot)
                st.caption(f"Live bivariate chart: '{bi_col}' vs '{inferred_target}'.")

    st.markdown(
        """
        This section presents the visual analysis generated by your pipeline:
        - **Target Distribution**: checks class balance.
        - **Univariate Plots**: individual feature patterns.
        - **Bivariate Stacked Charts**: relationship between features and adaptability level.
        - **Correlation Heatmap**: encoded feature correlation structure.
        """
    )

    target_dist = DEFAULT_PLOTS_DIR / "eda" / "target_distribution.png"
    corr_plot = DEFAULT_PLOTS_DIR / "correlation" / "correlation_heatmap_after_encoding.png"
    uni_files = sorted((DEFAULT_PLOTS_DIR / "eda" / "univariate_categoricals").glob("*.png"))
    bi_files = sorted((DEFAULT_PLOTS_DIR / "eda" / "bivariate_categorical_vs_target").glob("*.png"))

    st.markdown("### Target Distribution")
    if target_dist.exists():
        st.image(str(target_dist), width="stretch")
        st.caption(
            "This plot checks whether target classes (Low/Moderate/High) are balanced. "
            "Balanced classes usually help models learn fairly."
        )
    else:
        st.info("Target distribution plot not found.")

    st.markdown("### Correlation Heatmap")
    if corr_plot.exists():
        st.image(str(corr_plot), width="stretch")
        st.caption(
            "This heatmap shows how encoded features move together. "
            "Values near +1/-1 mean strong relationship; values near 0 mean weak relationship."
        )
    else:
        st.info("Correlation heatmap not found.")

    st.markdown("### Univariate Categorical Analysis")
    if uni_files:
        show_plot_grid(uni_files, columns=2, description_fn=describe_univariate_plot)
    else:
        st.info("Univariate plots not found.")

    st.markdown("### Bivariate Categorical vs Target Analysis")
    if bi_files:
        show_plot_grid(bi_files, columns=2, description_fn=describe_bivariate_plot)
    else:
        st.info("Bivariate plots not found.")

with tabs[3]:
    st.markdown("## Prediction Lab")
    if bundle is None:
        st.error("`best_model.joblib` not found. Train model first in the Train tab.")
    else:
        st.markdown(
            """
            Two ways to predict:
            1. **Manual single-student input** for quick testing.
            2. **Batch CSV upload** for multiple students.
            """
        )
        input_mode = st.radio("Choose input mode", ["Manual Input", "Upload CSV"], horizontal=True)

        feature_columns = bundle.get("feature_columns", [])
        if not feature_columns and not dataset_df.empty:
            fallback_target = "Adaptivity Level" if "Adaptivity Level" in dataset_df.columns else dataset_df.columns[-1]
            feature_columns = [c for c in dataset_df.columns if c != fallback_target]

        if input_mode == "Manual Input":
            if not feature_columns:
                st.warning("Could not infer feature columns.")
            else:
                st.markdown("### Enter Student Profile")
                input_data: Dict[str, str] = {}
                mcols = st.columns(3)
                for i, col in enumerate(feature_columns):
                    with mcols[i % 3]:
                        if not dataset_df.empty and col in dataset_df.columns:
                            options = [str(x) for x in dataset_df[col].dropna().astype(str).unique().tolist()]
                            options = sorted(set(options))
                            if len(options) >= 2 and len(options) <= 30:
                                input_data[col] = st.selectbox(col, options=options, index=0, key=f"manual_{col}")
                            else:
                                input_data[col] = st.text_input(col, value="", key=f"manual_{col}")
                        else:
                            input_data[col] = st.text_input(col, value="", key=f"manual_{col}")

                if st.button("Predict Adaptability Level", type="primary"):
                    pred_df = pd.DataFrame([input_data])
                    y_pred, y_prob, classes = predict_from_bundle(bundle=bundle, raw_input_df=pred_df)
                    st.success(f"Predicted Adaptability Level: **{y_pred[0]}**")

                    if y_prob is not None and classes is not None:
                        proba_df = pd.DataFrame({"Class": [str(c) for c in classes], "Probability": y_prob[0]})
                        st.markdown("#### Prediction Confidence")
                        st.dataframe(proba_df, width="stretch")
                        st.bar_chart(proba_df.set_index("Class"))
                        st.caption(
                            "This chart shows model confidence for each class for the current student. "
                            "The highest bar is the predicted class."
                        )

        else:
            st.markdown("### Upload CSV for Batch Prediction")
            st.caption("Upload a CSV containing the same feature columns used during training.")
            uploaded = st.file_uploader("Choose CSV file", type=["csv"])
            if uploaded is not None:
                upload_df = pd.read_csv(uploaded)
                st.markdown("#### Uploaded Data Preview")
                st.dataframe(upload_df.head(20), width="stretch")

                if st.button("Run Batch Prediction", type="primary"):
                    y_pred, y_prob, classes = predict_from_bundle(bundle=bundle, raw_input_df=upload_df)
                    result_df = upload_df.copy()
                    result_df["Predicted_Adaptivity_Level"] = y_pred

                    if y_prob is not None and classes is not None:
                        for idx, cls in enumerate(classes):
                            result_df[f"prob_{cls}"] = y_prob[:, idx]

                    st.markdown("#### Prediction Output")
                    st.dataframe(result_df, width="stretch")
                    csv_bytes = result_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download prediction results as CSV",
                        data=csv_bytes,
                        file_name="predictions_output.csv",
                        mime="text/csv",
                    )

with tabs[4]:
    st.markdown("## Train / Re-Train")
    st.markdown(
        """
        Use this section to regenerate:
        - model comparison CSV
        - EDA plots
        - confusion matrices
        - classification reports
        - model bundle (`best_model.joblib`)
        """
    )
    train_data_path = st.text_input("Dataset path", value=str(DEFAULT_DATA_PATH))
    train_target_col = st.text_input("Target column", value="Adaptivity Level")

    if st.button("Run Training Pipeline", type="primary"):
        with st.spinner("Training in progress..."):
            output_log = run_training(Path(train_data_path), train_target_col)
        st.success("Training command completed.")
        st.markdown("### Training Logs")
        st.code(output_log, language="text")
        st.info("If outputs were regenerated, refresh the page to reload cached files.")

