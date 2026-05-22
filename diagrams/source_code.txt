import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import seaborn as sns

# Use a non-interactive backend so plots work on servers / CI.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, label_binarize
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None


TARGET_DEFAULT = "Student Adaptability Level"


@dataclass
class FeatureEngineeringConfig:
    network_col: Optional[str] = None
    self_lms_col: Optional[str] = None


def load_dataset(path: str) -> pd.DataFrame:
    """
    Loads dataset from CSV or Excel files.
    Supported: .csv, .xlsx, .xls
    """
    if not path:
        raise ValueError("Dataset path is empty.")

    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext in {".xlsx", ".xls"}:
        return pd.read_excel(path)

    raise ValueError(
        f"Unsupported dataset format '{ext}'. Use a .csv, .xlsx, or .xls file."
    )


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """
    Simple frequency encoder for categorical features.

    Encodes each category with its relative frequency in the training data.
    Unseen categories during transform become 0.
    """

    def __init__(self, normalize: bool = True):
        self.normalize = normalize

    def fit(self, X, y=None):
        X_arr = self._to_numpy(X)
        n_features = X_arr.shape[1] if X_arr.ndim == 2 else 1
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        self.n_features_in_ = n_features
        self.freq_maps_: List[Dict[Any, float]] = []

        for j in range(n_features):
            col = pd.Series(X_arr[:, j])
            counts = col.value_counts(dropna=False)
            if self.normalize:
                denom = max(counts.sum(), 1)
                mapping = (counts / denom).to_dict()
            else:
                mapping = counts.to_dict()
            self.freq_maps_.append(mapping)
        return self

    def transform(self, X):
        X_arr = self._to_numpy(X)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        n_features = X_arr.shape[1]
        if n_features != self.n_features_in_:
            raise ValueError(
                f"FrequencyEncoder expected {self.n_features_in_} features, got {n_features}."
            )

        out = np.zeros((X_arr.shape[0], n_features), dtype=float)
        for j in range(n_features):
            mapping = self.freq_maps_[j]
            out[:, j] = [mapping.get(v, 0.0) for v in X_arr[:, j]]
        return out

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.array([f"x{i}_freq" for i in range(self.n_features_in_)], dtype=object)
        if len(input_features) != self.n_features_in_:
            # Fallback if sklearn couldn't pass matching names
            return np.array([f"x{i}_freq" for i in range(self.n_features_in_)], dtype=object)
        return np.array([f"{name}_freq" for name in input_features], dtype=object)

    @staticmethod
    def _to_numpy(X):
        if isinstance(X, pd.DataFrame):
            return X.values
        if isinstance(X, pd.Series):
            return X.to_numpy()
        return np.asarray(X)


def infer_column_by_keywords(df: pd.DataFrame, keywords: List[str]) -> Optional[str]:
    """
    Tries to find a column where all keywords appear in the normalized name.
    """
    normalized = {c: str(c).strip().lower().replace(" ", "_") for c in df.columns}
    for col, col_norm in normalized.items():
        if all(k in col_norm for k in keywords):
            return col
    # Second pass: allow any keyword match (broader)
    for col, col_norm in normalized.items():
        if any(k in col_norm for k in keywords):
            return col
    return None


def add_engineered_features(
    df: pd.DataFrame, config: FeatureEngineeringConfig
) -> Tuple[pd.DataFrame, FeatureEngineeringConfig]:
    """
    Adds:
      - network_score from 2G/3G/4G/5G
      - has_self_lms from Yes/No
    """
    df = df.copy()

    # Infer columns if not provided.
    if config.network_col is None:
        config.network_col = infer_column_by_keywords(df, ["network"])
    if config.self_lms_col is None:
        # Covers "self_lms", "self lms", etc.
        config.self_lms_col = infer_column_by_keywords(df, ["self", "lms"])

    # Always create the engineered columns so downstream steps don't fail.
    # If we can't infer the source column, they will remain NaN and be filled later.
    if "network_score" not in df.columns:
        df["network_score"] = np.nan
    if "has_self_lms" not in df.columns:
        df["has_self_lms"] = np.nan

    # network_score
    if config.network_col is not None and config.network_col in df.columns:
        def network_to_score(v):
            if pd.isna(v):
                return np.nan
            s = str(v).strip().upper()
            m = re.search(r"([2-5])\s*G", s)
            if m:
                return int(m.group(1)) - 1  # 2G->1, 3G->2, ...
            m2 = re.search(r"\b([2-5])\b", s)
            if m2:
                return int(m2.group(1)) - 1
            return np.nan

        df["network_score"] = df[config.network_col].apply(network_to_score)

    # has_self_lms
    if config.self_lms_col is not None and config.self_lms_col in df.columns:
        def yn_to_01(v):
            if pd.isna(v):
                return np.nan
            if isinstance(v, (int, float)) and v in [0, 1]:
                return int(v)
            s = str(v).strip().lower()
            if s in {"yes", "y", "true", "t", "1"}:
                return 1
            if s in {"no", "n", "false", "f", "0"}:
                return 0
            return np.nan

        df["has_self_lms"] = df[config.self_lms_col].apply(yn_to_01)

    return df, config


def separate_numeric_categorical(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Returns numeric and categorical columns for modeling.
    """
    numeric_cols: List[str] = []
    categorical_cols: List[str] = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)
    return numeric_cols, categorical_cols


def fill_missing_values(df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)

    for col in numeric_cols:
        med = pd.to_numeric(df[col], errors="coerce").median(skipna=True)
        if pd.isna(med):
            # If the entire column is NaN, fall back to 0.0.
            med = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(med)

    for col in categorical_cols:
        df[col] = df[col].fillna("Unknown").astype(str)
    return df


def determine_low_high_cardinality(
    df: pd.DataFrame, categorical_cols: List[str], threshold: int = 10
) -> Tuple[List[str], List[str]]:
    low_card = []
    high_card = []
    for col in categorical_cols:
        # categorical columns should already have no NaN; treat "Unknown" as category
        nunique = df[col].nunique(dropna=False)
        if nunique <= threshold:
            low_card.append(col)
        else:
            high_card.append(col)
    return low_card, high_card


def make_preprocessor(
    X_train: pd.DataFrame,
    numeric_cols: List[str],
    low_card_cols: List[str],
    high_card_cols: List[str],
    scale_numeric: bool,
) -> ColumnTransformer:
    numeric_transformer = "passthrough"
    if scale_numeric:
        numeric_transformer = Pipeline([("scaler", StandardScaler())])

    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_transformer, numeric_cols))
    if low_card_cols:
        transformers.append(
            (
                "low_cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                low_card_cols,
            )
        )
    if high_card_cols:
        transformers.append(("high_cat", FrequencyEncoder(normalize=True), high_card_cols))

    # Drop everything not explicitly listed.
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)
    return preprocessor


def get_target_order(y: pd.Series) -> List[str]:
    preferred = ["Low", "Moderate", "High"]
    y_unique = list(pd.unique(y))
    if all(v in y_unique for v in preferred):
        return preferred
    # Fallback: keep existing order
    return sorted(y_unique)


def plot_target_distribution(df: pd.DataFrame, target_col: str, out_path: str):
    target_order = get_target_order(df[target_col])
    plt.figure(figsize=(7, 4))
    sns.countplot(data=df, x=target_col, order=target_order)
    plt.title("Target Distribution: Student Adaptability Level")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_univariate_categoricals(
    df_work: pd.DataFrame,
    categorical_cols: List[str],
    target_col: str,
    out_dir: str,
    max_plots: int = 8,
):
    os.makedirs(out_dir, exist_ok=True)
    # Limit plots to avoid huge outputs.
    cols = categorical_cols[:max_plots]
    for col in cols:
        plt.figure(figsize=(10, 4))
        order = df_work[col].value_counts().index[:20]
        sns.countplot(data=df_work, x=col, order=order)
        plt.title(f"Univariate Distribution: {col}")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"univariate_countplot__{col}.png"), dpi=200)
        plt.close()


def plot_bivariate_categorical_vs_target_stacked(
    df_work: pd.DataFrame,
    categorical_cols: List[str],
    target_col: str,
    out_dir: str,
    max_plots: int = 6,
):
    os.makedirs(out_dir, exist_ok=True)
    target_order = get_target_order(df_work[target_col])
    cols = categorical_cols[:max_plots]

    for col in cols:
        top_categories = df_work[col].value_counts().index[:15]
        df_sub = df_work[df_work[col].isin(top_categories)]
        ct = pd.crosstab(df_sub[col], df_sub[target_col]).reindex(columns=target_order, fill_value=0)

        ax = ct.plot(kind="bar", stacked=True, figsize=(12, 5), colormap="viridis")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        ax.set_title(f"Bivariate: {col} vs {target_col}")
        plt.xticks(rotation=45, ha="right")
        plt.legend(title=target_col, bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"bivariate_stacked__{col}.png"), dpi=200)
        plt.close()


def plot_correlation_heatmap_after_encoding(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    out_path: str,
    corr_max_features: int = 30,
):
    """
    Correlation heatmap after encoding.

    For high-dimensional one-hot features, we subset to a manageable number of
    columns using variance as a proxy.
    """
    X_enc = preprocessor.transform(X_train)
    feature_names = preprocessor.get_feature_names_out()
    enc_df = pd.DataFrame(X_enc, columns=feature_names)

    if enc_df.shape[1] > corr_max_features:
        variances = enc_df.var(axis=0).sort_values(ascending=False)
        keep_cols = list(variances.head(corr_max_features).index)
        enc_df = enc_df[keep_cols]

    corr = enc_df.corr(method="pearson")
    plt.figure(figsize=(max(8, corr.shape[1] * 0.35), max(6, corr.shape[1] * 0.25)))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap (After Encoding)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray],
    classes: List[Any],
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["precision_macro"] = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["recall_macro"] = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    # ROC-AUC (One-vs-Rest)
    if y_prob is not None:
        # Ensure order matches `classes`.
        try:
            y_true_bin = label_binarize(y_true, classes=classes)
            roc_auc = roc_auc_score(
                y_true_bin,
                y_prob,
                multi_class="ovr",
                average="macro",
            )
            metrics["roc_auc_ovr_macro"] = float(roc_auc)
        except Exception:
            metrics["roc_auc_ovr_macro"] = np.nan
    else:
        metrics["roc_auc_ovr_macro"] = np.nan
    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: List[str],
    out_path: str,
):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    cm_df = pd.DataFrame(cm, index=[f"True_{c}" for c in classes], columns=[f"Pred_{c}" for c in classes])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def safe_predict_proba(model, X):
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)
        except Exception:
            return None
    return None


def evaluate_pipeline(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    classes: List[Any],
) -> Tuple[Dict[str, Any], str, str]:
    """
    Fits the pipeline and evaluates it on the test split.
    Returns:
      metrics dict, classification_report_text, confusion_matrix_text
    """
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = safe_predict_proba(pipeline, X_test)

    metrics = compute_metrics(y_test, y_pred, y_prob, classes)
    report = classification_report(y_test, y_pred, digits=4)
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    cm_text = pd.DataFrame(cm, index=[f"True_{c}" for c in classes], columns=[f"Pred_{c}" for c in classes]).to_string()
    return metrics, report, cm_text


def predict_from_bundle(bundle: Dict[str, Any], raw_input_df: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray], List[Any]]:
    """
    Predicts 'Student Adaptability Level' for new input rows.

    The input dataframe should contain the same columns as the original dataset
    (pre-feature-engineering). Missing columns are filled with NaN, then
    imputed using the script's rules.
    """
    pipeline: Pipeline = bundle["pipeline"]
    feature_config: FeatureEngineeringConfig = bundle.get("feature_config", FeatureEngineeringConfig())
    expected_feature_columns: Optional[List[str]] = bundle.get("feature_columns", None)

    df = raw_input_df.copy()
    df, _ = add_engineered_features(df, feature_config)

    if expected_feature_columns is not None:
        for c in expected_feature_columns:
            if c not in df.columns:
                df[c] = np.nan
        df = df[expected_feature_columns]

    numeric_cols, categorical_cols = separate_numeric_categorical(df)
    df = fill_missing_values(df, numeric_cols, categorical_cols)

    y_pred = pipeline.predict(df)
    y_prob = safe_predict_proba(pipeline, df)

    classes = bundle.get("classes", None)
    if classes is None:
        classes = list(getattr(pipeline.named_steps["model"], "classes_", np.unique(y_pred)))
    return y_pred, y_prob, classes


def build_model_pipelines(
    preprocessor_scaled: ColumnTransformer,
    preprocessor_unscaled: ColumnTransformer,
    y_train: np.ndarray,
):
    """
    Returns dict of model_name -> pipeline
    """
    # Logistic Regression
    log_reg = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
    )
    pipe_log_reg = Pipeline([("preprocess", preprocessor_scaled), ("model", log_reg)])

    # SVM (RBF)
    svm = SVC(kernel="rbf", probability=True, class_weight="balanced", gamma="scale")
    pipe_svm = Pipeline([("preprocess", preprocessor_scaled), ("model", svm)])

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=400,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    pipe_rf = Pipeline([("preprocess", preprocessor_unscaled), ("model", rf)])

    # XGBoost
    pipelines = {
        "Logistic Regression": pipe_log_reg,
        "SVM (RBF)": pipe_svm,
        "Random Forest": pipe_rf,
    }
    if XGBClassifier is not None:
        xgb = XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            num_class=len(np.unique(y_train)),
            eval_metric="mlogloss",
            random_state=42,
        )
        pipe_xgb = Pipeline([("preprocess", preprocessor_unscaled), ("model", xgb)])
        pipelines["XGBoost"] = pipe_xgb

    return pipelines


def fit_xgb_with_balanced_weights(pipeline: Pipeline, X_train, y_train):
    """
    XGBoost doesn't support class_weight directly for multiclass in the sklearn wrapper.
    We compute sample weights from balanced class frequencies and pass sample_weight to fit.
    """
    # Compute balanced sample weights
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    pipeline.fit(X_train, y_train, model__sample_weight=sample_weight)


def main():
    parser = argparse.ArgumentParser(description="End-to-end ML pipeline for Student Adaptability Level prediction.")
    parser.add_argument("--data", type=str, default="", help="Path to dataset file (.csv/.xlsx/.xls), required for training.")
    parser.add_argument("--target-col", type=str, default=TARGET_DEFAULT, help="Target column name.")
    parser.add_argument("--network-col", type=str, default="", help="Column containing network type (e.g., 2G/3G/4G/5G).")
    parser.add_argument("--self-lms-col", type=str, default="", help="Column containing Yes/No for self LMS usage.")
    parser.add_argument("--test-size", type=float, default=0.25, help="Test size fraction (default 0.25).")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Where to save plots and results.")
    parser.add_argument("--save-model", type=str, default="", help="If set, saves the best model bundle to this path.")
    parser.add_argument("--eda-max-plots", type=int, default=10, help="Max categorical plots to generate.")
    parser.add_argument("--corr-max-features", type=int, default=30, help="Max features in correlation heatmap.")

    # Prediction-only modes
    parser.add_argument("--predict-file", type=str, default="", help="CSV with one row of input features.")
    parser.add_argument("--predict-json", type=str, default="", help="JSON string of a single input row.")
    parser.add_argument("--model-bundle", type=str, default="", help="Path to a saved model bundle (.joblib).")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    plots_dir = os.path.join(args.output_dir, "plots")
    eda_dir = os.path.join(plots_dir, "eda")
    os.makedirs(eda_dir, exist_ok=True)

    # Prediction mode
    if args.predict_file or args.predict_json:
        if not args.model_bundle:
            raise ValueError("For prediction, you must provide --model-bundle (path to saved joblib).")
        bundle = joblib.load(args.model_bundle)

        if args.predict_file:
            raw_df = pd.read_csv(args.predict_file)
        else:
            pred_row = json.loads(args.predict_json)
            raw_df = pd.DataFrame([pred_row])

        y_pred, y_prob, classes = predict_from_bundle(bundle=bundle, raw_input_df=raw_df)

        print("\nPrediction result")
        print("------------------")
        print("Predicted Student Adaptability Level:", y_pred[0])
        if y_prob is not None:
            if classes is None:
                classes = list(range(y_prob.shape[1]))
            probs = {str(cls): float(p) for cls, p in zip(classes, y_prob[0])}
            print("Class probabilities (ovr):")
            for k, v in probs.items():
                print(f"  {k}: {v:.6f}")
        return

    # Training mode
    if not args.data:
        raise ValueError("Missing --data. Provide --data to train, or use --predict-file/--predict-json with --model-bundle.")
    df = load_dataset(args.data)
    if args.target_col not in df.columns:
        raise ValueError(f"Target column '{args.target_col}' not found in dataset columns: {list(df.columns)}")

    print("Dataset preview (first 5 rows):")
    print(df.head())
    print("\nDataset preview (last 5 rows):")
    print(df.tail())
    print("\nDataset info / shape:")
    print(df.info())
    print("Shape:", df.shape)
    print("\nSummary statistics:")
    print(df.describe(include="all"))

    df_work = df.copy(deep=True)

    # Feature engineering
    network_col = args.network_col.strip() or None
    self_lms_col = args.self_lms_col.strip() or None
    config = FeatureEngineeringConfig(network_col=network_col, self_lms_col=self_lms_col)
    df_work, config = add_engineered_features(df_work, config)

    # Remove infinite values and split columns for missing value handling
    # Convert boolean columns to numeric for safety
    for col in df_work.columns:
        if pd.api.types.is_bool_dtype(df_work[col]):
            df_work[col] = df_work[col].astype(int)

    numeric_cols_all, categorical_cols_all = separate_numeric_categorical(df_work.drop(columns=[args.target_col]))

    # Handle missing values (required)
    df_work = df_work.replace([np.inf, -np.inf], np.nan)
    df_work = fill_missing_values(
        df_work,
        numeric_cols=numeric_cols_all,
        categorical_cols=categorical_cols_all,
    )

    # Ensure target is string labels (for consistent class order)
    df_work[args.target_col] = df_work[args.target_col].astype(str)

    # Split
    X = df_work.drop(columns=[args.target_col])
    y = df_work[args.target_col].values
    class_labels = list(pd.unique(df_work[args.target_col]))
    classes_order = get_target_order(df_work[args.target_col])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    # EDA (required). Use the training-prepared full df_work for plots.
    plot_target_distribution(df_work, args.target_col, out_path=os.path.join(eda_dir, "target_distribution.png"))

    # Univariate and bivariate categorical analyses (limited to keep output size manageable)
    # Use categorical columns as derived from the prepared full dataframe.
    numeric_cols_wo_target, categorical_cols_wo_target = separate_numeric_categorical(X_train)
    plot_univariate_categoricals(
        df_work=df_work,
        categorical_cols=categorical_cols_wo_target,
        target_col=args.target_col,
        out_dir=os.path.join(eda_dir, "univariate_categoricals"),
        max_plots=args.eda_max_plots,
    )
    plot_bivariate_categorical_vs_target_stacked(
        df_work=df_work,
        categorical_cols=categorical_cols_wo_target,
        target_col=args.target_col,
        out_dir=os.path.join(eda_dir, "bivariate_categorical_vs_target"),
        max_plots=max(6, min(8, args.eda_max_plots)),
    )

    # Feature engineering: identify low/high-cardinality categoricals on training set
    low_card_cols, high_card_cols = determine_low_high_cardinality(
        X_train, categorical_cols_wo_target, threshold=10
    )

    # Prepare preprocessors (scaled only for LR & SVM)
    preprocessor_unscaled = make_preprocessor(
        X_train=X_train,
        numeric_cols=numeric_cols_wo_target,
        low_card_cols=low_card_cols,
        high_card_cols=high_card_cols,
        scale_numeric=False,
    )
    preprocessor_scaled = make_preprocessor(
        X_train=X_train,
        numeric_cols=numeric_cols_wo_target,
        low_card_cols=low_card_cols,
        high_card_cols=high_card_cols,
        scale_numeric=True,
    )

    # Correlation heatmap after encoding (required)
    os.makedirs(os.path.join(plots_dir, "correlation"), exist_ok=True)
    # Fit preprocessor first so transform works
    preprocessor_unscaled.fit(X_train, y_train)
    plot_correlation_heatmap_after_encoding(
        preprocessor=preprocessor_unscaled,
        X_train=X_train,
        out_path=os.path.join(plots_dir, "correlation", "correlation_heatmap_after_encoding.png"),
        corr_max_features=args.corr_max_features,
    )

    # Build pipelines
    pipelines = build_model_pipelines(
        preprocessor_scaled=preprocessor_scaled,
        preprocessor_unscaled=preprocessor_unscaled,
        y_train=y_train,
    )

    results: List[Dict[str, Any]] = []
    best_name: Optional[str] = None
    best_f1: float = -1.0
    best_pipeline: Optional[Pipeline] = None

    # Train & evaluate each model
    for name, pipe in pipelines.items():
        print(f"\nTraining model: {name}")
        try:
            # Special handling for XGBoost sample weights
            if name == "XGBoost":
                # Fit preprocess+model pipeline with weights on the estimator step.
                # We need to call fit manually to inject weights.
                pipe.fit(X_train, y_train, model__sample_weight=compute_sample_weight(class_weight="balanced", y=y_train))
                y_pred = pipe.predict(X_test)
                y_prob = safe_predict_proba(pipe, X_test)
                model_classes = list(getattr(pipe.named_steps["model"], "classes_", np.unique(y_train)))
                metrics = compute_metrics(y_test, y_pred, y_prob, model_classes)
                report = classification_report(y_test, y_pred, digits=4)
                cm_text = confusion_matrix(y_test, y_pred, labels=model_classes)
            else:
                metrics, report, cm_text = evaluate_pipeline(
                    pipeline=pipe,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    classes=list(getattr(pipe.named_steps["model"], "classes_", np.unique(y_train))),
                )
        except Exception as e:
            print(f"Skipping model '{name}' due to error: {e}")
            continue

        # Print key results
        print("Accuracy:", metrics["accuracy"])
        print("F1 Macro:", metrics["f1_macro"])
        print("ROC-AUC OVR Macro:", metrics["roc_auc_ovr_macro"])

        # Save classification report
        report_path = os.path.join(plots_dir, f"classification_report__{name.replace(' ', '_')}.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"Model: {name}\n\n")
            f.write(report)

        # Confusion matrix plot
        cm_out_path = os.path.join(plots_dir, f"confusion_matrix__{name.replace(' ', '_')}.png")
        # Ensure label array ordering used in confusion matrix plot
        y_pred = pipe.predict(X_test)
        model_classes = list(getattr(pipe.named_steps["model"], "classes_", np.unique(y_train)))
        plot_confusion_matrix(
            y_true=y_test,
            y_pred=y_pred,
            classes=model_classes,
            out_path=cm_out_path,
        )

        row = {"model": name}
        row.update(metrics)
        results.append(row)

        if metrics["f1_macro"] > best_f1:
            best_f1 = float(metrics["f1_macro"])
            best_name = name
            best_pipeline = pipe

    results_df = pd.DataFrame(results).sort_values(by="f1_macro", ascending=False)
    results_csv = os.path.join(args.output_dir, "model_comparison_results.csv")
    results_df.to_csv(results_csv, index=False)

    print("\nModel comparison (sorted by F1 Macro):")
    print(results_df[["model", "accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc_ovr_macro"]])

    if best_pipeline is None or best_name is None:
        raise RuntimeError("No model selected as best. Check training outputs.")

    print(f"\nBest model selected by F1 Macro: {best_name} (F1 Macro={best_f1:.6f})")

    # Retrain best model on full training data (X_train)
    # Rebuild pipeline to avoid any accidental state from previous fit.
    # Rebuild the best pipeline from scratch (fresh estimator params)
    # (Scaling decisions are handled inside `build_model_pipelines`.)
    rebuild_pipelines = build_model_pipelines(
        preprocessor_scaled=preprocessor_scaled,
        preprocessor_unscaled=preprocessor_unscaled,
        y_train=y_train,
    )
    best_pipeline = rebuild_pipelines[best_name]

    if best_name == "XGBoost":
        fit_xgb_with_balanced_weights(best_pipeline, X_train, y_train)
    else:
        best_pipeline.fit(X_train, y_train)

    # Save model bundle (bonus)
    feature_columns = list(X.columns)  # columns after feature engineering, before encoding
    if args.save_model:
        bundle = {
            "pipeline": best_pipeline,
            "target_col": args.target_col,
            "feature_config": config,
            "feature_columns": feature_columns,
            "classes": list(np.unique(y_train)),
            "best_model_name": best_name,
        }
        joblib.dump(bundle, args.save_model)
        print(f"\nSaved best model bundle to: {args.save_model}")

    # Simple input-based example (bonus): interactive prompt
    # This only triggers if user provided --save-model; otherwise we just finish training.
    if args.save_model:
        print("\nQuick prediction example")
        print("-------------------------")
        print("To predict, re-run with --predict-file or --predict-json.")
        print("Example:")
        print(
            '  python student_adaptability_ml.py --data "your.csv" --save-model "best.joblib"'
        )
        print(
            '  python student_adaptability_ml.py --predict-json \'{"Age": 20, "network": "4G", "...": "..."}\' --model-bundle "best.joblib"'
        )


if __name__ == "__main__":
    sns.set_theme(style="whitegrid")
    main()

