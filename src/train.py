"""
Model training pipeline for multi-label ECG classification.

Trains a Logistic Regression baseline and an XGBoost classifier
using the PTB-XL patient-level train/validation/test splits.
All experiments are tracked with MLflow.
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, classification_report
from xgboost import XGBClassifier

SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]

# PTB-XL official fold assignments
TRAIN_FOLDS = [1, 2, 3, 4, 5, 6, 7, 8]
VAL_FOLD    = 9
TEST_FOLD   = 10


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load features and labels, then split by official PTB-XL folds.

    Returns:
        Tuple of (features DataFrame, label matrix DataFrame, metadata DataFrame)
        all indexed by ecg_id.
    """
    from data_loader import load_metadata, load_scp_statements
    from labels import build_label_matrix, filter_labeled_records

    features = pd.read_csv("data/features.csv", index_col="ecg_id")
    df       = load_metadata()
    scp      = load_scp_statements()

    label_matrix         = build_label_matrix(df, scp)
    df_labeled, labels   = filter_labeled_records(df, label_matrix)

    # Align all three DataFrames to the same index
    common_idx = features.index.intersection(labels.index)
    features   = features.loc[common_idx]
    labels     = labels.loc[common_idx]
    df_labeled = df_labeled.loc[common_idx]

    return features, labels, df_labeled


def split_by_fold(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    metadata: pd.DataFrame,
) -> tuple:
    """
    Split data into train, validation, and test sets using official PTB-XL folds.

    Folds 1-8: training | Fold 9: validation | Fold 10: test
    This assignment ensures no patient appears in more than one split.
    """
    train_idx = metadata[metadata["strat_fold"].isin(TRAIN_FOLDS)].index
    val_idx   = metadata[metadata["strat_fold"] == VAL_FOLD].index
    test_idx  = metadata[metadata["strat_fold"] == TEST_FOLD].index

    X_train, y_train = features.loc[train_idx], labels.loc[train_idx]
    X_val,   y_val   = features.loc[val_idx],   labels.loc[val_idx]
    X_test,  y_test  = features.loc[test_idx],  labels.loc[test_idx]

    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_features(
    X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize features to zero mean and unit variance.

    The scaler is fitted on training data only, then applied to
    validation and test sets. This prevents data leakage.
    """
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)
    return X_train, X_val, X_test, scaler

def impute_features(
    X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Replace NaN values with the column median computed on the training set.

    Median imputation is used instead of mean because physiological features
    can have skewed distributions where the median is more representative.
    The median is computed on training data only to prevent data leakage.
    """
    from sklearn.impute import SimpleImputer

    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_val   = imputer.transform(X_val)
    X_test  = imputer.transform(X_test)
    return X_train, X_val, X_test    


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, split: str):
    """
    Compute and print macro-averaged AUC and F1 for a given split.

    Args:
        y_true: ground truth binary matrix (n_samples, n_classes)
        y_pred: predicted binary matrix after threshold application
        y_prob: predicted probability matrix (n_samples, n_classes)
        split:  label for printing ('val' or 'test')
    """
    auc = roc_auc_score(y_true, y_prob, average="macro")
    f1  = f1_score(y_true, y_pred, average="macro", zero_division=0)
    print(f"\n{split.upper()} — Macro AUC: {auc:.4f} | Macro F1: {f1:.4f}")
    print(classification_report(y_true, y_pred, target_names=SUPERCLASSES, zero_division=0))
    return auc, f1


def train_logistic_regression(
    X_train, X_val, X_test, y_train, y_val, y_test
):
    """Train a One-vs-Rest Logistic Regression baseline and log it to MLflow."""
    print("\n--- Logistic Regression ---")

    with mlflow.start_run(run_name="logistic_regression"):
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("strategy", "OneVsRest")
        mlflow.log_param("max_iter", 1000)

        model = OneVsRestClassifier(
            LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
        )
        model.fit(X_train, y_train)

        y_prob = model.predict_proba(X_val)
        y_pred = (y_prob >= 0.5).astype(int)

        auc, f1 = evaluate(y_val.values, y_pred, y_prob, "val")
        mlflow.log_metric("val_macro_auc", auc)
        mlflow.log_metric("val_macro_f1",  f1)

        # Final evaluation on test set
        y_prob_test = model.predict_proba(X_test)
        y_pred_test = (y_prob_test >= 0.5).astype(int)
        auc_test, f1_test = evaluate(y_test.values, y_pred_test, y_prob_test, "test")
        mlflow.log_metric("test_macro_auc", auc_test)
        mlflow.log_metric("test_macro_f1",  f1_test)

        mlflow.sklearn.log_model(model, "model")

    return model


def train_xgboost(
    X_train, X_val, X_test, y_train, y_val, y_test
):
    """Train a One-vs-Rest XGBoost classifier and log it to MLflow."""
    print("\n--- XGBoost ---")

    with mlflow.start_run(run_name="xgboost"):
        mlflow.log_param("model", "XGBoost")
        mlflow.log_param("strategy", "OneVsRest")
        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("max_depth", 6)
        mlflow.log_param("learning_rate", 0.1)

        base = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric="logloss",
            verbosity=0,
        )
        model = OneVsRestClassifier(base)
        model.fit(X_train, y_train)

        y_prob = model.predict_proba(X_val)
        y_pred = (y_prob >= 0.5).astype(int)

        auc, f1 = evaluate(y_val.values, y_pred, y_prob, "val")
        mlflow.log_metric("val_macro_auc", auc)
        mlflow.log_metric("val_macro_f1",  f1)

        y_prob_test = model.predict_proba(X_test)
        y_pred_test = (y_prob_test >= 0.5).astype(int)
        auc_test, f1_test = evaluate(y_test.values, y_pred_test, y_prob_test, "test")
        mlflow.log_metric("test_macro_auc", auc_test)
        mlflow.log_metric("test_macro_f1",  f1_test)

        mlflow.xgboost.log_model(model.estimators_[0], "model")

    return model


if __name__ == "__main__":
    mlflow.set_experiment("ecg-classification")

    print("Loading data...")
    features, labels, metadata = load_data()

    print("Splitting by fold...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_by_fold(
        features, labels, metadata
    )

    print("Scaling features...")
    X_train, X_val, X_test, scaler = scale_features(X_train, X_val, X_test)

    print("Imputing missing values...")
    X_train, X_val, X_test = impute_features(X_train, X_val, X_test)
    
    train_logistic_regression(X_train, X_val, X_test, y_train, y_val, y_test)
    train_xgboost(X_train, X_val, X_test, y_train, y_val, y_test)

    print("\nAll runs logged. Launch MLflow UI with:")
    print("  mlflow ui")