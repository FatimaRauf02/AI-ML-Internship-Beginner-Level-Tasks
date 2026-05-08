# =============================================================================
# Task 3: Heart Disease Prediction
# DevelopersHub Corporation - AI/ML Engineering Internship
# =============================================================================
# Objective: Build a model to predict whether a person is at risk of heart
# disease based on their health data.
# Dataset: Heart Disease UCI Dataset
# =============================================================================

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay
)
from sklearn.utils import resample

# ── Output Directory ──────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# 1. LOAD DATASET
# =============================================================================
def load_dataset():
    """
    Load the Heart Disease UCI dataset.
    Tries a local CSV first; falls back to fetching from UCI repository.
    """
    local_path = os.path.join(os.path.dirname(__file__), "heart.csv")

    if os.path.exists(local_path):
        print(f"[INFO] Loading dataset from: {local_path}")
        df = pd.read_csv(local_path)
    else:
        # Download directly from UCI ML Repository mirror
        print("[INFO] heart.csv not found locally. Downloading from UCI repository...")
        url = (
            "https://archive.ics.uci.edu/ml/machine-learning-databases/"
            "heart-disease/processed.cleveland.data"
        )
        col_names = [
            "age", "sex", "cp", "trestbps", "chol", "fbs",
            "restecg", "thalach", "exang", "oldpeak", "slope",
            "ca", "thal", "target"
        ]
        df = pd.read_csv(url, header=None, names=col_names, na_values="?")
        # Binarise target: 0 = no disease, 1 = disease
        df["target"] = (df["target"] > 0).astype(int)
        print("[INFO] Dataset downloaded successfully.")

    return df


# =============================================================================
# 2. DATA INSPECTION
# =============================================================================
def inspect_data(df):
    print("\n" + "=" * 60)
    print("STEP 1: DATA INSPECTION")
    print("=" * 60)
    print(f"\nShape        : {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\nColumn names : {list(df.columns)}")
    print("\nFirst 5 rows:")
    print(df.head())
    print("\n.info():")
    df.info()
    print("\n.describe():")
    print(df.describe())
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nTarget distribution:\n{df['target'].value_counts()}")


# =============================================================================
# 3. DATA CLEANING / PREPROCESSING
# =============================================================================
def preprocess(df):
    """Handle missing values, encode & scale features."""
    print("\n" + "=" * 60)
    print("STEP 2: DATA CLEANING & PREPROCESSING")
    print("=" * 60)

    # Drop rows with missing values (very few in this dataset)
    before = len(df)
    df = df.dropna()
    print(f"Dropped {before - len(df)} rows with missing values. Remaining: {len(df)}")

    # Ensure correct dtypes for known categorical columns
    cat_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(float).astype(int)

    # Feature / target split
    X = df.drop("target", axis=1)
    y = df["target"]

    # Train / test split (80 / 20, stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale continuous features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    print(f"Train samples: {X_train_scaled.shape[0]}  |  Test samples: {X_test_scaled.shape[0]}")
    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns.tolist(), df


# =============================================================================
# 4. EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================
def run_eda(df):
    print("\n" + "=" * 60)
    print("STEP 3: EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    sns.set_theme(style="whitegrid", palette="muted")

    # ── 4a. Target distribution ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["target"].value_counts()
    bars = ax.bar(
        ["No Disease (0)", "Heart Disease (1)"],
        counts.values,
        color=["#4C72B0", "#DD8452"],
        edgecolor="white", width=0.5
    )
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            str(int(bar.get_height())),
            ha="center", va="bottom", fontweight="bold"
        )
    ax.set_title("Target Class Distribution", fontsize=14, fontweight="bold")
    ax.set_ylabel("Count")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "01_target_distribution.png"), dpi=150)
    plt.close(fig)
    print("  Saved: 01_target_distribution.png")

    # ── 4b. Correlation heatmap ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 9))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
        linewidths=0.5, ax=ax, vmin=-1, vmax=1,
        annot_kws={"size": 8}
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "02_correlation_heatmap.png"), dpi=150)
    plt.close(fig)
    print("  Saved: 02_correlation_heatmap.png")

    # ── 4c. Age distribution by target ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, color in zip([0, 1], ["#4C72B0", "#DD8452"]):
        subset = df[df["target"] == label]["age"]
        ax.hist(subset, bins=20, alpha=0.7, label=f"{'No Disease' if label==0 else 'Heart Disease'}",
                color=color, edgecolor="white")
    ax.set_title("Age Distribution by Target", fontsize=14, fontweight="bold")
    ax.set_xlabel("Age"); ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "03_age_distribution.png"), dpi=150)
    plt.close(fig)
    print("  Saved: 03_age_distribution.png")

    # ── 4d. Chest pain type vs target ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    cp_target = df.groupby(["cp", "target"]).size().unstack(fill_value=0)
    cp_target.plot(kind="bar", ax=ax, color=["#4C72B0", "#DD8452"],
                   edgecolor="white", rot=0)
    ax.set_title("Chest Pain Type vs Heart Disease", fontsize=14, fontweight="bold")
    ax.set_xlabel("Chest Pain Type (0-3)"); ax.set_ylabel("Count")
    ax.legend(["No Disease", "Heart Disease"])
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "04_cp_vs_target.png"), dpi=150)
    plt.close(fig)
    print("  Saved: 04_cp_vs_target.png")

    # ── 4e. Box plots for numerical features ─────────────────────────────
    num_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    fig, axes = plt.subplots(1, len(num_features), figsize=(16, 5))
    for ax, feat in zip(axes, num_features):
        df.boxplot(column=feat, by="target", ax=ax,
                   boxprops=dict(color="#4C72B0"),
                   medianprops=dict(color="#DD8452", linewidth=2))
        ax.set_title(feat, fontsize=11, fontweight="bold")
        ax.set_xlabel("Target (0=No, 1=Yes)")
    fig.suptitle("Box Plots: Numerical Features by Target", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "05_boxplots.png"), dpi=150)
    plt.close(fig)
    print("  Saved: 05_boxplots.png")

    # ── 4f. Scatter: Age vs Max Heart Rate ───────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, color, marker in zip([0, 1], ["#4C72B0", "#DD8452"], ["o", "^"]):
        subset = df[df["target"] == label]
        ax.scatter(subset["age"], subset["thalach"], alpha=0.6, s=60,
                   label=f"{'No Disease' if label==0 else 'Heart Disease'}",
                   color=color, marker=marker, edgecolors="white", linewidths=0.4)
    ax.set_title("Age vs Max Heart Rate (thalach)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Age"); ax.set_ylabel("Max Heart Rate")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "06_scatter_age_thalach.png"), dpi=150)
    plt.close(fig)
    print("  Saved: 06_scatter_age_thalach.png")


# =============================================================================
# 5. MODEL TRAINING & EVALUATION
# =============================================================================
def train_and_evaluate(X_train, X_test, y_train, y_test, feature_names):
    print("\n" + "=" * 60)
    print("STEP 4: MODEL TRAINING & EVALUATION")
    print("=" * 60)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree":       DecisionTreeClassifier(max_depth=5, random_state=42)
    }

    results = {}

    for name, model in models.items():
        print(f"\n── {name} ──")
        model.fit(X_train, y_train)
        y_pred  = model.predict(X_test)
        y_prob  = model.predict_proba(X_test)[:, 1]

        acc     = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)

        print(f"  Accuracy : {acc:.4f}")
        print(f"  ROC-AUC  : {roc_auc:.4f}")
        print(f"\n  Classification Report:\n{classification_report(y_test, y_pred)}")

        results[name] = {
            "model":   model,
            "y_pred":  y_pred,
            "y_prob":  y_prob,
            "acc":     acc,
            "roc_auc": roc_auc
        }

    return results


# =============================================================================
# 6. VISUALISE EVALUATION RESULTS
# =============================================================================
def plot_evaluation(results, y_test, feature_names, X_train):
    print("\n" + "=" * 60)
    print("STEP 5: GENERATING EVALUATION PLOTS")
    print("=" * 60)

    colors = {"Logistic Regression": "#4C72B0", "Decision Tree": "#DD8452"}

    # ── 6a. Confusion matrices ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (name, res) in zip(axes, results.items()):
        cm = confusion_matrix(y_test, res["y_pred"])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=["No Disease", "Heart Disease"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"{name}\nAccuracy: {res['acc']:.4f}", fontsize=12, fontweight="bold")
    fig.suptitle("Confusion Matrices", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "07_confusion_matrices.png"), dpi=150)
    plt.close(fig)
    print("  Saved: 07_confusion_matrices.png")

    # ── 6b. ROC curves ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random (AUC = 0.50)")
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        ax.plot(fpr, tpr, lw=2, color=colors[name],
                label=f"{name} (AUC = {res['roc_auc']:.4f})")
    ax.set_title("ROC Curves", fontsize=14, fontweight="bold")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "08_roc_curves.png"), dpi=150)
    plt.close(fig)
    print("  Saved: 08_roc_curves.png")

    # ── 6c. Feature importance (Logistic Regression coefficients) ────────
    lr_model = results["Logistic Regression"]["model"]
    coefs    = pd.Series(np.abs(lr_model.coef_[0]), index=feature_names).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    coefs.plot(kind="barh", ax=ax, color="#4C72B0", edgecolor="white")
    ax.set_title("Feature Importance\n(Logistic Regression |Coefficient|)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("|Coefficient|")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "09_feature_importance_lr.png"), dpi=150)
    plt.close(fig)
    print("  Saved: 09_feature_importance_lr.png")

    # ── 6d. Feature importance (Decision Tree Gini) ───────────────────────
    dt_model = results["Decision Tree"]["model"]
    dt_imp   = pd.Series(dt_model.feature_importances_, index=feature_names).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    dt_imp.plot(kind="barh", ax=ax, color="#DD8452", edgecolor="white")
    ax.set_title("Feature Importance\n(Decision Tree Gini Importance)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "10_feature_importance_dt.png"), dpi=150)
    plt.close(fig)
    print("  Saved: 10_feature_importance_dt.png")

    # ── 6e. Model comparison bar chart ────────────────────────────────────
    model_names = list(results.keys())
    accs        = [results[n]["acc"]     for n in model_names]
    roc_aucs    = [results[n]["roc_auc"] for n in model_names]

    x = np.arange(len(model_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, accs,     width, label="Accuracy", color="#4C72B0", edgecolor="white")
    bars2 = ax.bar(x + width/2, roc_aucs, width, label="ROC-AUC",  color="#DD8452", edgecolor="white")

    for bar in bars1 + bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x); ax.set_xticklabels(model_names, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title("Model Comparison: Accuracy vs ROC-AUC", fontsize=13, fontweight="bold")
    ax.set_ylabel("Score"); ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "11_model_comparison.png"), dpi=150)
    plt.close(fig)
    print("  Saved: 11_model_comparison.png")


# =============================================================================
# 7. SUMMARY REPORT
# =============================================================================
def print_summary(results):
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    best = max(results.items(), key=lambda x: x[1]["roc_auc"])
    print(f"\n  Best Model   : {best[0]}")
    print(f"  Accuracy     : {best[1]['acc']:.4f}")
    print(f"  ROC-AUC      : {best[1]['roc_auc']:.4f}")
    print(f"\n  All outputs saved to: {OUTPUT_DIR}/")
    print("\nKey Findings:")
    print("  • Features like 'thal', 'cp', 'ca', and 'thalach' are strong predictors.")
    print("  • Older patients with typical angina (cp=0) are at higher risk.")
    print("  • Higher max heart rate (thalach) is generally protective.")
    print("=" * 60)


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print(" Task 3: Heart Disease Prediction")
    print(" DevelopersHub Corporation - AI/ML Internship")
    print("=" * 60)

    df = load_dataset()
    inspect_data(df)
    X_train, X_test, y_train, y_test, feature_names, df_clean = preprocess(df)
    run_eda(df_clean)
    results = train_and_evaluate(X_train, X_test, y_train, y_test, feature_names)
    plot_evaluation(results, y_test, feature_names, X_train)
    print_summary(results)