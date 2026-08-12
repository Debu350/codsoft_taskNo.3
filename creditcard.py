import argparse
import warnings
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
 
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
 
warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
RANDOM_STATE = 42

# 1. LOAD & EXPLORE

def load_and_explore(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
 
    print("=" * 70)
    print("DATA OVERVIEW")
    print("=" * 70)
    print(f"Shape: {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum()}")
 
    class_counts = df["Class"].value_counts()
    fraud_pct = class_counts[1] / len(df) * 100
    print(f"\nClass distribution:")
    print(f"  Genuine (0): {class_counts[0]:,}")
    print(f"  Fraud   (1): {class_counts[1]:,}  ({fraud_pct:.3f}% of all transactions)")
    print(f"\nAmount stats:\n{df['Amount'].describe()}")
 
    # Visualize class imbalance and amount distribution
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
 
    sns.countplot(x="Class", data=df, ax=axes[0], hue="Class",
                  palette=["#4C72B0", "#C44E52"], legend=False)
    axes[0].set_title("Class Distribution (0 = Genuine, 1 = Fraud)")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Count (log scale)")
 
    sns.boxplot(x="Class", y="Amount", data=df, ax=axes[1], hue="Class",
                palette=["#4C72B0", "#C44E52"], legend=False)
    axes[1].set_title("Transaction Amount by Class")
    axes[1].set_ylim(0, 500)  # zoom in, amounts are heavily right-skewed
 
    plt.tight_layout()
    plt.savefig("01_data_overview.png", dpi=120)
    print("\nSaved: 01_data_overview.png")
    plt.close()
 
    return df
 
# 2. PREPROCESS

def preprocess(df: pd.DataFrame):
    df = df.copy()
 
    # V1-V28 are already PCA components (pre-scaled). Only Time and Amount
    # are on a different scale, so we standardize just those two.
    scaler = StandardScaler()
    df[["Time", "Amount"]] = scaler.fit_transform(df[["Time", "Amount"]])
 
    X = df.drop(columns=["Class"])
    y = df["Class"]
 
    # Stratified split preserves the ~0.17% fraud ratio in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
 
    print("\n" + "=" * 70)
    print("TRAIN/TEST SPLIT")
    print("=" * 70)
    print(f"Train: {X_train.shape[0]:,} rows  |  Frauds: {y_train.sum()} "
          f"({y_train.mean()*100:.3f}%)")
    print(f"Test:  {X_test.shape[0]:,} rows  |  Frauds: {y_test.sum()} "
          f"({y_test.mean()*100:.3f}%)")
 
    return X_train, X_test, y_train, y_test

# 3. RESAMPLING STRATEGIES

def resample_data(X_train, y_train, strategy: str):
    """
    strategy: 'none' | 'smote' | 'undersample'
    Only ever resample the TRAINING set — never touch the test set,
    or evaluation metrics become meaningless.
    """
    if strategy == "none":
        return X_train, y_train
 
    if strategy == "smote":
        sampler = SMOTE(random_state=RANDOM_STATE)
    elif strategy == "undersample":
        sampler = RandomUnderSampler(random_state=RANDOM_STATE)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
 
    X_res, y_res = sampler.fit_resample(X_train, y_train)
    print(f"\n[{strategy.upper()}] Resampled training set: {X_res.shape[0]:,} rows "
          f"({y_res.sum():,} fraud / {(y_res == 0).sum():,} genuine)")
    return X_res, y_res
 
# 4. TRAIN MODELS

def train_models(X_train, y_train):
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=12, n_jobs=-1, random_state=RANDOM_STATE
        ),
    }
 
    fitted = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted[name] = model
    return fitted
 
# 5. EVALUATE

def evaluate_model(name, model, X_test, y_test, strategy_label, results_log):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
 
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
 
    print(f"\n--- {name} | resampling: {strategy_label} ---")
    print(classification_report(y_test, y_pred, target_names=["Genuine", "Fraud"]))
    print(f"ROC-AUC: {roc_auc:.4f}  |  PR-AUC: {pr_auc:.4f}")
 
    results_log.append({
        "Model": name,
        "Resampling": strategy_label,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc,
    })
 
    return y_pred, y_proba, confusion_matrix(y_test, y_pred)
 
 
def plot_confusion_matrices(cms: dict):
    fig, axes = plt.subplots(1, len(cms), figsize=(6 * len(cms), 5))
    if len(cms) == 1:
        axes = [axes]
    for ax, (title, cm) in zip(axes, cms.items()):
        sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", ax=ax,
                    xticklabels=["Genuine", "Fraud"],
                    yticklabels=["Genuine", "Fraud"])
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig("02_confusion_matrices.png", dpi=120)
    print("\nSaved: 02_confusion_matrices.png")
    plt.close()
 
 
def plot_pr_curves(curves: dict):
    plt.figure(figsize=(7, 6))
    for label, (y_test, y_proba) in curves.items():
        prec, rec, _ = precision_recall_curve(y_test, y_proba)
        ap = average_precision_score(y_test, y_proba)
        plt.plot(rec, prec, label=f"{label} (AP={ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves (best model per resampling strategy)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("03_precision_recall_curves.png", dpi=120)
    print("Saved: 03_precision_recall_curves.png")
    plt.close()
 
# MAIN

def main(path: str):
    df = load_and_explore(path)
    X_train, X_test, y_train, y_test = preprocess(df)
 
    strategies = ["none", "undersample", "smote"]
    results_log = []
    cms_to_plot = {}
    pr_curves_to_plot = {}
 
    for strategy in strategies:
        X_res, y_res = resample_data(X_train, y_train, strategy)
        models = train_models(X_res, y_res)
 
        for name, model in models.items():
            y_pred, y_proba, cm = evaluate_model(
                name, model, X_test, y_test, strategy, results_log
            )
            cms_to_plot[f"{name}\n({strategy})"] = cm
 
            # Keep only Random Forest curves for the PR plot (cleaner chart)
            if name == "Random Forest":
                pr_curves_to_plot[strategy] = (y_test, y_proba)
 
    # Summary table
    
    results_df = pd.DataFrame(results_log).sort_values("F1", ascending=False)
    print("\n" + "=" * 70)
    print("SUMMARY — sorted by F1-score")
    print("=" * 70)
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    results_df.to_csv("model_results_summary.csv", index=False)
    print("\nSaved: model_results_summary.csv")
 
    # Plot only the 6 confusion matrices in a readable grid (3 strategies x RF only, for space)
    rf_cms = {k: v for k, v in cms_to_plot.items() if "Random Forest" in k}
    plot_confusion_matrices(rf_cms)
    plot_pr_curves(pr_curves_to_plot)
 
    print("\nDone. Best model by F1:")
    best = results_df.iloc[0]
    print(f"  {best['Model']} + {best['Resampling']} resampling "
          f"-> F1={best['F1']:.4f}, Precision={best['Precision']:.4f}, "
          f"Recall={best['Recall']:.4f}")
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Credit Card Fraud Detection")
    parser.add_argument(
        "--data", type=str, default="creditcard.csv",
        help="Path to creditcard.csv"
    )
    args = parser.parse_args()
    main(args.data)