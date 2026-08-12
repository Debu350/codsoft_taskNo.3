# 💳 Credit Card Fraud Detection Using Machine Learning

A machine learning project that detects **fraudulent credit card transactions** using classification algorithms and techniques for handling highly imbalanced datasets.

The project compares **Logistic Regression** and **Random Forest** models under three different data-resampling strategies:

- No resampling
- Random Undersampling
- SMOTE (Synthetic Minority Oversampling Technique)

The models are evaluated using metrics such as **Precision, Recall, F1-Score, ROC-AUC, and PR-AUC**.

---

## 📌 Project Overview

Credit card fraud detection is a binary classification problem where the goal is to identify whether a transaction is:

- `0` → Genuine transaction
- `1` → Fraudulent transaction

Fraud detection is challenging because fraudulent transactions represent only a very small percentage of all transactions. This creates a **highly imbalanced dataset**, where a model can achieve high accuracy while still failing to detect many fraudulent transactions.

This project focuses on handling this class imbalance and evaluating models using metrics that are more meaningful for fraud detection.

---

## 🎯 Objectives

The main objectives of this project are:

- Load and explore credit card transaction data.
- Analyze the distribution of genuine and fraudulent transactions.
- Check for missing values.
- Analyze transaction amount statistics.
- Standardize the `Time` and `Amount` features.
- Split the dataset using stratified train-test splitting.
- Train Logistic Regression and Random Forest classifiers.
- Handle class imbalance using:
  - Random Undersampling
  - SMOTE
- Evaluate model performance using:
  - Precision
  - Recall
  - F1-Score
  - ROC-AUC
  - PR-AUC
- Generate confusion matrices.
- Generate Precision-Recall curves.
- Compare all model and resampling combinations.
- Identify the best model based on F1-Score.

---

## 📊 Dataset

The project uses the **Credit Card Fraud Detection dataset**.

The dataset contains transactions made by European cardholders.

### Dataset Features

The dataset contains **284,807 transactions** and **31 columns**.

| Feature | Description |
|---|---|
| `Time` | Seconds elapsed between each transaction and the first transaction |
| `V1`–`V28` | Principal Component Analysis (PCA) transformed features |
| `Amount` | Transaction amount |
| `Class` | Target variable |

### Target Variable

| Class | Meaning |
|---|---|
| `0` | Genuine transaction |
| `1` | Fraudulent transaction |

The dataset contains approximately **0.17% fraudulent transactions**, making it highly imbalanced.

> **Note:** The dataset file `creditcard.csv` is not included in this repository if its redistribution is restricted. Download the dataset separately and place it in the project directory.

---

## 🧠 Machine Learning Approach

The project follows the following pipeline:

```text
Credit Card Dataset
        │
        ▼
Data Loading & Exploration
        │
        ▼
Missing Value Check
        │
        ▼
Class Distribution Analysis
        │
        ▼
Feature Scaling
(Time & Amount)
        │
        ▼
Stratified Train-Test Split
        │
        ▼
Training Data
        │
        ├───────────────┐
        ▼               ▼
   No Resampling   Resampling
                    │
             ┌──────┴──────┐
             ▼             ▼
       Undersampling     SMOTE
             │             │
             └──────┬──────┘
                    ▼
          Machine Learning Models
                    │
             ┌──────┴──────┐
             ▼             ▼
      Logistic Regression  Random Forest
             │             │
             └──────┬──────┘
                    ▼
              Model Evaluation
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
    Precision     Recall      F1-Score
        │           │           │
        └───────────┼───────────┘
                    ▼
             ROC-AUC / PR-AUC
                    │
                    ▼
              Best Model
```

---

## 🔍 Data Exploration

The project performs an initial analysis of the dataset, including:

- Dataset shape
- Missing value detection
- Class distribution
- Fraud percentage
- Transaction amount statistics

Two visualizations are generated:

### 1. Class Distribution

Shows the severe imbalance between genuine and fraudulent transactions.

### 2. Transaction Amount by Class

A boxplot is used to compare transaction amounts between genuine and fraudulent transactions.

The generated visualization is saved as:

```text
01_data_overview.png
```

---

## ⚙️ Data Preprocessing

### Feature Scaling

The dataset's `V1`–`V28` features are already PCA-transformed.

The `Time` and `Amount` features are standardized using `StandardScaler`.

```python
scaler = StandardScaler()

df[["Time", "Amount"]] = scaler.fit_transform(
    df[["Time", "Amount"]]
)
```

### Train-Test Split

The dataset is divided into:

- 80% training data
- 20% testing data

A **stratified split** is used to preserve the fraud-to-genuine transaction ratio.

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)
```

---

# ⚖️ Handling Class Imbalance

Because fraudulent transactions are extremely rare, training directly on the original dataset can cause the model to favor the majority class.

This project compares three strategies.

## 1. No Resampling

The original training dataset is used without modification.

```text
Original Training Data
        ↓
Model Training
```

---

## 2. Random Undersampling

Random undersampling reduces the number of genuine transactions so that the classes become more balanced.

```text
Majority Class ──► Reduced
Minority Class ──► Preserved
```

This can improve fraud detection but may discard useful information from genuine transactions.

---

## 3. SMOTE

SMOTE stands for:

**Synthetic Minority Oversampling Technique**

Instead of simply duplicating existing fraud examples, SMOTE generates synthetic minority-class samples.

```text
Original Fraud Samples
        ↓
SMOTE
        ↓
Synthetic Fraud Samples
        ↓
Balanced Training Dataset
```

SMOTE is applied **only to the training set**.

The test set is never resampled because doing so would produce misleading evaluation results.

---

# 🤖 Machine Learning Models

Two classification algorithms are used.

## Logistic Regression

Logistic Regression provides a strong and interpretable baseline for binary classification.

Configuration:

```python
LogisticRegression(
    max_iter=1000,
    random_state=42
)
```

---

## Random Forest

Random Forest is an ensemble learning algorithm that combines multiple decision trees.

Configuration:

```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    n_jobs=-1,
    random_state=42
)
```

Random Forest can capture nonlinear relationships between transaction features and fraud patterns.

---

# 📈 Model Evaluation

Because this is a highly imbalanced classification problem, **accuracy alone is not sufficient**.

The project evaluates models using the following metrics.

## Precision

Precision measures how many transactions predicted as fraud are actually fraudulent.

```text
Precision = TP / (TP + FP)
```

High precision means fewer genuine transactions are incorrectly flagged as fraud.

---

## Recall

Recall measures how many actual fraudulent transactions were successfully detected.

```text
Recall = TP / (TP + FN)
```

Recall is especially important in fraud detection because missing a fraudulent transaction can be costly.

---

## F1-Score

F1-Score combines Precision and Recall.

```text
F1 = 2 × (Precision × Recall)
     -------------------------
       Precision + Recall
```

The project uses **F1-Score as the primary metric for selecting the best model**.

---

## ROC-AUC

ROC-AUC measures the model's ability to distinguish between genuine and fraudulent transactions across different classification thresholds.

---

## PR-AUC

Precision-Recall AUC is particularly useful for highly imbalanced datasets because it focuses on the relationship between precision and recall for the minority class.

---

# 📊 Visualizations

The project generates the following visualizations.

### Data Overview

```text
01_data_overview.png
```

Contains:

- Class distribution
- Transaction amount distribution by class

### Confusion Matrices

```text
02_confusion_matrices.png
```

Contains confusion matrices for the Random Forest model under the different resampling strategies.

### Precision-Recall Curves

```text
03_precision_recall_curves.png
```

Compares the Precision-Recall performance of Random Forest under:

- No resampling
- Random Undersampling
- SMOTE

---

# 📋 Model Comparison

The program evaluates all combinations of:

| Model | Resampling |
|---|---|
| Logistic Regression | None |
| Logistic Regression | Undersampling |
| Logistic Regression | SMOTE |
| Random Forest | None |
| Random Forest | Undersampling |
| Random Forest | SMOTE |

The results are sorted by F1-Score.

The final results are saved to:

```text
model_results_summary.csv
```

Example structure:

| Model | Resampling | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---:|---:|---:|---:|---:|
| Random Forest | SMOTE | ... | ... | ... | ... | ... |
| Random Forest | Undersampling | ... | ... | ... | ... | ... |
| Logistic Regression | SMOTE | ... | ... | ... | ... | ... |

The actual values will depend on the dataset and execution environment.

---

# 🛠️ Technologies Used

- **Python**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Seaborn**
- **Scikit-learn**
- **Imbalanced-learn (imblearn)**

### Machine Learning

- Logistic Regression
- Random Forest
- SMOTE
- Random Undersampling

### Evaluation

- Precision
- Recall
- F1-Score
- ROC-AUC
- PR-AUC
- Confusion Matrix
- Precision-Recall Curve

---

# 📦 Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/Credit-Card-Fraud-Detection.git
```

Move into the project directory:

```bash
cd Credit-Card-Fraud-Detection
```

Install the required packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
```

---

# 📁 Project Structure

```text
Credit-Card-Fraud-Detection/
│
├── creditcard.csv
├── fraud_detection.py
├── README.md
│
├── 01_data_overview.png
├── 02_confusion_matrices.png
├── 03_precision_recall_curves.png
│
└── model_results_summary.csv
```

If the dataset is not included in the repository:

```text
Credit-Card-Fraud-Detection/
│
├── fraud_detection.py
├── README.md
├── 01_data_overview.png
├── 02_confusion_matrices.png
├── 03_precision_recall_curves.png
└── model_results_summary.csv
```

Place `creditcard.csv` in the root directory before running the program.

---

# ▶️ How to Run

If your dataset is named:

```text
creditcard.csv
```

run:

```bash
python fraud_detection.py
```

Or specify a custom dataset path:

```bash
python fraud_detection.py --data path/to/creditcard.csv
```

The program will:

1. Load the dataset.
2. Display dataset information.
3. Analyze class imbalance.
4. Generate the data overview visualization.
5. Preprocess the data.
6. Create a stratified train-test split.
7. Apply different resampling strategies.
8. Train Logistic Regression and Random Forest models.
9. Evaluate every model.
10. Generate confusion matrices.
11. Generate Precision-Recall curves.
12. Save the model comparison results.
13. Display the best model according to F1-Score.

---

# 📤 Output Files

After successful execution, the following files are generated:

```text
01_data_overview.png
02_confusion_matrices.png
03_precision_recall_curves.png
model_results_summary.csv
```

---

# 🔑 Key Insights

This project demonstrates several important concepts in real-world machine learning:

- Fraud detection datasets can be extremely imbalanced.
- Accuracy can be misleading for rare-event classification.
- Precision and Recall provide more useful information.
- F1-Score provides a balance between Precision and Recall.
- PR-AUC can be particularly informative for imbalanced datasets.
- Resampling should be applied only to the training data.
- SMOTE can increase representation of the minority fraud class.
- Different algorithms respond differently to class imbalance.
- Model selection should depend on the actual business objective.

---

# 🚀 Future Improvements

Possible improvements for future versions include:

- Hyperparameter tuning using `GridSearchCV` or `RandomizedSearchCV`
- Threshold optimization for fraud classification
- XGBoost or LightGBM models
- Cost-sensitive learning
- Feature importance analysis
- SHAP-based model explainability
- Real-time fraud prediction API using Flask or FastAPI
- Interactive fraud detection dashboard
- Deployment using Streamlit
- Model monitoring and drift detection
- Automated model retraining

---

# 🎓 Learning Outcomes

Through this project, I gained practical experience in:

- Binary classification
- Exploratory Data Analysis
- Feature preprocessing
- Feature scaling
- Imbalanced dataset handling
- SMOTE
- Random undersampling
- Logistic Regression
- Random Forest
- Model evaluation
- Precision-Recall analysis
- ROC-AUC analysis
- Confusion matrix interpretation
- Machine learning pipeline development

---

# 👨‍💻 Author

**Debabrata Mazumder**

Machine Learning / AI Enthusiast

---

# ⭐ If You Find This Project Useful

If you found this project useful for learning machine learning and fraud detection, consider giving the repository a ⭐.