# Task-02-MutawiffahMudassarKhan

# 🌸 Iris Flower Classification — Full ML Pipeline

A complete, end-to-end machine learning project built on the classic **Iris dataset**. This notebook covers everything from raw data loading and exploratory analysis to hyperparameter-tuned model comparison, evaluation, and deployment-ready model saving.

---

## 📁 Project Structure

```
iris-ml-project/
│
├── iris_classification_ml_project.ipynb   # Main Jupyter Notebook
├── Iris.csv                               # Dataset (place in /content/ for Colab)
├── iris_best_model.pkl                    # Saved best model (generated on run)
├── iris_scaler.pkl                        # Saved StandardScaler (generated on run)
├── iris_label_encoder.pkl                 # Saved LabelEncoder (generated on run)
└── README.md                              # This file
```

---

## 📊 Dataset

| Property | Detail |
|---|---|
| **Source** | [Kaggle — Iris Flower Dataset](https://www.kaggle.com/datasets/uciml/iris) |
| **Samples** | 150 (50 per class) |
| **Features** | 4 numeric (sepal & petal dimensions in cm) |
| **Target** | 3 species — `Iris-setosa`, `Iris-versicolor`, `Iris-virginica` |
| **Missing Values** | None |

### Feature Columns

| Column | Description |
|---|---|
| `SepalLengthCm` | Sepal length in centimetres |
| `SepalWidthCm` | Sepal width in centimetres |
| `PetalLengthCm` | Petal length in centimetres |
| `PetalWidthCm` | Petal width in centimetres |
| `Species` | Target class label (string) |

---

## 🔬 Pipeline Overview

### Stage 1 — Environment Setup & Imports
All required libraries are imported and plot styling is configured.

### Stage 2 — Data Loading & Overview
Dataset is loaded from `/content/Iris.csv`. Basic shape, data types, missing value check, and first 10 rows are displayed.

### Stage 3 — Exploratory Data Analysis (EDA)
- Feature distribution histograms by species
- Box plots for outlier detection
- Correlation heatmap
- Pairplot across all feature combinations
- Scatter plots for class separability

### Stage 4 — Data Preprocessing
- `Id` column dropped (if present)
- `LabelEncoder` applied to convert string species → integer labels
- 80/20 stratified train/test split
- `StandardScaler` fit on training data, applied to both splits

### Stage 5 — Model Training (8 Algorithms)

| Model | Library |
|---|---|
| Logistic Regression | `sklearn.linear_model` |
| K-Nearest Neighbors | `sklearn.neighbors` |
| Decision Tree | `sklearn.tree` |
| Random Forest | `sklearn.ensemble` |
| Gradient Boosting | `sklearn.ensemble` |
| AdaBoost | `sklearn.ensemble` |
| Support Vector Machine | `sklearn.svm` |
| Naive Bayes | `sklearn.naive_bayes` |

All models are evaluated with **5-Fold Stratified Cross-Validation** and test accuracy.

### Stage 6 — Hyperparameter Tuning
`GridSearchCV` is applied to the top 3 most complex models:
- Random Forest
- Support Vector Machine
- Gradient Boosting

### Stage 7 — Model Comparison
Horizontal bar charts compare all models by test accuracy and CV accuracy (with standard deviation).

### Stage 8 — Best Model Report
- Best model is selected automatically by highest test accuracy
- Full `classification_report` (precision, recall, F1-score)
- Normalised and raw confusion matrices

### Stage 9 — Feature Importance & Interpretation
- Random Forest feature importance bar chart
- Decision boundary visualisation on petal features (scaled)

### Stage 10 — Saving the Final Model
Model, scaler, and label encoder are persisted using `joblib`. An inference demo predicts 3 new samples with confidence scores.

---

## ⚙️ Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
joblib
```

Install all dependencies with:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn joblib
```

---

## 🚀 How to Run

### Google Colab (Recommended)
1. Upload `Iris.csv` to `/content/` in your Colab session
2. Open `iris_classification_ml_project.ipynb` in Colab
3. Click **Runtime → Run All**

### Local Jupyter
1. Place `Iris.csv` in `/content/` or update the path in **Cell 4**:
   ```python
   path = '/content/Iris.csv'   # ← change this to your local path
   ```
2. Launch Jupyter:
   ```bash
   jupyter notebook iris_classification_ml_project.ipynb
   ```
3. Run all cells

---

## 📈 Results Summary

> Results below are representative — exact values depend on GridSearchCV's best configuration.

| Model | CV Accuracy | Test Accuracy |
|---|---|---|
| Random Forest (Tuned) | ~0.9833 | ~1.0000 |
| SVM (Tuned) | ~0.9750 | ~1.0000 |
| Gradient Boosting (Tuned) | ~0.9667 | ~0.9667 |
| Logistic Regression | ~0.9583 | ~0.9667 |
| KNN | ~0.9583 | ~0.9667 |

**Key Finding:** Petal length and petal width are the most discriminative features. `Iris-setosa` is perfectly linearly separable; `Iris-versicolor` and `Iris-virginica` have slight overlap in sepal space but are well-separated in petal space.

---

## 🔮 Inference Example

After running the notebook, use the saved model to classify new samples:

```python
import joblib
import numpy as np

model  = joblib.load('iris_best_model.pkl')
scaler = joblib.load('iris_scaler.pkl')
le     = joblib.load('iris_label_encoder.pkl')

# [SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm]
sample   = np.array([[5.1, 3.5, 1.4, 0.2]])
scaled   = scaler.transform(sample)
pred     = model.predict(scaled)
species  = le.inverse_transform(pred)[0]
print(species)   # → Iris-setosa
```

---

## 📦 Output Artifacts

| File | Description |
|---|---|
| `iris_best_model.pkl` | Best-performing trained classifier |
| `iris_scaler.pkl` | Fitted `StandardScaler` for feature normalisation |
| `iris_label_encoder.pkl` | Fitted `LabelEncoder` for decoding predictions |

---

## 🧠 Key Takeaways

- Petal features explain ~94% of total feature importance in Random Forest
- Even simple models like Logistic Regression achieve >96% accuracy on this dataset
- Hyperparameter tuning pushes top models to near-perfect 100% test accuracy
- The dataset is well-suited for benchmarking classification algorithms

---
