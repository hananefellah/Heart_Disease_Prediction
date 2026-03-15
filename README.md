# 🫀 Heart Disease Prediction — Comparative ML Analysis
### Binary Classification | GridSearchCV Tuning | SHAP & LIME Explainability | 5 Models

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![Logistic Regression](https://img.shields.io/badge/Best_Model-Logistic_Regression-orange)
![SHAP](https://img.shields.io/badge/Explainability-SHAP-green)
![LIME](https://img.shields.io/badge/Explainability-LIME-yellowgreen)
![GridSearchCV](https://img.shields.io/badge/Tuning-GridSearchCV-red)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## 📌 Problem Statement

Heart disease is the leading cause of death globally, responsible for approximately **17.9 million deaths per year** according to the WHO. Early detection is critical — patients identified as high-risk can receive timely interventions that significantly improve survival outcomes.

This project builds and compares multiple machine learning models that predict heart disease from routine clinical measurements. The goal is not just accuracy, but also **clinical explainability** — using SHAP and LIME to reveal exactly why the model flags a patient as high-risk.

> **Dataset:** Heart Disease Prediction — 270 patients, 14 clinical features  
> **Domain:** Healthcare AI / Clinical Decision Support  
> **Target:** Binary — Absence / Presence of heart disease  
> **Clinical priority:** Recall for the Presence class — missing a sick patient is far more dangerous than a false alarm

---

## 📊 Results

### Baseline (Default Parameters)

| Model | Test Accuracy |
|-------|:---:|
| SVM | 87.0% |
| XGBoost | 87.0% |
| **Logistic Regression** | **92.6%** |

### After GridSearchCV Tuning

| Model | Best CV Accuracy | Test Accuracy |
|-------|:---:|:---:|
| XGBoost (Tuned) | 82.0% | 87.0% |
| SVM (Tuned) | 82.9% | 87.0% |
| Logistic Regression (Tuned) | 82.9% | 90.7% |
| Neural Network | — | 85.2% |
| Random Forest (Tuned) | **83.4%** | 83.3% |
| **Logistic Regression (Baseline)** ✅ | — | **92.6%** |

> **Best model: Logistic Regression — 92.6% accuracy, strongest on both precision and recall**  
> Logistic Regression outperforms all complex models, indicating that clinical features have strong **linear relationships** with heart disease in this dataset

### Per-Class Performance — Best Model (Logistic Regression)

| Class | Precision | Recall | F1 |
|---|:---:|:---:|:---:|
| Absence | High | High | High |
| **Presence** | High | **High** | High |

> Recall for Presence is the most critical clinical metric — the model successfully catches the majority of actual heart disease cases

---

## 🔍 SHAP Explainability — Top Clinical Drivers

SHAP analysis on the XGBoost model reveals which features most strongly drive predictions:

| Feature | Direction | Clinical Meaning |
|---|---|---|
| **Number of vessels fluro** | More blocked → higher risk | Direct marker of coronary artery disease severity |
| **Thallium** | Abnormal result → higher risk | Stress test detecting reduced blood flow to heart muscle |
| **Chest pain type** | Asymptomatic (type 4) → higher risk | Silent ischemia — most dangerous and easily missed |
| **Max HR** | Low max HR → higher risk | Inability to achieve high heart rate signals cardiac limitation |
| **ST depression** | Higher depression → higher risk | Exercise-induced ST changes indicate ischemia |
| **Exercise angina** | Present → higher risk | Chest pain during exertion is a direct disease indicator |

> These SHAP findings align with established cardiology knowledge, validating that the model has learned clinically meaningful patterns

---

## 🧠 Key Technical Decisions

### ✅ GridSearchCV with 5-Fold Cross-Validation
All 4 models tuned with exhaustive grid search. On a small 270-patient dataset, cross-validation is essential — a single train/test split is not reliable enough to guide clinical decisions.

### ✅ Two-Stage Scaling
- **MinMaxScaler** applied before splitting to normalize numerical feature ranges to [0,1]
- **StandardScaler** applied after splitting (fit on train only) for zero mean and unit variance — required for distance-based and gradient models (SVM, LR, NN)

### ✅ SHAP — Global Explainability
SHAP (SHapley Additive exPlanations) reveals which features drive predictions across the entire test set, and how each feature value pushes individual predictions. Every SHAP finding in this project is backed by clinical cardiology literature.

### ✅ LIME — Local Explainability
LIME (Local Interpretable Model-Agnostic Explanations) explains a single patient's prediction in plain terms — telling a doctor not just "this patient has heart disease" but exactly which clinical measurements triggered the flag. This is what makes AI clinically actionable.

### ✅ Why Logistic Regression Wins
Complex models (XGBoost, Random Forest, Neural Network) do not outperform Logistic Regression here. This is a meaningful finding — the clinical features in this dataset have strong linear separability, and simpler models generalize better on small datasets (270 patients).

---

## 🔑 Key EDA Findings

| Feature | Finding |
|---|---|
| **Average age** | 54.4 years — predominantly middle-aged to older adults |
| **Average BP** | 131.3 mmHg — above the normal threshold of 120 |
| **Average cholesterol** | 249.7 mg/dl — above the recommended limit of 200 |
| **Chest pain type 4** | Asymptomatic patients are counterintuitively at highest risk |
| **Max HR** | Lower maximum heart rate during exercise correlates with disease |
| **Class balance** | 150 Absence / 120 Presence — balanced, no SMOTE needed |

---

## 📁 Project Structure

```
Heart_Disease_Prediction/
├── Heart_Disease_Prediction.ipynb    # Main notebook (fully executed)
└── README.md
```

> The dataset is loaded automatically via `kagglehub`. To run locally without a Kaggle account, download `Heart_Disease_Prediction.csv` and load with `pd.read_csv("Heart_Disease_Prediction.csv")`.

---

## ⚙️ How to Run

```bash
# 1. Clone
git clone https://github.com/hananefellah/Heart_Disease_Prediction
cd Heart_Disease_Prediction

# 2. Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn xgboost shap lime kagglehub

# 3. Run
jupyter notebook Heart_Disease_Prediction.ipynb
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.12 | Core language |
| Pandas / NumPy | Data manipulation |
| Scikit-learn | ML models, preprocessing, GridSearchCV |
| XGBoost | Gradient boosting model |
| SHAP | Global model explainability |
| LIME | Local patient-level explainability |
| Matplotlib / Seaborn | Visualizations |
| kagglehub | Dataset loading |

---

## 💼 Clinical Recommendations

1. **Prioritize fluoroscopy and thallium tests** — the strongest predictors; ensure these are part of standard screening protocols
2. **Monitor asymptomatic patients carefully** — type 4 chest pain (silent ischemia) is the highest-risk presentation and is easily overlooked
3. **Track ST depression during exercise tests** — a consistent predictor across all models and confirmed by SHAP
4. **Use LIME explanations in clinical workflow** — patient-specific explanations increase physician trust and support informed consent
5. **Flag patients with low max HR during stress tests** — inability to achieve target heart rate is a strong independent risk marker

---

## 🚀 Future Work

- [ ] Validate on a larger, multi-center dataset to confirm generalization
- [ ] Add calibration curve to assess probability reliability for clinical use
- [ ] Deploy as a clinical decision support API with FastAPI
- [ ] Explore deep learning on larger cardiac datasets (ECG signals, imaging data)

---
## 📄 License

MIT License


## 👩‍💻 Author

**Fellah Hanane** — Data Scientist  
🌐 [GitHub](https://github.com/hananefellah) 
📧 Email: hananefellah35@gmail.com
· Open to Remote Roles