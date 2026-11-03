# ✈️ Predicting Airline Passenger Satisfaction

A machine learning project built for a Kaggle competition. The goal is to predict whether an airline passenger was **satisfied** or **neutral/dissatisfied** with their flight, based on their profile, trip details, and ratings of in-flight services.

> 🏆 **Kaggle Competition:** [Kaggle Playground Series – Season 6, Episode 10](https://www.kaggle.com/competitions/playground-series-s6e10)
> 📊 **Public LB Score:** `0.95816` &nbsp;|&nbsp; **Private LB Score:** `0.95893` &nbsp;|&nbsp; **Rank:** `~2700 / 3800`

---

## 📌 Problem Statement

Customer satisfaction is a key driver of airline loyalty and revenue. Given passenger information and survey responses, build a binary classifier that predicts satisfaction so airlines can identify which factors matter most and where to improve.

- **Task:** Binary classification
- **Target:** `satisfaction` (satisfied vs. neutral or dissatisfied)
- **Evaluation metric:** ROC-AUC *(verify against the competition's Evaluation tab)*

---

## 🗂️ Repository Structure

```
Predicting-Airline-Satisfaction/
├── raw_data/                         # Original files downloaded from Kaggle
├── processed_data/                   # Cleaned & feature-engineered datasets
├── submission_file/                  # Final predictions for Kaggle submission
├── data_loading_&_preparation.ipynb  # Data loading, cleaning, preprocessing
├── EDA.ipynb                         # Exploratory data analysis & visualisations
├── model_training.ipynb              # Model training, tuning, evaluation, submission
└── README.md
```

---

## 📊 Dataset

The data comes from the Kaggle competition page linked above. Typical features include:

| Category | Example features |
|---|---|
| **Passenger profile** | Gender, Age, Customer Type (loyal / disloyal) |
| **Trip details** | Type of Travel, Class, Flight Distance |
| **Service ratings (1–5)** | Seat comfort, Inflight wifi, Food & drink, Online boarding, Inflight entertainment, Cleanliness, Leg room, Baggage handling, Check-in service, etc. |
| **Delays** | Departure delay, Arrival delay (minutes) |
| **Target** | Satisfaction |

> Update this table to match the exact columns in your dataset.

The raw files are stored in `raw_data/`. Download them again from Kaggle if needed.

---

## 🔄 Project Workflow

1. **Data Loading & Preparation** (`data_loading_&_preparation.ipynb`)
   - Load train/test files
   - Handle missing values (e.g., arrival delay)
   - Encode categorical variables
   - Save cleaned data to `processed_data/`

2. **Exploratory Data Analysis** (`EDA.ipynb`)
   - Target distribution and class balance
   - Distributions and correlations of numeric features
   - Satisfaction patterns by class, travel type, customer type, and service ratings
   - Key insights that guided feature engineering

3. **Modelling** (`model_training.ipynb`)
   - Baseline and advanced models
   - Cross-validation and hyperparameter tuning
   - Model comparison and feature importance
   - Generating the final `submission.csv` in `submission_file/`

---

## 🤖 Models & Results

| Model | Public LB | Private LB |
|---|---|---|
| Logistic Regression (baseline) | `0.92217` | `0.92159` |
| XGBoost | `0.95780` | `0.95884` |
| **CatBoost (final)** | **`0.95816`** | **`0.95893`** |

**Final model hyperparameters (CatBoost):**

```python
CatBoostClassifier(
    iterations=900,
    depth=8,
    learning_rate=0.08,
    loss_function="Logloss",
    eval_metric="AUC",
    random_state=42,
    l2_leaf_reg=5,
    random_strength=0.5,
    verbose=False,
    allow_writing_files=False,
)
```

**Other models tried:**
- `XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42)`
- `LogisticRegression(C=1.0, solver='lbfgs', max_iter=2000, random_state=42)`

Gradient boosting models clearly outperformed the linear baseline (~3.6 points), with CatBoost edging out XGBoost on both leaderboards.

---

## 🔍 Key Insights

- 💡 *Example:* Online boarding and in-flight entertainment ratings are among the strongest predictors of satisfaction.
- 💡 *Example:* Business class passengers and loyal customers are significantly more likely to be satisfied.
- 💡 *Example:* Poor in-flight wifi is a frequent driver of dissatisfaction.

> Replace with the real findings from your EDA and feature importance analysis.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Jupyter Notebook or JupyterLab

### Installation

```bash
git clone https://github.com/Rishabh2170/Predicting-Airline-Satisfaction.git
cd Predicting-Airline-Satisfaction
pip install pandas numpy matplotlib seaborn scikit-learn xgboost catboost jupyter
```

Core libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`, `catboost`.

### Run the notebooks in order

1. `data_loading_&_preparation.ipynb`
2. `EDA.ipynb`
3. `model_training.ipynb`

The final predictions are written to `submission_file/`, ready to upload to Kaggle.

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?logo=kaggle&logoColor=white)

---

## 🔮 Future Improvements

- Ensemble / blend CatBoost and XGBoost (and possibly LightGBM)
- More advanced feature engineering and interaction features
- Bayesian hyperparameter optimisation (Optuna) instead of manual tuning
- Model explainability with SHAP

---

## 👤 Author

**Rishabh**
GitHub: [@Rishabh2170](https://github.com/Rishabh2170)
Kaggle: [rishabhsingh2222](https://www.kaggle.com/rishabhsingh2222)

---

## 📄 License

This project is for educational and competition purposes. Dataset © its original owners / Kaggle competition hosts.
