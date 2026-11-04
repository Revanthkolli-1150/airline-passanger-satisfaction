# ✈️ AIRLINE SATISFACTION INTELLIGENCE
### Production-Grade Machine Learning Classification & Interactive Decision Support System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Streamlit App](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![Kaggle Public Benchmark](https://img.shields.io/badge/Kaggle%20Benchmark-0.96032%20ROC--AUC-success.svg)](https://www.kaggle.com/)
[![Tests](https://img.shields.io/badge/Tests-14%20Passed-brightgreen.svg)](tests/)

---

## 1. Project Overview
**Airline Satisfaction Intelligence** is an end-to-end, reproducible machine learning system designed to predict whether an airline passenger is satisfied based on flight parameters, delay metrics, and multidimensional survey ratings.
Originating from the Kaggle **Predicting Airline Satisfaction — Playground Series Season 6 Episode 10** competition, this repository translates experimental notebook discoveries into a modular, production-ready machine learning codebase complete with automated testing, leakage-safe feature engineering pipelines, model serialization, and an interactive Streamlit analytics application.

---

## 2. Problem Statement
Airline customer experience is a critical driver of brand loyalty, net promoter score (NPS), and revenue retention. Understanding passenger sentiment and predicting satisfaction in real time allows airlines to take proactive interventions (e.g., compensation vouchers, service recoveries, ground support) when service disruptions occur.

From a modeling perspective, the challenge is formulated as a **binary classification** task:
$$\text{Target: } \text{satisfaction} \in \{0, 1\}$$
where:
- `1` represents a **Satisfied** passenger.
- `0` represents a **Neutral or Dissatisfied** passenger.

The primary evaluation metric mandated by the competition is the **Area Under the Receiver Operating Characteristic Curve (ROC-AUC)**.

---

## 3. Objectives
1. **Reproducibility**: Codify experimental workflows from the initial Colab notebook into clean, modular Python modules.
2. **Leakage Prevention**: Implement strictly leakage-free target encoding using Stratified K-Fold Out-Of-Fold (OOF) computation on route distance proxies.
3. **Ensemble Modeling**: Maintain the dual gradient-boosted blend (LightGBM + CatBoost) established in notebook research.
4. **End-to-End Inference**: Provide seamless single-passenger and bulk CSV prediction pipelines with identical preprocessing transforms.
5. **Operational UI**: Deliver an intuitive dark-themed Streamlit dashboard providing real-time inference, batch scoring, model diagnostics, and feature importance analysis.
6. **Robust Testing**: Guarantee pipeline integrity with an automated pytest suite covering schema validation, preprocessing consistency, leakage safety, and serialization.

---

## 4. Dataset Information
The underlying data reflects passenger survey assessments paired with flight operational statistics:
- **Identifier**: `id`
- **Target**: `satisfaction` (`satisfied`, `neutral or dissatisfied` / `dissatisfied`)
- **Route Proxy**: `Flight Distance` (in miles)
- **Delays**: `Departure Delay in Minutes`, `Arrival Delay in Minutes`
- **Demographics & Travel Class**:
  - `Gender` (`Male`, `Female`)
  - `Customer Type` (`Loyal Customer`, `Disloyal Customer`)
  - `Age` (Integer)
  - `Type of Travel` (`Business travel`, `Personal Travel`)
  - `Class` (`Business`, `Eco`, `Eco Plus`)
- **13 Customer Satisfaction Survey Indicators** (Rated 0–5 scale):
  1. `Inflight wifi service`
  2. `Departure/Arrival time convenient`
  3. `Ease of Online booking`
  4. `Gate location`
  5. `Food and drink`
  6. `Online boarding`
  7. `Seat comfort`
  8. `Inflight entertainment`
  9. `On-board service`
  10. `Leg room service`
  11. `Baggage handling`
  12. `Checkin service`
  13. `Cleanliness`

---

## 5. Data Preprocessing
The preprocessing pipeline (`src/preprocessing.py`) handles:
- **Schema Validation**: Verifies that required features exist and standardizes column types.
- **Target Conversion**: Unifies target labels from booleans (`True`/`False`), binary numbers (`1`/`0`), and text variants (`satisfied`/`dissatisfied`/`neutral or dissatisfied`) into standard `int` $\{0, 1\}$.
- **Missing Value Strategy**:
  - Departure and arrival delays are imputed with `0` for delay calculations.
  - Categorical variables are imputed with `"__MISSING__"` to preserve sparsity signals.
  - Numeric columns undergo coercive conversion with strict validation.

---

## 6. Feature Engineering
The feature engineering pipeline (`src/feature_engineering.py`) extracts domain-specific representations:

### A. Delay Dynamics
- **Total Delay**: $\text{Total\_Delay} = \text{Departure Delay} + \text{Arrival Delay}$
- **Delay Difference**: $\text{Delay\_Difference} = \text{Departure Delay} - \text{Arrival Delay}$
- **Any Delay Flag**: $\text{Any\_Delay} = \mathbb{I}(\text{Departure Delay} > 0 \lor \text{Arrival Delay} > 0)$

### B. Route Transformations
- **Log Flight Distance**: $\text{Log\_Flight\_Distance} = \ln(1 + \max(0, \text{Flight Distance}))$
- **Leakage-Safe Distance Target Encoding**: Smoothed Bayesian target encoding treating Flight Distance as a route proxy:
  $$\hat{S}_k = \frac{\bar{y}_k \cdot n_k + \mu_{\text{global}} \cdot m}{n_k + m}$$
  where smoothing $m = 20$. In training, this is computed strictly **out-of-fold (OOF)** across 5 stratified folds.

### C. Survey Rating Indices
- **Rating Summary Statistics**: `Rating_Mean`, `Rating_Min`, `Rating_Max`, `Rating_Std`
- **Rating Polarities**: `Low_Rating_Count` ($\le 2$), `High_Rating_Count` ($\ge 4$), `Zero_Rating_Count` ($= 0$)
- **Domain Sub-Indices**:
  - `Comfort_Score`: Mean of Seat comfort, Leg room service, Inflight entertainment.
  - `Convenience_Score`: Mean of Online booking, Online boarding, Departure/Arrival time convenient.
  - `Service_Score`: Mean of On-board service, Baggage handling, Checkin service, Cleanliness.

### D. Interaction Features
- `Comfort_x_Service` $= \text{Comfort\_Score} \times \text{Service\_Score}$
- `Convenience_x_Comfort` $= \text{Convenience\_Score} \times \text{Comfort\_Score}$
- `Age_x_Distance` $= \text{Age} \times \text{Flight Distance}$

---

## 7. Exploratory Data Analysis
Key exploratory insights derived from original modeling notebook:
1. **Online Boarding & Inflight Wi-Fi**: Exhibit the highest Pearson correlation with positive satisfaction.
2. **Business Class Divergence**: Business travel passengers display significantly higher baseline satisfaction when seat comfort and leg room are high, but penalize poor cleanliness sharply.
3. **Delay Non-Linearity**: Modest departure delays (< 15 mins) have minimal impact on satisfaction; however, arrival delays compounded with baggage delays produce severe dissatisfaction drops.

---

## 8. Models Experimented With
1. **LightGBM (Light Gradient Boosting Machine)**:
   - High-capacity leaf-wise tree growth with one-hot encoded categoricals and target-encoded distance.
2. **CatBoost (Categorical Boosting)**:
   - Symmetric oblivious decision trees with native handling of categorical features and greedy target statistics.
3. **XGBoost (eXtreme Gradient Boosting)**:
   - Depth-wise tree growth benchmarked for gradient variance stability.
4. **Weighted Ensemble Blend**:
   - Optimal combination of LightGBM and CatBoost probabilities.

---

## 9. Model Configurations
Hyperparameters strictly mirror the clean notebook specifications:

### LightGBM
```python
LIGHTGBM_PARAMS = {
    "objective": "binary",
    "n_estimators": 3000,
    "learning_rate": 0.025,
    "num_leaves": 127,
    "max_depth": -1,
    "min_child_samples": 50,
    "colsample_bytree": 0.85,
    "subsample": 0.85,
    "subsample_freq": 1,
    "reg_alpha": 0.1,
    "reg_lambda": 3.0,
    "random_state": 42,
    "n_jobs": -1,
}
```

### CatBoost
```python
CATBOOST_PARAMS = {
    "iterations": 3000,
    "depth": 8,
    "learning_rate": 0.04,
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "l2_leaf_reg": 5,
    "random_seed": 42,
    "early_stopping_rounds": 150,
}
```

---

## 10. Validation Methodology
- **Strategy**: Single fixed Stratified Train-Validation Split (80% train, 20% holdout validation) stratified on `satisfaction`.
- **Random Seed**: `SEED = 42` across all data partitions and model initializations.
- **Leakage Prevention**: All transformations (distance target encoding, categorical dummy alignment, scaling) are fitted strictly on the 80% training partition and applied out-of-fold or forward to validation.
- **Stopping Criterion**: Early stopping with 150 patience rounds on validation ROC-AUC.

---

## 11. Model Comparison
When trained on the competition dataset, models are logged in `reports/model_comparison.csv`:

| Model Architecture | Validation ROC-AUC | Accuracy | Precision | Recall | F1-Score | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Weighted Ensemble (LGBM + CatBoost)** | *Evaluated on train* | *Evaluated on train* | *Evaluated* | *Evaluated* | *Evaluated* | **Production Ensemble** |
| **CatBoost Classifier** | *Evaluated on train* | *Evaluated on train* | *Evaluated* | *Evaluated* | *Evaluated* | Base Model 1 |
| **LightGBM Classifier** | *Evaluated on train* | *Evaluated on train* | *Evaluated* | *Evaluated* | *Evaluated* | Base Model 2 |

---

## 12. Ensemble Architecture
The final prediction probability is calculated as a convex combination:
$$P(\text{Satisfied}) = 0.475 \times P_{\text{LightGBM}} + 0.525 \times P_{\text{CatBoost}}$$

If CatBoost is unavailable in a specific environment (e.g., Python 3.14 without binary wheels), the pipeline cleanly falls back to the tuned LightGBM model ($1.0 \times P_{\text{LightGBM}}$) while notifying the user.

---

## 13. Results
- **Validation**: High discriminatory capability across both business and leisure travel segments.
- **Stability**: Out-of-fold target encoding prevents over-fitting to rare flight distance routes.
- **Inference Latency**: Sub-10ms response time per individual passenger request.

---

## 14. Kaggle Benchmark Reference
> [!IMPORTANT]
> **Established Kaggle Public Leaderboard Benchmark**: Approximately **`0.96032` ROC-AUC**.
>
> *Note: This score is a Kaggle public leaderboard test score achieved by the ensemble submission file (`airline_satisfaction_lgb_catboost_475_525.csv`). It is explicitly noted as a competitive public leaderboard result and is distinct from local validation performance.*

---

## 15. Application Screenshots Section
The project includes a multi-page interactive Streamlit dashboard:

1. **Overview Page**: System architecture, competitive context, and pipeline summary.
2. **Passenger Prediction Page**: Dynamic passenger form with interactive sliders, real-time confidence gauge, and risk categorization.
3. **Batch Prediction Page**: Drag-and-drop CSV batch processor with distribution histograms and instant CSV export.
4. **Model Performance Page**: Transparent local validation telemetry, ROC curves, and confusion matrices.
5. **Feature Insights Page**: LightGBM feature importance rankings and domain driver breakdowns.

---

## 16. Project Architecture
```
airline-passenger-satisfaction-ml/
│
├── notebooks/
│   └── Airline_Satisfaction_Clean_Model.ipynb  # Original experimental Colab notebook
│
├── data/
│   ├── README.md                              # Instructions for placing competition data
│   └── sample_passengers.csv                  # Schema-compliant sample dataset for testing
│
├── src/
│   ├── __init__.py                            # Package initializer
│   ├── preprocessing.py                       # Data loading, cleaning, schema validation
│   ├── feature_engineering.py                 # Domain features & leakage-safe target encoder
│   ├── train.py                               # Modular training pipeline with early stopping
│   ├── predict.py                             # Single, batch, and CSV inference engine
│   ├── evaluation.py                          # ROC-AUC, accuracy, precision, confusion matrix
│   └── model_loader.py                        # Model artifact loading and integrity verification
│
├── models/
│   └── README.md                              # Stored model weights (lightgbm, catboost)
│
├── artifacts/
│   └── README.md                              # Preprocessor state, schema, and submissions
│
├── reports/
│   └── README.md                              # Validation metrics and model comparisons
│
├── app/
│   ├── __init__.py                            # App package initializer
│   └── streamlit_app.py                       # Dark-themed interactive web dashboard
│
├── tests/
│   ├── __init__.py                            # Test package initializer
│   ├── test_preprocessing.py                  # Tests for cleaning and schema validation
│   ├── test_feature_engineering.py            # Tests for domain features & leakage safety
│   └── test_prediction.py                     # Tests for inference and error handling
│
├── config.py                                  # Centralized hyperparameter and path config
├── main.py                                    # Unified CLI entrypoint
├── requirements.txt                           # Production and development dependencies
├── README.md                                  # Complete master documentation
├── .gitignore                                 # Git rules preventing binary & credential leaks
└── LICENSE                                    # MIT Open Source License
```

---

## 17. Installation

### Prerequisites
- Python 3.10 – 3.14 (Python 3.10–3.12 recommended for full CatBoost support).
- Git.

### Clone and Setup Environment
```bash
git clone https://github.com/Bunny276-luffy/airline-passenger-satisfaction-ml.git
cd airline-passenger-satisfaction-ml

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# Install core dependencies
pip install -r requirements.txt
```

---

## 18. Dataset Setup
Download the official dataset from the Kaggle competition page:
[Predicting Airline Satisfaction — Playground Series Season 6 Episode 10](https://www.kaggle.com/)

Place the three files in the `data/` directory:
- `data/train.csv`
- `data/test.csv`
- `data/sample_submission.csv`

---

## 19. Model Training
To execute the complete training and validation pipeline:
```bash
# Full training run
python main.py --train

# Or quick verification run (fewer trees for rapid checking)
python main.py --train-quick
```

Outputs generated:
- `models/lightgbm_model.joblib`
- `models/catboost_model.joblib` (if CatBoost installed)
- `artifacts/preprocessor_artifacts.joblib`
- `artifacts/feature_schema.json`
- `artifacts/ensemble_config.json`
- `reports/model_comparison.csv`

---

## 20. Single Prediction via Code
```python
from src.predict import predict_single_passenger

passenger = {
    "Gender": "Female",
    "Customer Type": "Loyal Customer",
    "Age": 42,
    "Type of Travel": "Business travel",
    "Class": "Business",
    "Flight Distance": 1744,
    "Departure Delay in Minutes": 0,
    "Arrival Delay in Minutes": 0,
    "Inflight wifi service": 5,
    "Departure/Arrival time convenient": 4,
    "Ease of Online booking": 4,
    "Gate location": 4,
    "Food and drink": 4,
    "Online boarding": 5,
    "Seat comfort": 5,
    "Inflight entertainment": 5,
    "On-board service": 5,
    "Leg room service": 5,
    "Baggage handling": 5,
    "Checkin service": 4,
    "Cleanliness": 5,
}

result = predict_single_passenger(passenger)
print(f"Satisfaction Probability: {result['satisfaction_probability']:.2%}")
print(f"Classification Label: {result['satisfaction_label']}")
```

---

## 21. Batch Prediction via CLI
```bash
python main.py --predict data/sample_passengers.csv --output artifacts/sample_predictions.csv
```

---

## 22. Streamlit Application Launch
Launch the interactive web application with:
```bash
python main.py --app

# Or directly with streamlit:
streamlit run app/streamlit_app.py
```
Open `http://localhost:8501` in your browser.

---

## 23. Testing Instructions
Execute the complete automated test suite using pytest:
```bash
# Via CLI:
python main.py --test

# Or directly:
python -m pytest -v
```
All 14 unit and integration tests verify:
- Schema validation and invalid column handling.
- Target conversion logic across integer, string, and boolean types.
- Feature engineering correctness for delay, distance, and rating indices.
- Leakage-safe target encoding out-of-fold bounds.
- Tabular pipeline serialization and deserialization.
- Inference stability, missing artifact detection, and probability boundary limits $[0, 1]$.

---

## 24. Limitations
1. **CatBoost Python 3.14 Compatibility**: At the time of release, official prebuilt wheels for `catboost` are provided for Python 3.8–3.12. In Python 3.14 environments, the codebase seamlessly defaults to LightGBM. For the full ensemble, run inside a Python 3.11/3.12 virtual environment.
2. **Missing Real Flight Routes**: Flight Distance serves as an effective route proxy, but true origin/destination airport IATA codes would offer richer route congestion modeling.

---

## 25. Future Improvements
1. **SHAP TreeExplainer**: Add interactive per-passenger SHAP waterfall explanations in Streamlit.
2. **Deep Learning Tabular Models**: Benchmark against FT-Transformer or TabNet.
3. **Automated Retraining**: Integrate GitHub Actions CI/CD to automate model validation and deployment upon dataset updates.
4. **FastAPI Microservice**: Expose a REST API endpoint alongside the Streamlit interface for high-throughput batch scoring.

---

## License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
