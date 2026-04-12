💳 Credit Card Fraud Detection
Machine learning project to detect fraudulent credit card transactions. Built with Logistic Regression, Random Forest, and XGBoost. Includes SMOTE for imbalanced data, threshold tuning, SHAP explainability, ROC-AUC evaluation, and model deployment using FastAPI. Dataset: Kaggle Credit Card Fraud Detection.

📌 Problem Statement
Credit card fraud is a major financial threat. In real-world datasets, fraudulent transactions make up less than 1% of all transactions — making this a classic imbalanced classification problem.
The goal is to build a model that can accurately detect fraud without simply predicting "not fraud" for everything.

📂 Dataset

Source: Kaggle — Credit Card Fraud Detection
284,807 transactions
492 frauds (0.17%)
Features: Time, Amount, and V1–V28 (PCA-transformed anonymized features)
Target: Class → 0 = Normal, 1 = Fraud


🧠 Project Workflow
Raw Data
   ↓
Exploratory Data Analysis (EDA)
   ↓
Data Cleaning (duplicates, nulls)
   ↓
Feature Scaling (StandardScaler)
   ↓
Train-Test Split (stratified)
   ↓
SMOTE (handle imbalanced data)
   ↓
Model Training (LR, RF, XGBoost)
   ↓
Threshold Tuning
   ↓
Evaluation (Precision, Recall, F1, ROC-AUC)
   ↓
SHAP Explainability
   ↓
Model Saved (joblib)

⚙️ Tech Stack
CategoryToolsLanguagePython 3DataPandas, NumPyVisualizationMatplotlibML ModelsScikit-learn, XGBoostImbalance Handlingimbalanced-learn (SMOTE)ExplainabilitySHAPModel SavingJoblibEnvironmentGoogle Colab

🤖 Models Used
ModelImbalance HandlingLogistic RegressionSMOTERandom ForestSMOTEXGBoostscale_pos_weight
All three models are compared using the same evaluation framework at multiple thresholds (0.3, 0.5, 0.7, 0.9).

📊 Evaluation Metrics
Accuracy is not used as the primary metric — a model predicting "not fraud" always would achieve 99%+ accuracy but be completely useless.
Instead the project uses:

Precision — of all predicted frauds, how many were actually fraud
Recall — of all actual frauds, how many did the model catch
F1-Score — balance between precision and recall
ROC-AUC — overall model discrimination ability
Confusion Matrix — visual breakdown of predictions


🎯 Threshold Tuning
Instead of using the default 0.5 probability threshold, the project tests multiple thresholds:

Lower threshold (0.3) → catches more frauds, more false alarms
Higher threshold (0.7) → fewer false alarms, may miss some frauds

The best threshold is selected based on the F1-score for the fraud class.

🔍 SHAP Explainability
SHAP (SHapley Additive exPlanations) is used on the XGBoost model to answer:

"Why did the model flag this transaction as fraud?"

Two SHAP plots are included:

Summary plot — which features matter most overall
Force plot — explanation for a single transaction


📁 Project Structure
credit-card-fraud-detection/
│
├── Credit_card_fraud_detection.ipynb   # Main notebook
├── fraud_model.pkl                     # Saved XGBoost model
├── scaler_amount.pkl                   # Scaler for Amount feature
├── scaler_time.pkl                     # Scaler for Time feature
└── README.md

🚀 How to Run

Clone the repository

bashgit clone https://github.com/yourusername/credit-card-fraud-detection.git

Download the dataset from Kaggle and place creditcard.csv.zip in the project folder
Open the notebook in Google Colab or Jupyter and run all cells top to bottom


📌 Key Learnings

How to handle severely imbalanced datasets using SMOTE
Why accuracy is a misleading metric for fraud detection
How to compare multiple models using a structured evaluation table
How to tune prediction thresholds for better recall
How to explain model predictions using SHAP


🙋 Author
Made with ❤️ as a end-to-end ML project covering data preprocessing, model building, evaluation, and explainability.
