# 💳 Credit Card Fraud Detection

A machine learning project that detects fraudulent credit card transactions using a trained classification model. The project includes:

* 📊 Data analysis and model development in Jupyter Notebook
* 🌐 Interactive Streamlit web application
* ⚡ FastAPI prediction API
* 🤖 Fraud probability prediction for credit card transactions

---

## 📌 Project Overview

Credit card fraud detection is a highly imbalanced classification problem where fraudulent transactions represent a very small percentage of total transactions.

This project demonstrates:

* Data exploration and preprocessing
* Fraud detection model training
* Prediction through a web interface
* Prediction through a REST API

---

## 📂 Repository Structure

```text
credit-card-fraud-detection/
│
├── Credit_card_fraud__detection.ipynb   # Model development notebook
├── streamlit_app.py                     # Streamlit web application
├── app.py                               # FastAPI backend API
├── README.md                            # Project documentation
└── .gitignore
```

---

## 🛠 Technologies Used

| Category            | Tools         |
| ------------------- | ------------- |
| Language            | Python        |
| Data Analysis       | Pandas, NumPy |
| Machine Learning    | Scikit-learn  |
| Web App             | Streamlit     |
| API                 | FastAPI       |
| Model Serialization | Joblib        |

---

## 🚀 Streamlit Web Application

The Streamlit application allows users to test transactions through a simple user interface.

### Features

* Input all transaction features (V1–V28, Amount, Time)
* Load sample fraud transaction
* Load sample normal transaction
* Predict transaction status
* Display fraud probability

### Run Locally

```bash
pip install streamlit joblib numpy
streamlit run streamlit_app.py
```

---

## ⚡ FastAPI Prediction API

The project also includes a REST API for integration with external applications.

### Run API

```bash
pip install fastapi uvicorn joblib numpy pydantic

uvicorn app:app --reload
```

### API Endpoint

#### Home

```http
GET /
```

Response:

```json
{
  "message": "Fraud Detection API is running"
}
```

#### Prediction

```http
POST /predict
```

Example Request:

```json
{
  "V1": 0.1,
  "V2": 0.2,
  "V3": 0.3,
  "V4": 0.4,
  "V5": 0.5,
  "V6": 0.6,
  "V7": 0.7,
  "V8": 0.8,
  "V9": 0.9,
  "V10": 1.0,
  "V11": 1.1,
  "V12": 1.2,
  "V13": 1.3,
  "V14": 1.4,
  "V15": 1.5,
  "V16": 1.6,
  "V17": 1.7,
  "V18": 1.8,
  "V19": 1.9,
  "V20": 2.0,
  "V21": 2.1,
  "V22": 2.2,
  "V23": 2.3,
  "V24": 2.4,
  "V25": 2.5,
  "V26": 2.6,
  "V27": 2.7,
  "V28": 2.8,
  "Amount": 100,
  "Time": 50000
}
```

Example Response:

```json
{
  "prediction": 0,
  "label": "NOT FRAUD",
  "fraud_probability": 0.0234
}
```

---

## 📊 Notebook

The notebook contains:

* Dataset loading
* Exploratory Data Analysis (EDA)
* Fraud vs Normal transaction analysis
* Data preprocessing
* Model training
* Model evaluation

---

## ⚠ Important Note

This repository currently contains only the application code and notebook.

The following files are not included and must be added separately before running predictions:

```text
fraud_model.pkl
scaler.pkl
```

If you trained the model locally, place these files in the project root directory.

---

## ▶ Installation

```bash
git clone <repository-url>

cd credit-card-fraud-detection

pip install -r requirements.txt
```

Or install manually:

```bash
pip install pandas numpy scikit-learn streamlit fastapi uvicorn joblib pydantic
```

---

## 👨‍💻 Author

Developed as a machine learning project for credit card fraud detection using Python, Scikit-learn, Streamlit, and FastAPI.
