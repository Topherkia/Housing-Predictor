# 🏠 Melbourne Housing Price Predictor

A machine learning project for predicting Melbourne property prices using the **Melbourne Housing dataset**. The project includes data cleaning and preprocessing, K-Means clustering, multiple regression models, model evaluation, and an interactive Streamlit interface.

## ✨ Features

* CSV data preprocessing and validation
* Automatic handling of missing and invalid values
* Optional removal of unusually high property prices
* Numerical missing-value imputation using the median
* Categorical missing-value imputation using the most frequent value
* Date parsing and feature engineering
* Duplicate-row removal
* K-Means clustering
* Multiple regression algorithms:

  * Linear Regression
  * Decision Tree Regressor
  * Random Forest Regressor
  * Gradient Boosting Regressor
  * XGBoost Regressor
* One-hot encoding of categorical features
* Model evaluation using:

  * MAE
  * RMSE
  * R²
* Command-line pipeline
* Interactive Streamlit web application

---

## 📁 Project Structure

```text
Housing-Predictor/
│
├── app.py                         # Command-line ML pipeline
├── streamlit_app.py               # Streamlit web interface
├── data_processor.py              # CSV cleaning and preprocessing
│
├── data/
│   ├── raw/
│   │   └── melb_data.csv          # Original dataset
│   │
│   └── processed/
│       ├── melb_data_processed.csv
│       └── melb_data_processed_rm.csv
│
└── src/
    ├── clustering_stage.py        # K-Means clustering
    ├── model_linear_regression.py
    ├── model_decision_tree.py
    ├── model_random_forest.py
    ├── model_gradient_boosting.py
    └── model_xgboost.py
```

---

## 🧰 Technologies

The project is written in **Python** and uses:

* [Python](https://www.python.org/)
* [Pandas](https://pandas.pydata.org/)
* [NumPy](https://numpy.org/)
* [Scikit-learn](https://scikit-learn.org/)
* [XGBoost](https://xgboost.readthedocs.io/)
* [Streamlit](https://streamlit.io/)

---

## 📊 Dataset

The project uses the **[Melbourne Housing Snapshot](https://www.kaggle.com/datasets/dansbecker/melbourne-housing-snapshot)** dataset provided by the user **[DanB](https://www.kaggle.com/dansbecker)**.

The expected input CSV contains the following columns:

```text
Suburb
Address
Rooms
Type
Price
Method
SellerG
Date
Distance
Postcode
Bedroom2
Bathroom
Car
Landsize
BuildingArea
YearBuilt
CouncilArea
Lattitude
Longtitude
Regionname
Propertycount
```

The target variable is:

```text
Price
```

The project currently includes the dataset under:

```text
data/raw/melb_data.csv
```

---


# 🤖 Machine Learning Pipeline

The main pipeline is implemented in:

```text
app.py
```

The pipeline performs the following steps:

```text
CSV Dataset
    │
    ▼
Data Loading
    │
    ▼
Feature Filtering
    │
    ▼
K-Means Clustering
    │
    ▼
Feature Preparation
    │
    ▼
Missing-Value Imputation
    │
    ▼
Categorical Encoding
    │
    ▼
Train/Test Split
    │
    ▼
Regression Model
    │
    ▼
Predictions
    │
    ▼
Model Evaluation
```

---

# 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/Topherkia/Housing-Predictor.git
cd Housing-Predictor
```

Install the required packages:

```bash
pip install pandas numpy scikit-learn xgboost streamlit
```

---

# ▶️ Running the Command-Line Pipeline

From the project root:

```bash
python app.py
```

---

# 🧠 Project Architecture

The project separates the machine learning components into individual modules.

```text
                     ┌──────────────────────┐
                     │    CSV Dataset       │
                     └──────────┬───────────┘
                                │
                                ▼
                     ┌──────────────────────┐
                     │   Data Processor     │
                     │ data_processor.py    │
                     └──────────┬───────────┘
                                │
                                ▼
                     ┌──────────────────────┐
                     │   Feature Cleaning   │
                     │ & Preprocessing      │
                     └──────────┬───────────┘
                                │
                                ▼
                     ┌──────────────────────┐
                     │    K-Means           │
                     │    Clustering        │
                     └──────────┬───────────┘
                                │
                                ▼
                     ┌──────────────────────┐
                     │ Regression Pipeline  │
                     └──────────┬───────────┘
                                │
              ┌─────────────────┼─────────────────┐
              │        │        │        │         │
              ▼        ▼        ▼        ▼         ▼
           Linear   Decision  Random   Gradient  XGBoost
          Regression  Tree     Forest   Boosting
              │        │        │        │         │
              └────────┴────────┴────────┴─────────┘
                                │
                                ▼
                     ┌──────────────────────┐
                     │ Model Evaluation     │
                     │ MAE / RMSE / R²      │
                     └──────────────────────┘
```

---


# 📌 Notes

* The project is intended as a machine learning/data-processing project for experimentation with housing-price prediction.
* The model performance depends heavily on the dataset and preprocessing configuration.
* The `Price` column is the prediction target.
* The data processor does **not** automatically alter valid price values.

---


## 📄 License

Educational Use Only License

Copyright (c) 2026 Kiavash Montazeri

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to use, copy, modify, and merge the Software solely for non-commercial educational and academic purposes.

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

RESTRICTIONS:

The Software may not be used for any commercial, production, or revenue-generating purposes.

Distribution, re-licensing, or selling of the Software outside of educational institutions or academic contexts is strictly prohibited.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---

## 👤 Authors

**Topherkia**

GitHub:
https://github.com/Topherkia

Repository:
https://github.com/Topherkia/Housing-Predictor
