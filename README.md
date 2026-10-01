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
* Interactive Folium map of every property in Melbourne

---

## 📁 Project Structure

```text
Housing-Predictor/
├── Docs
│   ├── Melbourne Housing Price Predictor.pptx
│   └── Melbourne_housing 2.csv
├── LICENSE
├── README.md
├── app.py
├── data
│   ├── processed
│   │   ├── melb_data_processed.csv
│   │   └── melb_data_processed_rm.csv
│   └── raw
│       └── melb_data.csv
├── data_processor.py
├── requirements.txt
├── src
│   ├── __pycache__
│   │   ├── clustering_stage.cpython-314.pyc
│   │   ├── map_generator.cpython-314.pyc
│   │   ├── model_decision_tree.cpython-314.pyc
│   │   ├── model_gradient_boosting.cpython-314.pyc
│   │   ├── model_linear_regression.cpython-314.pyc
│   │   ├── model_random_forest.cpython-314.pyc
│   │   └── model_xgboost.cpython-314.pyc
│   ├── clustering_stage.py
│   ├── map_generator.py
│   ├── model_decision_tree.py
│   ├── model_gradient_boosting.py
│   ├── model_linear_regression.py
│   ├── model_random_forest.py
│   └── model_xgboost.py
└── streamlit_app.py
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
* [Folium](https://python-visualization.github.io/folium/)

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
Map HTML Generation (MapGenerator)
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
pip install pandas numpy scikit-learn xgboost streamlit folium
```

---

# ▶️ Running the Command-Line Pipeline

From the project root:

```bash
python app.py
```

---

# 🗺️ Map Demo

`map_generator.py` loads an interactive map of Melbourne and plots every property of the
chosen dataset using its `Lattitude` / `Longtitude` columns.

* Clustered markers (fast, even with all ~13.5k properties) – click one for its details
* Marker colour shows the sale-price band, with a legend
* Toggle layers for House / Unit / Townhouse and an optional price heat-map
* Several base maps (Esri streets, light grey, satellite, OpenStreetMap) and full-screen mode

The map is one feature of the project: it is shown in the **Streamlit app**
(`streamlit run streamlit_app.py`, section "Property Map", using the dataset chosen in the
sidebar) and can also be run on its own.

In the Streamlit app the map is connected to the ML pipeline:

* **Colour by sale price, K-Means cluster or prediction error.** Cluster and error colours
  become available after pressing *Run Pipeline*; the legend lists each cluster's size and
  median price.
* **Error map** – test-set properties coloured by `(predicted - actual) / actual`
  (under-estimated / within ±10% / over-estimated), with summary metrics and the suburbs
  where the model is least accurate. Popups show the predicted price and error.
* **Sidebar map filters** – property type, region, price, rooms, distance to CBD and year built.


---

# 🧠 AI-Enhanced Architecture

The project now combines traditional machine learning with
hyperparameter optimization, explainable AI, retrieval-augmented
generation and a local Qwen language model.

```text
                         CSV Dataset
                              │
                              ▼
                       Data Processing
                              │
                              ▼
                       Train/Test Split
                         /          \
                        /            \
                       ▼              ▼
              Training Data       Test Data
                    │                 │
                    ▼                 │
             K-Means Fitting         │
                    │                 │
                    ▼                 │
              Preprocessing          │
                    │                 │
                    ▼                 │
             Optuna XGBoost           │
                    │                 │
                    ▼                 │
             Tuned XGBoost            │
                    │                 │
                    └──────┬──────────┘
                           │
                           ▼
                    Model Evaluation
                     MAE / RMSE / R²
                           │
                           ▼
                         SHAP
                           │
                           │
User ──► Qwen ─────────────┤
        │                  │
        │                  ▼
        │             XGBoost
        │                  │
        ▼                  ▼
       RAG              Prediction
        │                  │
        └────────┬─────────┘
                 ▼
               Qwen
                 │
                 ▼
        AI Housing Explanation
                 │
                 ▼
             Streamlit
```

---


# 📌 Notes

* The project is intended as a machine learning/data-processing project for experimentation with housing-price prediction.
* The model performance depends heavily on the dataset and preprocessing configuration.
* The `Price` column is the prediction target.
* The data processor does **not** automatically alter valid price values.

---


## 📄 License

* This project is licensed under the Educational Use Only License - see the [LICENSE](LICENSE) file for details.
---

## 👤 Authors

**[Kiavash Montazeri](https://github.com/Topherkia)** ,
**[Eyob Talew](https://github.com/iyyoba)** , 
**[Meeraf Mergia Diribssa](https://github.com/megerafe)**

Repository:
https://github.com/Topherkia/Housing-Predictor
