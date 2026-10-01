# 🏠 Housing Predictor

An end-to-end Melbourne housing price prediction project combining classical machine learning, clustering, explainability, retrieval-augmented generation (RAG), a Qwen-powered AI assistant, and an interactive Streamlit application.

## 📋 Overview

This project predicts residential property prices from the Melbourne housing dataset and provides several ways to inspect and explain the results.

The application combines:

- Multiple supervised regression models
- K-Means clustering
- XGBoost hyperparameter optimization
- SHAP-based model explainability
- A Qwen 2.5 instruction model for natural-language housing queries
- Retrieval-Augmented Generation (RAG) for contextual explanations
- Interactive Folium maps
- Streamlit dashboards and pipeline demonstrations

The AI assistant follows this general flow:

```text
Natural-language question
        ↓
Qwen feature extraction
        ↓
Structured housing features
        ↓
Input validation
        ↓
Missing values filled with training medians
        ↓
XGBoost price prediction
        ↓
RAG document retrieval
        ↓
Qwen-generated explanation
```

---

## 🌟 Features

### 1. Regression Models

The project supports several regression algorithms:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor

Models are evaluated using:

- **MAE** — Mean Absolute Error
- **RMSE** — Root Mean Squared Error
- **R²** — coefficient of determination

The evaluation uses a train/test split so that test-set observations are not used to fit the predictive model.

---

### 2. Clustering

K-Means clustering is used to identify groups of properties with similar characteristics.

The clustering stage can be used to explore patterns such as:

- Property size
- Number of rooms
- Bathrooms
- Bedrooms
- Distance
- Land size
- Building area

Cluster assignments can also be displayed on the interactive map.

### Important preprocessing rule

Preprocessing parameters such as imputation values and scaling parameters should be learned from the training data and then reused for validation, test, and inference data.

This avoids test-set leakage and makes the reported model metrics reproducible.

---

### 3. XGBoost

XGBoost is used as one of the main high-performance regression models.

The project can use hyperparameter optimization to search for a stronger configuration rather than relying exclusively on default XGBoost parameters.

The final model can also be inspected using SHAP.

---

### 4. SHAP Explainability

SHAP is used to explain how individual features contribute to model predictions.

This provides two complementary views:

- **Global explanations** — which features are generally important to the model
- **Local explanations** — which features influenced a particular prediction

When interpreting SHAP plots, feature importance should not be confused with causality. A high SHAP contribution indicates model influence, not that changing the feature will necessarily cause the predicted price to change by the same amount in the real world.

---

## 🤖 AI Housing Assistant

The project includes a natural-language housing assistant powered by Qwen.

A user can ask a question such as:

```text
Estimate the price of a 3-bedroom property with 2 bathrooms,
5 rooms, 8 km distance, 500 m² land and 120 m² building area.
```

The assistant extracts structured values from the request and passes them to the prediction model.

### 🤖 Supported AI features

The assistant extracts:

| Feature | Description |
|---|---|
| `Rooms` | Number of rooms |
| `Distance` | Distance from the relevant reference point |
| `Bedroom2` | Number of bedrooms |
| `Bathroom` | Number of bathrooms |
| `Car` | Number of car spaces |
| `Landsize` | Land size |
| `BuildingArea` | Building area |

If the user does not provide one of these values, the assistant fills the missing value using a **training-data median** rather than inventing a value with the language model.

This is important because the language model should interpret the request, while the statistical model should remain responsible for the numerical prediction.

---

## 🤖 Qwen

The assistant uses:

```text
Qwen/Qwen2.5-3B-Instruct
```

Qwen is used for:

1. Natural-language feature extraction
2. Explanation generation

The extraction prompt requires structured JSON and explicitly tells the model not to guess missing values.

The implementation also handles:

- Chat-template formatting
- CPU/GPU execution
- Automatic device mapping where supported
- Deterministic generation for structured extraction
- Markdown-fenced JSON responses
- JSON extraction from model output
- Input truncation

## 🖥️ Hardware

Running a 3B-parameter language model locally can require significant memory.

GPU inference is recommended when available.

On CPU, inference may be considerably slower.

---

## 📚 Retrieval-Augmented Generation

The assistant can retrieve relevant documents from the project's knowledge base before generating an explanation.

The RAG pipeline uses:

- Sentence Transformers
- `all-MiniLM-L6-v2`
- FAISS

The conceptual flow is:

```text
User question
      ↓
Embedding
      ↓
FAISS similarity search
      ↓
Relevant documents
      ↓
Qwen
      ↓
Context-aware explanation
```

RAG is used to provide contextual information rather than allowing the language model to rely entirely on its pretrained knowledge.

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

```text
                            CSV Dataset
                                 │
                                 ▼
                            Data Loading
                                 │
                                 ▼
                         Feature Filtering
                                 │
                     ┌───────────┴───────────┐
                     ▼                       ▼
            K-Means Clustering       Feature Preparation
                     │                       │
                     ▼                       ▼
            Map HTML Generation    Missing-Value Imputation
              (MapGenerator)                 │
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
            ========================================================================
            LLM & RAG INTEGRATION PIPELINE
            [ Unstructured Docs / Reports ]               [ Conversational QA Data ]
                        │                                         │
                        ▼                                         ▼
            Chunking & Vector Embeddings                   Qwen Fine-Tuning (QLoRA)
                        │                                         │
                        ▼                                         ▼
                Vector DB (FAISS/Chroma)                    Fine-Tuned Qwen Weights
                        │                                         │
                        └────────────────────┬────────────────────┘
                                                ▼
                                RAG Contextual Chat Assistant
```

---
---

## 🌐 Streamlit Application

The main application provides an interactive interface for running the pipeline.

Typical workflow:

1. Select the dataset.
2. Select a regression model.
3. Configure the number of clusters.
4. Run the ML pipeline.
5. Inspect evaluation metrics.
6. Inspect predictions and errors.
7. Explore the interactive map.
8. Optionally enable the AI assistant.
9. Ask a natural-language housing question.
10. Inspect the extracted features, prediction, retrieved context, and generated explanation.

---

## 📲 Installation

Clone the repository:

```bash
git clone https://github.com/Topherkia/Housing-Predictor.git
cd Housing-Predictor
```

Create a virtual environment

Install dependencies:

```bash
pip install -r requirements.txt
```

If the Qwen implementation uses automatic Hugging Face device mapping, make sure `accelerate` is installed:

```bash
pip install accelerate
```

---

## 🚀 Running the Streamlit Application

From the repository root:

```bash
streamlit run streamlit_app.py
```

Then open the local Streamlit URL shown in the terminal.

---

## 📁 Project Structure

A typical project structure is:

```text
Housing-Predictor/
├── Docs
│   └── Melbourne_housing 2.csv
├── __pycache__
│   └── test_ai_assistant.cpython-314-pytest-9.0.3.pyc
├── data
│   ├── processed
│   │   ├── melb_data_processed.csv
│   │   └── melb_data_processed_rm.csv
│   └── raw
│       └── melb_data.csv
├── knowledge_base
│   ├── features.md
│   ├── melbourn_housing.md
│   ├── methodology.md
│   └── model.md
├── models
│   └── xgboost_best_params.json
├── src
│   ├── __init__.py
│   ├── ai_housing_assistant.py
│   ├── clustering_stage.py
│   ├── hyperparameter_tuning.py
│   ├── map_generator.py
│   ├── model_decision_tree.py
│   ├── model_gradient_boosting.py
│   ├── model_linear_regression.py
│   ├── model_random_forest.py
│   ├── model_xgboost.py
│   ├── qwen.py
│   ├── rag.py
│   ├── shap_explainer.py
│   ├── test_shap.py
│   ├── train_ai_model.py
│   └── tuned_xgboost.py
├── LICENSE
├── README.md
├── README2.md
├── app.py
├── requirements.txt
├── data_processor.py
├── streamlit_app.py
└── test_ai_assistant.py
```

The exact contents can vary as the project evolves.

---

## 📊 Model Evaluation

The main evaluation metrics are:

### MAE

```text
MAE = average(|actual - predicted|)
```

MAE is expressed in the same monetary units as the target variable and is easy to interpret as an average absolute prediction error.

### RMSE

```text
RMSE = sqrt(average((actual - predicted)²))
```

RMSE penalizes larger errors more strongly than MAE.

### R²

R² describes the proportion of target variance explained by the model relative to a constant-mean baseline.

An R² value should be interpreted together with the other metrics and the evaluation dataset.

---

## 📈 Pipeline Demo Notes

The pipeline demonstration should make it clear which data are used at each stage.

A recommended sequence is:

```text
Load data
   ↓
Clean data
   ↓
Split train/test
   ↓
Fit preprocessing on train only
   ↓
Transform train/test
   ↓
Fit clustering on train
   ↓
Transform test using fitted clustering
   ↓
Fit regression model on train
   ↓
Predict test
   ↓
Calculate metrics
   ↓
Visualize results
```

---

## 🧠 AI-Enhanced Architecture

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

## 🚀 Future Improvements

Potential improvements include:

- Persisting fitted preprocessing pipelines with `joblib`.
- Persisting trained XGBoost models.
- Adding automated regression tests for the full Streamlit pipeline.
- Adding explicit data/schema validation.
- Adding confidence or prediction intervals.
- Comparing multiple train/test splits or cross-validation.
- Adding time-aware validation if historical dates are available.
- Improving geographic features.
- Adding model monitoring.
- Adding automated dataset quality checks.
- Improving RAG document evaluation.
- Supporting additional local or hosted LLMs.
- Adding a dedicated model-card section for each trained model.

---

## 🛡️ Responsible Use

This project demonstrates how machine learning and generative AI can be combined for housing-price analysis.

Predictions and explanations should be treated as analytical outputs, not as guarantees about the value of a property.

Particular care should be taken when using housing models for decisions that may affect individuals or communities. Model performance should be evaluated across relevant geographic and demographic segments where appropriate, and potentially sensitive features should be reviewed before deployment.

---

## 🛠️ Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Optuna
- SHAP
- Streamlit
- Folium
- Hugging Face Transformers
- Qwen 2.5
- Sentence Transformers
- FAISS
- PyTorch

---

# 📌 Notes

* The project is intended as a machine learning/data-processing project with tuning for experimentation with housing-price prediction.
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
