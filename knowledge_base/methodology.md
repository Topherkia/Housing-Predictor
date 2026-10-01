# Methodology

The housing pipeline separates training and test data before model fitting.

K-Means clustering should be fitted using training data only.

The fitted scaler and K-Means model can then be used to transform test data.

Numerical model preprocessing uses median imputation.

Categorical variables use most-frequent-value imputation and one-hot encoding.

The test set is used only for final model evaluation.

SHAP is used to explain how features contribute to individual XGBoost predictions.

RAG is used to retrieve project and housing-domain information.

Qwen is used for natural-language feature extraction and explanation.