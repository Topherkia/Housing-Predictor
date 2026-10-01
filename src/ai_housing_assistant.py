from __future__ import annotations

from typing import Any

import pandas as pd

from src.qwen import HOUSING_FEATURES


class HousingAIAssistant:
    """Coordinate Qwen, XGBoost and the local RAG system."""

    def __init__(
        self,
        xgboost_model,
        qwen_model,
        rag,
        feature_medians: dict[str, float],
    ):
        self.xgboost_model = xgboost_model
        self.qwen_model = qwen_model
        self.rag = rag
        self.feature_medians = feature_medians

        missing_medians = [
            feature
            for feature in HOUSING_FEATURES
            if feature not in feature_medians
        ]

        if missing_medians:
            raise ValueError(
                "Missing training medians for: "
                + ", ".join(missing_medians)
            )

    def extract_features(
        self,
        query: str,
    ) -> tuple[dict[str, float], list[str]]:
        """
        Extract model features and fill omitted values using training medians.
        """

        extracted = self.qwen_model.extract_features(
            query
        )

        features: dict[str, float] = {}
        missing_features: list[str] = []

        for feature in HOUSING_FEATURES:
            value = extracted.get(feature)

            if value is None:
                features[feature] = float(
                    self.feature_medians[feature]
                )
                missing_features.append(feature)
            else:
                features[feature] = float(value)

        return features, missing_features

    def predict(
        self,
        features: dict[str, float],
    ) -> float:
        """Run the numerical XGBoost model."""

        X = pd.DataFrame(
            [
                [
                    features[feature]
                    for feature in HOUSING_FEATURES
                ]
            ],
            columns=HOUSING_FEATURES,
        )

        prediction = self.xgboost_model.predict(X)

        return float(prediction[0])

    def answer(
        self,
        query: str,
    ) -> dict[str, Any]:
        """Complete natural-language housing prediction workflow."""

        if not query or not query.strip():
            raise ValueError(
                "A housing question is required."
            )

        features, missing_features = (
            self.extract_features(query)
        )

        prediction = self.predict(
            features
        )

        retrieved = self.rag.retrieve(
            query,
            k=4,
        )

        explanation = (
            self.qwen_model.explain_prediction(
                query=query,
                features=features,
                prediction=prediction,
                retrieved_documents=retrieved,
                missing_features=missing_features,
            )
        )

        return {
            "prediction": prediction,
            "features": features,
            "missing_filled": missing_features,
            "retrieved": retrieved,
            "explanation": explanation,
        }