import pandas as pd

from src.ai_housing_assistant import HousingAIAssistant
from src.qwen import HOUSING_FEATURES


class FakeQwen:

    def extract_features(self, query):

        return {
            "Rooms": 4,
            "Distance": None,
            "Bedroom2": 4,
            "Bathroom": 2,
            "Car": None,
            "Landsize": 600,
            "BuildingArea": None,
        }

    def explain_prediction(
        self,
        query,
        features,
        prediction,
        retrieved_documents,
        missing_features,
    ):

        return "Test explanation."


class FakeModel:

    def predict(self, X):

        assert list(X.columns) == HOUSING_FEATURES

        assert len(X) == 1

        return [750000.0]


class FakeRAG:

    def retrieve(self, query, k=4):

        return [
            {
                "source": "test.md",
                "text": "Test knowledge.",
                "score": 0.9,
            }
        ]


def test_assistant():

    medians = {
        feature: 1.0
        for feature in HOUSING_FEATURES
    }

    assistant = HousingAIAssistant(
        xgboost_model=FakeModel(),
        qwen_model=FakeQwen(),
        rag=FakeRAG(),
        feature_medians=medians,
    )

    result = assistant.answer(
        "Estimate a 4 bedroom property."
    )

    assert result["prediction"] == 750000.0

    assert set(
        result["features"].keys()
    ) == set(HOUSING_FEATURES)

    assert "Distance" in result["missing_filled"]

    assert "Car" in result["missing_filled"]

    assert "BuildingArea" in result["missing_filled"]

    assert len(result["retrieved"]) == 1

    assert result["explanation"] == "Test explanation."