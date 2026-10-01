from __future__ import annotations

import json
import re
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


HOUSING_FEATURES = [
    "Rooms",
    "Distance",
    "Bedroom2",
    "Bathroom",
    "Car",
    "Landsize",
    "BuildingArea",
]


class QwenModel:
    """Wrapper around Qwen2.5-Instruct for local generation."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        max_input_tokens: int = 2048,
    ):
        self.model_name = model_name
        self.max_input_tokens = max_input_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        # Qwen normally has a valid pad token, but explicitly setting it
        # avoids generation warnings/failures with some tokenizer versions.
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if torch.cuda.is_available():
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                trust_remote_code=True,
            )
            self.model.to("cpu")

        self.model.eval()

    @property
    def input_device(self) -> torch.device:
        """
        Return the device where model inputs should initially be placed.

        With device_map='auto', model.device may not represent every device,
        but the embedding/input device is available from the model's first
        parameter.
        """
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 350,
        temperature: float = 0.0,
    ) -> str:
        """Generate a deterministic response from Qwen."""

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a Melbourne housing machine-learning assistant. "
                    "Use only information supplied in the prompt. "
                    "Do not invent property facts, prices, locations, "
                    "market statistics, or model inputs. "
                    "A model prediction is an estimate, not a professional valuation."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )

        inputs = {
            key: value.to(self.input_device)
            for key, value in inputs.items()
        }

        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        if temperature > 0:
            generation_kwargs["temperature"] = temperature

        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                **generation_kwargs,
            )

        input_length = inputs["input_ids"].shape[1]

        generated_tokens = output[0][input_length:]

        return self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        ).strip()

    @staticmethod
    def _extract_json_object(text: str) -> dict[str, Any]:
        """
        Extract the first JSON object from an LLM response.

        Qwen sometimes adds a short explanation despite being instructed to
        return JSON only, so this parser deliberately handles that case.
        """

        # Remove markdown code fences if Qwen adds them.
        cleaned = re.sub(
            r"```(?:json)?",
            "",
            text,
            flags=re.IGNORECASE,
        )
        cleaned = cleaned.replace("```", "").strip()

        match = re.search(
            r"\{.*\}",
            cleaned,
            flags=re.DOTALL,
        )

        if not match:
            raise ValueError(
                "Qwen did not return a JSON object.\n\n"
                f"Raw response:\n{text}"
            )

        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Qwen returned malformed JSON.\n\n"
                f"Raw response:\n{text}"
            ) from exc

        if not isinstance(parsed, dict):
            raise ValueError(
                "Qwen JSON response was not an object."
            )

        return parsed

    def extract_features(
        self,
        user_query: str,
    ) -> dict[str, float | None]:
        """
        Extract only the seven numerical model features.

        Missing information is represented as None.
        The model is explicitly instructed not to guess missing values.
        """

        prompt = f"""
Extract housing prediction inputs from the user's request.

Return ONLY one valid JSON object.

The object MUST contain exactly these keys:

Rooms
Distance
Bedroom2
Bathroom
Car
Landsize
BuildingArea

Rules:

- Use a numeric value when the user explicitly provides one.
- Use null when the user does not provide the value.
- Never guess or infer missing values.
- Do not include units in the numeric values.
- Do not add extra keys.
- Do not calculate the property price.
- Distance is distance from Melbourne CBD in kilometres.
- Landsize and BuildingArea are square metres.

User request:

{user_query}
"""

        raw = self.generate(
            prompt,
            max_new_tokens=220,
        )

        data = self._extract_json_object(raw)

        extracted = {}

        for feature in HOUSING_FEATURES:
            value = data.get(feature)

            if value is None:
                extracted[feature] = None
                continue

            try:
                extracted[feature] = float(value)
            except (TypeError, ValueError):
                extracted[feature] = None

        return extracted

    def explain_prediction(
        self,
        query: str,
        features: dict[str, Any],
        prediction: float,
        retrieved_documents: list[dict[str, Any]],
        missing_features: list[str],
    ) -> str:
        """Generate a grounded explanation for a numerical prediction."""

        knowledge = "\n\n".join(
            (
                f"Source: {document.get('source', 'unknown')}\n"
                f"{document.get('text', '')}"
            )
            for document in retrieved_documents
        )

        missing_text = (
            ", ".join(missing_features)
            if missing_features
            else "None"
        )

        prompt = f"""
Explain the following Melbourne housing prediction.

IMPORTANT:
- The numerical prediction was produced by an XGBoost regression model.
- You did NOT calculate the prediction.
- Do not change the predicted price.
- Do not invent facts.
- Explain what inputs were used.
- Clearly identify inputs that were filled using training-data medians.
- Mention that this is a historical-data model estimate, not a professional valuation.
- Use the retrieved project documentation as supporting context.
- Keep the explanation concise and useful.

User question:
{query}

Model inputs:
{json.dumps(features, indent=2)}

Predicted price:
AUD ${prediction:,.0f}

Inputs filled using training medians:
{missing_text}

Retrieved project documentation:
{knowledge}
"""

        return self.generate(
            prompt,
            max_new_tokens=350,
        )