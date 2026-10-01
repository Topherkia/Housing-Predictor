import json
import re

import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


class QwenModel:

    def __init__(
        self,
        model_name=(
            "Qwen/Qwen2.5-3B-Instruct"
        ),
    ):

        self.model_name = model_name

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                model_name
            )
        )

        if torch.cuda.is_available():

            self.model = (
                AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                )
            )

        else:

            self.model = (
                AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                )
            )

            self.model.to("cpu")

    def generate(
        self,
        prompt,
        max_new_tokens=350,
    ):

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a Melbourne housing "
                    "machine-learning assistant. "
                    "Use only the supplied information. "
                    "Do not invent property data."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        prompt_text = (
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )

        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(
                self.model.device
            )
            for key, value in inputs.items()
        }

        with torch.no_grad():

            output = (
                self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
            )

        generated = (
            output[0][
                inputs["input_ids"].shape[1]:
            ]
        )

        return (
            self.tokenizer.decode(
                generated,
                skip_special_tokens=True,
            )
            .strip()
        )

    def extract_features(
        self,
        user_query,
    ):

        prompt = f"""
Extract housing prediction inputs from this user request.

Return ONLY valid JSON.

The JSON must contain exactly these keys:

Rooms
Distance
Bedroom2
Bathroom
Car
Landsize
BuildingArea

Use null when the user did not provide a value.

DO NOT guess missing values.

User request:

{user_query}
"""

        raw = self.generate(
            prompt,
            max_new_tokens=180,
        )

        match = re.search(
            r"\{.*\}",
            raw,
            re.DOTALL,
        )

        if not match:

            raise ValueError(
                "Qwen did not return valid JSON.\n"
                f"Raw response:\n{raw}"
            )

        data = json.loads(
            match.group(0)
        )

        keys = [
            "Rooms",
            "Distance",
            "Bedroom2",
            "Bathroom",
            "Car",
            "Landsize",
            "BuildingArea",
        ]

        return {
            key: data.get(key)
            for key in keys
        }