import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)


class QwenModel:

    def __init__(
        self,
        model_name="Qwen/Qwen2.5-3B-Instruct"
    ):

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=(
                torch.float16
                if torch.cuda.is_available()
                else torch.float32
            ),
            device_map="auto"
        )

    def generate(self, prompt):

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a Melbourne housing "
                    "price prediction assistant."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(
            text,
            return_tensors="pt"
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=300
        )

        response = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        return response