"""
LLM model loading and inference.

Supports:
- Local Llama-2 GPTQ models (requires NVIDIA GPU)
- Local lightweight models (runs on CPU/MPS for Mac)
- HuggingFace Inference API
"""
import argparse
import requests
import os
from typing import List, Optional


class LLMInterface:
    """Base interface for language model generation."""
    def generate(self, prompt: str, max_new_tokens: int = 512,
                 temperature: float = 0.7, num_return_sequences: int = 1) -> List[str]:
        raise NotImplementedError


class LocalLlamaModel(LLMInterface):
    """Loads Llama-2 GPTQ model locally. Requires NVIDIA GPU."""
    def __init__(self, size: str = '7'):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        model_name = f'TheBloke/Llama-2-{size}B-chat-GPTQ'
        print(f"  Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto",
            trust_remote_code=True, revision="main"
        )

    def generate(self, prompt: str, max_new_tokens: int = 512,
                 temperature: float = 0.7, num_return_sequences: int = 1) -> List[str]:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        gen_kwargs = dict(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=temperature > 0.0, num_return_sequences=num_return_sequences
        )
        if temperature > 0.0:
            gen_kwargs["temperature"] = temperature
        outputs = self.model.generate(**gen_kwargs)
        results = []
        for out in outputs:
            text = self.tokenizer.decode(out, skip_special_tokens=True)
            # Remove the prompt portion
            results.append(text[len(self.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):])
        return results


class LocalLightModel(LLMInterface):
    """
    Loads a lightweight causal LM that can run on CPU or Apple MPS.
    Used for reproduction on machines without NVIDIA GPUs.
    Default: TinyLlama-1.1B-Chat (fits in ~2GB RAM).
    """
    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        print(f"  Loading lightweight model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Use MPS on Apple Silicon if available, else CPU
        if torch.backends.mps.is_available():
            self.device = "mps"
            print("  Using Apple MPS backend")
        else:
            self.device = "cpu"
            print("  Using CPU backend")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32
        ).to(self.device)
        self.model.eval()
        print(f"  Model loaded on {self.device}")

    def generate(self, prompt: str, max_new_tokens: int = 256,
                 temperature: float = 0.7, num_return_sequences: int = 1) -> List[str]:
        import torch

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                                max_length=1024).to(self.device)
        input_len = inputs["input_ids"].shape[1]

        results = []
        for _ in range(num_return_sequences):
            gen_kwargs = dict(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=temperature > 0.0, pad_token_id=self.tokenizer.eos_token_id
            )
            if temperature > 0.0:
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"] = 0.9

            with torch.no_grad():
                output = self.model.generate(**gen_kwargs)

            generated = self.tokenizer.decode(output[0][input_len:], skip_special_tokens=True)
            results.append(generated.strip())

        return results

    def generate_chat(self, question: str, max_new_tokens: int = 150,
                      temperature: float = 0.7) -> tuple:
        import torch
        messages = [
            {"role": "system", "content": "You are a precise reasoning assistant. Think step by step and end your response with 'Answer: Yes' or 'Answer: No'."},
            {"role": "user", "content": f"Question: {question}"}
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0.0),
                temperature=max(temperature, 0.01) if temperature > 0.0 else 1.0,
                top_p=0.9 if temperature > 0.0 else 1.0,
                pad_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True
            )

        text = self.tokenizer.decode(outputs.sequences[0][input_len:], skip_special_tokens=True).strip()
        try:
            transition_scores = self.model.compute_transition_scores(
                outputs.sequences, outputs.scores, normalize_logits=True
            )
            avg_log_prob = float(transition_scores[0].mean().cpu().numpy()) if len(transition_scores[0]) > 0 else -10.0
        except Exception:
            avg_log_prob = -10.0

        return text, avg_log_prob

    def predict_given_reasoning(self, reasoning: str, question: str) -> str:
        import torch, re
        messages = [
            {"role": "system", "content": "Based ONLY on the provided reasoning, answer the question with 'Answer: Yes' or 'Answer: No'."},
            {"role": "user", "content": f"Reasoning: {reasoning}\nQuestion: {question}"}
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=25, do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        text = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip().lower()
        
        match = re.search(r'answer:\s*(yes|no)', text)
        if match:
            return match.group(1)
        if text.startswith('yes'):
            return 'yes'
        if text.startswith('no'):
            return 'no'
        if re.search(r'\byes\b', text):
            return 'yes'
        if re.search(r'\bno\b', text):
            return 'no'
        return 'yes'


class APIModel(LLMInterface):
    """Uses HuggingFace Inference API."""
    def __init__(self, model_name: str):
        self.api_key = os.environ.get("HF_API_KEY") or os.environ.get("HF_TOKEN")
        if not self.api_key:
            raise ValueError("HF_API_KEY or HF_TOKEN environment variable required for API mode.")
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        self.headers = {"Authorization": f"Bearer {self.api_key}"}

    def generate(self, prompt: str, max_new_tokens: int = 512,
                 temperature: float = 0.7, num_return_sequences: int = 1) -> List[str]:
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_new_tokens,
                "temperature": max(temperature, 0.01),
                "num_return_sequences": num_return_sequences,
                "return_full_text": False
            }
        }
        response = requests.post(self.api_url, headers=self.headers, json=payload)
        response.raise_for_status()
        data = response.json()

        if isinstance(data, list):
            return [item.get('generated_text', '') for item in data]
        return [data.get('generated_text', '')]


def get_model(use_api: bool = False, size: str = '7',
              api_model: Optional[str] = None,
              lightweight: bool = False,
              lightweight_model: Optional[str] = None) -> LLMInterface:
    """
    Get an LLM instance.

    Args:
        use_api: Use HuggingFace Inference API
        size: Llama-2 model size (7, 13, 70)
        api_model: Custom API model name
        lightweight: Use lightweight local model (for CPU/Mac)
        lightweight_model: Custom lightweight model name
    """
    if use_api:
        if api_model is None:
            api_model = f'meta-llama/Llama-2-{size}b-chat-hf'
        return APIModel(api_model)
    elif lightweight:
        model_name = lightweight_model or "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        return LocalLightModel(model_name)
    else:
        return LocalLlamaModel(size)


def load_model(model_size: str = '70B', use_api: bool = False,
               api_model: Optional[str] = None,
               lightweight: bool = False) -> LLMInterface:
    """Convenience wrapper matching run_generation.py's interface."""
    size = model_size.replace('B', '')
    return get_model(use_api=use_api, size=size, api_model=api_model, lightweight=lightweight)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-api', action='store_true', help='Use HF API')
    parser.add_argument('--api-model', type=str, default=None)
    parser.add_argument('--size', type=str, default='7', choices=['7', '13', '70'])
    parser.add_argument('--lightweight', action='store_true',
                        help='Use lightweight model for CPU/Mac')
    args = parser.parse_args()

    model = get_model(args.use_api, args.size, args.api_model, args.lightweight)
    result = model.generate("What is 2+2? Let's think step by step.", max_new_tokens=100)
    print("Generated:", result[0])


if __name__ == '__main__':
    main()
