"""
LLM model loading and inference.
"""
import argparse
import requests
from typing import List, Optional

class LLMInterface:
    def generate(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.7, num_return_sequences: int = 1) -> List[str]:
        raise NotImplementedError

class LocalLlamaModel(LLMInterface):
    def __init__(self, size: str = '7'):
        """Loads Llama-2 model using transformers + auto-gptq."""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
        except ImportError:
            raise ImportError("Please install transformers and auto-gptq")
        
        model_name = f'TheBloke/Llama-2-{size}B-chat-GPTQ'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            device_map="auto", 
            trust_remote_code=True,
            revision="main"
        )
        
    def generate(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.7, num_return_sequences: int = 1) -> List[str]:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens, 
            temperature=temperature,
            do_sample=temperature > 0.0,
            num_return_sequences=num_return_sequences
        )
        results = []
        for out in outputs:
            text = self.tokenizer.decode(out, skip_special_tokens=True)
            results.append(text[len(prompt):])
        return results


class APIModel(LLMInterface):
    def __init__(self, model_name: str):
        """Uses HuggingFace Inference API."""
        import os
        self.api_key = os.environ.get("HF_API_KEY")
        if not self.api_key:
            raise ValueError("HF_API_KEY environment variable is required for API mode.")
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        self.headers = {"Authorization": f"Bearer {self.api_key}"}

    def generate(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.7, num_return_sequences: int = 1) -> List[str]:
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
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


def get_model(use_api: bool = False, size: str = '7', api_model: Optional[str] = None) -> LLMInterface:
    if use_api:
        if api_model is None:
            api_model = f'meta-llama/Llama-2-{size}b-chat-hf'
        return APIModel(api_model)
    else:
        return LocalLlamaModel(size)


def load_model(model_size: str = '70B', use_api: bool = False, api_model: Optional[str] = None) -> LLMInterface:
    """Convenience wrapper matching run_generation.py's interface."""
    size = model_size.replace('B', '')
    return get_model(use_api=use_api, size=size, api_model=api_model)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-api', action='store_true', help='Use HF API')
    parser.add_argument('--api-model', type=str, default=None, help='API model name')
    parser.add_argument('--size', type=str, default='7', choices=['7', '13', '70'])
    args = parser.parse_args()
    
    model = get_model(args.use_api, args.size, args.api_model)
    print(model.generate("What is 2+2? Let's think step by step."))

if __name__ == '__main__':
    main()
