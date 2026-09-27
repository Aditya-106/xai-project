"""
Generate paraphrased explanations using OpenAI API.
"""
import os
import json
import openai
from typing import List, Dict

PARAPHRASE_PROMPT = """Paraphrase the following reasoning explanation while preserving its meaning. Only change the wording, not the logic or conclusions.

Original:
{explanation}

Paraphrased:
"""

def generate_paraphrase(explanation: str) -> str:
    """Generates paraphrased explanation."""
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    prompt = PARAPHRASE_PROMPT.format(explanation=explanation)
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo", # or gpt-4
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

def process_paraphrase_dataset(dataset_name: str, explanations: List[Dict]):
    """Processes and saves paraphrased dataset."""
    os.makedirs(f'data/{dataset_name}/perturbations', exist_ok=True)
    out_path = f'data/{dataset_name}/perturbations/paraphrase.jsonl'
    
    with open(out_path, 'w', encoding='utf-8') as f:
        for item in explanations:
            paraphrased = generate_paraphrase(item['explanation'])
            item['paraphrased_explanation'] = paraphrased
            f.write(json.dumps(item) + '\n')
