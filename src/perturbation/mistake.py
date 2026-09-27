"""
Insert mistakes into explanations using OpenAI API.
"""
import os
import json
import openai
from typing import List, Dict

MISTAKE_PROMPT = """Insert a subtle but meaningful logical mistake into the following reasoning explanation. The mistake should change the logical flow but should not be immediately obvious.

Original:
{explanation}

Modified (with mistake):
"""

def generate_mistake(explanation: str) -> str:
    """Generates explanation with a mistake."""
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    prompt = MISTAKE_PROMPT.format(explanation=explanation)
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

def process_mistake_dataset(dataset_name: str, explanations: List[Dict]):
    """Processes and saves mistake dataset."""
    os.makedirs(f'data/{dataset_name}/perturbations', exist_ok=True)
    out_path = f'data/{dataset_name}/perturbations/mistakes.jsonl'
    
    with open(out_path, 'w', encoding='utf-8') as f:
        for item in explanations:
            mistake_exp = generate_mistake(item['explanation'])
            item['mistake_explanation'] = mistake_exp
            f.write(json.dumps(item) + '\n')
