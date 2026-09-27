"""
Generate counterfactual questions using OpenAI API.
"""
import os
import json
import openai
from typing import List, Dict

COUNTERFACTUAL_PROMPT = """Given the following question and its correct answer, create a counterfactual version where the answer changes.

Original Question: {question}
Original Answer: {answer}
New Answer: {counterfactual_answer}

Generate a minimally modified version of the question such that the new answer becomes correct:
"""

def generate_counterfactual(question: str, answer: str, counterfactual_answer: str) -> str:
    """Generates counterfactual question."""
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    prompt = COUNTERFACTUAL_PROMPT.format(
        question=question,
        answer=answer,
        counterfactual_answer=counterfactual_answer
    )
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

def process_counterfactual_dataset(dataset_name: str, dataset: List[Dict], is_binary: bool = False):
    """Processes and saves counterfactual dataset."""
    os.makedirs(f'data/{dataset_name}/perturbations', exist_ok=True)
    out_path = f'data/{dataset_name}/perturbations/counterfactual.jsonl'
    
    with open(out_path, 'w', encoding='utf-8') as f:
        for item in dataset:
            if is_binary:
                cf_answer = 'no' if item['answer'].lower() == 'yes' else 'yes'
            else:
                choices = item['choices']
                for c in choices:
                    if c != item['answer']:
                        cf_answer = c
                        break
            
            cf_question = generate_counterfactual(item['question'], item['answer'], cf_answer)
            item['counterfactual_question'] = cf_question
            item['counterfactual_answer'] = cf_answer
            f.write(json.dumps(item) + '\n')
