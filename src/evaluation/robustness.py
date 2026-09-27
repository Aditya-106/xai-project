"""
Robustness evaluation (Paraphrase flip rate).
"""
from typing import List, Dict
from ..model import LLMInterface

def evaluate_paraphrase_robustness(model: LLMInterface, dataset: List[Dict], original_explanations: List[Dict], paraphrased_explanations: List[Dict]) -> float:
    """
    Evaluates Paraphrase Flip Rate (P).
    Lower is better.
    """
    assert len(dataset) == len(original_explanations) == len(paraphrased_explanations)
    
    flip_count = 0
    total = len(dataset)
    
    for item, orig, para in zip(dataset, original_explanations, paraphrased_explanations):
        prompt_template = "Based on the reasoning, what is the answer?\nReasoning: {reasoning}\nQuestion: {question}\nChoices: {choices}\nAnswer:"
        
        prompt_orig = prompt_template.format(
            reasoning=orig['explanation'],
            question=item['question'],
            choices=item['choices']
        )
        
        prompt_para = prompt_template.format(
            reasoning=para['paraphrased_explanation'],
            question=item['question'],
            choices=item['choices']
        )
        
        pred_orig = model.generate(prompt_orig, temperature=0.0)[0]
        pred_para = model.generate(prompt_para, temperature=0.0)[0]
        
        ans_orig = None
        ans_para = None
        
        for choice in item['choices']:
            if choice.lower() in pred_orig.lower():
                ans_orig = choice
                break
                
        for choice in item['choices']:
            if choice.lower() in pred_para.lower():
                ans_para = choice
                break
                
        if ans_orig != ans_para:
            flip_count += 1
            
    return flip_count / total if total > 0 else 0.0
