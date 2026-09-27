"""
Self-Consistent Chain-of-Thought.
"""
from collections import Counter
from typing import List, Tuple
from ..model import LLMInterface
from .cot import STRATEGY_QA_TEMPLATE, OPENBOOK_QA_TEMPLATE, parse_cot_output

def generate_sc_cot(model: LLMInterface, question: str, choices: List[str], dataset_type: str, n_paths: int = 5) -> Tuple[str, str]:
    """Generates Self-Consistent CoT explanation and answer."""
    if dataset_type == 'strategyqa':
        prompt = STRATEGY_QA_TEMPLATE.format(
            question=question,
            choices=str(choices)
        )
    else:
        if len(choices) >= 4:
            prompt = OPENBOOK_QA_TEMPLATE.format(
                question=question,
                choice_a=choices[0],
                choice_b=choices[1],
                choice_c=choices[2],
                choice_d=choices[3]
            )
        else:
            prompt = STRATEGY_QA_TEMPLATE.format(
                question=question,
                choices=str(choices)
            )
            
    # Sample N paths
    outputs = model.generate(prompt, temperature=0.7, num_return_sequences=n_paths)
    
    paths = []
    answers = []
    for out in outputs:
        reasoning, ans = parse_cot_output(out, choices)
        paths.append(reasoning)
        answers.append(ans)
        
    # Determine majority answer
    ans_counts = Counter(answers)
    majority_answer = ans_counts.most_common(1)[0][0]
    
    selected_reasoning = ""
    for path, ans in zip(paths, answers):
        if ans == majority_answer:
            selected_reasoning = path
            break
            
    return selected_reasoning, majority_answer
