"""
Self-Entailment-Alignment Chain-of-Thought.
"""
from collections import Counter
from typing import List, Tuple
from ..model import LLMInterface
from .cot import STRATEGY_QA_TEMPLATE, OPENBOOK_QA_TEMPLATE, parse_cot_output
from ..scoring.entailment import calculate_entailment_score
from ..scoring.overlap import calculate_overlap_score

def generate_sea_cot(model: LLMInterface, question: str, choices: List[str], dataset_type: str, n_paths: int = 5) -> Tuple[str, str]:
    """Generates SEA-CoT explanation and answer."""
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
            
    # 1. Generate N reasoning paths
    outputs = model.generate(prompt, temperature=0.7, num_return_sequences=n_paths)
    
    paths = []
    answers = []
    for out in outputs:
        reasoning, ans = parse_cot_output(out, choices)
        paths.append(reasoning)
        answers.append(ans)
        
    # 2. Determine majority answer
    ans_counts = Counter(answers)
    majority_answer = ans_counts.most_common(1)[0][0]
    
    # 3. For each candidate supporting majority answer, calculate scores
    best_reasoning = ""
    best_score = -1.0
    
    hypothesis = f"{question} {majority_answer}"
    
    for path, ans in zip(paths, answers):
        if ans == majority_answer:
            # 3a. Entailment score S_e
            s_e = calculate_entailment_score(path, hypothesis)
            
            # 3b. Overlap score S_o
            s_o = calculate_overlap_score(path, hypothesis)
            
            # 3c. Total score
            s_t = s_e + s_o
            
            if s_t > best_score:
                best_score = s_t
                best_reasoning = path
                
    # 4. Return best explanation
    return best_reasoning, majority_answer
