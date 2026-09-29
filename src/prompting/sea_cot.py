"""
Self-Entailment-Alignment Chain-of-Thought.
"""
from collections import Counter
from typing import List, Tuple
from ..model import LLMInterface
from .cot import STRATEGY_QA_TEMPLATE, OPENBOOK_QA_TEMPLATE, parse_cot_output
from ..scoring.entailment import calculate_entailment_score
from ..scoring.overlap import calculate_overlap_score

def generate_sea_cot(model: LLMInterface, question: str, choices: List[str], dataset_type: str = 'strategyqa', n_paths: int = 5, strategy: str = 'oe') -> Tuple[str, str]:
    """Generates SEA-CoT explanation and answer using specified selection strategy."""
    paths = []
    answers = []
    log_probs = []
    
    # 1. Generate N reasoning paths
    for _ in range(n_paths):
        if hasattr(model, 'generate_chat'):
            text, avg_log_prob = model.generate_chat(question, max_new_tokens=120)
            ans = parse_answer(text)
            paths.append(text)
            answers.append(ans)
            log_probs.append(avg_log_prob)
        else:
            prompt = STRATEGY_QA_TEMPLATE.format(question=question, choices=str(choices))
            out = model.generate(prompt, temperature=0.7)[0]
            reasoning, ans = parse_cot_output(out, choices)
            paths.append(reasoning)
            answers.append(ans)
            log_probs.append(0.0)
            
    # 2. Determine majority answer
    ans_counts = Counter(answers)
    majority_answer = ans_counts.most_common(1)[0][0]
    
    # 3. Filter candidates supporting majority answer
    supporting = []
    for i, (path, ans) in enumerate(zip(paths, answers)):
        if ans == majority_answer:
            supporting.append((path, log_probs[i]))
            
    if not supporting:
        supporting = list(zip(paths, log_probs))
        
    hypothesis = f"Question: {question} Answer: {majority_answer}"
    
    # 4. Select according to strategy
    if strategy == 'random':
        best_reasoning = supporting[0][0]
    elif strategy == 'max':
        best_reasoning = max(supporting, key=lambda x: x[1])[0]
    elif strategy == 'overlap':
        best_reasoning = max(supporting, key=lambda x: calculate_overlap_score(x[0], hypothesis))[0]
    elif strategy == 'entailment':
        best_reasoning = max(supporting, key=lambda x: calculate_entailment_score(x[0], hypothesis))[0]
    else:  # 'oe' / SEA-CoT
        best_reasoning = max(
            supporting,
            key=lambda x: calculate_entailment_score(x[0], hypothesis) + calculate_overlap_score(x[0], hypothesis)
        )[0]
        
    return best_reasoning, majority_answer

