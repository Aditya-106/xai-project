"""
Self-Refine prompting.
"""
from typing import List, Tuple
from ..model import LLMInterface
from .cot import STRATEGY_QA_TEMPLATE, parse_cot_output

REFINE_TEMPLATE = """Review and refine the following reasoning. Fix any errors and improve clarity.

Question: {question}
Answer Choices: {choices}

Original Reasoning:
{original_reasoning}

Refined Reasoning:
"""

def generate_self_refine(model: LLMInterface, question: str, choices: List[str], dataset_type: str) -> Tuple[str, str]:
    """Generates Self-Refine explanation and answer."""
    # Stage 1: Generate
    if dataset_type == 'strategyqa':
        prompt1 = STRATEGY_QA_TEMPLATE.format(
            question=question,
            choices=str(choices)
        )
    else:
        # Simplification for demo
        prompt1 = STRATEGY_QA_TEMPLATE.format(
            question=question,
            choices=str(choices)
        )
            
    output1 = model.generate(prompt1, temperature=0.0)[0]
    
    # Stage 2: Refine
    prompt2 = REFINE_TEMPLATE.format(
        question=question,
        choices=str(choices),
        original_reasoning=output1
    )
    
    output2 = model.generate(prompt2, temperature=0.0)[0]
    
    reasoning, answer = parse_cot_output(output2, choices)
    return reasoning, answer
