"""
Chain-of-Thought prompting.
"""
from typing import List, Tuple
from ..model import LLMInterface

STRATEGY_QA_TEMPLATE = """Answer the following question by reasoning step-by-step.
Question: {question}
Answer Choices: {choices}
Let's think step by step.
"""

OPENBOOK_QA_TEMPLATE = """Answer the following question by reasoning step-by-step.
Question: {question}
Answer Choices: (A) {choice_a} (B) {choice_b} (C) {choice_c} (D) {choice_d}
Let's think step by step.
"""

def generate_cot(model: LLMInterface, question: str, choices: List[str], dataset_type: str) -> Tuple[str, str]:
    """Generates CoT explanation and answer."""
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
            
    output = model.generate(prompt, temperature=0.0)[0]
    reasoning, answer = parse_cot_output(output, choices)
    return reasoning, answer

def parse_cot_output(output: str, choices: List[str]) -> Tuple[str, str]:
    """Parses CoT output to extract reasoning and answer."""
    answer = ""
    for choice in choices:
        if choice.lower() in output.lower():
            answer = choice
            break
            
    reasoning = output
    return reasoning, answer
