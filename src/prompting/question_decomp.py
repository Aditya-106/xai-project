"""
Question Decomposition prompting.
"""
from typing import List, Tuple
from ..model import LLMInterface

DECOMP_TEMPLATE = """Decompose the following question into simpler sub-questions and answer each to arrive at the final answer.
Question: {question}
Answer Choices: {choices}

Sub-questions:
"""

def generate_question_decomp(model: LLMInterface, question: str, choices: List[str]) -> Tuple[str, str]:
    """Generates Question Decomposition explanation and answer."""
    prompt = DECOMP_TEMPLATE.format(
        question=question,
        choices=str(choices)
    )
            
    output = model.generate(prompt, temperature=0.0)[0]
    reasoning, answer = parse_decomp_output(output, choices)
    return reasoning, answer

def parse_decomp_output(output: str, choices: List[str]) -> Tuple[str, str]:
    """Parses decomp output to extract reasoning and answer."""
    answer = ""
    for choice in choices:
        if choice.lower() in output.lower():
            answer = choice
            break
            
    reasoning = output
    return reasoning, answer
