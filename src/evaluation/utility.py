"""
Utility evaluation using Leakage-Adjusted Simulatability (LAS).
"""
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from typing import List, Dict

class T5Student:
    def __init__(self, model_name: str = 'google-t5/t5-base'):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def predict(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=50)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

def evaluate_las(dataset_train: List[Dict], dataset_test: List[Dict], explanations: List[str]) -> float:
    """
    Evaluates Leakage-Adjusted Simulatability (LAS).
    LAS = P(correct with explanation) - P(correct without explanation)
    """
    student = T5Student()
    
    las_sum = 0
    total = len(dataset_test)
    
    for i, item in enumerate(dataset_test):
        question = item['question']
        choices = item['choices']
        answer = item['answer']
        explanation = explanations[i]
        
        prompt_without = f"Question: {question} Choices: {choices} Answer:"
        prompt_with = f"Explanation: {explanation} Question: {question} Choices: {choices} Answer:"
        
        pred_without = student.predict(prompt_without)
        pred_with = student.predict(prompt_with)
        
        correct_without = answer.lower() in pred_without.lower()
        correct_with = answer.lower() in pred_with.lower()
        
        if correct_with and not correct_without:
            las_sum += 1
        elif not correct_with and correct_without:
            las_sum -= 1
            
    return las_sum / total if total > 0 else 0.0
